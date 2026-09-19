/*
 * vaist_tokens.c -- Full BPE + SentencePiece + byte tokenizer implementation.
 *
 * Implements the public API declared in vaist_tokens.h. No external libraries
 * beyond the C runtime. All three algorithm families are self-contained:
 *
 *   - VAIST_TOKENIZER_BPE: GPT-2 style byte-level BPE.
 *       * Loads vocab.json  (JSON: {"token-str": id, ...})
 *       * Loads merges.txt  (ranked "left right" pairs, one per line)
 *       * Byte->unicode encoding table (byte 33..126 -> self; rest -> U+0100..).
 *       * Greedy longest-merge BPE over pre-tokenized words.
 *   - VAIST_TOKENIZER_SENTENCEPIECE: SentencePiece (unigram / BPE-word).
 *       * Parses the .model protobuf (wire format) for the pieces list.
 *       * Reads TrainerSpec.model_type ("unigram" | "bpe" | "char" | "word").
 *       * Greedy longest-match decode for unigram; sampling via lattice Viterbi.
 *       * Special pieces (<s>, </s>, <unk>, <pad>) are detected by name; the
 *         SentencePiece "type" field (CONTROL=3, UNKNOWN=2, USER_DEFINED=4).
 *   - VAIST_TOKENIZER_BYTES: one token per byte, id = byte + offset.
 *
 * File I/O follows the vaist_model.c pattern (fopen/fread/fseek/ftell/fclose).
 * Heap allocations use calloc/free; every create has a matching destroy.
 */
#include "vaist_tokens.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

/* =========================================================================
 *   Shared helpers
 * ========================================================================= */

static char *read_file(const char *path, size_t *out_len) {
    FILE *f;
    long n;
    char *buf;
    size_t got;
    if (!path || !out_len) { return NULL; }
    f = fopen(path, "rb");
    if (!f) { return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    buf = (char *)malloc((size_t)n + 1u);
    if (!buf) { fclose(f); return NULL; }
    got = fread(buf, 1u, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[(size_t)n] = '\0';
    *out_len = (size_t)n;
    return buf;
}

/* =========================================================================
 *   Byte->unicode (GPT-2) table
 * =========================================================================
 * GPT-2 byte-level BPE maps each of the 256 bytes to a single Unicode
 * codepoint so that tokenization never produces control characters.
 *   - printable ASCII (0x21..0x7E) plus 0x20 (space) map to themselves.
 *   - bytes 0x00..0x1F, 0x7F..0xA0 map to U+0100 + <rank>.
 *   - bytes 0xA1..0xFF are kept as-is (printable).
 * This is the canonical llama.cpp / HuggingFace mapping.
 */

static uint16_t g_byte_to_uni[VAIST_BYTE_MAP_SIZE];
static uint16_t g_uni_to_byte[256 * 2]; /* reverse: unicode codepoint -> byte */
static int g_byte_table_ready = 0;

static void init_byte_tables(void) {
    size_t i;
    uint16_t uni;
    if (g_byte_table_ready) { return; }
    uni = 256u; /* first remapped slot */
    for (i = 0; i < VAIST_BYTE_MAP_SIZE; i++) {
        if (i >= 33 && i <= 126) {              /* printable ASCII */
            g_byte_to_uni[i] = (uint16_t)i;
        } else if (i >= 161 && i <= 255) {      /* 0xA1..0xFF kept as-is */
            g_byte_to_uni[i] = (uint16_t)i;
        } else {                                /* 0..32, 127..160 -> high slots */
            g_byte_to_uni[i] = uni++;
        }
    }
    /* Build reverse map. */
    for (i = 0; i < sizeof(g_uni_to_byte) / sizeof(g_uni_to_byte[0]); i++) {
        g_uni_to_byte[i] = 0xFFFFu;
    }
    for (i = 0; i < VAIST_BYTE_MAP_SIZE; i++) {
        uint16_t u = g_byte_to_uni[i];
        if (u < (sizeof(g_uni_to_byte) / sizeof(g_uni_to_byte[0]))) {
            g_uni_to_byte[u] = (uint16_t)i;
        }
    }
    g_byte_table_ready = 1;
}

/* Encode raw bytes into the unicode-mapped char sequence used internally by
 * BPE. Returns number of unicode chars written (<= cap). One uint16_t per byte. */
static size_t bytes_to_unicode_str(const char *bytes, size_t len,
                                   uint16_t *out, size_t cap) {
    size_t i, n = 0;
    init_byte_tables();
    for (i = 0; i < len && n < cap; i++) {
        out[n++] = g_byte_to_uni[(uint8_t)bytes[i]];
    }
    return n;
}

/* Decode a unicode-mapped char sequence back to raw bytes. */
static size_t unicode_str_to_bytes(const uint16_t *chars, size_t len,
                                   char *out, size_t cap) {
    size_t i, n = 0;
    init_byte_tables();
    for (i = 0; i < len && n < cap; i++) {
        uint16_t u = chars[i];
        if (u < (sizeof(g_uni_to_byte) / sizeof(g_uni_to_byte[0]))) {
            uint16_t b = g_uni_to_byte[u];
            if (b == 0xFFFFu) { continue; } /* unmapped: skip */
            out[n++] = (char)b;
        }
    }
    return n;
}

/* =========================================================================
 *   Minimal JSON parser (objects with string keys -> string or int values)
 * ========================================================================= */

typedef struct json_val {
    char *key;       /* NUL-terminated key (object member) */
    char *str;       /* string value (for string members)  */
    int64_t num;     /* numeric value if parsed as number  */
    int is_num;      /* 1 if num is valid                    */
    struct json_val *next; /* sibling linked list */
} json_val;

static const char *skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) { p++; }
    return p;
}

static char *parse_string(const char **pp) {
    const char *p;
    char *out;
    size_t cap = 16, len = 0;
    if (!pp || !*pp) { return NULL; }
    p = skip_ws(*pp);
    if (*p != '"') { return NULL; }
    p++;
    out = (char *)malloc(cap);
    if (!out) { return NULL; }
    while (*p) {
        char c = *p;
        if (c == '"') {
            p++;
            out[len] = '\0';
            *pp = p;
            return out;
        }
        if (c == '\\') {
            p++;
            switch (*p) {
                case '"':  c = '"'; break;
                case '\\': c = '\\'; break;
                case '/':  c = '/'; break;
                case 'n':  c = '\n'; break;
                case 't':  c = '\t'; break;
                case 'r':  c = '\r'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'u': {
                    unsigned int cp = 0;
                    int d;
                    for (d = 0; d < 4 && p[1 + d]; d++) {
                        char h = p[1 + d];
                        int v;
                        if (h >= '0' && h <= '9') { v = h - '0'; }
                        else if (h >= 'a' && h <= 'f') { v = h - 'a' + 10; }
                        else if (h >= 'A' && h <= 'F') { v = h - 'A' + 10; }
                        else { break; }
                        cp = (cp << 4) | (unsigned int)v;
                    }
                    if (d != 4) { free(out); return NULL; }
                    p += 4;
                    if (cp < 0x80) {
                        if (len + 1 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
                        out[len++] = (char)cp;
                    } else if (cp < 0x800) {
                        if (len + 2 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
                        out[len++] = (char)(0xC0 | (cp >> 6));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        if (len + 3 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
                        out[len++] = (char)(0xE0 | (cp >> 12));
                        out[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default:
                    free(out);
                    return NULL;
            }
        } else {
            if (len + 1 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
            out[len++] = c;
        }
        p++;
    }
    free(out);
    return NULL; /* unterminated */
}

static json_val *parse_object(const char **pp) {
    const char *p;
    json_val *head = NULL, *tail = NULL;
    char *key;
    if (!pp || !*pp) { return NULL; }
    p = skip_ws(*pp);
    if (*p != '{') { return NULL; }
    p++;
    p = skip_ws(p);
    if (*p == '}') { p++; *pp = p; return NULL; } /* empty object */
    while (*p) {
        json_val *m = NULL;
        p = skip_ws(p);
        key = parse_string(&p);
        if (!key) { goto fail; }
        p = skip_ws(p);
        if (*p != ':') { free(key); goto fail; }
        p++;
        p = skip_ws(p);
        m = (json_val *)calloc(1, sizeof(json_val));
        if (!m) { free(key); goto fail; }
        m->key = key;
        if (*p == '"') {
            char *v = parse_string(&p);
            if (!v) { free(key); free(m); goto fail; }
            m->str = v;
        } else {
            char *e = NULL;
            long v = strtol(p, &e, 10);
            if (e == p) { free(key); free(m); goto fail; }
            m->num = (int64_t)v;
            m->is_num = 1;
            p = e;
        }
        if (!head) { head = m; tail = m; }
        else { tail->next = m; tail = m; }
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        goto fail;
    }
    *pp = p;
    return head;
fail:
    {
        json_val *c = head;
        while (c) { json_val *n = c->next; free(c->key); free(c->str); free(c); c = n; }
    }
    return NULL;
}

static void json_val_free(json_val *v) {
    json_val *n;
    while (v) {
        n = v->next;
        free(v->key);
        free(v->str);
        free(v);
        v = n;
    }
}

/* =========================================================================
 *   Protobuf minimal wire-format decoder (SentencePiece ModelProto)
 * =========================================================================
 * ModelProto { 1:repeated SentencePiece pieces; 2:TrainerSpec; 3/5 normalizer;
 *   16:model_version }.  SentencePiece { 1:piece(string); 3:score(float);
 *   4:type(uint32) }.  TrainerSpec { 3:model_type(string) }.
 */

typedef struct sp_piece {
    char    *piece;
    size_t   piece_len;
    float    score;
    uint32_t type;
    uint32_t id;
} sp_piece;

typedef struct sp_model {
    char    *model_type;
    size_t   piece_count;
    size_t   piece_cap;
    sp_piece *pieces;
    int32_t  bos_id;
    int32_t  eos_id;
    int32_t  unk_id;
    int32_t  pad_id;
    uint32_t model_version;
    uint32_t max_length;
} sp_model;

static int pb_read_varint(const uint8_t *p, size_t cap, size_t *off, uint64_t *out) {
    uint64_t v = 0;
    size_t i = *off;
    uint32_t shift = 0;
    while (i < cap) {
        uint8_t b = p[i++];
        v |= ((uint64_t)(b & 0x7F)) << shift;
        if ((b & 0x80) == 0) {
            *off = i;
            *out = v;
            return 1;
        }
        shift += 7;
        if (shift >= 64) { return 0; }
    }
    return 0;
}

static int pb_read_tag(const uint8_t *p, size_t cap, size_t *off,
                       uint32_t *field, uint32_t *wt) {
    uint64_t v;
    if (!pb_read_varint(p, cap, off, &v)) { return 0; }
    *field = (uint32_t)(v >> 3);
    *wt = (uint32_t)(v & 7u);
    return 1;
}

static int pb_skip_value(const uint8_t *p, size_t cap, size_t *off, uint32_t wt) {
    switch (wt) {
        case 0: { uint64_t v; return pb_read_varint(p, cap, off, &v); }
        case 1: if (*off + 8 > cap) { return 0; } *off += 8; return 1;
        case 2: {
            uint64_t n;
            if (!pb_read_varint(p, cap, off, &n)) { return 0; }
            if (*off + n > cap) { return 0; }
            *off += (size_t)n;
            return 1;
        }
        case 5: if (*off + 4 > cap) { return 0; } *off += 4; return 1;
        default: return 0;
    }
}

static int sp_load_pieces(const uint8_t *p, size_t cap, sp_model *m) {
    size_t scan;
    size_t idx = 0;
    /* First pass: count top-level field-1 entries. */
    m->piece_count = 0;
    scan = 0;
    while (scan < cap) {
        uint32_t field, wt;
        if (!pb_read_tag(p, cap, &scan, &field, &wt)) { return 0; }
        if (field == 1 && wt == 2) {
            uint64_t n;
            if (!pb_read_varint(p, cap, &scan, &n)) { return 0; }
            if (scan + n > cap) { return 0; }
            scan += (size_t)n;
            m->piece_count++;
        } else {
            if (!pb_skip_value(p, cap, &scan, wt)) { return 0; }
        }
    }
    m->piece_cap = m->piece_count;
    if (m->piece_cap == 0) {
        m->pieces = NULL;
        /* Still parse trainer / version. */
        scan = 0;
        while (scan < cap) {
            uint32_t field, wt;
            if (!pb_read_tag(p, cap, &scan, &field, &wt)) { break; }
            if (field == 2 && wt == 2) { /* trainer_spec */
                uint64_t n; size_t s;
                if (!pb_read_varint(p, cap, &scan, &n)) { break; }
                if (scan + n > cap) { break; }
                s = scan; scan += (size_t)n;
                while (s < scan) {
                    uint32_t tf, twt;
                    if (!pb_read_tag(p, cap, &s, &tf, &twt)) { break; }
                    if (tf == 3 && twt == 2) { /* model_type */
                        uint64_t ln;
                        if (!pb_read_varint(p, cap, &s, &ln)) { break; }
                        if (s + ln > scan) { break; }
                        free(m->model_type);
                        m->model_type = (char *)malloc(ln + 1u);
                        if (m->model_type) {
                            memcpy(m->model_type, p + s, ln);
                            m->model_type[ln] = '\0';
                        }
                        s += (size_t)ln;
                    } else {
                        if (!pb_skip_value(p, cap, &s, twt)) { break; }
                    }
                }
            } else if (field == 16 && wt == 0) {
                uint64_t v;
                if (!pb_read_varint(p, cap, &scan, &v)) { break; }
                m->model_version = (uint32_t)v;
            } else {
                if (!pb_skip_value(p, cap, &scan, wt)) { break; }
            }
        }
        return 1;
    }
    m->pieces = (sp_piece *)calloc(m->piece_cap, sizeof(sp_piece));
    if (!m->pieces) { return 0; }
    /* Second pass: populate. */
    scan = 0;
    while (scan < cap) {
        uint32_t field, wt;
        if (!pb_read_tag(p, cap, &scan, &field, &wt)) { return 0; }
        if (field == 1 && wt == 2) {
            uint64_t n;
            size_t sub_off;
            if (!pb_read_varint(p, cap, &scan, &n)) { return 0; }
            if (scan + n > cap) { return 0; }
            sub_off = scan;
            scan += (size_t)n;
            {
                sp_piece *pp = &m->pieces[idx];
                pp->id = (uint32_t)idx;
                while (sub_off < scan) {
                    uint32_t sf, swt;
                    if (!pb_read_tag(p, cap, &sub_off, &sf, &swt)) { return 0; }
                    if (sf == 1 && swt == 2) { /* piece: string */
                        uint64_t ln;
                        if (!pb_read_varint(p, cap, &sub_off, &ln)) { return 0; }
                        if (sub_off + ln > scan) { return 0; }
                        pp->piece_len = (size_t)ln;
                        pp->piece = (char *)malloc(pp->piece_len + 1u);
                        if (!pp->piece) { return 0; }
                        memcpy(pp->piece, p + sub_off, pp->piece_len);
                        pp->piece[pp->piece_len] = '\0';
                        sub_off += (size_t)ln;
                    } else if (sf == 3 && swt == 5) { /* score: float */
                        uint32_t bits;
                        float fv;
                        if (sub_off + 4 > cap) { return 0; }
                        bits = (uint32_t)p[sub_off] | ((uint32_t)p[sub_off+1]<<8) |
                               ((uint32_t)p[sub_off+2]<<16) | ((uint32_t)p[sub_off+3]<<24);
                        memcpy(&fv, &bits, 4);
                        pp->score = fv;
                        sub_off += 4;
                    } else if (sf == 4 && swt == 0) { /* type: varint */
                        uint64_t v;
                        if (!pb_read_varint(p, cap, &sub_off, &v)) { return 0; }
                        pp->type = (uint32_t)v;
                    } else {
                        if (!pb_skip_value(p, cap, &sub_off, swt)) { return 0; }
                    }
                }
                idx++;
            }
        } else if (field == 2 && wt == 2) { /* trainer_spec */
            uint64_t n;
            size_t s;
            if (!pb_read_varint(p, cap, &scan, &n)) { return 0; }
            if (scan + n > cap) { return 0; }
            s = scan; scan += (size_t)n;
            while (s < scan) {
                uint32_t tf, twt;
                if (!pb_read_tag(p, cap, &s, &tf, &twt)) { break; }
                if (tf == 3 && twt == 2) { /* model_type */
                    uint64_t ln;
                    if (!pb_read_varint(p, cap, &s, &ln)) { break; }
                    if (s + ln > scan) { break; }
                    free(m->model_type);
                    m->model_type = (char *)malloc(ln + 1u);
                    if (m->model_type) {
                        memcpy(m->model_type, p + s, ln);
                        m->model_type[ln] = '\0';
                    }
                    s += (size_t)ln;
                } else {
                    if (!pb_skip_value(p, cap, &s, twt)) { break; }
                }
            }
        } else if (field == 16 && wt == 0) {
            uint64_t v;
            if (!pb_read_varint(p, cap, &scan, &v)) { return 0; }
            m->model_version = (uint32_t)v;
        } else {
            if (!pb_skip_value(p, cap, &scan, wt)) { return 0; }
        }
    }
    return 1;
}

static void sp_identify_specials(sp_model *m) {
    size_t i;
    m->bos_id = -1; m->eos_id = -1; m->unk_id = -1; m->pad_id = -1;
    for (i = 0; i < m->piece_count; i++) {
        sp_piece *pp = &m->pieces[i];
        const char *s = pp->piece;
        if (strcmp(s, "<s>") == 0) { m->bos_id = (int32_t)i; }
        else if (strcmp(s, "</s>") == 0) { m->eos_id = (int32_t)i; }
        else if (strcmp(s, "<unk>") == 0) { m->unk_id = (int32_t)i; }
        else if (strcmp(s, "<pad>") == 0) { m->pad_id = (int32_t)i; }
    }
}

static void sp_model_free(sp_model *m) {
    size_t i;
    if (!m) { return; }
    for (i = 0; i < m->piece_count; i++) { free(m->pieces[i].piece); }
    free(m->pieces);
    free(m->model_type);
    memset(m, 0, sizeof(*m));
}

/* =========================================================================
 *   Hash table for BPE vocab (unicode-mapped string -> token id)
 * ========================================================================= */

typedef struct {
    char    *key;
    size_t   key_len;
    uint32_t id;
    size_t   next; /* index into the buckets' linked list; SIZE_MAX = end */
} bpe_entry;

typedef struct {
    bpe_entry *buckets;
    size_t     bucket_count;
    size_t     count;
    size_t     *chain; /* bucket head index or SIZE_MAX */
} bpe_map;

static size_t hash_bytes(const char *s, size_t len) {
    size_t h = 14695981039346656037ull;
    size_t i;
    for (i = 0; i < len; i++) {
        h ^= (size_t)(uint8_t)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int bpe_map_init(bpe_map *m, size_t cap) {
    size_t i;
    m->buckets = (bpe_entry *)calloc(cap, sizeof(bpe_entry));
    m->chain = (size_t *)malloc(cap * sizeof(size_t));
    if (!m->buckets || !m->chain) { return 0; }
    for (i = 0; i < cap; i++) { m->chain[i] = SIZE_MAX; }
    m->bucket_count = cap;
    m->count = 0;
    return 1;
}

static void bpe_map_free(bpe_map *m) {
    size_t i;
    if (!m) { return; }
    for (i = 0; i < m->count; i++) { free(m->buckets[i].key); }
    free(m->buckets);
    free(m->chain);
    memset(m, 0, sizeof(*m));
}

static int bpe_map_put(bpe_map *m, const char *key, size_t key_len, uint32_t id) {
    size_t h, idx;
    if (m->count >= m->bucket_count) { return -1; } /* caller sizes up */
    h = hash_bytes(key, key_len) % m->bucket_count;
    idx = m->chain[h];
    while (idx != SIZE_MAX) {
        if (m->buckets[idx].key_len == key_len &&
            memcmp(m->buckets[idx].key, key, key_len) == 0) {
            m->buckets[idx].id = id;
            return 0;
        }
        idx = m->buckets[idx].next;
    }
    m->buckets[m->count].key = (char *)malloc(key_len + 1u);
    if (!m->buckets[m->count].key) { return -1; }
    memcpy(m->buckets[m->count].key, key, key_len);
    m->buckets[m->count].key[key_len] = '\0';
    m->buckets[m->count].key_len = key_len;
    m->buckets[m->count].id = id;
    m->buckets[m->count].next = m->chain[h];
    m->chain[h] = m->count;
    m->count++;
    return 1;
}

static int bpe_map_get(const bpe_map *m, const char *key, size_t key_len, uint32_t *out) {
    size_t h, idx;
    if (!m->bucket_count) { return 0; }
    h = hash_bytes(key, key_len) % m->bucket_count;
    idx = m->chain[h];
    while (idx != SIZE_MAX) {
        if (m->buckets[idx].key_len == key_len &&
            memcmp(m->buckets[idx].key, key, key_len) == 0) {
            if (out) { *out = m->buckets[idx].id; }
            return 1;
        }
        idx = m->buckets[idx].next;
    }
    return 0;
}

/* =========================================================================
 *   BPE merge rule list
 * ========================================================================= */

typedef struct {
    uint32_t a;
    uint32_t b;
    uint32_t rank; /* lower rank = higher priority (applied first) */
} bpe_merge;

static int merge_cmp(const void *x, const void *y) {
    const bpe_merge *a = (const bpe_merge *)x;
    const bpe_merge *b = (const bpe_merge *)y;
    if (a->rank < b->rank) { return -1; }
    if (a->rank > b->rank) { return 1; }
    return 0;
}

/* =========================================================================
 *   Internal tokenizer struct (opaque handle)
 * ========================================================================= */

struct VaistTokenizer {
    TokenizerType type;
    /* BPE data: */
    bpe_map       vocab;        /* unicode-token-string -> id            */
    char         **id_str;      /* dense id -> NUL-terminated piece      */
    size_t       *id_str_len;   /* dense id -> length                    */
    uint32_t      vocab_size;
    bpe_merge    *merges;
    size_t        merge_count;
    uint32_t      byte_offset;
    int32_t       bos_id;
    int32_t       eos_id;
    int32_t       unk_id;
    int32_t       pad_id;
    uint32_t      max_length;
    /* SentencePiece data: */
    sp_model      sp;
};

static VaistTokenizer *tok_alloc(TokenizerType type) {
    VaistTokenizer *t = (VaistTokenizer *)calloc(1, sizeof(VaistTokenizer));
    if (!t) { return NULL; }
    t->type = type;
    t->bos_id = -1; t->eos_id = -1; t->unk_id = -1; t->pad_id = -1;
    t->byte_offset = 0;
    t->max_length = 0;
    if (type == VAIST_TOKENIZER_BYTES) {
        t->vocab_size = 256u;
    }
    return t;
}

static void tok_free(VaistTokenizer *t) {
    size_t i;
    if (!t) { return; }
    switch (t->type) {
        case VAIST_TOKENIZER_BPE:
            bpe_map_free(&t->vocab);
            if (t->id_str) {
                for (i = 0; i < t->vocab_size; i++) { free(t->id_str[i]); }
                free(t->id_str);
            }
            free(t->id_str_len);
            free(t->merges);
            break;
        case VAIST_TOKENIZER_SENTENCEPIECE:
            sp_model_free(&t->sp);
            break;
        case VAIST_TOKENIZER_BYTES:
            break;
        default:
            break;
    }
    free(t);
}

static char *path_join(const char *dir, const char *name) {
    size_t dl = strlen(dir), nl = strlen(name);
    char *out = (char *)malloc(dl + 1 + nl + 1);
    if (!out) { return NULL; }
    memcpy(out, dir, dl);
    out[dl] = '/';
    memcpy(out + dl + 1, name, nl);
    out[dl + 1 + nl] = '\0';
    return out;
}

/* =========================================================================
 *   vocab.json loading (BPE)
 * ========================================================================= */

static VaistStatus load_bpe_vocab(VaistTokenizer *t, const char *json) {
    const char *p = json;
    json_val *v;
    size_t n = 0, idx = 0, i;
    char **id_str;
    size_t *id_len_arr;
    uint32_t max_id = 0;
    json_val *c;
    if (!json) { return VAIST_IO_ERROR; }
    v = parse_object(&p);
    if (!v) { return VAIST_MODEL_ERROR; }
    /* Count numeric entries. */
    for (c = v; c; c = c->next) { if (c->is_num) { n++; } }
    /* First pass: collect records {str,len,id}. */
    {
        typedef struct { char *str; size_t len; uint32_t id; } rec;
        rec *recs = (rec *)calloc(n ? n : 1, sizeof(rec));
        if (!recs) { json_val_free(v); return VAIST_OUT_OF_MEMORY; }
        for (c = v, idx = 0; c; c = c->next) {
            if (!c->is_num) {
                /* Special tokens: capture by name. */
                if (c->key && c->str) {
                    if (strcmp(c->key, "<|bos|>") == 0 || strcmp(c->key, "bos") == 0) { t->bos_id = (int32_t)c->num; }
                    else if (strcmp(c->key, "<|eos|>") == 0 || strcmp(c->key, "eos") == 0) { t->eos_id = (int32_t)c->num; }
                    else if (strcmp(c->key, "<|unk|>") == 0 || strcmp(c->key, "unk") == 0) { t->unk_id = (int32_t)c->num; }
                    else if (strcmp(c->key, "<|pad|>") == 0 || strcmp(c->key, "pad") == 0) { t->pad_id = (int32_t)c->num; }
                }
                continue;
            }
            {
                uint32_t id = (uint32_t)c->num;
                size_t klen = strlen(c->key);
                char *copy = (char *)malloc(klen + 1u);
                if (!copy) {
                    size_t j;
                    for (j = 0; j < idx; j++) { free(recs[j].str); }
                    free(recs); json_val_free(v);
                    return VAIST_OUT_OF_MEMORY;
                }
                memcpy(copy, c->key, klen + 1u);
                recs[idx].str = copy;
                recs[idx].len = klen;
                recs[idx].id = id;
                if (id > max_id) { max_id = id; }
                idx++;
            }
        }
        n = idx;
        /* Build dense id tables. */
        {
            size_t dense = (size_t)max_id + 1u;
            size_t alloc_cap = dense + (dense / 2) + 16;
            id_str = (char **)calloc(dense, sizeof(char *));
            id_len_arr = (size_t *)calloc(dense, sizeof(size_t));
            if (!id_str || !id_len_arr) {
                size_t j;
                for (j = 0; j < n; j++) { free(recs[j].str); }
                free(recs); free(id_str); free(id_len_arr); json_val_free(v);
                return VAIST_OUT_OF_MEMORY;
            }
            if (!bpe_map_init(&t->vocab, alloc_cap)) {
                size_t j;
                for (j = 0; j < n; j++) { free(recs[j].str); }
                free(recs); free(id_str); free(id_len_arr); json_val_free(v);
                return VAIST_OUT_OF_MEMORY;
            }
            for (i = 0; i < n; i++) {
                uint32_t id = recs[i].id;
                int r;
                r = bpe_map_put(&t->vocab, recs[i].str, recs[i].len, id);
                (void)r; /* ignore collision (dup keys) */
                if (id < dense) {
                    id_str[id] = recs[i].str;       /* take ownership */
                    id_len_arr[id] = recs[i].len;
                    recs[i].str = NULL;             /* prevent double-free */
                }
            }
            t->id_str = id_str;
            t->id_str_len = id_len_arr;
            t->vocab_size = max_id + 1u;
            free(recs);
        }
    }
    json_val_free(v);
    return VAIST_OK;
}

/* =========================================================================
 *   merges.txt loading (BPE)
 * ========================================================================= */

static VaistStatus load_bpe_merges(VaistTokenizer *t, const char *txt) {
    const char *p = txt;
    size_t rank = 0;
    size_t cap = 64, count = 0;
    bpe_merge *m = (bpe_merge *)malloc(cap * sizeof(bpe_merge));
    if (!m) { return VAIST_OUT_OF_MEMORY; }
    t->merges = NULL;
    t->merge_count = 0;
    while (*p) {
        /* Skip comments and blank lines. */
        if (*p == '#') {
            while (*p && *p != '\n') { p++; }
            continue;
        }
        if (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') {
            p++;
            continue;
        }
        {
            const char *line = p;
            const char *nl = line;
            const char *q;
            const char *s;
            char toks[2][VAIST_BYTE_MAP_SIZE * 4];
            size_t ti, k;
            uint32_t ids[2];
            /* find line end */
            while (*nl && *nl != '\n' && *nl != '\r') { nl++; }
            /* Token 1 */
            q = line;
            while (q < nl && (*q == ' ' || *q == '\t')) { q++; }
            s = q;
            while (q < nl && *q != ' ' && *q != '\t') { q++; }
            k = (size_t)(q - s);
            if (k == 0 || k >= sizeof(toks[0])) { p = (*nl) ? nl + 1 : nl; continue; }
            memcpy(toks[0], s, k); toks[0][k] = '\0';
            /* Token 2 */
            while (q < nl && (*q == ' ' || *q == '\t')) { q++; }
            s = q;
            while (q < nl && *q != ' ' && *q != '\t') { q++; }
            k = (size_t)(q - s);
            if (k == 0 || k >= sizeof(toks[1])) { p = (*nl) ? nl + 1 : nl; continue; }
            memcpy(toks[1], s, k); toks[1][k] = '\0';
            if (!bpe_map_get(&t->vocab, toks[0], strlen(toks[0]), &ids[0])) {
                p = (*nl) ? nl + 1 : nl; continue;
            }
            if (!bpe_map_get(&t->vocab, toks[1], strlen(toks[1]), &ids[1])) {
                p = (*nl) ? nl + 1 : nl; continue;
            }
            if (count + 1 > cap) {
                size_t ncap = cap * 2;
                bpe_merge *nm = (bpe_merge *)realloc(m, ncap * sizeof(bpe_merge));
                if (!nm) { free(m); return VAIST_OUT_OF_MEMORY; }
                m = nm; cap = ncap;
            }
            m[count].a = ids[0];
            m[count].b = ids[1];
            m[count].rank = (uint32_t)rank;
            count++;
            rank++;
            p = (*nl) ? nl + 1 : nl;
            (void)ti;
        }
    }
    t->merges = m;
    t->merge_count = count;
    if (count > 1) {
        qsort(t->merges, count, sizeof(bpe_merge), merge_cmp);
    }
    return VAIST_OK;
}

/* =========================================================================
 *   BPE tokenization (greedy longest-merge)
 * ========================================================================= */

static uint32_t bpe_merge_rank(const bpe_merge *m, size_t n, uint32_t a, uint32_t b) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (m[i].a == a && m[i].b == b) { return m[i].rank; }
    }
    return UINT32_MAX;
}

static void bpe_encode_word(const uint16_t *chars, size_t nchars,
                            VaistTokenizer *t,
                            uint32_t *out, size_t *out_count, size_t cap) {
    typedef struct { char *s; size_t len; uint32_t id; } sym;
    sym *syms = NULL;
    size_t sym_cap = 0, sym_n = 0;
    size_t o = 0;
    size_t i;
    /* Build initial symbols: one per unicode char, each encoded as UTF-8 of that
     * codepoint, looked up in vocab to get its id. */
    sym_cap = nchars + 16;
    syms = (sym *)malloc(sym_cap * sizeof(sym));
    if (!syms) { *out_count = 0; return; }
    for (i = 0; i < nchars; i++) {
        uint16_t u = chars[i];
        char buf[4];
        size_t bl;
        uint32_t id;
        if (u < 0x80) { buf[0] = (char)u; bl = 1; }
        else if (u < 0x800) { buf[0] = (char)(0xC0 | (u >> 6)); buf[1] = (char)(0x80 | (u & 0x3F)); bl = 2; }
        else { buf[0] = (char)(0xE0 | (u >> 12)); buf[1] = (char)(0x80 | ((u >> 6) & 0x3F)); buf[2] = (char)(0x80 | (u & 0x3F)); bl = 3; }
        if (sym_n + 1 > sym_cap) {
            size_t nc = sym_cap * 2;
            sym *ns = (sym *)realloc(syms, nc * sizeof(sym));
            if (!ns) { goto done; }
            syms = ns; sym_cap = nc;
        }
        if (bpe_map_get(&t->vocab, buf, bl, &id)) {
            char *cstr = (char *)malloc(bl + 1u);
            if (!cstr) { goto done; }
            memcpy(cstr, buf, bl); cstr[bl] = '\0';
            syms[sym_n].s = cstr; syms[sym_n].len = bl; syms[sym_n].id = id;
        } else {
            /* Unknown char -> use unk or id -1. */
            syms[sym_n].s = NULL; syms[sym_n].len = 0; syms[sym_n].id = (uint32_t)-1;
        }
        sym_n++;
    }
    /* Merge loop: repeatedly find the lowest-rank merge among adjacent pairs. */
    for (;;) {
        size_t best = SIZE_MAX;
        uint32_t best_rank = UINT32_MAX;
        size_t j;
        if (sym_n < 2) { break; }
        for (j = 0; j + 1 < sym_n; j++) {
            uint32_t a = syms[j].id, b = syms[j + 1].id;
            uint32_t r = bpe_merge_rank(t->merges, t->merge_count, a, b);
            if (r < best_rank) { best_rank = r; best = j; }
        }
        if (best == SIZE_MAX) { break; }
        /* Merge syms[best] and syms[best+1]. */
        {
            size_t alen = syms[best].len, blen = syms[best + 1].len;
            size_t nlen = alen + blen;
            char *merged = (char *)malloc(nlen + 1u);
            uint32_t mid;
            if (!merged) { break; }
            memcpy(merged, syms[best].s, alen);
            memcpy(merged + alen, syms[best + 1].s, blen);
            merged[nlen] = '\0';
            if (bpe_map_get(&t->vocab, merged, nlen, &mid)) {
                free(merged);
                free(syms[best].s);
                free(syms[best + 1].s);
                syms[best].s = NULL; syms[best].len = 0; syms[best].id = mid;
                memmove(syms + best + 1, syms + best + 2, (sym_n - best - 2) * sizeof(sym));
                sym_n--;
            } else {
                free(merged);
                break; /* no vocab entry for combined pair */
            }
        }
    }
    /* Emit final ids. */
    for (i = 0; i < sym_n; i++) {
        if (syms[i].id != (uint32_t)-1) {
            if (o < cap) { out[o++] = syms[i].id; }
        }
    }
    *out_count = o;
done:
    /* cleanup */
    for (i = 0; i < sym_n; i++) { free(syms[i].s); }
    free(syms);
}

/* Pre-tokenize: split on whitespace runs (simplified GPT-2 regex). */
static size_t pretokenize(const char *text, char ***words_out) {
    char **w = (char **)malloc(16 * sizeof(char *));
    size_t cap = 16, n = 0;
    const char *p;
    if (!w) { *words_out = NULL; return 0; }
    *words_out = w;
    p = text;
    while (*p) {
        size_t start, blen;
        char *buf;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) { p++; }
        if (!*p) { break; }
        start = (size_t)(p - text);
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { p++; }
        blen = (size_t)(p - text) - start;
        buf = (char *)malloc(blen + 1u);
        if (!buf) { break; }
        memcpy(buf, text + start, blen); buf[blen] = '\0';
        if (n + 1 > cap) {
            size_t nc = cap * 2;
            char **nw = (char **)realloc(w, nc * sizeof(char *));
            if (!nw) { free(buf); break; }
            w = nw; cap = nc; *words_out = w;
        }
        w[n++] = buf;
    }
    return n;
}

/* =========================================================================
 *   SentencePiece greedy longest-match tokenization
 * ========================================================================= */

/* Build a SentencePiece-normalized copy of text: split on whitespace and
 * prepend the word-boundary marker U+2581 (UTF-8 0xE2 0x96 0x81) to the first
 * byte of each word, matching SentencePiece's encode pre-normalization. The
 * result is written to out (cap incl. NUL). Returns the length (excl. NUL),
 * or SIZE_MAX on alloc failure. */
static size_t sp_normalize(const char *text, char *out, size_t cap) {
    size_t o = 0;
    const char *p = text;
    static const char *mk = "\xe2\x96\x81"; /* U+2581 "▁" */
    while (*p) {
        /* skip leading whitespace */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) { p++; }
        if (!*p) { break; }
        /* prepend marker for the next word */
        if (o + 3 + 1 > cap) { return SIZE_MAX; }
        out[o++] = mk[0]; out[o++] = mk[1]; out[o++] = mk[2];
        /* copy the word */
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
            if (o + 1 > cap) { return SIZE_MAX; }
            out[o++] = *p;
            p++;
        }
    }
    if (o + 1 > cap) { return SIZE_MAX; }
    out[o] = '\0';
    return o;
}

static void sp_greedy_tokenize(const sp_model *m, const char *text, size_t text_len,
                               int32_t unk_id, uint32_t *out, size_t cap, size_t *count) {
    size_t pos = 0, o = 0;
    while (pos < text_len) {
        size_t best = 0, best_len = 0;
        size_t i;
        int found = 0;
        for (i = 0; i < m->piece_count; i++) {
            const sp_piece *pp = &m->pieces[i];
            if (pp->type == 3) { continue; } /* skip control */
            if (pp->piece_len == 0) { continue; }
            if (pp->piece_len > text_len - pos) { continue; }
            if (memcmp(pp->piece, text + pos, pp->piece_len) == 0) {
                if (pp->piece_len > best_len) { best_len = pp->piece_len; best = i; found = 1; }
            }
        }
        if (found && best_len > 0) {
            if (o < cap) { out[o++] = (uint32_t)best; }
            pos += best_len;
        } else {
            if (unk_id >= 0) {
                if (o < cap) { out[o++] = (uint32_t)unk_id; }
            } else {
                if (o < cap) { out[o++] = 0; } /* fallback unk id */
            }
            pos++;
        }
    }
    *count = o;
}

/* =========================================================================
 *   SentencePiece unigram lattice sampling
 * ========================================================================= */

static uint64_t sp_rng(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static VaistStatus sp_sample_tokenize(VaistTokenizer *t,
                                      const char *norm,
                                      size_t text_len,
                                      uint64_t seed,
                                      uint32_t nbest,
                                      uint32_t *tokens,
                                      size_t cap,
                                      size_t *count) {
    const sp_model *m = &t->sp;
    size_t i;
    typedef struct {
        size_t pos;
        float score;
        uint32_t *toks;
        size_t ntok;
        size_t cap;
    } beam_state;
    beam_state *beam = NULL, *next_beam = NULL;
    size_t beam_len = 0, beam_cap = 0;
    nbest = nbest ? nbest : 1;
    if (nbest > 64) { nbest = 64; }
    beam_cap = nbest + 2;
    beam = (beam_state *)calloc(beam_cap, sizeof(beam_state));
    if (!beam) { return VAIST_OUT_OF_MEMORY; }
    beam[0].pos = 0; beam[0].score = 0.0f; beam[0].toks = NULL; beam[0].ntok = 0; beam[0].cap = 0;
    beam_len = 1;
    for (i = 0; i < text_len; i++) {
        size_t ccap = nbest * (m->piece_count + 1) + 4;
        size_t nlen = 0;
        size_t b;
        next_beam = (beam_state *)calloc(ccap, sizeof(beam_state));
        if (!next_beam) {
            for (b = 0; b < beam_len; b++) { free(beam[b].toks); }
            free(beam);
            return VAIST_OUT_OF_MEMORY;
        }
        for (b = 0; b < beam_len; b++) {
            beam_state *st = &beam[b];
            size_t pos = st->pos;
            size_t j;
            if (pos >= text_len) {
                /* Completed path: carry forward unmodified (transfer toks ownership). */
                if (nlen < ccap) {
                    next_beam[nlen] = *st;
                    st->toks = NULL; /* prevent double-free in old-beam cleanup */
                    nlen++;
                }
                continue;
            }
            for (j = 0; j < m->piece_count; j++) {
                const sp_piece *pp = &m->pieces[j];
                if (pp->type == 3) { continue; }
                if (pp->piece_len == 0) { continue; }
                if (pp->piece_len > text_len - pos) { continue; }
                if (memcmp(pp->piece, norm + pos, pp->piece_len) == 0) {
                    float ns = st->score + pp->score;
                    size_t need = st->ntok + 1;
                    beam_state *nb = &next_beam[nlen];
                    nb->pos = pos + pp->piece_len;
                    nb->score = ns;
                    nb->cap = need;
                    nb->toks = (uint32_t *)malloc(need * sizeof(uint32_t));
                    if (!nb->toks) {
                        size_t k;
                        for (k = 0; k < nlen; k++) { free(next_beam[k].toks); }
                        free(next_beam);
                        for (k = 0; k < beam_len; k++) { free(beam[k].toks); }
                        free(beam);
                        return VAIST_OUT_OF_MEMORY;
                    }
                    memcpy(nb->toks, st->toks, st->ntok * sizeof(uint32_t));
                    nb->toks[st->ntok] = (uint32_t)j;
                    nb->ntok = need;
                    nlen++;
                    if (nlen >= ccap) { break; }
                }
            }
        }
        if (nlen > nbest) {
            size_t k;
            for (k = 0; k < nbest; k++) {
                size_t best = k, j2;
                for (j2 = k + 1; j2 < nlen; j2++) {
                    if (next_beam[j2].score > next_beam[best].score) { best = j2; }
                }
                if (best != k) {
                    beam_state tmp = next_beam[k]; next_beam[k] = next_beam[best]; next_beam[best] = tmp;
                }
            }
            for (k = nbest; k < nlen; k++) { free(next_beam[k].toks); }
            nlen = nbest;
        }
        {
            size_t k;
            for (k = 0; k < beam_len; k++) { free(beam[k].toks); }
            for (k = 0; k < nlen; k++) { beam[k] = next_beam[k]; }
            for (k = nlen; k < beam_cap; k++) { beam[k].pos = 0; beam[k].toks = NULL; beam[k].ntok = 0; }
            beam_len = nlen;
        }
        free(next_beam);
        next_beam = NULL;
        /* Exit when every surviving path has consumed the full text. */
        {
            size_t done = 0, k;
            for (k = 0; k < beam_len; k++) { if (beam[k].pos >= text_len) { done++; } }
            if (done == beam_len && beam_len > 0) { break; }
            if (beam_len == 0) {
                free(beam);
                return VAIST_MODEL_ERROR;
            }
        }
    }
    /* Sample one path proportionally to exp(score). */
    {
        float best = beam[0].score;
        size_t b;
        for (b = 1; b < beam_len; b++) { if (beam[b].score > best) { best = beam[b].score; } }
        {
            float *prob = (float *)calloc(beam_len, sizeof(float));
            float total = 0;
            uint64_t r;
            size_t pick;
            if (!prob) {
                for (b = 0; b < beam_len; b++) { free(beam[b].toks); }
                free(beam);
                return VAIST_OUT_OF_MEMORY;
            }
            for (b = 0; b < beam_len; b++) {
                prob[b] = expf(beam[b].score - best);
                total += prob[b];
            }
            if (total <= 0.0f) {
                pick = 0;
            } else {
                r = sp_rng(&seed);
                {
                    float thr = ((float)(r & 0xFFFFFFull) / 16777216.0f) * total;
                    float acc = 0;
                    pick = beam_len;
                    for (b = 0; b < beam_len; b++) {
                        acc += prob[b];
                        if (acc >= thr) { pick = b; break; }
                    }
                    if (pick >= beam_len) { pick = beam_len - 1; }
                }
            }
            free(prob);
            {
                size_t nn = beam[pick].ntok;
                if (nn > cap) {
                    *count = nn;
                    for (b = 0; b < beam_len; b++) { free(beam[b].toks); }
                    free(beam);
                    return VAIST_OUT_OF_MEMORY;
                }
                memcpy(tokens, beam[pick].toks, nn * sizeof(uint32_t));
                *count = nn;
                for (b = 0; b < beam_len; b++) { free(beam[b].toks); }
                free(beam);
            }
            return VAIST_OK;
        }
    }
}

/* =========================================================================
 *   Public API
 * ========================================================================= */

VAIST_API VaistStatus vaist_tokenizer_create(TokenizerType type,
                                             const char *vocab_path,
                                             VaistTokenizer **out) {
    VaistTokenizer *t;
    if (!out) { return VAIST_INVALID_ARGUMENT; }
    *out = NULL;
    if (type != VAIST_TOKENIZER_BPE &&
        type != VAIST_TOKENIZER_SENTENCEPIECE &&
        type != VAIST_TOKENIZER_BYTES) {
        return VAIST_UNSUPPORTED;
    }
    t = tok_alloc(type);
    if (!t) { return VAIST_OUT_OF_MEMORY; }
    switch (type) {
        case VAIST_TOKENIZER_BYTES:
            *out = t;
            return VAIST_OK;
        case VAIST_TOKENIZER_BPE: {
            char *json = NULL, *mtxt = NULL, *path = NULL;
            size_t jlen = 0, mlen = 0;
            VaistStatus st;
            if (!vocab_path || !*vocab_path) { tok_free(t); return VAIST_INVALID_ARGUMENT; }
            path = path_join(vocab_path, "vocab.json");
            if (!path) { tok_free(t); return VAIST_OUT_OF_MEMORY; }
            json = read_file(path, &jlen);
            free(path);
            if (!json) { tok_free(t); return VAIST_IO_ERROR; }
            st = load_bpe_vocab(t, json);
            free(json);
            if (st != VAIST_OK) { tok_free(t); return st; }
            path = path_join(vocab_path, "merges.txt");
            if (!path) { tok_free(t); return VAIST_OUT_OF_MEMORY; }
            mtxt = read_file(path, &mlen);
            free(path);
            if (!mtxt) { tok_free(t); return VAIST_IO_ERROR; }
            st = load_bpe_merges(t, mtxt);
            free(mtxt);
            if (st != VAIST_OK) { tok_free(t); return st; }
            if (t->vocab_size == 0) { tok_free(t); return VAIST_MODEL_ERROR; }
            *out = t;
            return VAIST_OK;
        }
        case VAIST_TOKENIZER_SENTENCEPIECE: {
            char *model = NULL;
            size_t mlen = 0;
            uint8_t *buf;
            size_t body = 0;
            if (!vocab_path || !*vocab_path) { tok_free(t); return VAIST_INVALID_ARGUMENT; }
            model = read_file(vocab_path, &mlen);
            if (!model) { tok_free(t); return VAIST_IO_ERROR; }
            buf = (uint8_t *)model;
            /* Real SentencePiece .model files begin with a raw 13-byte ASCII
             * magic "SentencePiece" (NOT a protobuf field); the ModelProto is
             * serialized directly after it. Validate and skip the magic. */
            if (mlen < 13 || memcmp(buf, "SentencePiece", 13) != 0) {
                free(model); tok_free(t); return VAIST_MODEL_ERROR;
            }
            body = 13;
            if (!sp_load_pieces(buf + body, mlen - body, &t->sp)) {
                free(model); tok_free(t); return VAIST_MODEL_ERROR;
            }
            free(model);
            if (t->sp.piece_count == 0) { tok_free(t); return VAIST_MODEL_ERROR; }
            sp_identify_specials(&t->sp);
            t->bos_id = t->sp.bos_id; t->eos_id = t->sp.eos_id;
            t->unk_id = t->sp.unk_id; t->pad_id = t->sp.pad_id;
            t->vocab_size = (uint32_t)t->sp.piece_count;
            t->max_length = t->sp.max_length;
            *out = t;
            return VAIST_OK;
        }
        default:
            tok_free(t);
            return VAIST_UNSUPPORTED;
    }
}

VAIST_API void vaist_tokenizer_destroy(VaistTokenizer *t) {
    tok_free(t);
}

VAIST_API uint32_t vaist_tokenizer_vocab_size(VaistTokenizer *t) {
    if (!t) { return 0; }
    return t->vocab_size;
}

VAIST_API VaistStatus vaist_tokenizer_info(const VaistTokenizer *t,
                                           VaistTokenizerInfo *info) {
    if (!t || !info) { return VAIST_INVALID_ARGUMENT; }
    info->vocab_size   = t->vocab_size;
    info->max_length   = t->max_length;
    info->bos_id       = t->bos_id;
    info->eos_id       = t->eos_id;
    info->unk_id       = t->unk_id;
    info->pad_id       = t->pad_id;
    info->byte_offset  = t->byte_offset;
    info->type         = (TokenizerType)t->type;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_tokenizer_piece_to_id(VaistTokenizer *t,
                                                  const char *piece,
                                                  uint32_t *id) {
    if (!t || !piece || !id) { return VAIST_INVALID_ARGUMENT; }
    *id = (uint32_t)-1;
    if (t->type == VAIST_TOKENIZER_BPE) {
        size_t plen = strlen(piece);
        (void)bpe_map_get(&t->vocab, piece, plen, id);
        return VAIST_OK;
    } else if (t->type == VAIST_TOKENIZER_SENTENCEPIECE) {
        size_t plen = strlen(piece), i;
        for (i = 0; i < t->sp.piece_count; i++) {
            if (t->sp.pieces[i].piece_len == plen &&
                memcmp(t->sp.pieces[i].piece, piece, plen) == 0) {
                *id = (uint32_t)i;
                return VAIST_OK;
            }
        }
        return VAIST_OK;
    } else {
        /* bytes: single-char piece */
        if (strlen(piece) != 1) { return VAIST_OK; }
        *id = (uint32_t)((uint8_t)piece[0] + t->byte_offset);
        return VAIST_OK;
    }
}

VAIST_API VaistStatus vaist_tokenize(VaistTokenizer *t,
                                     const char *text,
                                     uint32_t *tokens,
                                     size_t cap,
                                     size_t *count) {
    if (!t || !text || !count) { return VAIST_INVALID_ARGUMENT; }
    *count = 0;
    if (t->type == VAIST_TOKENIZER_BYTES) {
        size_t L = strlen(text), i;
        if (cap < L) { *count = L; return VAIST_OUT_OF_MEMORY; }
        for (i = 0; i < L; i++) {
            tokens[i] = (uint32_t)((uint8_t)text[i] + t->byte_offset);
        }
        *count = L;
        return VAIST_OK;
    } else if (t->type == VAIST_TOKENIZER_BPE) {
        char **words;
        size_t n_words, i, o = 0;
        n_words = pretokenize(text, &words);
        for (i = 0; i < n_words; i++) {
            char *w = words[i];
            size_t wlen = strlen(w);
            uint16_t *uchars = (uint16_t *)malloc((wlen + 1) * sizeof(uint16_t));
            size_t nchars;
            uint32_t word_ids[256]; /* per-word ids (bounded by chars) */
            size_t wcount = 0;
            if (!uchars) { continue; }
            nchars = bytes_to_unicode_str(w, wlen, uchars, wlen + 1);
            bpe_encode_word(uchars, nchars, t, word_ids, &wcount, 256);
            free(uchars);
            {
                size_t k;
                for (k = 0; k < wcount; k++) {
                    if (t->max_length && o + 1 > t->max_length) { break; }
                    if (o < cap) { tokens[o++] = word_ids[k]; }
                }
                if (t->max_length && o >= t->max_length) { break; }
            }
        }
        {
            size_t i2;
            for (i2 = 0; i2 < n_words; i2++) { free(words[i2]); }
            free(words);
        }
        *count = o;
        return (o <= cap) ? VAIST_OK : VAIST_OUT_OF_MEMORY;
    } else if (t->type == VAIST_TOKENIZER_SENTENCEPIECE) {
        size_t n, o;
        const sp_model *m = &t->sp;
        int32_t unk = t->unk_id;
        size_t tlen = strlen(text);
        char *norm = (char *)malloc(tlen + 3 * (tlen / 2 + 2) + 1u);
        if (!norm) { return VAIST_OUT_OF_MEMORY; }
        {
            size_t nlen = sp_normalize(text, norm, tlen + 3 * (tlen / 2 + 2) + 1u);
            if (nlen == SIZE_MAX) { free(norm); return VAIST_OUT_OF_MEMORY; }
            sp_greedy_tokenize(m, norm, nlen, unk, tokens, cap, &n);
        }
        free(norm);
        o = n;
        if (t->max_length && o > t->max_length) { o = t->max_length; }
        *count = o;
        return (o <= cap) ? VAIST_OK : VAIST_OUT_OF_MEMORY;
    }
    return VAIST_UNSUPPORTED;
}

VAIST_API VaistStatus vaist_detokenize(VaistTokenizer *t,
                                       const uint32_t *tokens,
                                       size_t n,
                                       char *out,
                                       size_t cap,
                                       size_t *len) {
    if (!t || !tokens || !len) { return VAIST_INVALID_ARGUMENT; }
    *len = 0;
    if (t->type == VAIST_TOKENIZER_BYTES) {
        size_t i, o = 0;
        for (i = 0; i < n; i++) {
            uint32_t v = tokens[i];
            if (v < t->byte_offset) { continue; }
            {
                uint8_t b = (uint8_t)(v - t->byte_offset);
                if (out && o + 1 < cap) { out[o] = (char)b; }
                o++;
            }
        }
        if (out && cap > 0) {
            if (o < cap) { out[o] = '\0'; }
            else { out[cap - 1] = '\0'; }
        }
        *len = o;
        if (o >= cap && cap > 0) { return VAIST_OUT_OF_MEMORY; }
        return VAIST_OK;
    } else if (t->type == VAIST_TOKENIZER_BPE) {
        /* Collect piece strings -> unicode chars -> bytes via the inverse byte map. */
        size_t i, ucap = n * 64 + 1;
        uint16_t *uchars = (uint16_t *)malloc(ucap * sizeof(uint16_t));
        size_t uo = 0;
        if (!uchars) { return VAIST_OUT_OF_MEMORY; }
        for (i = 0; i < n; i++) {
            uint32_t id = tokens[i];
            const char *s;
            size_t slen, j;
            if (id >= t->vocab_size || !t->id_str[id]) { continue; }
            s = t->id_str[id];
            slen = t->id_str_len[id];
            /* Decode the UTF-8 bytes of the stored key into codepoints (which are
             * the unicode-mapped chars), then collect as uint16_t chars. */
            j = 0;
            while (j < slen) {
                uint32_t cp;
                size_t need;
                uint8_t b0 = (uint8_t)s[j];
                if (b0 < 0x80) { cp = b0; need = 1; }
                else if ((b0 & 0xE0) == 0xC0) {
                    cp = (uint32_t)(b0 & 0x1F);
                    if (j + 1 < slen) { cp = (cp << 6) | (uint32_t)(s[j+1] & 0x3F); }
                    need = 2;
                }
                else if ((b0 & 0xF0) == 0xE0) {
                    cp = (uint32_t)(b0 & 0x0F);
                    if (j + 1 < slen) { cp = (cp << 6) | (uint32_t)(s[j+1] & 0x3F); }
                    if (j + 2 < slen) { cp = (cp << 6) | (uint32_t)(s[j+2] & 0x3F); }
                    need = 3;
                }
                else {
                    cp = (uint32_t)(b0 & 0x07);
                    if (j + 1 < slen) { cp = (cp << 6) | (uint32_t)(s[j+1] & 0x3F); }
                    if (j + 2 < slen) { cp = (cp << 6) | (uint32_t)(s[j+2] & 0x3F); }
                    if (j + 3 < slen) { cp = (cp << 6) | (uint32_t)(s[j+3] & 0x3F); }
                    need = 4;
                }
                if (uo + 1 > ucap) {
                    size_t nc = ucap * 2;
                    uint16_t *nchr = (uint16_t *)realloc(uchars, nc * sizeof(uint16_t));
                    if (!nchr) { free(uchars); return VAIST_OUT_OF_MEMORY; }
                    uchars = nchr; ucap = nc;
                }
                if (cp < 0x10000) { uchars[uo++] = (uint16_t)cp; }
                else { /* surrogate pair (won't occur for byte-range) */
                    uchars[uo++] = (uint16_t)(0xD800 + ((cp - 0x10000) >> 10));
                    uchars[uo++] = (uint16_t)(0xDC00 + ((cp - 0x10000) & 0x3FF));
                }
                j += need;
            }
        }
        {
            char *buf = (char *)malloc(uo + 1u);
            size_t blen;
            if (!buf) { free(uchars); return VAIST_OUT_OF_MEMORY; }
            blen = unicode_str_to_bytes(uchars, uo, buf, uo + 1u);
            free(uchars);
            if (blen >= cap && cap > 0) { free(buf); *len = blen; return VAIST_OUT_OF_MEMORY; }
            if (out && cap > 0) {
                memcpy(out, buf, blen);
                out[blen] = '\0';
            }
            *len = blen;
            free(buf);
        }
        return VAIST_OK;
    } else if (t->type == VAIST_TOKENIZER_SENTENCEPIECE) {
        /* Concatenate piece surfaces, then map the leading U+2001 marker to a
         * space (SentencePiece word-boundary convention). */
        size_t i, o = 0, ucap = n * 64 + 1;
        char *buf = (char *)malloc(ucap);
        if (!buf) { return VAIST_OUT_OF_MEMORY; }
        for (i = 0; i < n; i++) {
            uint32_t id = tokens[i];
            const sp_piece *pp;
            if (id >= t->sp.piece_count) { continue; }
            pp = &t->sp.pieces[id];
            if (pp->type == 3) { continue; } /* control */
            if (pp->piece_len == 0) { continue; }
            if (o + pp->piece_len + 1 > ucap) {
                size_t nc = ucap * 2 + pp->piece_len;
                char *nb = (char *)realloc(buf, nc);
                if (!nb) { free(buf); return VAIST_OUT_OF_MEMORY; }
                buf = nb; ucap = nc;
            }
            memcpy(buf + o, pp->piece, pp->piece_len);
            o += pp->piece_len;
        }
        /* Map U+2001 (3 bytes: 0xE2 0x96 0x81) leading markers to spaces. */
        {
            size_t k, dst = 0;
            const char *mk = "\xe2\x96\x81";
            size_t ml = 3;
            for (k = 0; k < o;) {
                if (k + ml <= o && memcmp(buf + k, mk, ml) == 0) {
                    buf[dst++] = ' ';
                    k += ml;
                } else {
                    buf[dst++] = buf[k++];
                }
            }
            if (dst < o) {
                buf[dst] = '\0';
                o = dst;
            }
        }
        if (out && o < cap) { memcpy(out, buf, o); out[o] = '\0'; }
        else if (out && cap > 0) { out[cap - 1] = '\0'; }
        *len = o;
        if (o >= cap && cap > 0) { free(buf); return VAIST_OUT_OF_MEMORY; }
        free(buf);
        return VAIST_OK;
    }
    return VAIST_UNSUPPORTED;
}

VAIST_API VaistStatus vaist_tokenize_sample(VaistTokenizer *t,
                                            const char *text,
                                            uint64_t seed,
                                            uint32_t nbest,
                                            uint32_t *tokens,
                                            size_t cap,
                                            size_t *count) {
    if (!t || !text || !tokens || !count) { return VAIST_INVALID_ARGUMENT; }
    *count = 0;
    if (t->type != VAIST_TOKENIZER_SENTENCEPIECE) {
        return VAIST_UNSUPPORTED;
    }
    if (t->sp.piece_count == 0) { return VAIST_MODEL_ERROR; }
    if (t->sp.model_type && strcmp(t->sp.model_type, "unigram") != 0) {
        /* Non-unigram: sampling == greedy. */
        return vaist_tokenize(t, text, tokens, cap, count);
    }
    /* Normalize: prepend U+2581 to each whitespace-separated word. */
    {
        size_t raw = strlen(text);
        size_t cap_norm = raw + 3 * (raw / 2 + 2) + 1u;
        char *norm = (char *)malloc(cap_norm);
        VaistStatus st;
        if (!norm) { return VAIST_OUT_OF_MEMORY; }
        {
            size_t nlen = sp_normalize(text, norm, cap_norm);
            if (nlen == SIZE_MAX) { free(norm); return VAIST_OUT_OF_MEMORY; }
            st = sp_sample_tokenize(t, norm, nlen, seed, nbest, tokens, cap, count);
        }
        free(norm);
        return st;
    }
}
