/**
 * \file vkmodel.c
 * \brief VKModel implementation: GGUF host parser + per-tensor GPU upload.
 *
 * Design:
 *  - Host parsing is pure C99 + FILE* I/O (no Vulkan). A little-endian byte
 *    reader walks the header, the metadata key-value pairs (all 13 GGUF value
 *    types incl. nested arrays, stored as typed heap copies), and the tensor
 *    infos. Tensor data is never loaded whole-file: each tensor is read in
 *    bounded VKMODEL_STREAM_CHUNK slices via fseek/fread and uploaded with
 *    vkr_upload() into a device buffer obtained from vkr_malloc().
 *  - One command pool + one command buffer are created at load and reused
 *    across every tensor upload (vkr_upload() resets it after each submit).
 *    The queue is resolved as family 0 / index 0, the stack-wide convention
 *    used by every harness when creating the runtime.
 *  - Tensor byte sizes follow the stack's canonical (classic ggml) block
 *    layout — the same layout VKQuant dequantizes (Q8_0 = 36 B f32 scale,
 *    Q4_0 = 20 B f32 scale, K-format super-blocks unchanged). The gguf
 *    per-tensor offset field is used verbatim to locate data.
 */
#include "vkmodel_internal.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define VKMODEL_FSEEK _fseeki64
#else
#define VKMODEL_FSEEK fseeko
#endif

/* ===========================================================================
 * ggml_type tables (elements per block, bytes per block)
 *
 * Enum values are the GGUF-spec ggml_type numbers (Q5_0=6, Q5_1=7, Q8_0=8,
 * Q8_1=9, Q2_K=10 ... Q6_K=14). Byte sizes match the stack's VKQuant block
 * layouts for the formats it dequantizes and the classic ggml layouts for the
 * rest. Removed/unknown enum slots default to 0 (treated as an error).
 * ========================================================================== */

static const uint32_t vkmodel_blck_elems_tbl[] = {
    [0]  = 1,    /* F32     */
    [1]  = 1,    /* F16     */
    [2]  = 32,   /* Q4_0    */
    [3]  = 32,   /* Q4_1    */
    [6]  = 32,   /* Q5_0    */
    [7]  = 32,   /* Q5_1    */
    [8]  = 32,   /* Q8_0    */
    [9]  = 32,   /* Q8_1    */
    [10] = 256,  /* Q2_K    */
    [11] = 256,  /* Q3_K    */
    [12] = 256,  /* Q4_K    */
    [13] = 256,  /* Q5_K    */
    [14] = 256,  /* Q6_K    */
    [15] = 256,  /* Q8_K    */
    [16] = 256,  /* IQ2_XXS */
    [17] = 256,  /* IQ2_XS  */
    [18] = 256,  /* IQ3_XXS */
    [19] = 256,  /* IQ1_S   */
    [20] = 32,   /* IQ4_NL  */
    [21] = 256,  /* IQ3_S   */
    [22] = 256,  /* IQ2_S   */
    [23] = 256,  /* IQ4_XS  */
    [24] = 1,    /* I8      */
    [25] = 1,    /* I16     */
    [26] = 1,    /* I32     */
    [27] = 1,    /* I64     */
    [28] = 1,    /* F64     */
    [29] = 256,  /* IQ1_M   */
    [30] = 1,    /* BF16    */
    [34] = 256,  /* TQ1_0   */
    [35] = 256,  /* TQ2_0   */
    [39] = 32,   /* MXFP4   */
    [40] = 64,   /* NVFP4   */
    [41] = 128,  /* Q1_0    */
    [42] = 64,   /* Q2_0    */
};

static const uint32_t vkmodel_blck_bytes_tbl[] = {
    [0]  = 4,     /* F32     */
    [1]  = 2,     /* F16     */
    [2]  = 20,    /* Q4_0    (f32 scale + 16 nibbles)                 */
    [3]  = 24,    /* Q4_1    (f32 d + f32 m + 16 nibbles)             */
    [6]  = 24,    /* Q5_0    (f32 d + qh + 16 nibbles)                */
    [7]  = 28,    /* Q5_1    (f32 d + f32 m + qh + 16 nibbles)        */
    [8]  = 36,    /* Q8_0    (f32 scale + 32 x int8)                  */
    [9]  = 40,    /* Q8_1    (f32 d + f32 s + 32 x int8)              */
    [10] = 84,    /* Q2_K    */
    [11] = 110,   /* Q3_K    */
    [12] = 144,   /* Q4_K    (f16 d/dmin + 12 scales + 128 nibbles)   */
    [13] = 176,   /* Q5_K    */
    [14] = 210,   /* Q6_K    (ql + qh + 16 int8 scales + f16 d)       */
    [15] = 258,   /* Q8_K    */
    [16] = 42,    /* IQ2_XXS */
    [17] = 42,    /* IQ2_XS  */
    [18] = 82,    /* IQ3_XXS */
    [19] = 44,    /* IQ1_S   */
    [20] = 18,    /* IQ4_NL  */
    [21] = 110,   /* IQ3_S   */
    [22] = 50,    /* IQ2_S   */
    [23] = 136,   /* IQ4_XS  (f16 d + u16 scales_h + scales_l + qs)   */
    [24] = 1,     /* I8      */
    [25] = 2,     /* I16     */
    [26] = 4,     /* I32     */
    [27] = 8,     /* I64     */
    [28] = 8,     /* F64     */
    [29] = 56,    /* IQ1_M   */
    [30] = 2,     /* BF16    */
    [34] = 54,    /* TQ1_0   */
    [35] = 66,    /* TQ2_0   */
    [39] = 17,    /* MXFP4   */
    [40] = 36,    /* NVFP4   */
    [41] = 18,    /* Q1_0    */
    [42] = 18,    /* Q2_0    */
};

static uint32_t vkmodel_tbl_entry(const uint32_t* tbl, uint32_t count,
                                  uint32_t type)
{
    return type < count ? tbl[type] : 0;
}

static uint32_t vkmodel_blck_elems_of(uint32_t type)
{
    return vkmodel_tbl_entry(vkmodel_blck_elems_tbl,
                             (uint32_t)(sizeof(vkmodel_blck_elems_tbl) /
                                        sizeof(vkmodel_blck_elems_tbl[0])),
                             type);
}

static uint32_t vkmodel_blck_bytes_of(uint32_t type)
{
    return vkmodel_tbl_entry(vkmodel_blck_bytes_tbl,
                             (uint32_t)(sizeof(vkmodel_blck_bytes_tbl) /
                                        sizeof(vkmodel_blck_bytes_tbl[0])),
                             type);
}

static VkDeviceSize vkmodel_tensor_bytes(uint32_t type, uint64_t nelems)
{
    uint32_t blck = vkmodel_blck_elems_of(type);
    uint32_t bsz  = vkmodel_blck_bytes_of(type);
    if (blck == 0 || bsz == 0) return 0;
    uint64_t blocks = (nelems + blck - 1) / blck;
    return (VkDeviceSize)blocks * (VkDeviceSize)bsz;
}

static uint64_t vkmodel_align_up(uint64_t value, uint64_t align)
{
    if (align == 0) return value;
    uint64_t rem = value % align;
    return rem == 0 ? value : value + (align - rem);
}

static char* vkmodel_strdup(const char* s)
{
    if (!s) return NULL;
    size_t n = strlen(s);
    char* d = (char*)malloc(n + 1);
    if (!d) return NULL;
    memcpy(d, s, n + 1);
    return d;
}

static VkResult vkmodel_create_upload_state(VkModel* m)
{
    VkResult res = vkr_create_command_pool(m->rt, 0, &m->upload_pool);
    if (res != VK_SUCCESS) return res;

    VkCommandBufferAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m->upload_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(vkr_get_device(m->rt), &ai, &m->upload_cmd);
    if (res != VK_SUCCESS) {
        vkDestroyCommandPool(vkr_get_device(m->rt), m->upload_pool, NULL);
        m->upload_pool = VK_NULL_HANDLE;
        return res;
    }
    return VK_SUCCESS;
}

/* ===========================================================================
 * Little-endian file reader
 * ========================================================================== */

typedef struct {
    FILE    *fp;
    uint64_t pos;   /**< Sequential read position (parse phase only). */
} vkmodel_reader_t;

static VkResult vkmodel_rd_fill(vkmodel_reader_t* r, void* dst, uint64_t n)
{
    if (n > (uint64_t)(size_t)-1) return VK_ERROR_INITIALIZATION_FAILED;
    if (fread(dst, 1, (size_t)n, r->fp) != n) return VK_ERROR_INITIALIZATION_FAILED;
    r->pos += n;
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_u16(vkmodel_reader_t* r, uint16_t* out)
{
    uint8_t b[2];
    VkResult res = vkmodel_rd_fill(r, b, 2);
    if (res != VK_SUCCESS) return res;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_u32(vkmodel_reader_t* r, uint32_t* out)
{
    uint8_t b[4];
    VkResult res = vkmodel_rd_fill(r, b, 4);
    if (res != VK_SUCCESS) return res;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_u64(vkmodel_reader_t* r, uint64_t* out)
{
    uint8_t b[8];
    VkResult res = vkmodel_rd_fill(r, b, 8);
    if (res != VK_SUCCESS) return res;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    *out = v;
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_f32(vkmodel_reader_t* r, float* out)
{
    uint32_t u;
    VkResult res = vkmodel_rd_u32(r, &u);
    if (res != VK_SUCCESS) return res;
    memcpy(out, &u, sizeof(u));
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_f64(vkmodel_reader_t* r, double* out)
{
    uint64_t u;
    VkResult res = vkmodel_rd_u64(r, &u);
    if (res != VK_SUCCESS) return res;
    memcpy(out, &u, sizeof(u));
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_str(vkmodel_reader_t* r, char** out)
{
    uint64_t len = 0;
    VkResult res = vkmodel_rd_u64(r, &len);
    if (res != VK_SUCCESS) return res;
    if (len > VKMODEL_MAX_STRING) return VK_ERROR_INITIALIZATION_FAILED;

    char* s = (char*)malloc((size_t)len + 1);
    if (!s) return VK_ERROR_OUT_OF_HOST_MEMORY;
    res = vkmodel_rd_fill(r, s, len);
    if (res != VK_SUCCESS) {
        free(s);
        return res;
    }
    s[len] = '\0';
    *out = s;
    return VK_SUCCESS;
}

/* ===========================================================================
 * Minimal JSON parser (recursive descent, self-contained)
 *
 * Parses the safetensors header (a JSON object mapping tensor names to
 * {dtype, shape, data_offsets} plus an optional "__metadata__" object). The
 * parser is deliberately small: it handles objects, arrays, strings (incl.
 * \uXXXX escapes with surrogate pairs, encoded to UTF-8), non-negative
 * integers, and the true/false/null literals. Fractions/exponents are
 * consumed but truncated (the loader only needs integer dims/offsets). Every
 * node stores its values in host memory owned by the node tree, released with
 * vkmodel_json_free(). No external dependency.
 * ========================================================================== */

typedef enum {
    VKMODEL_JSON_OBJECT,
    VKMODEL_JSON_ARRAY,
    VKMODEL_JSON_STRING,
    VKMODEL_JSON_NUMBER,
    VKMODEL_JSON_BOOL,
    VKMODEL_JSON_NULL
} VkModelJsonType;

typedef struct VkModelJsonNode VkModelJsonNode;

struct VkModelJsonNode {
    VkModelJsonType  type;      /**< Node kind.                              */
    char            *key;       /**< Object member key (owned) or NULL.      */
    char            *str;       /**< STRING: NUL-terminated decoded value.   */
    unsigned long long num;     /**< NUMBER: integer value.                  */
    int              boolean;   /**< BOOL: 0/1.                              */
    size_t           count;     /**< OBJECT/ARRAY: child count.              */
    size_t           cap;       /**< Allocated capacity of members[].        */
    VkModelJsonNode *members;   /**< OBJECT/ARRAY: children.                 */
};

typedef struct {
    const char *s;              /**< Header text.                            */
    size_t      len;            /**< Header byte length.                     */
    size_t      pos;            /**< Current cursor.                         */
    int         failed;         /**< Parse failed flag.                      */
} VkModelJsonParser;

static void vkmodel_json_free_children(VkModelJsonNode* n)
{
    if (!n) return;
    free(n->key);
    if (n->type == VKMODEL_JSON_STRING) free(n->str);
    if (n->type == VKMODEL_JSON_OBJECT || n->type == VKMODEL_JSON_ARRAY) {
        for (size_t i = 0; i < n->count; i++) {
            vkmodel_json_free_children(&n->members[i]);
        }
        free(n->members);
    }
}

static void vkmodel_json_free(VkModelJsonNode* n)
{
    if (!n) return;
    /* Children are stored as array slots inside n->members, never malloc'd
     * individually, so only n itself may be free()'d here. */
    vkmodel_json_free_children(n);
    free(n);
}

static void vkmodel_json_skip_ws(VkModelJsonParser* p)
{
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
}

static int vkmodel_json_parse_lit(VkModelJsonParser* p, const char* lit)
{
    size_t n = strlen(lit);
    if (p->pos + n > p->len || strncmp(p->s + p->pos, lit, n) != 0) return 0;
    p->pos += n;
    return 1;
}

static int vkmodel_json_putc(VkModelJsonParser* p, char** buf, size_t* cap,
                             size_t* len, unsigned char b)
{
    if (*len + 1 >= *cap) {
        size_t ncap = *cap * 2;
        char* nb = (char*)realloc(*buf, ncap);
        if (!nb) { p->failed = 1; return 0; }
        *buf = nb;
        *cap = ncap;
    }
    (*buf)[(*len)++] = (char)b;
    return 1;
}

static int vkmodel_json_parse_string(VkModelJsonParser* p, char** out)
{
    if (p->pos >= p->len || p->s[p->pos] != '"') { p->failed = 1; return 0; }
    p->pos++;
    size_t cap = 32, len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) { p->failed = 1; return 0; }

    for (;;) {
        if (p->pos >= p->len) { free(buf); p->failed = 1; return 0; }
        unsigned char c = (unsigned char)p->s[p->pos];

        if (c == '"') {
            p->pos++;
            buf[len] = '\0';
            *out = buf;
            return 1;
        }
        if (c == '\\') {
            p->pos++;
            if (p->pos >= p->len) { free(buf); p->failed = 1; return 0; }
            char e = p->s[p->pos];
            p->pos++;

            if (e == 'u') {
                if (p->pos + 4 > p->len) { free(buf); p->failed = 1; return 0; }
                unsigned long cp = 0;
                for (int k = 0; k < 4; k++) {
                    char h = p->s[p->pos++];
                    unsigned hv;
                    if (h >= '0' && h <= '9')      hv = (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') hv = (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') hv = (unsigned)(h - 'A' + 10);
                    else { free(buf); p->failed = 1; return 0; }
                    cp = (cp << 4) | hv;
                }
                /* combine a valid surrogate pair */
                if (cp >= 0xD800 && cp <= 0xDBFF && p->pos + 6 <= p->len &&
                    p->s[p->pos] == '\\' && p->s[p->pos + 1] == 'u') {
                    unsigned long lo = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = p->s[p->pos + 2 + k];
                        unsigned hv;
                        if (h >= '0' && h <= '9')      hv = (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') hv = (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') hv = (unsigned)(h - 'A' + 10);
                        else { free(buf); p->failed = 1; return 0; }
                        lo = (lo << 4) | hv;
                    }
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        p->pos += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                }
                /* encode code point as UTF-8 */
                if (cp < 0x80) {
                    if (!vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)cp)) { free(buf); return 0; }
                } else if (cp < 0x800) {
                    if (!vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0xC0 | (cp >> 6))) ||
                        !vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0x80 | (cp & 0x3F)))) {
                        free(buf); return 0;
                    }
                } else if (cp < 0x10000) {
                    if (!vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0xE0 | (cp >> 12))) ||
                        !vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0x80 | ((cp >> 6) & 0x3F))) ||
                        !vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0x80 | (cp & 0x3F)))) {
                        free(buf); return 0;
                    }
                } else {
                    if (!vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0xF0 | (cp >> 18))) ||
                        !vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0x80 | ((cp >> 12) & 0x3F))) ||
                        !vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0x80 | ((cp >> 6) & 0x3F))) ||
                        !vkmodel_json_putc(p, &buf, &cap, &len,
                                           (unsigned char)(0x80 | (cp & 0x3F)))) {
                        free(buf); return 0;
                    }
                }
                continue;
            }

            char ch;
            switch (e) {
            case '"':  ch = '"';  break;
            case '\\': ch = '\\'; break;
            case '/':  ch = '/';  break;
            case 'b':  ch = '\b'; break;
            case 'f':  ch = '\f'; break;
            case 'n':  ch = '\n'; break;
            case 'r':  ch = '\r'; break;
            case 't':  ch = '\t'; break;
            default:   free(buf); p->failed = 1; return 0;
            }
            if (!vkmodel_json_putc(p, &buf, &cap, &len, (unsigned char)ch)) {
                free(buf); return 0;
            }
            continue;
        }
        /* raw byte (UTF-8 bytes pass through untouched) */
        if (!vkmodel_json_putc(p, &buf, &cap, &len, c)) { free(buf); return 0; }
        p->pos++;
    }
}

static int vkmodel_json_add_child(VkModelJsonParser* p, VkModelJsonNode* parent,
                                  VkModelJsonNode* child)
{
    if (parent->count >= parent->cap) {
        size_t ncap = parent->cap == 0 ? 4 : parent->cap * 2;
        VkModelJsonNode* nm = (VkModelJsonNode*)realloc(
            parent->members, ncap * sizeof(VkModelJsonNode));
        if (!nm) { p->failed = 1; return 0; }
        parent->members = nm;
        parent->cap = ncap;
    }
    parent->members[parent->count] = *child;
    parent->count++;
    return 1;
}

static VkModelJsonNode* vkmodel_json_parse_value(VkModelJsonParser* p);

static VkModelJsonNode* vkmodel_json_parse_object(VkModelJsonParser* p)
{
    VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
    if (!node) { p->failed = 1; return NULL; }
    node->type = VKMODEL_JSON_OBJECT;
    p->pos++;
    vkmodel_json_skip_ws(p);
    if (p->pos < p->len && p->s[p->pos] == '}') { p->pos++; return node; }

    for (;;) {
        vkmodel_json_skip_ws(p);
        char* key = NULL;
        if (p->pos >= p->len || !vkmodel_json_parse_string(p, &key)) {
            vkmodel_json_free(node);
            return NULL;
        }
        vkmodel_json_skip_ws(p);
        if (p->pos >= p->len || p->s[p->pos] != ':') {
            free(key);
            vkmodel_json_free(node);
            p->failed = 1;
            return NULL;
        }
        p->pos++;
        VkModelJsonNode* val = vkmodel_json_parse_value(p);
        if (!val) {
            free(key);
            vkmodel_json_free(node);
            return NULL;
        }
        val->key = key;
        if (!vkmodel_json_add_child(p, node, val)) {
            vkmodel_json_free(val);
            vkmodel_json_free(node);
            return NULL;
        }
        free(val);
        vkmodel_json_skip_ws(p);
        if (p->pos >= p->len) { vkmodel_json_free(node); p->failed = 1; return NULL; }
        if (p->s[p->pos] == ',') { p->pos++; continue; }
        if (p->s[p->pos] == '}') { p->pos++; return node; }
        vkmodel_json_free(node);
        p->failed = 1;
        return NULL;
    }
}

static VkModelJsonNode* vkmodel_json_parse_array(VkModelJsonParser* p)
{
    VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
    if (!node) { p->failed = 1; return NULL; }
    node->type = VKMODEL_JSON_ARRAY;
    p->pos++;
    vkmodel_json_skip_ws(p);
    if (p->pos < p->len && p->s[p->pos] == ']') { p->pos++; return node; }

    for (;;) {
        VkModelJsonNode* val = vkmodel_json_parse_value(p);
        if (!val) { vkmodel_json_free(node); return NULL; }
        if (!vkmodel_json_add_child(p, node, val)) {
            vkmodel_json_free(val);
            vkmodel_json_free(node);
            return NULL;
        }
        free(val);
        vkmodel_json_skip_ws(p);
        if (p->pos >= p->len) { vkmodel_json_free(node); p->failed = 1; return NULL; }
        if (p->s[p->pos] == ',') { p->pos++; continue; }
        if (p->s[p->pos] == ']') { p->pos++; return node; }
        vkmodel_json_free(node);
        p->failed = 1;
        return NULL;
    }
}

static VkModelJsonNode* vkmodel_json_parse_value(VkModelJsonParser* p)
{
    vkmodel_json_skip_ws(p);
    if (p->pos >= p->len) { p->failed = 1; return NULL; }
    char c = p->s[p->pos];

    if (c == '{') return vkmodel_json_parse_object(p);
    if (c == '[') return vkmodel_json_parse_array(p);

    if (c == '"') {
        VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
        if (!node) { p->failed = 1; return NULL; }
        node->type = VKMODEL_JSON_STRING;
        if (!vkmodel_json_parse_string(p, &node->str)) { free(node); return NULL; }
        return node;
    }
    if (c == 't') {
        if (!vkmodel_json_parse_lit(p, "true")) { p->failed = 1; return NULL; }
        VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
        if (!node) { p->failed = 1; return NULL; }
        node->type = VKMODEL_JSON_BOOL;
        node->boolean = 1;
        return node;
    }
    if (c == 'f') {
        if (!vkmodel_json_parse_lit(p, "false")) { p->failed = 1; return NULL; }
        VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
        if (!node) { p->failed = 1; return NULL; }
        node->type = VKMODEL_JSON_BOOL;
        return node;
    }
    if (c == 'n') {
        if (!vkmodel_json_parse_lit(p, "null")) { p->failed = 1; return NULL; }
        VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
        if (!node) { p->failed = 1; return NULL; }
        node->type = VKMODEL_JSON_NULL;
        return node;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        VkModelJsonNode* node = (VkModelJsonNode*)calloc(1, sizeof(*node));
        if (!node) { p->failed = 1; return NULL; }
        node->type = VKMODEL_JSON_NUMBER;
        if (c == '-') p->pos++;
        unsigned long long val = 0;
        int digits = 0;
        while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') {
            if (val > 0x0FFFFFFFFFFFFFFULL) { free(node); p->failed = 1; return NULL; }
            val = val * 10 + (unsigned long long)(p->s[p->pos] - '0');
            p->pos++;
            digits++;
        }
        if (!digits) { free(node); p->failed = 1; return NULL; }
        /* fraction/exponent: consume (loader needs integers only) */
        if (p->pos < p->len && p->s[p->pos] == '.') {
            p->pos++;
            while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
        }
        if (p->pos < p->len && (p->s[p->pos] == 'e' || p->s[p->pos] == 'E')) {
            p->pos++;
            if (p->pos < p->len && (p->s[p->pos] == '+' || p->s[p->pos] == '-')) p->pos++;
            while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
        }
        node->num = val;
        return node;
    }
    p->failed = 1;
    return NULL;
}

/**
 * \brief Parse a complete JSON document (single top-level value).
 *
 * \param s   NUL-terminated JSON text.
 * \param len Byte length of \p s (may be shorter than strlen(s)).
 * \return Root node, or NULL on malformed input / trailing garbage.
 */
static VkModelJsonNode* vkmodel_json_parse(const char* s, size_t len)
{
    VkModelJsonParser p;
    p.s = s;
    p.len = len;
    p.pos = 0;
    p.failed = 0;
    VkModelJsonNode* root = vkmodel_json_parse_value(&p);
    if (!root || p.failed) { vkmodel_json_free(root); return NULL; }
    vkmodel_json_skip_ws(&p);
    if (p.pos != p.len) { vkmodel_json_free(root); return NULL; }
    return root;
}

static VkModelJsonNode* vkmodel_json_find(VkModelJsonNode* obj, const char* key)
{
    if (!obj || obj->type != VKMODEL_JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->count; i++) {
        VkModelJsonNode* m = &obj->members[i];
        if (m->key && strcmp(m->key, key) == 0) return m;
    }
    return NULL;
}

/* ===========================================================================
 * Metadata value parsing
 * ========================================================================== */

void vkmodel_value_free(VkModelValue* value)
{
    if (!value) return;
    switch (value->type) {
    case VKMODEL_VAL_STRING:
        free(value->v.str);
        break;
    case VKMODEL_VAL_ARRAY:
        for (uint64_t i = 0; i < value->v.array.count; i++) {
            vkmodel_value_free(&value->v.array.elems[i]);
        }
        free(value->v.array.elems);
        break;
    default:
        break;
    }
    memset(value, 0, sizeof(*value));
}

static VkResult vkmodel_parse_value(vkmodel_reader_t* r, uint32_t type,
                                    VkModelValue* out)
{
    memset(out, 0, sizeof(*out));
    out->type = type;

    switch (type) {
    case VKMODEL_VAL_UINT8: {
        uint8_t v;
        VkResult res = vkmodel_rd_fill(r, &v, 1);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT8: {
        uint8_t b;
        VkResult res = vkmodel_rd_fill(r, &b, 1);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = (uint64_t)(int64_t)(int8_t)b;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_UINT16: {
        uint16_t v;
        VkResult res = vkmodel_rd_u16(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT16: {
        uint16_t u;
        VkResult res = vkmodel_rd_u16(r, &u);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = (uint64_t)(int64_t)(int16_t)u;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_UINT32: {
        uint32_t v;
        VkResult res = vkmodel_rd_u32(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT32: {
        uint32_t u;
        VkResult res = vkmodel_rd_u32(r, &u);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = (uint64_t)(int64_t)(int32_t)u;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_UINT64: {
        uint64_t v;
        VkResult res = vkmodel_rd_u64(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT64: {
        uint64_t u;
        VkResult res = vkmodel_rd_u64(r, &u);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = u; /* two's complement preserved */
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_FLOAT32: {
        float v;
        VkResult res = vkmodel_rd_f32(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.f64 = (double)v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_FLOAT64: {
        double v;
        VkResult res = vkmodel_rd_f64(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.f64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_BOOL: {
        uint8_t b;
        VkResult res = vkmodel_rd_fill(r, &b, 1);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = b ? 1 : 0;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_STRING:
        return vkmodel_rd_str(r, &out->v.str);
    case VKMODEL_VAL_ARRAY: {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        VkResult res = vkmodel_rd_u32(r, &elem_type);
        if (res != VK_SUCCESS) return res;
        res = vkmodel_rd_u64(r, &count);
        if (res != VK_SUCCESS) return res;
        if (count > (1u << 24)) return VK_ERROR_INITIALIZATION_FAILED;

        out->v.array.elem_type = elem_type;
        out->v.array.count = count;
        if (count == 0) {
            out->v.array.elems = NULL;
            return VK_SUCCESS;
        }
        out->v.array.elems =
            (VkModelValue*)calloc((size_t)count, sizeof(VkModelValue));
        if (!out->v.array.elems) return VK_ERROR_OUT_OF_HOST_MEMORY;
        for (uint64_t i = 0; i < count; i++) {
            res = vkmodel_parse_value(r, elem_type,
                                      &out->v.array.elems[i]);
            if (res != VK_SUCCESS) {
                vkmodel_value_free(out);
                return res;
            }
        }
        return VK_SUCCESS;
    }
    default:
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

/* ===========================================================================
 * Section parsing
 * ========================================================================== */

static VkResult vkmodel_parse_header(vkmodel_reader_t* r, VkModel* m)
{
    uint32_t magic = 0;
    VkResult res = vkmodel_rd_u32(r, &magic);
    if (res != VK_SUCCESS) return res;
    if (magic != VKMODEL_MAGIC_LE) return VK_ERROR_INITIALIZATION_FAILED;

    uint32_t version = 0;
    res = vkmodel_rd_u32(r, &version);
    if (res != VK_SUCCESS) return res;
    if (version < 2 || version > VKMODEL_GGUF_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint64_t tc = 0, kc = 0;
    res = vkmodel_rd_u64(r, &tc);
    if (res != VK_SUCCESS) return res;
    res = vkmodel_rd_u64(r, &kc);
    if (res != VK_SUCCESS) return res;
    if (tc > (1u << 20) || kc > (1u << 20)) return VK_ERROR_INITIALIZATION_FAILED;

    m->tensor_count = (uint32_t)tc;
    m->kv_count = (uint32_t)kc;

    if (m->tensor_count > 0) {
        m->tensors = (VkModelTensor*)calloc(m->tensor_count,
                                            sizeof(VkModelTensor));
        if (!m->tensors) return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (m->kv_count > 0) {
        m->kv = (VkModelKV*)calloc(m->kv_count, sizeof(VkModelKV));
        if (!m->kv) return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return VK_SUCCESS;
}

static VkResult vkmodel_parse_metadata(vkmodel_reader_t* r, VkModel* m)
{
    for (uint64_t i = 0; i < m->kv_count; i++) {
        char* key = NULL;
        VkResult res = vkmodel_rd_str(r, &key);
        if (res != VK_SUCCESS) return res;

        uint32_t vtype = 0;
        res = vkmodel_rd_u32(r, &vtype);
        if (res != VK_SUCCESS) {
            free(key);
            return res;
        }

        VkModelValue val;
        res = vkmodel_parse_value(r, vtype, &val);
        if (res != VK_SUCCESS) {
            free(key);
            return res;
        }

        m->kv[i].key = key;
        m->kv[i].value = val;

        if (vtype == VKMODEL_VAL_UINT32 || vtype == VKMODEL_VAL_UINT64) {
            if (strcmp(key, "general.alignment") == 0) {
                uint64_t a = val.v.u64;
                if (a < 8 || (a % 8) != 0) return VK_ERROR_INITIALIZATION_FAILED;
                m->alignment = a;
            }
        }
    }
    return VK_SUCCESS;
}

static VkResult vkmodel_parse_tensors(vkmodel_reader_t* r, VkModel* m)
{
    for (uint64_t i = 0; i < m->tensor_count; i++) {
        VkModelTensor* t = &m->tensors[i];

        VkResult res = vkmodel_rd_str(r, &t->name);
        if (res != VK_SUCCESS) return res;

        uint32_t n_dims = 0;
        res = vkmodel_rd_u32(r, &n_dims);
        if (res != VK_SUCCESS) return res;
        if (n_dims == 0 || n_dims > VKMODEL_MAX_DIMS)
            return VK_ERROR_INITIALIZATION_FAILED;
        t->n_dims = n_dims;

        t->nelems = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = 0;
            res = vkmodel_rd_u64(r, &dim);
            if (res != VK_SUCCESS) return res;
            if (dim == 0) return VK_ERROR_INITIALIZATION_FAILED;
            t->dims[d] = dim;
            t->nelems *= dim;
        }

        uint32_t dtype = 0;
        res = vkmodel_rd_u32(r, &dtype);
        if (res != VK_SUCCESS) return res;
        t->dtype = dtype;

        uint64_t offset = 0;
        res = vkmodel_rd_u64(r, &offset);
        if (res != VK_SUCCESS) return res;
        t->offset = offset;

        t->size = vkmodel_tensor_bytes(dtype, t->nelems);
        if (t->size == 0) return VK_ERROR_INITIALIZATION_FAILED;
    }

    m->data_offset = vkmodel_align_up(r->pos, m->alignment);
    return VK_SUCCESS;
}

/* ===========================================================================
 * Tensor data upload (streamed)
 * ========================================================================== */

static VkResult vkmodel_upload_tensor(VkModel* m, vkmodel_reader_t* r,
                                      uint32_t idx)
{
    VkModelTensor* t = &m->tensors[idx];
    if (t->size == 0) return VK_SUCCESS;

    VkResult res = vkr_malloc(m->rt, t->size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &t->buffer, &t->memory);
    if (res != VK_SUCCESS) return res;

    uint64_t file_off = m->data_offset + t->offset;
    if (VKMODEL_FSEEK(r->fp, (long long)file_off, SEEK_SET) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    void* chunk = (void*)malloc(VKMODEL_STREAM_CHUNK);
    if (!chunk) return VK_ERROR_OUT_OF_HOST_MEMORY;

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(vkr_get_device(m->rt), 0, 0, &queue);
    if (queue == VK_NULL_HANDLE) {
        free(chunk);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint64_t remaining = t->size;
    VkDeviceSize upload_off = 0;
    while (remaining > 0) {
        uint64_t n = remaining < (uint64_t)VKMODEL_STREAM_CHUNK
                         ? remaining
                         : (uint64_t)VKMODEL_STREAM_CHUNK;
        if (fread(chunk, 1, (size_t)n, r->fp) != n) {
            free(chunk);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        res = vkr_upload(m->rt, m->upload_cmd, queue, chunk, t->buffer,
                         upload_off, (VkDeviceSize)n);
        if (res != VK_SUCCESS) {
            free(chunk);
            return res;
        }
        remaining -= n;
        upload_off += (VkDeviceSize)n;
    }

    free(chunk);
    return VK_SUCCESS;
}

/* ===========================================================================
 * Public API: lifecycle
 * ========================================================================== */

VkResult vkmodel_load(VkRuntime* rt, const char* path, VkModel** pModel)
{
    if (!rt || !path || !pModel) return VK_ERROR_INITIALIZATION_FAILED;
    *pModel = NULL;

    VkModel* m = (VkModel*)calloc(1, sizeof(VkModel));
    if (!m) return VK_ERROR_OUT_OF_HOST_MEMORY;
    m->rt = rt;
    m->alignment = VKMODEL_DEFAULT_ALIGN;
    m->upload_pool = VK_NULL_HANDLE;
    m->upload_cmd = VK_NULL_HANDLE;

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        free(m);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkmodel_reader_t reader;
    memset(&reader, 0, sizeof(reader));
    reader.fp = fp;

    VkResult res = vkmodel_parse_header(&reader, m);
    if (res == VK_SUCCESS) res = vkmodel_parse_metadata(&reader, m);
    if (res == VK_SUCCESS) res = vkmodel_parse_tensors(&reader, m);
    if (res != VK_SUCCESS) {
        fclose(fp);
        vkmodel_destroy(m);
        return res;
    }

    res = vkmodel_create_upload_state(m);
    if (res != VK_SUCCESS) {
        fclose(fp);
        vkmodel_destroy(m);
        return res;
    }

    for (uint32_t i = 0; i < m->tensor_count; i++) {
        res = vkmodel_upload_tensor(m, &reader, i);
        if (res != VK_SUCCESS) {
            fclose(fp);
            vkmodel_destroy(m);
            return res;
        }
    }

    fclose(fp);
    *pModel = m;
    return VK_SUCCESS;
}

/* ===========================================================================
 * Safetensors loader
 *
 * Format: 8-byte little-endian header length N, then N bytes of UTF-8 JSON
 * (object: tensor name -> {dtype, shape, data_offsets}, plus an optional
 * "__metadata__" object), then tensor data. Tensor k is stored at absolute
 * file offset 8 + N + data_offsets[0].
 *
 * The JSON is parsed with the self-contained parser above; each tensor is
 * stored into the same VkModelTensor array as GGUF, with data_offset =
 * 8 + N so the shared streamed-upload path locates bytes verbatim. __metadata__
 * string entries are exposed as regular VKModel KV pairs (non-string values
 * are skipped; get_kv_string() returns NULL for them, matching GGUF).
 * ========================================================================== */

typedef struct {
    const char* name;    /**< Safetensors dtype name.                       */
    uint32_t     ggml;   /**< ggml_type mapping (VKMODEL_DTYPE_UNKNOWN if none). */
    uint32_t     esize;  /**< Element size in bytes.                        */
} VkModelStDtype;

static const VkModelStDtype vkmodel_st_dtypes[] = {
    { "F32",      0,                  4 },
    { "F16",      1,                  2 },
    { "BF16",     30,                 2 },
    { "F64",      28,                 8 },
    { "I8",       24,                 1 },
    { "I16",      25,                 2 },
    { "I32",      26,                 4 },
    { "I64",      27,                 8 },
    { "U8",       VKMODEL_DTYPE_UNKNOWN, 1 },
    { "U16",      VKMODEL_DTYPE_UNKNOWN, 2 },
    { "U32",      VKMODEL_DTYPE_UNKNOWN, 4 },
    { "U64",      VKMODEL_DTYPE_UNKNOWN, 8 },
    { "BOOL",     VKMODEL_DTYPE_UNKNOWN, 1 },
    { "F8_E4M3",  VKMODEL_DTYPE_UNKNOWN, 1 },
    { "F8_E5M2",  VKMODEL_DTYPE_UNKNOWN, 1 },
};

static int vkmodel_st_dtype_lookup(const char* name, uint32_t* out_ggml,
                                   uint32_t* out_esize, const char** out_name)
{
    for (size_t i = 0; i < sizeof(vkmodel_st_dtypes) / sizeof(vkmodel_st_dtypes[0]); i++) {
        if (strcmp(name, vkmodel_st_dtypes[i].name) == 0) {
            *out_ggml  = vkmodel_st_dtypes[i].ggml;
            *out_esize = vkmodel_st_dtypes[i].esize;
            *out_name  = vkmodel_st_dtypes[i].name;
            return 1;
        }
    }
    return 0;
}

/**
 * \brief Count tensors and metadata string entries in a parsed header.
 *
 * \param root     Parsed header (JSON object).
 * \param p_tensor Receives the tensor count.
 * \param p_kv     Receives the metadata KV count (string entries only).
 */
static void vkmodel_st_count(VkModelJsonNode* root, uint32_t* p_tensor,
                             uint32_t* p_kv)
{
    uint32_t tc = 0, kc = 0;
    for (size_t i = 0; i < root->count; i++) {
        VkModelJsonNode* m = &root->members[i];
        if (!m->key) continue;
        if (strcmp(m->key, "__metadata__") == 0) {
            if (m->type == VKMODEL_JSON_OBJECT) {
                for (size_t j = 0; j < m->count; j++) {
                    if (m->members[j].key &&
                        m->members[j].type == VKMODEL_JSON_STRING) kc++;
                }
            }
            continue;
        }
        if (m->type == VKMODEL_JSON_OBJECT) tc++;
    }
    *p_tensor = tc;
    *p_kv = kc;
}

static VkResult vkmodel_st_fill(VkModelJsonNode* root, VkModel* m)
{
    uint32_t ti = 0, ki = 0;
    for (size_t i = 0; i < root->count; i++) {
        VkModelJsonNode* memb = &root->members[i];
        if (!memb->key) continue;

        if (strcmp(memb->key, "__metadata__") == 0) {
            if (memb->type != VKMODEL_JSON_OBJECT)
                return VK_ERROR_INITIALIZATION_FAILED;
            for (size_t j = 0; j < memb->count; j++) {
                VkModelJsonNode* kv = &memb->members[j];
                if (!kv->key || kv->type != VKMODEL_JSON_STRING) continue;
                m->kv[ki].key = vkmodel_strdup(kv->key);
                if (!m->kv[ki].key) return VK_ERROR_OUT_OF_HOST_MEMORY;
                m->kv[ki].value.type = VKMODEL_VAL_STRING;
                m->kv[ki].value.v.str = vkmodel_strdup(kv->str);
                if (!m->kv[ki].value.v.str) return VK_ERROR_OUT_OF_HOST_MEMORY;
                ki++;
            }
            continue;
        }

        /* tensor entry */
        if (memb->type != VKMODEL_JSON_OBJECT)
            return VK_ERROR_INITIALIZATION_FAILED;

        VkModelJsonNode* dt    = vkmodel_json_find(memb, "dtype");
        VkModelJsonNode* shape = vkmodel_json_find(memb, "shape");
        VkModelJsonNode* offs  = vkmodel_json_find(memb, "data_offsets");
        if (!dt || dt->type != VKMODEL_JSON_STRING)
            return VK_ERROR_INITIALIZATION_FAILED;
        if (!shape || shape->type != VKMODEL_JSON_ARRAY)
            return VK_ERROR_INITIALIZATION_FAILED;
        if (!offs || offs->type != VKMODEL_JSON_ARRAY || offs->count != 2)
            return VK_ERROR_INITIALIZATION_FAILED;

        uint32_t ggml = 0, esize = 0;
        const char* dname = NULL;
        if (!vkmodel_st_dtype_lookup(dt->str, &ggml, &esize, &dname))
            return VK_ERROR_INITIALIZATION_FAILED;   /* unsupported dtype */

        if (shape->count == 0 || shape->count > VKMODEL_MAX_DIMS)
            return VK_ERROR_INITIALIZATION_FAILED;

        VkModelTensor* t = &m->tensors[ti];
        t->n_dims = (uint32_t)shape->count;
        t->nelems = 1;
        for (size_t d = 0; d < shape->count; d++) {
            VkModelJsonNode* dim = &shape->members[d];
            if (dim->type != VKMODEL_JSON_NUMBER || dim->num == 0)
                return VK_ERROR_INITIALIZATION_FAILED;
            if (t->nelems > UINT64_MAX / dim->num)
                return VK_ERROR_INITIALIZATION_FAILED;   /* overflow */
            t->dims[d] = dim->num;
            t->nelems *= dim->num;
        }

        if (offs->members[0].type != VKMODEL_JSON_NUMBER ||
            offs->members[1].type != VKMODEL_JSON_NUMBER)
            return VK_ERROR_INITIALIZATION_FAILED;
        uint64_t start = offs->members[0].num;
        uint64_t end   = offs->members[1].num;
        if (end <= start) return VK_ERROR_INITIALIZATION_FAILED;
        if (end - start != t->nelems * esize)
            return VK_ERROR_INITIALIZATION_FAILED;   /* size/dtype mismatch */

        t->name = vkmodel_strdup(memb->key);
        if (!t->name) return VK_ERROR_OUT_OF_HOST_MEMORY;
        t->dtype      = ggml;
        t->dtype_name = dname;                       /* static, not owned */
        t->offset     = start;
        t->size       = (VkDeviceSize)(end - start);
        ti++;
    }
    return VK_SUCCESS;
}

VkResult vkmodel_load_safetensors(VkRuntime* rt, const char* path,
                                  VkModel** pModel)
{
    if (!rt || !path || !pModel) return VK_ERROR_INITIALIZATION_FAILED;
    *pModel = NULL;

    VkModel* m = (VkModel*)calloc(1, sizeof(VkModel));
    if (!m) return VK_ERROR_OUT_OF_HOST_MEMORY;
    m->rt = rt;

    FILE* fp = fopen(path, "rb");
    if (!fp) { free(m); return VK_ERROR_INITIALIZATION_FAILED; }

    /* 8-byte little-endian header length */
    uint8_t hdr[8];
    if (fread(hdr, 1, 8, fp) != 8) {
        fclose(fp); free(m); return VK_ERROR_INITIALIZATION_FAILED;
    }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)hdr[i] << (8 * i);
    if (hlen == 0 || hlen > VKMODEL_MAX_STRING) {
        fclose(fp); free(m); return VK_ERROR_INITIALIZATION_FAILED;
    }

    char* json = (char*)malloc((size_t)hlen + 1);
    if (!json) { fclose(fp); free(m); return VK_ERROR_OUT_OF_HOST_MEMORY; }
    if (fread(json, 1, (size_t)hlen, fp) != hlen) {
        free(json); fclose(fp); free(m); return VK_ERROR_INITIALIZATION_FAILED;
    }
    json[hlen] = '\0';

    VkModelJsonNode* root = vkmodel_json_parse(json, hlen);
    if (!root || root->type != VKMODEL_JSON_OBJECT) {
        vkmodel_json_free(root);
        free(json); fclose(fp); free(m);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    /* first pass: allocate exact host arrays */
    uint32_t tc = 0, kc = 0;
    vkmodel_st_count(root, &tc, &kc);
    if (tc > (1u << 20) || kc > (1u << 20)) {
        vkmodel_json_free(root);
        free(json); fclose(fp); free(m);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (tc > 0) {
        m->tensors = (VkModelTensor*)calloc(tc, sizeof(VkModelTensor));
        if (!m->tensors) {
            vkmodel_json_free(root);
            free(json); fclose(fp); free(m);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    if (kc > 0) {
        m->kv = (VkModelKV*)calloc(kc, sizeof(VkModelKV));
        if (!m->kv) {
            vkmodel_json_free(root);
            free(json); fclose(fp); free(m);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    m->tensor_count = tc;
    m->kv_count = kc;

    VkResult res = vkmodel_st_fill(root, m);
    vkmodel_json_free(root);
    free(json);
    if (res != VK_SUCCESS) {
        fclose(fp);
        vkmodel_destroy(m);
        return res;
    }

    /* data region starts right after the header */
    m->data_offset = 8 + hlen;

    res = vkmodel_create_upload_state(m);
    if (res != VK_SUCCESS) {
        fclose(fp);
        vkmodel_destroy(m);
        return res;
    }

    vkmodel_reader_t reader;
    memset(&reader, 0, sizeof(reader));
    reader.fp = fp;
    for (uint32_t i = 0; i < m->tensor_count; i++) {
        res = vkmodel_upload_tensor(m, &reader, i);
        if (res != VK_SUCCESS) {
            fclose(fp);
            vkmodel_destroy(m);
            return res;
        }
    }

    fclose(fp);
    *pModel = m;
    return VK_SUCCESS;
}

/* ===========================================================================
 * OpenVINO IR loader (.xml topology + .bin weights)
 *
 * Format: a UTF-8 XML file describing the model topology and a raw
 * little-endian .bin blob holding every constant's bytes. Two idioms are
 * materialized as tensors (the loader never graph-builds):
 *   1. IR v10-v12: a <layer type="Const"> whose <data> carries element_type,
 *      shape, offset, size -> one tensor named after the layer; offset/size
 *      point into the .bin verbatim.
 *   2. Legacy: <weights offset size/> / <biases offset size/> child elements
 *      inside a layer -> one tensor per element, named "<layer>_<tag>"; the
 *      element type/shape come from the layer's <data> attributes, falling
 *      back to the first <output><port> precision/dims.
 * Parameter layers carry no offset/size and are skipped (they are graph
 * inputs, not weights). The .bin is the data region: data_offset = 0 so the
 * shared streamed upload path reads each tensor's slice at its verbatim
 * offset. Tensor byte size must equal shape-product x element-size, and the
 * referenced slice must lie inside the .bin (hardened against the offset+size
 * overflow class of bug OpenVINO itself shipped).
 *
 * The XML scanner below is self-contained (no external dependency): it walks
 * tags/attributes, is self-closing-tag aware, tracks nesting depth, and
 * decodes the five standard entities plus numeric char references.
 * ========================================================================== */

typedef struct {
    const char* tag;        /**< "weights" / "biases" (static string).   */
    uint64_t    offset;     /**< Byte offset into the .bin (verbatim).   */
    uint64_t    size;       /**< Byte size of the .bin slice.            */
} VkModelOvBlob;

typedef struct {
    char*       name;           /**< Layer name (owned).                 */
    int         is_const;       /**< type is Const/Constant.             */
    char*       element_type;   /**< <data element_type> (owned).        */
    char*       shape;          /**< <data shape> (owned; "" = scalar).  */
    uint64_t    offset;         /**< <data offset> (bytes into .bin).    */
    uint64_t    size;           /**< <data size> (bytes).                */
    int         has_data_blob;  /**< <data> carried offset + size.       */
    uint32_t    blob_count;     /**< weights/biases child element count. */
    VkModelOvBlob blobs[VKMODEL_OV_MAX_BLOBS];
    uint64_t    out_dims[VKMODEL_MAX_DIMS]; /**< First <output> port dims.*/
    uint32_t    out_n_dims;     /**< Number of captured out_dims.        */
    char*       port_precision; /**< <output><port precision> (owned).   */
} VkModelOvLayer;

typedef struct {
    const char* name;           /**< Canonical IR element-type name.     */
    uint32_t    ggml;           /**< ggml_type (UNKNOWN sentinel if none).*/
    uint32_t    esize;          /**< Element size in bytes.              */
    int         supported;      /**< 1 = whole-byte esize, uploadable.   */
} VkModelOvDtype;

static const VkModelOvDtype vkmodel_ov_dtypes[] = {
    { "f32",      0,                    4, 1 },
    { "f16",      1,                    2, 1 },
    { "bf16",     30,                   2, 1 },
    { "f64",      28,                   8, 1 },
    { "i8",       24,                   1, 1 },
    { "i16",      25,                   2, 1 },
    { "i32",      26,                   4, 1 },
    { "i64",      27,                   8, 1 },
    { "u8",       VKMODEL_DTYPE_UNKNOWN, 1, 1 },
    { "u16",      VKMODEL_DTYPE_UNKNOWN, 2, 1 },
    { "u32",      VKMODEL_DTYPE_UNKNOWN, 4, 1 },
    { "u64",      VKMODEL_DTYPE_UNKNOWN, 8, 1 },
    { "boolean",  VKMODEL_DTYPE_UNKNOWN, 1, 1 },
    { "f8e4m3",   VKMODEL_DTYPE_UNKNOWN, 1, 1 },
    { "f8e5m2",   VKMODEL_DTYPE_UNKNOWN, 1, 1 },
    { "f8e8m0",   VKMODEL_DTYPE_UNKNOWN, 1, 1 },
    /* Sub-byte packed / variable / non-static: no clean element size.   */
    { "i4",        0, 0, 0 }, { "u1", 0, 0, 0 }, { "u2", 0, 0, 0 },
    { "u3",        0, 0, 0 }, { "u4", 0, 0, 0 }, { "u6", 0, 0, 0 },
    { "nf4",       0, 0, 0 }, { "f4e2m1", 0, 0, 0 }, { "string", 0, 0, 0 },
    { "dynamic",   0, 0, 0 }, { "undefined", 0, 0, 0 },
};

static int vkmodel_ascii_ieq(const char* a, const char* b)
{
    while (*a && *b) {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

/* Port-precision strings written by the OpenVINO model exporter on <port>
 * elements differ from IR element-type names ("FP32" vs "f32", "BOOL" vs
 * "boolean", "FP8E4M3" vs "f8e4m3", ...). Map them to the canonical names. */
static const struct {
    const char* pp;
    const char* name;
} vkmodel_ov_port_aliases[] = {
    { "FP32", "f32" },       { "FP16", "f16" },   { "BF16", "bf16" },
    { "F64", "f64" },
    { "I4", "i4" },          { "I8", "i8" },      { "I16", "i16" },
    { "I32", "i32" },        { "I64", "i64" },
    { "U1", "u1" },          { "U2", "u2" },      { "U3", "u3" },
    { "U4", "u4" },          { "U6", "u6" },      { "U8", "u8" },
    { "U16", "u16" },        { "U32", "u32" },    { "U64", "u64" },
    { "BOOL", "boolean" },
    { "NF4", "nf4" },        { "FP8E4M3", "f8e4m3" },
    { "FP8E5M2", "f8e5m2" }, { "FP8E8M0", "f8e8m0" },
    { "F4E2M1", "f4e2m1" },
    { "STRING", "string" },  { "DYNAMIC", "dynamic" },
};

static int vkmodel_all_digits(const char* s)
{
    if (!s || !s[0]) return 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
    }
    return 1;
}

/**
 * \brief Resolve an OpenVINO element type to the dtype table entry.
 *
 * Accepts the IR string name ("f32", "bf16", ...) case-insensitively, or a
 * numeric string holding the ov::element::Type_t enum value (dynamic=0,
 * boolean=1, bf16=2, f16=3, f32=4, f64=5, i4=6, i8=7, i16=8, i32=9, i64=10,
 * u1=11, u2=12, u3=13, u4=14, u6=15, u8=16, u16=17, u32=18, u64=19, nf4=20,
 * f8e4m3=21, f8e5m2=22, string=23, f4e2m1=24, f8e8m0=25).
 *
 * \param name Element type as written in the IR file.
 * \return Table entry, or NULL for unknown/out-of-range types.
 */
static const VkModelOvDtype* vkmodel_ov_dtype_lookup(const char* name)
{
    if (!name) return NULL;
    if (vkmodel_all_digits(name)) {
        uint64_t num = 0;
        for (const char* p = name; *p; p++) {
            if (num > (UINT64_MAX - 9) / 10) return NULL;
            num = num * 10 + (uint64_t)(*p - '0');
        }
        switch (num) {
        case 1:  return vkmodel_ov_dtype_lookup("boolean");
        case 2:  return vkmodel_ov_dtype_lookup("bf16");
        case 3:  return vkmodel_ov_dtype_lookup("f16");
        case 4:  return vkmodel_ov_dtype_lookup("f32");
        case 5:  return vkmodel_ov_dtype_lookup("f64");
        case 6:  return vkmodel_ov_dtype_lookup("i4");
        case 7:  return vkmodel_ov_dtype_lookup("i8");
        case 8:  return vkmodel_ov_dtype_lookup("i16");
        case 9:  return vkmodel_ov_dtype_lookup("i32");
        case 10: return vkmodel_ov_dtype_lookup("i64");
        case 11: return vkmodel_ov_dtype_lookup("u1");
        case 12: return vkmodel_ov_dtype_lookup("u2");
        case 13: return vkmodel_ov_dtype_lookup("u3");
        case 14: return vkmodel_ov_dtype_lookup("u4");
        case 15: return vkmodel_ov_dtype_lookup("u6");
        case 16: return vkmodel_ov_dtype_lookup("u8");
        case 17: return vkmodel_ov_dtype_lookup("u16");
        case 18: return vkmodel_ov_dtype_lookup("u32");
        case 19: return vkmodel_ov_dtype_lookup("u64");
        case 20: return vkmodel_ov_dtype_lookup("nf4");
        case 21: return vkmodel_ov_dtype_lookup("f8e4m3");
        case 22: return vkmodel_ov_dtype_lookup("f8e5m2");
        case 23: return vkmodel_ov_dtype_lookup("string");
        case 24: return vkmodel_ov_dtype_lookup("f4e2m1");
        case 25: return vkmodel_ov_dtype_lookup("f8e8m0");
        default: return NULL;   /* 0 = dynamic, higher = unknown */
        }
    }
    for (size_t i = 0; i < sizeof(vkmodel_ov_dtypes) /
                                sizeof(vkmodel_ov_dtypes[0]); i++) {
        if (vkmodel_ascii_ieq(name, vkmodel_ov_dtypes[i].name))
            return &vkmodel_ov_dtypes[i];
    }
    for (size_t i = 0; i < sizeof(vkmodel_ov_port_aliases) /
                                sizeof(vkmodel_ov_port_aliases[0]); i++) {
        if (vkmodel_ascii_ieq(name, vkmodel_ov_port_aliases[i].pp))
            return vkmodel_ov_dtype_lookup(vkmodel_ov_port_aliases[i].name);
    }
    return NULL;
}

static void vkmodel_ov_layer_free(VkModelOvLayer* L)
{
    free(L->name);
    free(L->element_type);
    free(L->shape);
    free(L->port_precision);
    memset(L, 0, sizeof(*L));
}

/* ===========================================================================
 * Minimal XML scanner (self-contained)
 * ========================================================================== */

typedef struct {
    const char *s;              /**< XML text.                            */
    size_t      len;            /**< Text byte length.                    */
    size_t      pos;            /**< Current cursor.                      */
    int         failed;         /**< Scan failed flag.                    */
} VkModelXmlParser;

typedef struct {
    char *key;                  /**< Attribute name (owned).              */
    char *val;                  /**< Attribute value (owned).             */
} VkModelXmlAttr;

typedef struct {
    char          *name;        /**< Element name (owned).                */
    VkModelXmlAttr *attrs;      /**< Attribute list.                      */
    uint32_t       attr_count;  /**< Number of attributes.                */
    uint32_t       attr_cap;    /**< Allocated attr capacity.             */
    int            is_end;      /**< '</name>' close tag.                 */
    int            self_closed; /**< '<name ... />' self-closing.         */
} VkModelXmlTag;

static int vkmodel_xml_skip_ws(VkModelXmlParser* p)
{
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
    return 1;
}

static int vkmodel_xml_putc(VkModelXmlParser* p, char** buf, size_t* cap,
                            size_t* len, unsigned char b)
{
    if (*len + 1 >= *cap) {
        size_t ncap = *cap * 2;
        char* nb = (char*)realloc(*buf, ncap);
        if (!nb) { p->failed = 1; return 0; }
        *buf = nb;
        *cap = ncap;
    }
    (*buf)[(*len)++] = (char)b;
    return 1;
}

static int vkmodel_xml_parse_name(VkModelXmlParser* p, char** out)
{
    size_t start = p->pos;
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
            c == '/' || c == '>' || c == '=') break;
        p->pos++;
    }
    if (p->pos == start) { p->failed = 1; return 0; }
    size_t n = p->pos - start;
    char* name = (char*)malloc(n + 1);
    if (!name) { p->failed = 1; return 0; }
    memcpy(name, p->s + start, n);
    name[n] = '\0';
    *out = name;
    return 1;
}

/**
 * \brief Parse one quoted attribute value (with entity decoding).
 *
 * Handles the five standard XML entities (&amp; &lt; &gt; &quot; &apos;)
 * and decimal/hex numeric char refs; unknown entities are kept verbatim.
 *
 * \retval 1 on success, 0 on parse failure.
 */
static int vkmodel_xml_parse_attr_val(VkModelXmlParser* p, char** out)
{
    vkmodel_xml_skip_ws(p);
    if (p->pos >= p->len ||
        (p->s[p->pos] != '"' && p->s[p->pos] != '\'')) {
        p->failed = 1;
        return 0;
    }
    char q = p->s[p->pos];
    p->pos++;
    size_t cap = 32, len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) { p->failed = 1; return 0; }

    for (;;) {
        if (p->pos >= p->len) { free(buf); p->failed = 1; return 0; }
        unsigned char c = (unsigned char)p->s[p->pos];
        if ((char)c == q) {
            p->pos++;
            buf[len] = '\0';
            *out = buf;
            return 1;
        }
        if (c == '&') {
            p->pos++;
            size_t es = p->pos;
            while (p->pos < p->len && p->s[p->pos] != ';') p->pos++;
            if (p->pos >= p->len) { free(buf); p->failed = 1; return 0; }
            size_t elen = p->pos - es;
            const char* e = p->s + es;
            p->pos++;

            unsigned char outbyte;
            int known = 0;
            if (elen == 3 && memcmp(e, "amp", 3) == 0)     { outbyte = '&';  known = 1; }
            else if (elen == 2 && memcmp(e, "lt", 2) == 0) { outbyte = '<';  known = 1; }
            else if (elen == 2 && memcmp(e, "gt", 2) == 0) { outbyte = '>';  known = 1; }
            else if (elen == 4 && memcmp(e, "quot", 4) == 0) { outbyte = '"'; known = 1; }
            else if (elen == 4 && memcmp(e, "apos", 4) == 0) { outbyte = '\''; known = 1; }
            if (known) {
                if (!vkmodel_xml_putc(p, &buf, &cap, &len, outbyte)) {
                    free(buf); return 0;
                }
                continue;
            }
            if (elen >= 2 && e[0] == '#') {
                int base = 10;
                const char* digits = e + 1;
                size_t dlen = elen - 1;
                if (dlen >= 2 && (e[1] == 'x' || e[1] == 'X')) {
                    base = 16;
                    digits = e + 2;
                    dlen -= 1;
                }
                unsigned long cp = 0;
                int ok = 1;
                for (size_t i = 0; i < dlen; i++) {
                    char h = digits[i];
                    unsigned hv;
                    if (h >= '0' && h <= '9')      hv = (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') hv = (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') hv = (unsigned)(h - 'A' + 10);
                    else { ok = 0; break; }
                    cp = cp * (unsigned long)base + hv;
                    if (cp > 0x10FFFFuL) { ok = 0; break; }
                }
                if (ok && cp >= 1) {
                    int r = 0;
                    if (cp < 0x80) {
                        r = vkmodel_xml_putc(p, &buf, &cap, &len, (unsigned char)cp) ? 1 : -1;
                    } else if (cp < 0x800) {
                        r = (vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0xC0 | (cp >> 6))) &&
                             vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0x80 | (cp & 0x3F)))) ? 1 : -1;
                    } else if (cp < 0x10000) {
                        r = (vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0xE0 | (cp >> 12))) &&
                             vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0x80 | ((cp >> 6) & 0x3F))) &&
                             vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0x80 | (cp & 0x3F)))) ? 1 : -1;
                    } else {
                        r = (vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0xF0 | (cp >> 18))) &&
                             vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0x80 | ((cp >> 12) & 0x3F))) &&
                             vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0x80 | ((cp >> 6) & 0x3F))) &&
                             vkmodel_xml_putc(p, &buf, &cap, &len,
                                              (unsigned char)(0x80 | (cp & 0x3F)))) ? 1 : -1;
                    }
                    if (r < 0) { free(buf); return 0; }
                    continue;
                }
            }
            /* Unknown entity: keep "&...;" verbatim. */
            if (!vkmodel_xml_putc(p, &buf, &cap, &len, '&')) { free(buf); return 0; }
            for (size_t i = 0; i < elen; i++) {
                if (!vkmodel_xml_putc(p, &buf, &cap, &len,
                                      (unsigned char)e[i])) { free(buf); return 0; }
            }
            if (!vkmodel_xml_putc(p, &buf, &cap, &len, ';')) { free(buf); return 0; }
            continue;
        }
        if (!vkmodel_xml_putc(p, &buf, &cap, &len, c)) { free(buf); return 0; }
        p->pos++;
    }
}

static void vkmodel_xml_tag_free(VkModelXmlTag* t)
{
    free(t->name);
    for (uint32_t i = 0; i < t->attr_count; i++) {
        free(t->attrs[i].key);
        free(t->attrs[i].val);
    }
    free(t->attrs);
    memset(t, 0, sizeof(*t));
}

static const char* vkmodel_xml_attr_find(const VkModelXmlTag* t,
                                         const char* key)
{
    for (uint32_t i = 0; i < t->attr_count; i++) {
        if (strcmp(t->attrs[i].key, key) == 0) return t->attrs[i].val;
    }
    return NULL;
}

static int vkmodel_xml_attr_u64(const VkModelXmlTag* t, const char* key,
                                uint64_t* out)
{
    const char* v = vkmodel_xml_attr_find(t, key);
    if (!v || !v[0] || !vkmodel_all_digits(v)) return 0;
    uint64_t val = 0;
    for (; *v; v++) {
        if (val > (UINT64_MAX - 9) / 10) return 0;
        val = val * 10 + (uint64_t)(*v - '0');
    }
    *out = val;
    return 1;
}

/**
 * \brief Skip an XML declaration, comment, CDATA section, or DOCTYPE.
 *
 * Must be called with p->s[p->pos] == '<'. Returns 1 if a special construct
 * was consumed (and p->failed set on unterminated input), 0 if the following
 * text is a regular tag.
 */
static int vkmodel_xml_skip_special(VkModelXmlParser* p)
{
    if (p->pos + 1 >= p->len) return 0;
    char c = p->s[p->pos + 1];
    if (c == '?') {
        p->pos += 2;
        while (p->pos + 1 < p->len &&
               !(p->s[p->pos] == '?' && p->s[p->pos + 1] == '>')) p->pos++;
        if (p->pos + 1 >= p->len) { p->failed = 1; return 1; }
        p->pos += 2;
        return 1;
    }
    if (c == '!') {
        if (p->pos + 3 < p->len && p->s[p->pos + 2] == '-' &&
            p->s[p->pos + 3] == '-') {
            p->pos += 4;
            while (p->pos + 2 < p->len &&
                   !(p->s[p->pos] == '-' && p->s[p->pos + 1] == '-' &&
                     p->s[p->pos + 2] == '>')) p->pos++;
            if (p->pos + 2 >= p->len) { p->failed = 1; return 1; }
            p->pos += 3;
            return 1;
        }
        if (p->pos + 8 < p->len && strncmp(p->s + p->pos + 2, "[CDATA[", 7) == 0) {
            p->pos += 9;
            while (p->pos + 2 < p->len &&
                   !(p->s[p->pos] == ']' && p->s[p->pos + 1] == ']' &&
                     p->s[p->pos + 2] == '>')) p->pos++;
            if (p->pos + 2 >= p->len) { p->failed = 1; return 1; }
            p->pos += 3;
            return 1;
        }
        /* <!DOCTYPE ...> or any other <!...> construct */
        p->pos += 2;
        while (p->pos < p->len && p->s[p->pos] != '>') p->pos++;
        if (p->pos >= p->len) { p->failed = 1; return 1; }
        p->pos++;
        return 1;
    }
    return 0;
}

/**
 * \brief Parse one XML tag (open, self-closing, or end).
 *
 * Precondition: p->s[p->pos] == '<'. On success the tag's name/attributes are
 * heap-owned; the caller frees them with vkmodel_xml_tag_free(). Open/end
 * nesting is not validated here (the caller tracks a name stack).
 *
 * \retval 1 on success, 0 on malformed input.
 */
static int vkmodel_xml_next_tag(VkModelXmlParser* p, VkModelXmlTag* tag)
{
    memset(tag, 0, sizeof(*tag));
    if (p->pos >= p->len || p->s[p->pos] != '<') { p->failed = 1; return 0; }
    p->pos++;
    if (p->pos >= p->len) { p->failed = 1; return 0; }

    if (p->s[p->pos] == '/') {
        p->pos++;
        if (!vkmodel_xml_parse_name(p, &tag->name)) return 0;
        tag->is_end = 1;
        vkmodel_xml_skip_ws(p);
        if (p->pos >= p->len || p->s[p->pos] != '>') { p->failed = 1; return 0; }
        p->pos++;
        return 1;
    }

    if (!vkmodel_xml_parse_name(p, &tag->name)) return 0;
    for (;;) {
        vkmodel_xml_skip_ws(p);
        if (p->pos >= p->len) { p->failed = 1; return 0; }
        char c = p->s[p->pos];
        if (c == '>') { p->pos++; return 1; }
        if (c == '/') {
            p->pos++;
            if (p->pos >= p->len || p->s[p->pos] != '>') { p->failed = 1; return 0; }
            p->pos++;
            tag->self_closed = 1;
            return 1;
        }
        char* key = NULL;
        if (!vkmodel_xml_parse_name(p, &key)) return 0;
        vkmodel_xml_skip_ws(p);
        if (p->pos >= p->len || p->s[p->pos] != '=') {
            free(key);
            p->failed = 1;
            return 0;
        }
        p->pos++;
        char* val = NULL;
        if (!vkmodel_xml_parse_attr_val(p, &val)) { free(key); return 0; }
        if (tag->attr_count >= tag->attr_cap) {
            size_t ncap = tag->attr_cap == 0 ? 8 : tag->attr_cap * 2;
            VkModelXmlAttr* na = (VkModelXmlAttr*)realloc(
                tag->attrs, ncap * sizeof(VkModelXmlAttr));
            if (!na) {
                free(key); free(val);
                p->failed = 1;
                return 0;
            }
            tag->attrs = na;
            tag->attr_cap = (uint32_t)ncap;
        }
        tag->attrs[tag->attr_count].key = key;
        tag->attrs[tag->attr_count].val = val;
        tag->attr_count++;
    }
}

/**
 * \brief Scan the XML document, collecting constant-bearing layer records.
 *
 * Walks every tag while tracking a nesting stack. Inside each <layer> it
 * records <data> element_type/shape/offset/size, <weights>/<biases> blobs,
 * and the first <output><port> precision/dims. The <net> version is captured
 * for the v10-v12 gate.
 *
 * \param xml        NUL-terminated XML text.
 * \param len        Byte length of \p xml.
 * \param out_layers Receives the heap layer array (caller frees each layer
 *                   with vkmodel_ov_layer_free() then free()s the array).
 * \param out_count  Receives the layer count.
 * \param out_version Receives the <net> version attribute.
 */
static VkResult vkmodel_ov_scan_layers(const char* xml, size_t len,
                                       VkModelOvLayer** out_layers,
                                       uint32_t* out_count,
                                       int* out_version)
{
    VkModelXmlParser p;
    p.s = xml;
    p.len = len;
    p.pos = 0;
    p.failed = 0;

    VkModelOvLayer* layers = NULL;
    uint32_t count = 0, cap = 0;
    int active = -1;
    int net_version = -1;
    int in_output = 0, in_first_port = 0;
    char* stack[VKMODEL_OV_MAX_DEPTH];
    int depth = 0;
    char dim_text[64];
    size_t dim_len = 0;
    int ok = 1;
    VkResult res = VK_SUCCESS;

    for (;;) {
        /* capture text between tags (only inside a first-port <dim>) */
        size_t ts = p.pos;
        while (p.pos < p.len && p.s[p.pos] != '<') p.pos++;
        if (in_first_port && depth > 0 &&
            strcmp(stack[depth - 1], "dim") == 0 && p.pos > ts) {
            size_t n = p.pos - ts;
            if (dim_len + n > sizeof(dim_text) - 1) { ok = 0; break; }
            memcpy(dim_text + dim_len, p.s + ts, n);
            dim_len += n;
        }
        if (p.pos >= p.len) break;
        if (vkmodel_xml_skip_special(&p)) {
            if (p.failed) { ok = 0; break; }
            continue;
        }

        VkModelXmlTag tag;
        if (!vkmodel_xml_next_tag(&p, &tag)) { ok = 0; break; }

        if (tag.is_end) {
            if (depth == 0 || strcmp(stack[depth - 1], tag.name) != 0) {
                vkmodel_xml_tag_free(&tag);
                ok = 0;
                break;
            }
            depth--;
            if (strcmp(tag.name, "layer") == 0) active = -1;
            else if (strcmp(tag.name, "output") == 0) in_output = 0;
            else if (strcmp(tag.name, "port") == 0) in_first_port = 0;
            else if (strcmp(tag.name, "dim") == 0 && active >= 0 &&
                     in_first_port) {
                VkModelOvLayer* L = &layers[active];
                if (L->out_n_dims < VKMODEL_MAX_DIMS) {
                    size_t b = 0, e = dim_len;
                    while (b < e && (dim_text[b] == ' ' || dim_text[b] == '\t' ||
                                     dim_text[b] == '\n' || dim_text[b] == '\r')) b++;
                    while (e > b && (dim_text[e - 1] == ' ' || dim_text[e - 1] == '\t' ||
                                     dim_text[e - 1] == '\n' || dim_text[e - 1] == '\r')) e--;
                    if (e == b) { vkmodel_xml_tag_free(&tag); ok = 0; break; }
                    uint64_t v = 0;
                    int bad = 0;
                    for (size_t i = b; i < e; i++) {
                        char c = dim_text[i];
                        if (c < '0' || c > '9') { bad = 1; break; }
                        if (v > (UINT64_MAX - 9) / 10) { bad = 1; break; }
                        v = v * 10 + (uint64_t)(c - '0');
                    }
                    if (bad || v == 0) { vkmodel_xml_tag_free(&tag); ok = 0; break; }
                    L->out_dims[L->out_n_dims++] = v;
                    dim_len = 0;    /* reset between <dim>s and across layers */
                }
            }
            free(stack[depth]);
            vkmodel_xml_tag_free(&tag);
            continue;
        }

        /* open tag */
        if (strcmp(tag.name, "net") == 0) {
            uint64_t v = 0;
            if (vkmodel_xml_attr_u64(&tag, "version", &v)) net_version = (int)v;
        } else if (strcmp(tag.name, "layer") == 0) {
            if (count >= VKMODEL_OV_MAX_LAYERS) {
                vkmodel_xml_tag_free(&tag);
                ok = 0;
                break;
            }
            if (count >= cap) {
                uint32_t ncap = cap == 0 ? 16 : cap * 2;
                if (ncap > VKMODEL_OV_MAX_LAYERS) ncap = VKMODEL_OV_MAX_LAYERS;
                VkModelOvLayer* nl = (VkModelOvLayer*)realloc(
                    layers, (size_t)ncap * sizeof(VkModelOvLayer));
                if (!nl) {
                    vkmodel_xml_tag_free(&tag);
                    res = VK_ERROR_OUT_OF_HOST_MEMORY;
                    ok = 0;
                    break;
                }
                layers = nl;
                cap = ncap;
            }
            VkModelOvLayer* L = &layers[count];
            memset(L, 0, sizeof(*L));
            const char* nm = vkmodel_xml_attr_find(&tag, "name");
            if (nm) {
                L->name = vkmodel_strdup(nm);
                if (!L->name) {
                    vkmodel_xml_tag_free(&tag);
                    res = VK_ERROR_OUT_OF_HOST_MEMORY;
                    ok = 0;
                    break;
                }
            }
            const char* ty = vkmodel_xml_attr_find(&tag, "type");
            L->is_const = (ty != NULL && (vkmodel_ascii_ieq(ty, "Const") ||
                                          vkmodel_ascii_ieq(ty, "Constant")));
            active = (int)count;
            count++;
        } else if (active >= 0) {
            VkModelOvLayer* L = &layers[active];
            if (strcmp(tag.name, "data") == 0) {
                const char* et = vkmodel_xml_attr_find(&tag, "element_type");
                if (et && et[0]) {
                    free(L->element_type);
                    L->element_type = vkmodel_strdup(et);
                }
                const char* sh = vkmodel_xml_attr_find(&tag, "shape");
                if (sh) {
                    free(L->shape);
                    L->shape = vkmodel_strdup(sh);
                }
                uint64_t off = 0, sz = 0;
                if (vkmodel_xml_attr_u64(&tag, "offset", &off) &&
                    vkmodel_xml_attr_u64(&tag, "size", &sz)) {
                    L->offset = off;
                    L->size = sz;
                    L->has_data_blob = 1;
                }
            } else if (strcmp(tag.name, "weights") == 0 ||
                       strcmp(tag.name, "biases") == 0) {
                uint64_t off = 0, sz = 0;
                if (!vkmodel_xml_attr_u64(&tag, "offset", &off) ||
                    !vkmodel_xml_attr_u64(&tag, "size", &sz) ||
                    L->blob_count >= VKMODEL_OV_MAX_BLOBS) {
                    vkmodel_xml_tag_free(&tag);
                    ok = 0;
                    break;
                }
                L->blobs[L->blob_count].tag =
                    (strcmp(tag.name, "weights") == 0) ? "weights" : "biases";
                L->blobs[L->blob_count].offset = off;
                L->blobs[L->blob_count].size = sz;
                L->blob_count++;
            } else if (strcmp(tag.name, "output") == 0) {
                in_output = 1;
                in_first_port = 0;
            } else if (strcmp(tag.name, "port") == 0) {
                if (in_output && !in_first_port) {
                    in_first_port = 1;
                    const char* pr = vkmodel_xml_attr_find(&tag, "precision");
                    if (pr && pr[0]) {
                        free(L->port_precision);
                        L->port_precision = vkmodel_strdup(pr);
                    }
                }
            }
        }

        /* push onto the nesting stack unless self-closing */
        if (!tag.self_closed) {
            if (depth >= (int)VKMODEL_OV_MAX_DEPTH) {
                vkmodel_xml_tag_free(&tag);
                ok = 0;
                break;
            }
            stack[depth++] = tag.name;
            tag.name = NULL;
        }
        vkmodel_xml_tag_free(&tag);
    }

    for (int i = 0; i < depth; i++) free(stack[i]);

    if (!ok) {
        for (uint32_t i = 0; i < count; i++) vkmodel_ov_layer_free(&layers[i]);
        free(layers);
        return res != VK_SUCCESS ? res : VK_ERROR_INITIALIZATION_FAILED;
    }
    if (net_version < (int)VKMODEL_OV_IR_VERSION_MIN ||
        net_version > (int)VKMODEL_OV_IR_VERSION_MAX) {
        for (uint32_t i = 0; i < count; i++) vkmodel_ov_layer_free(&layers[i]);
        free(layers);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *out_layers = layers;
    *out_count = count;
    *out_version = net_version;
    return VK_SUCCESS;
}

static char* vkmodel_strdup3(const char* a, const char* b, const char* c)
{
    size_t na = strlen(a), nb = strlen(b), nc = strlen(c);
    char* s = (char*)malloc(na + nb + nc + 1);
    if (!s) return NULL;
    memcpy(s, a, na);
    memcpy(s + na, b, nb);
    memcpy(s + na + nb, c, nc);
    s[na + nb + nc] = '\0';
    return s;
}

static int vkmodel_ov_parse_shape(const char* s, uint64_t* dims,
                                  uint32_t* n_out, uint64_t* nelems_out)
{
    uint32_t n = 0;
    uint64_t nelems = 1;
    const char* p = s;
    for (;;) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == '\0') break;
        if (*p < '0' || *p > '9') return 0;
        uint64_t v = 0;
        while (*p >= '0' && *p <= '9') {
            if (v > (UINT64_MAX - 9) / 10) return 0;
            v = v * 10 + (uint64_t)(*p - '0');
            p++;
        }
        if (v == 0) return 0;
        if (n >= VKMODEL_MAX_DIMS) return 0;
        if (nelems > UINT64_MAX / v) return 0;
        dims[n++] = v;
        nelems *= v;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == '\0') break;
        if (*p == ',') { p++; continue; }
        return 0;
    }
    if (n == 0) return 0;
    *n_out = n;
    *nelems_out = nelems;
    return 1;
}

/**
 * \brief Fill one VkModelTensor from an OpenVINO constant candidate.
 *
 * Resolves the element type (string name or numeric enum; opaque types are
 * stored with VKMODEL_DTYPE_UNKNOWN), the shape (<data shape>, then output
 * port dims, then scalar for shape=""), and validates size == nelems x
 * element-size and the .bin slice bounds.
 *
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED malformed/unsupported candidate.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY name allocation failed.
 */
static VkResult vkmodel_ov_fill_tensor(VkModel* m, const VkModelOvLayer* L,
                                      const char* tag, uint64_t offset,
                                      uint64_t size, uint32_t idx,
                                      uint64_t bin_size)
{
    if (!L->name) return VK_ERROR_INITIALIZATION_FAILED;

    const char* et = L->element_type;
    if (!et || !et[0]) et = L->port_precision;
    const VkModelOvDtype* dt = vkmodel_ov_dtype_lookup(et);
    if (!dt || !dt->supported) return VK_ERROR_INITIALIZATION_FAILED;

    uint64_t dims[VKMODEL_MAX_DIMS];
    uint32_t nd = 0;
    uint64_t nelems = 1;
    if (L->shape) {
        if (L->shape[0] == '\0') {
            nd = 1;
            dims[0] = 1;
            nelems = 1;                     /* 0-d scalar constant */
        } else if (!vkmodel_ov_parse_shape(L->shape, dims, &nd, &nelems)) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    } else if (L->out_n_dims > 0) {
        nd = L->out_n_dims;
        for (uint32_t d = 0; d < nd; d++) {
            dims[d] = L->out_dims[d];
            if (nelems > UINT64_MAX / dims[d]) return VK_ERROR_INITIALIZATION_FAILED;
            nelems *= dims[d];
        }
    } else {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (nelems > UINT64_MAX / dt->esize)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (size != nelems * dt->esize) return VK_ERROR_INITIALIZATION_FAILED;
    if (offset > UINT64_MAX - size || offset + size > bin_size)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkModelTensor* t = &m->tensors[idx];
    t->name = (tag && strcmp(tag, "data") == 0)
                  ? vkmodel_strdup(L->name)
                  : vkmodel_strdup3(L->name, "_", tag);
    if (!t->name) return VK_ERROR_OUT_OF_HOST_MEMORY;
    t->dtype      = dt->ggml;
    t->dtype_name = dt->name;               /* static, not owned */
    t->n_dims     = nd;
    for (uint32_t d = 0; d < nd; d++) t->dims[d] = dims[d];
    t->nelems     = nelems;
    t->offset     = offset;
    t->size       = size;
    return VK_SUCCESS;
}

VkResult vkmodel_load_openvino(VkRuntime* rt, const char* xml_path,
                               const char* bin_path, VkModel** pModel)
{
    if (!rt || !xml_path || !bin_path || !pModel)
        return VK_ERROR_INITIALIZATION_FAILED;
    *pModel = NULL;

    FILE* xf = fopen(xml_path, "rb");
    if (!xf) return VK_ERROR_INITIALIZATION_FAILED;
    if (VKMODEL_FSEEK(xf, 0, SEEK_END) != 0) {
        fclose(xf);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    long xsize = ftell(xf);
    if (xsize <= 0 || (uint64_t)xsize > VKMODEL_MAX_STRING) {
        fclose(xf);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (VKMODEL_FSEEK(xf, 0, SEEK_SET) != 0) {
        fclose(xf);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    char* xml = (char*)malloc((size_t)xsize + 1);
    if (!xml) {
        fclose(xf);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (fread(xml, 1, (size_t)xsize, xf) != (size_t)xsize) {
        free(xml);
        fclose(xf);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    xml[xsize] = '\0';
    fclose(xf);

    FILE* bf = fopen(bin_path, "rb");
    if (!bf) {
        free(xml);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (VKMODEL_FSEEK(bf, 0, SEEK_END) != 0) {
        fclose(bf);
        free(xml);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    long bsize = ftell(bf);
    if (bsize < 0) {
        fclose(bf);
        free(xml);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (VKMODEL_FSEEK(bf, 0, SEEK_SET) != 0) {
        fclose(bf);
        free(xml);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    uint64_t bin_size = (uint64_t)bsize;

    VkModelOvLayer* layers = NULL;
    uint32_t layer_count = 0;
    int net_version = 0;
    VkResult res = vkmodel_ov_scan_layers(xml, (size_t)xsize, &layers,
                                          &layer_count, &net_version);
    free(xml);
    if (res != VK_SUCCESS) {
        fclose(bf);
        return res;
    }

    uint32_t tc = 0;
    for (uint32_t i = 0; i < layer_count; i++) {
        VkModelOvLayer* L = &layers[i];
        uint32_t extra = (L->is_const && L->has_data_blob) ? 1u : 0u;
        if (extra > VKMODEL_OV_MAX_LAYERS - tc) { tc = 0; break; }
        tc += extra;
        if (L->blob_count > VKMODEL_OV_MAX_LAYERS - tc) { tc = 0; break; }
        tc += L->blob_count;
    }

    VkModel* m = (VkModel*)calloc(1, sizeof(VkModel));
    if (!m) {
        for (uint32_t i = 0; i < layer_count; i++) vkmodel_ov_layer_free(&layers[i]);
        free(layers);
        fclose(bf);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    m->rt = rt;
    m->data_offset = 0;                 /* .bin starts the data region */
    m->tensor_count = tc;
    if (tc > 0) {
        m->tensors = (VkModelTensor*)calloc(tc, sizeof(VkModelTensor));
        if (!m->tensors) {
            free(m);
            for (uint32_t i = 0; i < layer_count; i++) vkmodel_ov_layer_free(&layers[i]);
            free(layers);
            fclose(bf);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    uint32_t ti = 0;
    for (uint32_t i = 0; i < layer_count && res == VK_SUCCESS; i++) {
        VkModelOvLayer* L = &layers[i];
        if (L->is_const && L->has_data_blob) {
            res = vkmodel_ov_fill_tensor(m, L, "data", L->offset, L->size,
                                         ti, bin_size);
            ti++;
        }
        for (uint32_t b = 0; b < L->blob_count && res == VK_SUCCESS; b++) {
            res = vkmodel_ov_fill_tensor(m, L, L->blobs[b].tag,
                                         L->blobs[b].offset, L->blobs[b].size,
                                         ti, bin_size);
            ti++;
        }
    }
    for (uint32_t i = 0; i < layer_count; i++) vkmodel_ov_layer_free(&layers[i]);
    free(layers);

    if (res != VK_SUCCESS || ti != m->tensor_count) {
        vkmodel_destroy(m);
        fclose(bf);
        return res != VK_SUCCESS ? res : VK_ERROR_INITIALIZATION_FAILED;
    }

    res = vkmodel_create_upload_state(m);
    if (res != VK_SUCCESS) {
        vkmodel_destroy(m);
        fclose(bf);
        return res;
    }

    vkmodel_reader_t reader;
    memset(&reader, 0, sizeof(reader));
    reader.fp = bf;
    for (uint32_t i = 0; i < m->tensor_count; i++) {
        res = vkmodel_upload_tensor(m, &reader, i);
        if (res != VK_SUCCESS) {
            fclose(bf);
            vkmodel_destroy(m);
            return res;
        }
    }

    fclose(bf);
    *pModel = m;
    return VK_SUCCESS;
}

void vkmodel_destroy(VkModel* m)
{
    if (!m) return;

    if (m->upload_cmd != VK_NULL_HANDLE && m->upload_pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(vkr_get_device(m->rt), m->upload_pool, 1,
                             &m->upload_cmd);
    }
    if (m->upload_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vkr_get_device(m->rt), m->upload_pool, NULL);
    }

    for (uint32_t i = 0; i < m->kv_count; i++) {
        free(m->kv[i].key);
        vkmodel_value_free(&m->kv[i].value);
    }
    free(m->kv);

    for (uint32_t i = 0; i < m->tensor_count; i++) {
        free(m->tensors[i].name);
        if (m->tensors[i].buffer != VK_NULL_HANDLE) {
            vkr_free(m->rt, m->tensors[i].buffer, m->tensors[i].memory);
        }
    }
    free(m->tensors);

    free(m);
}

/* ===========================================================================
 * Public API: metadata access
 * ========================================================================== */

uint32_t vkmodel_get_kv_count(VkModel* m)
{
    return m ? m->kv_count : 0;
}

const char* vkmodel_get_kv_key(VkModel* m, uint32_t i)
{
    if (!m || i >= m->kv_count) return NULL;
    return m->kv[i].key;
}

const char* vkmodel_get_kv_string(VkModel* m, const char* key,
                                  const char* fallback)
{
    if (!m || !key) return fallback;
    for (uint32_t i = 0; i < m->kv_count; i++) {
        if (m->kv[i].key && strcmp(m->kv[i].key, key) == 0) {
            if (m->kv[i].value.type != VKMODEL_VAL_STRING) return NULL;
            return m->kv[i].value.v.str;
        }
    }
    return fallback;
}

/* ===========================================================================
 * Public API: tensor access
 * ========================================================================== */

uint32_t vkmodel_get_tensor_count(VkModel* m)
{
    return m ? m->tensor_count : 0;
}

const char* vkmodel_get_tensor_name(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return NULL;
    return m->tensors[i].name;
}

uint32_t vkmodel_get_tensor_dtype(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return 0;
    return m->tensors[i].dtype;
}

uint32_t vkmodel_get_tensor_nelems(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return 0;
    return (uint32_t)m->tensors[i].nelems;
}

VkBuffer vkmodel_get_tensor_buffer(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return VK_NULL_HANDLE;
    return m->tensors[i].buffer;
}

/* ggml_type -> canonical name (for GGUF-loaded tensors; safetensors tensors
 * carry their dtype name directly in dtype_name). */
static const struct { uint32_t ggml; const char* name; } vkmodel_ggml_names[] = {
    { 0,  "F32"     }, { 1,  "F16"    }, { 2,  "Q4_0"  }, { 3,  "Q4_1"  },
    { 6,  "Q5_0"    }, { 7,  "Q5_1"   }, { 8,  "Q8_0"  }, { 9,  "Q8_1"  },
    { 10, "Q2_K"    }, { 11, "Q3_K"   }, { 12, "Q4_K"  }, { 13, "Q5_K"  },
    { 14, "Q6_K"    }, { 15, "Q8_K"   }, { 16, "IQ2_XXS"}, { 17, "IQ2_XS"},
    { 18, "IQ3_XXS" }, { 19, "IQ1_S"  }, { 20, "IQ4_NL"}, { 21, "IQ3_S" },
    { 22, "IQ2_S"   }, { 23, "IQ4_XS" }, { 24, "I8"    }, { 25, "I16"   },
    { 26, "I32"     }, { 27, "I64"    }, { 28, "F64"   }, { 29, "IQ1_M" },
    { 30, "BF16"    }, { 34, "TQ1_0"  }, { 35, "TQ2_0" }, { 39, "MXFP4" },
    { 40, "NVFP4"   }, { 41, "Q1_0"   }, { 42, "Q2_0"  },
};

const char* vkmodel_get_tensor_dtype_name(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return NULL;
    VkModelTensor* t = &m->tensors[i];
    if (t->dtype_name) return t->dtype_name;
    if (t->dtype == VKMODEL_DTYPE_UNKNOWN) return NULL;
    for (size_t k = 0; k < sizeof(vkmodel_ggml_names) / sizeof(vkmodel_ggml_names[0]); k++) {
        if (vkmodel_ggml_names[k].ggml == t->dtype) return vkmodel_ggml_names[k].name;
    }
    return NULL;
}

VkDeviceSize vkmodel_get_tensor_size(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return 0;
    return m->tensors[i].size;
}

/* ===========================================================================
 * Public API: type mapping
 * ========================================================================== */

uint32_t vkmodel_block_elems(uint32_t ggml_type)
{
    return vkmodel_blck_elems_of(ggml_type);
}
