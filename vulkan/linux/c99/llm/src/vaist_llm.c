/**
 * \file vaist_llm.c
 * \brief LLM inference primitives: KV cache, sampler, tokenizer, paged KV cache.
 */
#include "vaist_llm.h"
#include "vaist_quant.h"
#include "vaist_runtime.h"
#include "vaist_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ======================================================================== */
/* Existing simple KV cache (unchanged)                                     */
/* ======================================================================== */

struct VaistKVCache { size_t cap, width, len; float *data; };

VAIST_API VaistStatus vaist_kv_create(size_t c, size_t w, VaistKVCache **o) {
    VaistKVCache *k;
    if (!o || !c || !w) return VAIST_INVALID_ARGUMENT;
    k = (VaistKVCache *)calloc(1, sizeof(*k));
    if (!k) return VAIST_OUT_OF_MEMORY;
    k->data = (float *)calloc(c * w, sizeof(float));
    if (!k->data) { free(k); return VAIST_OUT_OF_MEMORY; }
    k->cap = c; k->width = w;
    *o = k;
    return VAIST_OK;
}

VAIST_API void vaist_kv_destroy(VaistKVCache *k) {
    if (k) { free(k->data); free(k); }
}

VAIST_API VaistStatus vaist_kv_write(VaistKVCache *k, size_t p, const float *d) {
    if (!k || !d || p >= k->cap) return VAIST_INVALID_ARGUMENT;
    memcpy(k->data + p * k->width, d, k->width * sizeof(float));
    if (p + 1 > k->len) k->len = p + 1;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_read(const VaistKVCache *k, size_t p, float *o) {
    if (!k || !o || p >= k->len) return VAIST_INVALID_ARGUMENT;
    memcpy(o, k->data + p * k->width, k->width * sizeof(float));
    return VAIST_OK;
}

VAIST_API size_t vaist_kv_length(const VaistKVCache *k) { return k ? k->len : 0; }

/* ======================================================================== */
/* Existing simple sampler (unchanged)                                      */
/* ======================================================================== */

struct VaistSampler { float temperature, top_p; uint32_t top_k; };

VAIST_API VaistStatus vaist_sampler_create(float t, uint32_t k, float p, VaistSampler **o) {
    VaistSampler *s;
    if (!o || t < 0 || p <= 0 || p > 1) return VAIST_INVALID_ARGUMENT;
    s = (VaistSampler *)calloc(1, sizeof(*s));
    if (!s) return VAIST_OUT_OF_MEMORY;
    s->temperature = t; s->top_k = k; s->top_p = p;
    *o = s;
    return VAIST_OK;
}

VAIST_API void vaist_sampler_destroy(VaistSampler *s) { free(s); }

static uint64_t rng_state(uint64_t *x) {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    return *x;
}

VAIST_API VaistStatus vaist_sample_greedy(const float *l, size_t n, uint32_t *o) {
    size_t i, b = 0;
    if (!l || !o || !n) return VAIST_INVALID_ARGUMENT;
    for (i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    *o = (uint32_t)b;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_sample(const float *l, size_t n, VaistSampler *s, uint64_t *st, uint32_t *o) {
    size_t i; float sum = 0, r; uint64_t q;
    if (!l || !s || !st || !o || !n) return VAIST_INVALID_ARGUMENT;
    if (s->temperature == 0) return vaist_sample_greedy(l, n, o);
    float *m = (float *)malloc(n * sizeof(float));
    if (!m) return VAIST_OUT_OF_MEMORY;
    for (i = 0; i < n; i++) { m[i] = expf(l[i] / s->temperature); sum += m[i]; }
    q = rng_state(st);
    r = ((float)(q & 0xffffffu) / 16777216.0f) * sum;
    for (i = 0; i < n; i++) { r -= m[i]; if (r <= 0) break; }
    if (i >= n) i = n - 1;
    *o = (uint32_t)i;
    free(m);
    return VAIST_OK;
}

/* ======================================================================== */
/* Extended sampler: full CFG with penalties, top-k/p, min-p, Gumbel-max    */
/* ======================================================================== */

static uint64_t xorshift64(uint64_t *x) {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    return *x;
}

static float rand_uniform(uint64_t *x) {
    uint64_t v = xorshift64(x);
    return (float)(v >> 11) * (1.0f / 9007199254740992.0f);
}

static float gumbel_max(uint64_t *x) {
    float u = rand_uniform(x);
    if (u <= 0.0f) u = 1e-30f;
    if (u >= 1.0f) u = 1.0f - 1e-30f;
    return -logf(-logf(u));
}

static void count_token_occurrences(const uint32_t *input_ids, size_t input_len, size_t vocab_size, uint32_t *counts) {
    size_t i;
    for (i = 0; i < input_len; i++) {
        if (input_ids[i] < vocab_size) counts[input_ids[i]]++;
    }
}

static void softmax_inplace(float *l, size_t n) {
    size_t i; float max = l[0], sum = 0;
    for (i = 1; i < n; i++) if (l[i] > max) max = l[i];
    for (i = 0; i < n; i++) { l[i] = expf(l[i] - max); sum += l[i]; }
    if (sum > 0) for (i = 0; i < n; i++) l[i] /= sum;
}

static const float *g_work_logits = NULL;

static int cmp_desc_idx(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (g_work_logits[ia] < g_work_logits[ib]) return 1;
    if (g_work_logits[ia] > g_work_logits[ib]) return -1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

VAIST_API VaistStatus vaist_sampler_config_init(
    VaistSamplerConfig *cfg,
    float temp, uint32_t top_k, float top_p, float min_p,
    float presence_penalty, float frequency_penalty) {
    if (!cfg) return VAIST_INVALID_ARGUMENT;
    cfg->temperature = temp;
    cfg->top_k = top_k;
    cfg->top_p = top_p;
    cfg->min_p = min_p;
    cfg->presence_penalty = presence_penalty;
    cfg->frequency_penalty = frequency_penalty;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_sample_logits(const float *logits, size_t n, uint32_t *out) {
    size_t i, b = 0;
    if (!logits || !out || !n) return VAIST_INVALID_ARGUMENT;
    for (i = 1; i < n; i++) if (logits[i] > logits[b]) b = i;
    *out = (uint32_t)b;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_sample_cfg(
    const VaistSampler *s,
    const VaistSamplerConfig *cfg,
    const float *logits, size_t vocab_size,
    const uint32_t *input_ids, size_t input_len,
    uint32_t *out) {
    (void)s;
    float *work = NULL;
    uint32_t *counts = NULL;
    uint32_t *sorted_idx = NULL;
    uint64_t prng;
    size_t i, kept_count, rank;
    float max_logit, threshold, accum, max_prob, gumbel;

    if (!cfg || !logits || !out || !vocab_size) return VAIST_INVALID_ARGUMENT;

    work = (float *)malloc(vocab_size * sizeof(float));
    if (!work) return VAIST_OUT_OF_MEMORY;
    memcpy(work, logits, vocab_size * sizeof(float));

    /* Step 1: Apply presence and frequency penalties */
    if (cfg->presence_penalty > 0 || cfg->frequency_penalty > 0) {
        counts = (uint32_t *)calloc(vocab_size, sizeof(uint32_t));
        if (!counts) { free(work); return VAIST_OUT_OF_MEMORY; }
        if (input_ids && input_len > 0) count_token_occurrences(input_ids, input_len, vocab_size, counts);
        for (i = 0; i < vocab_size; i++) {
            float penalty = cfg->presence_penalty * (counts[i] > 0 ? 1.0f : 0.0f)
                          + cfg->frequency_penalty * (float)counts[i];
            work[i] -= penalty;
        }
        free(counts);
        counts = NULL;
    }

    /* Step 2: Subtract max for numerical stability */
    max_logit = work[0];
    for (i = 1; i < vocab_size; i++) if (work[i] > max_logit) max_logit = work[i];
    for (i = 0; i < vocab_size; i++) work[i] -= max_logit;

    /* Step 3: Divide by temperature */
    if (cfg->temperature > 0 && cfg->temperature < 1e30f) {
        float inv_temp = 1.0f / cfg->temperature;
        for (i = 0; i < vocab_size; i++) work[i] *= inv_temp;
    }

    /* Step 4: Top-K filtering */
    if (cfg->top_k > 0 && cfg->top_k < vocab_size) {
        sorted_idx = (uint32_t *)malloc(vocab_size * sizeof(uint32_t));
        if (!sorted_idx) { free(work); return VAIST_OUT_OF_MEMORY; }
        for (i = 0; i < vocab_size; i++) sorted_idx[i] = (uint32_t)i;
        g_work_logits = work;
        qsort(sorted_idx, vocab_size, sizeof(uint32_t), cmp_desc_idx);
        for (i = cfg->top_k; i < vocab_size; i++) work[sorted_idx[i]] = -INFINITY;
        free(sorted_idx);
        sorted_idx = NULL;
    }

    /* Step 5: Softmax */
    softmax_inplace(work, vocab_size);

    /* Step 6: Top-p (nucleus) filtering */
    if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
        sorted_idx = (uint32_t *)malloc(vocab_size * sizeof(uint32_t));
        if (!sorted_idx) { free(work); return VAIST_OUT_OF_MEMORY; }
        for (i = 0; i < vocab_size; i++) sorted_idx[i] = (uint32_t)i;
        g_work_logits = work;
        qsort(sorted_idx, vocab_size, sizeof(uint32_t), cmp_desc_idx);
        accum = 0.0f;
        kept_count = 0;
        for (i = 0; i < vocab_size; i++) {
            accum += work[sorted_idx[i]];
            work[sorted_idx[i]] = -INFINITY;
            kept_count++;
            if (accum >= cfg->top_p) break;
        }
        for (i = 0; i < kept_count; i++) work[sorted_idx[i]] = logits[sorted_idx[i]];
        free(sorted_idx);
        sorted_idx = NULL;
    }

    /* Step 7: Min-p filtering */
    if (cfg->min_p > 0.0f && cfg->min_p < 1.0f) {
        softmax_inplace(work, vocab_size);
        max_prob = 0.0f;
        for (i = 0; i < vocab_size; i++) if (work[i] > max_prob) max_prob = work[i];
        threshold = cfg->min_p * max_prob;
        for (i = 0; i < vocab_size; i++) if (work[i] < threshold) work[i] = -INFINITY;
    }

    /* Step 8: Final softmax */
    softmax_inplace(work, vocab_size);

    /* Step 9: Sample */
    if (cfg->temperature == 0.0f) {
        max_logit = work[0];
        rank = 0;
        for (i = 1; i < vocab_size; i++) if (work[i] > max_logit) { max_logit = work[i]; rank = i; }
        *out = (uint32_t)rank;
    } else {
        prng = (uint64_t)(uintptr_t)logits;
        if (prng == 0) prng = 0x123456789abcdef0ULL;
        gumbel = gumbel_max(&prng);
        max_logit = work[0] + gumbel;
        rank = 0;
        for (i = 1; i < vocab_size; i++) {
            float score = work[i] + gumbel_max(&prng);
            if (score > max_logit) { max_logit = score; rank = i; }
        }
        *out = (uint32_t)rank;
    }

    free(work);
    return VAIST_OK;
}

/* ======================================================================== */
/* Tokenizer (unchanged)                                                      */
/* ======================================================================== */

struct VaistTokenizer { uint32_t reserved; };

VAIST_API VaistStatus vaist_tokenizer_create(VaistTokenizer **o) {
    if (!o) return VAIST_INVALID_ARGUMENT;
    *o = (VaistTokenizer *)calloc(1, sizeof(VaistTokenizer));
    return *o ? VAIST_OK : VAIST_OUT_OF_MEMORY;
}

VAIST_API void vaist_tokenizer_destroy(VaistTokenizer *t) { free(t); }

VAIST_API VaistStatus vaist_tokenize_bytes(const char *x, uint32_t *o, size_t c, size_t *n) {
    size_t i, L;
    if (!x || !o || !n) return VAIST_INVALID_ARGUMENT;
    L = strlen(x);
    if (c < L) return VAIST_OUT_OF_MEMORY;
    for (i = 0; i < L; i++) o[i] = (uint32_t)(unsigned char)x[i];
    *n = L;
    return VAIST_OK;
}

/* ======================================================================== */
/* Paged KV Cache (Linux: CPU-only fallback, same logic)                     */
/* ======================================================================== */

#define VAIST_PAGED_SEQ_CAP_INIT 16

typedef struct {
    uint32_t seq_id;
    uint32_t *blocks;
    size_t   count;
    size_t   capacity;
} PagedSeqEntry;

struct VaistKVCachePaged {
    VaistKVCachePagedConfig cfg;
    uint32_t  *bitmap;
    uint32_t   max_blocks;
    uint8_t   *gpu_buf;
    size_t     gpu_buf_size;
    uint8_t    is_vulkan;
    void      *vk_buffer;
    void      *vk_memory;
    VaistBuffer *vk_buf_obj;
    VaistRuntime *rt;
    PagedSeqEntry *seq_table;
    size_t  seq_capacity;
    size_t  seq_count;
};

static uint32_t bit_scan_low(uint32_t v) {
    uint32_t i = 0;
    while (i < 32 && (v & (1u << i)) == 0) i++;
    return i;
}

static PagedSeqEntry *paged_seq_lookup(VaistKVCachePaged *kvc, uint32_t seq_id, int create) {
    size_t mask = kvc->seq_capacity - 1;
    size_t idx = (size_t)(seq_id & (uint32_t)mask);
    for (;;) {
        if (kvc->seq_table[idx].seq_id == seq_id && kvc->seq_table[idx].capacity > 0)
            return &kvc->seq_table[idx];
        if (kvc->seq_table[idx].capacity == 0) {
            if (!create) return NULL;
            kvc->seq_table[idx].seq_id = seq_id;
            kvc->seq_table[idx].blocks = NULL;
            kvc->seq_table[idx].count = 0;
            kvc->seq_table[idx].capacity = 0;
            kvc->seq_count++;
            return &kvc->seq_table[idx];
        }
        idx = (idx + 1) & mask;
    }
}

static VaistStatus paged_seq_grow(PagedSeqEntry *e, size_t min_cap) {
    size_t new_cap = e->capacity ? e->capacity : VAIST_PAGED_SEQ_CAP_INIT;
    while (new_cap < min_cap) new_cap *= 2;
    uint32_t *new_blocks = (uint32_t *)realloc(e->blocks, new_cap * sizeof(uint32_t));
    if (!new_blocks) return VAIST_OUT_OF_MEMORY;
    e->blocks = new_blocks;
    e->capacity = new_cap;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_create(
    struct VaistRuntime *rt,
    const VaistKVCachePagedConfig *cfg,
    VaistKVCachePaged **out) {

    VaistKVCachePaged *kvc;
    size_t per_block_bytes, per_layer_bytes, total_bytes;
    size_t elems_per_token;
    size_t elem_sz;

    if (!cfg || !out) return VAIST_INVALID_ARGUMENT;
    if (cfg->num_layers == 0 || cfg->num_kv_heads == 0 || cfg->head_dim == 0 ||
        cfg->block_size == 0 || cfg->max_blocks == 0) return VAIST_INVALID_ARGUMENT;
    if (cfg->kv_dtype != VAIST_F32 && cfg->kv_dtype != VAIST_F16 &&
        cfg->kv_dtype != VAIST_DTYPE_Q8_0) return VAIST_UNSUPPORTED;

    switch (cfg->kv_dtype) {
        case VAIST_F32:  elem_sz = sizeof(float);     break;
        case VAIST_F16:  elem_sz = sizeof(uint16_t);   break;
        case VAIST_DTYPE_Q8_0: elem_sz = 0; break;
        default: return VAIST_UNSUPPORTED;
    }

    kvc = (VaistKVCachePaged *)calloc(1, sizeof(*kvc));
    if (!kvc) return VAIST_OUT_OF_MEMORY;
    kvc->cfg = *cfg;
    kvc->rt = rt;
    kvc->max_blocks = cfg->max_blocks;

    kvc->bitmap = (uint32_t *)calloc((cfg->max_blocks + 31) / 32, sizeof(uint32_t));
    if (!kvc->bitmap) { free(kvc); return VAIST_OUT_OF_MEMORY; }

    elems_per_token = (size_t)cfg->num_kv_heads * cfg->head_dim;
    if (cfg->kv_dtype == VAIST_DTYPE_Q8_0) {
        size_t q8_blocks_per_token = (elems_per_token + 31) / 32;
        per_block_bytes = (size_t)cfg->block_size * q8_blocks_per_token * sizeof(vaist_block_q8_0);
    } else {
        per_block_bytes = (size_t)cfg->block_size * elems_per_token * elem_sz;
    }
    per_layer_bytes = (size_t)cfg->max_blocks * per_block_bytes * 2;
    total_bytes = (size_t)cfg->num_layers * per_layer_bytes;
    kvc->gpu_buf_size = total_bytes;

    /* Vulkan path */
    if (rt) {
        VaistRuntimeInfo rinfo;
        if (vaist_runtime_info(rt, &rinfo) == VAIST_OK &&
            rinfo.backend == VAIST_BACKEND_VULKAN && rinfo.vulkan_available) {
            void *vk_device = NULL, *vk_queue = NULL;
            uint32_t vf = 0;
            if (vaist_runtime_vk_state(rt, &vk_device, &vk_queue, &vf) == VAIST_OK) {
                VaistBuffer *buf = NULL;
                VaistStatus st = vaist_buffer_create(rt, total_bytes, &buf);
                if (st == VAIST_OK && buf) {
                    kvc->gpu_buf = (uint8_t *)vaist_buffer_data(buf);
                    kvc->vk_buf_obj = buf;
                    kvc->is_vulkan = 1;
                    vaist_buffer_gpu_handle(buf, &kvc->vk_buffer);
                }
            }
        }
    }

    if (!kvc->is_vulkan) {
        kvc->gpu_buf = (uint8_t *)malloc(total_bytes);
        if (!kvc->gpu_buf) {
            free(kvc->bitmap);
            free(kvc);
            return VAIST_OUT_OF_MEMORY;
        }
    }

    kvc->seq_capacity = VAIST_PAGED_SEQ_CAP_INIT;
    kvc->seq_table = (PagedSeqEntry *)calloc(kvc->seq_capacity, sizeof(PagedSeqEntry));
    if (!kvc->seq_table) {
        free(kvc->bitmap);
        if (kvc->is_vulkan) {
            vaist_buffer_destroy(kvc->vk_buf_obj);
            kvc->vk_buf_obj = NULL;
        } else {
            free(kvc->gpu_buf);
        }
        free(kvc);
        return VAIST_OUT_OF_MEMORY;
    }

    *out = kvc;
    return VAIST_OK;
}

VAIST_API void vaist_kv_paged_destroy(VaistKVCachePaged *kvc) {
    size_t i;
    if (!kvc) return;
    if (kvc->seq_table) {
        for (i = 0; i < kvc->seq_capacity; i++) {
            if (kvc->seq_table[i].capacity > 0) free(kvc->seq_table[i].blocks);
        }
        free(kvc->seq_table);
    }
    free(kvc->bitmap);
    if (kvc->is_vulkan && kvc->vk_buf_obj) {
        vaist_buffer_destroy(kvc->vk_buf_obj);
        kvc->vk_buf_obj = NULL;
    } else {
        free(kvc->gpu_buf);
    }
    free(kvc);
}

VAIST_API VaistStatus vaist_kv_paged_alloc_block(
    VaistKVCachePaged *kvc, uint32_t num_blocks, uint32_t *block_ids) {

    uint32_t allocated = 0, word_idx;
    if (!kvc || !block_ids || num_blocks == 0) return VAIST_INVALID_ARGUMENT;
    for (word_idx = 0; word_idx < (kvc->max_blocks + 31) / 32 && allocated < num_blocks; word_idx++) {
        uint32_t free_mask = ~kvc->bitmap[word_idx];
        while (free_mask && allocated < num_blocks) {
            uint32_t bit = bit_scan_low(free_mask);
            if (bit >= 32) break;
            uint32_t block_id = word_idx * 32 + bit;
            if (block_id >= kvc->max_blocks) break;
            kvc->bitmap[word_idx] |= (1u << bit);
            block_ids[allocated] = block_id;
            allocated++;
            kvc->cfg.total_blocks_used++;
            free_mask &= ~(1u << bit);
        }
    }
    if (allocated < num_blocks) {
        while (allocated > 0) {
            allocated--;
            uint32_t bid = block_ids[allocated];
            kvc->bitmap[bid / 32] &= ~(1u << (bid % 32));
            kvc->cfg.total_blocks_used--;
        }
        return VAIST_OUT_OF_MEMORY;
    }
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_free_block(VaistKVCachePaged *kvc, uint32_t block_id) {
    if (!kvc) return VAIST_INVALID_ARGUMENT;
    if (block_id >= kvc->max_blocks) return VAIST_INVALID_ARGUMENT;
    uint32_t word = block_id / 32;
    uint32_t bit = block_id % 32;
    if (!(kvc->bitmap[word] & (1u << bit))) return VAIST_OK;
    kvc->bitmap[word] &= ~(1u << bit);
    kvc->cfg.total_blocks_used--;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_get_block_table(
    VaistKVCachePaged *kvc, uint32_t seq_id,
    uint32_t *out_blocks, size_t cap, size_t *count) {

    PagedSeqEntry *e;
    size_t i;
    if (!kvc || !count) return VAIST_INVALID_ARGUMENT;
    *count = 0;
    if (!out_blocks && cap > 0) return VAIST_INVALID_ARGUMENT;
    e = paged_seq_lookup(kvc, seq_id, 0);
    if (!e) return VAIST_OK;
    if (out_blocks) {
        for (i = 0; i < e->count && i < cap; i++) out_blocks[i] = e->blocks[i];
    }
    *count = e->count;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_assign_blocks(
    VaistKVCachePaged *kvc, uint32_t seq_id,
    const uint32_t *blocks, size_t count) {

    PagedSeqEntry *e;
    size_t i;
    if (!kvc || !blocks || count == 0) return VAIST_INVALID_ARGUMENT;
    if (kvc->seq_count * 4 >= kvc->seq_capacity * 3) {
        size_t new_cap = kvc->seq_capacity * 2;
        PagedSeqEntry *new_table = (PagedSeqEntry *)calloc(new_cap, sizeof(PagedSeqEntry));
        if (!new_table) return VAIST_OUT_OF_MEMORY;
        size_t mask = new_cap - 1;
        for (i = 0; i < kvc->seq_capacity; i++) {
            if (kvc->seq_table[i].capacity > 0) {
                size_t idx = (size_t)(kvc->seq_table[i].seq_id & (uint32_t)mask);
                while (new_table[idx].capacity > 0) idx = (idx + 1) & mask;
                new_table[idx] = kvc->seq_table[i];
            }
        }
        free(kvc->seq_table);
        kvc->seq_table = new_table;
        kvc->seq_capacity = new_cap;
    }
    e = paged_seq_lookup(kvc, seq_id, 1);
    if (!e) return VAIST_INTERNAL_ERROR;
    if (e->count + count > e->capacity) {
        VaistStatus st = paged_seq_grow(e, e->count + count);
        if (st != VAIST_OK) return st;
    }
    for (i = 0; i < count; i++) e->blocks[e->count + i] = blocks[i];
    e->count += count;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_write_tensor(
    VaistKVCachePaged *kvc, uint32_t block_id, uint32_t layer,
    uint32_t token_idx, const void *data, size_t element_size) {

    size_t per_block_bytes, offset, elems_per_token;
    if (!kvc || !data || layer >= kvc->cfg.num_layers || block_id >= kvc->cfg.max_blocks)
        return VAIST_INVALID_ARGUMENT;
    if (token_idx >= kvc->cfg.block_size) return VAIST_INVALID_ARGUMENT;
    if (element_size == 0) return VAIST_INVALID_ARGUMENT;

    elems_per_token = (size_t)kvc->cfg.num_kv_heads * kvc->cfg.head_dim;

    if (kvc->cfg.kv_dtype == VAIST_DTYPE_Q8_0) {
        size_t blocks_per_token = (elems_per_token + 31) / 32;
        per_block_bytes = (size_t)kvc->cfg.block_size * blocks_per_token * sizeof(vaist_block_q8_0);
    } else {
        per_block_bytes = (size_t)kvc->cfg.block_size * elems_per_token * element_size;
    }

    size_t per_layer_bytes = (size_t)kvc->cfg.max_blocks * per_block_bytes * 2;
    offset = (size_t)layer * per_layer_bytes + (size_t)block_id * per_block_bytes * 2;

    if (kvc->cfg.kv_dtype == VAIST_DTYPE_Q8_0) {
        size_t blocks_per_token = (elems_per_token + 31) / 32;
        size_t tok_k = (size_t)token_idx * blocks_per_token * sizeof(vaist_block_q8_0);
        size_t tok_v = per_block_bytes + tok_k;
        size_t total_write = blocks_per_token * sizeof(vaist_block_q8_0);
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_k, data, total_write);
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_v, data, total_write);
    } else {
        size_t tok_k = (size_t)token_idx * elems_per_token * element_size;
        size_t tok_v = per_block_bytes + tok_k;
        size_t total_write = elems_per_token * element_size;
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_k, data, total_write);
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_v, data, total_write);
    }
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_gpu_buffer(
    VaistKVCachePaged *kvc, uint32_t layer, void **handle) {

    if (!kvc || !handle) return VAIST_INVALID_ARGUMENT;
    if (layer >= kvc->cfg.num_layers) return VAIST_INVALID_ARGUMENT;
    *handle = kvc->vk_buffer;
    return VAIST_OK;
}

VAIST_API void vaist_kv_paged_clear_seq(VaistKVCachePaged *kvc, uint32_t seq_id) {
    if (!kvc) return;
    PagedSeqEntry *e = paged_seq_lookup(kvc, seq_id, 0);
    if (!e) return;
    if (e->blocks && e->count > 0) {
        size_t i;
        for (i = 0; i < e->count; i++) vaist_kv_paged_free_block(kvc, e->blocks[i]);
        free(e->blocks);
        e->blocks = NULL;
    }
    e->count = 0;
    e->capacity = 0;
    e->seq_id = 0;
    kvc->seq_count--;
}
