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
#include <float.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
#ifndef INFINITY
#define INFINITY ((float)(FLT_MAX * 100.0f))
#endif

/* ======================================================================== */
/* Existing simple KV cache (unchanged)                                     */
/* ======================================================================== */

struct VaistKVCache { size_t cap, width, len; float *data; };

VAIST_API VaistStatus vaist_kv_create(size_t c, size_t w, VaistKVCache **o) {
    VaistKVCache *k;
    if (!o || !c || !w) { DBG_TRACE("path=guard-fail -> INVALID_ARGUMENT"); return VAIST_INVALID_ARGUMENT; }
    k = (VaistKVCache *)calloc(1, sizeof(*k));
    if (!k) return VAIST_OUT_OF_MEMORY;
    k->data = (float *)calloc(c * w, sizeof(float));
    if (!k->data) { free(k); return VAIST_OUT_OF_MEMORY; }
    k->cap = c; k->width = w;
    *o = k;
    DBG_TRACE("kv_create cap=%lu width=%lu -> OK", (unsigned long)c, (unsigned long)w);
    return VAIST_OK;
}

VAIST_API void vaist_kv_destroy(VaistKVCache *k) {
    if (k) { free(k->data); free(k); }
}

VAIST_API VaistStatus vaist_kv_write(VaistKVCache *k, size_t p, const float *d) {
    if (!k || !d || p >= k->cap) { DBG_TRACE("path=guard-fail p=%lu cap=%lu -> INVALID_ARGUMENT", (unsigned long)p, (unsigned long)(k ? k->cap : 0ul)); return VAIST_INVALID_ARGUMENT; }
    memcpy(k->data + p * k->width, d, k->width * sizeof(float));
    if (p + 1 > k->len) k->len = p + 1;
    DBG_TRACE("kv_write p=%lu len=%lu", (unsigned long)p, (unsigned long)k->len);
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_read(const VaistKVCache *k, size_t p, float *o) {
    if (!k || !o || p >= k->len) { DBG_TRACE("path=guard-fail p=%lu len=%lu -> INVALID_ARGUMENT", (unsigned long)p, (unsigned long)(k ? k->len : 0ul)); return VAIST_INVALID_ARGUMENT; }
    memcpy(o, k->data + p * k->width, k->width * sizeof(float));
    DBG_TRACE("kv_read p=%lu len=%lu", (unsigned long)p, (unsigned long)k->len);
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

/* ---- existing greedy + temp sample (unchanged) ---- */

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

/**
 * \brief Seeded xorshift PRNG for deterministic Gumbel noise.
 * \param x State pointer (updated in place).
 * \return Next random uint64.
 */
static uint64_t xorshift64(uint64_t *x) {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    return *x;
}

/**
 * \brief Generate a uniform[0,1) float from the PRNG.
 */
static float rand_uniform(uint64_t *x) {
    uint64_t v = xorshift64(x);
    return (float)(v >> 11) * (1.0f / 9007199254740992.0f); /* 2^53 */
}

/**
 * \brief Generate a Gumbel(0,1) sample: -log(-log(u)).
 */
static float gumbel_max(uint64_t *x) {
    float u = rand_uniform(x);
    if (u <= 0.0f) u = 1e-30f;
    if (u >= 1.0f) u = 1.0f - 1e-30f;
    return -logf(-logf(u));
}

/**
 * \brief Count occurrences of each token ID in input_ids for repetition penalty.
 * \param input_ids  Token ID array.
 * \param input_len  Length of input_ids.
 * \param counts     Output array of size vocab_size, zero-initialized by caller.
 */
static void count_token_occurrences(const uint32_t *input_ids, size_t input_len, size_t vocab_size, uint32_t *counts) {
    size_t i;
    for (i = 0; i < input_len; i++) {
        if (input_ids[i] < vocab_size) {
            counts[input_ids[i]]++;
        }
    }
}

/**
 * \brief Softmax over logits array, in place.
 * \param l  Logits array (modified in place to probabilities).
 * \param n  Number of elements.
 */
static void softmax_inplace(float *l, size_t n) {
    size_t i; float max = l[0], sum = 0;
    for (i = 1; i < n; i++) if (l[i] > max) max = l[i];
    for (i = 0; i < n; i++) { l[i] = expf(l[i] - max); sum += l[i]; }
    if (sum > 0) for (i = 0; i < n; i++) l[i] /= sum;
}

/**
 * \brief Global work array for qsort comparison (MSVC has no qsort_r).
 */
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
    DBG_TRACE("sampler_config_init temp=%f top_k=%u top_p=%f min_p=%f presence=%f freq=%f",
              temp, top_k, top_p, min_p, presence_penalty, frequency_penalty);
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_sample_logits(const float *logits, size_t n, uint32_t *out) {
    size_t i, b = 0;
    if (!logits || !out || !n) return VAIST_INVALID_ARGUMENT;
    for (i = 1; i < n; i++) if (logits[i] > logits[b]) b = i;
    *out = (uint32_t)b;
    DBG_TRACE("sample_logits n=%lu winner=%u logit=%f", (unsigned long)n, (uint32_t)b, logits[b]);
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
    uint32_t *sorted_idx = NULL, *counts = NULL;
    uint64_t prng;
    size_t i, kept_count, rank;
    float max_logit, threshold, accum, max_prob, gumbel;

    if (!cfg || !logits || !out || !vocab_size) return VAIST_INVALID_ARGUMENT;

    /* Allocate working copies */
    work = (float *)malloc(vocab_size * sizeof(float));
    if (!work) return VAIST_OUT_OF_MEMORY;
    memcpy(work, logits, vocab_size * sizeof(float));

    /* ---- Step 1: Apply presence and frequency penalties ---- */
    if (cfg->presence_penalty > 0 || cfg->frequency_penalty > 0) {
        counts = (uint32_t *)calloc(vocab_size, sizeof(uint32_t));
        if (!counts) { free(work); return VAIST_OUT_OF_MEMORY; }
        if (input_ids && input_len > 0) {
            count_token_occurrences(input_ids, input_len, vocab_size, counts);
        }
        for (i = 0; i < vocab_size; i++) {
            float penalty = cfg->presence_penalty * (counts[i] > 0 ? 1.0f : 0.0f)
                          + cfg->frequency_penalty * (float)counts[i];
            work[i] -= penalty;
        }
        free(counts);
        counts = NULL;
    }

    /* ---- Step 2: Subtract max for numerical stability ---- */
    max_logit = work[0];
    for (i = 1; i < vocab_size; i++) if (work[i] > max_logit) max_logit = work[i];
    for (i = 0; i < vocab_size; i++) work[i] -= max_logit;

    /* ---- Step 3: Divide by temperature ---- */
    if (cfg->temperature > 0 && cfg->temperature < 1e30f) {
        float inv_temp = 1.0f / cfg->temperature;
        for (i = 0; i < vocab_size; i++) work[i] *= inv_temp;
    }

    /* ---- Step 4: Top-K filtering ---- */
    if (cfg->top_k > 0 && cfg->top_k < vocab_size) {
        sorted_idx = (uint32_t *)malloc(vocab_size * sizeof(uint32_t));
        if (!sorted_idx) { free(work); return VAIST_OUT_OF_MEMORY; }
        for (i = 0; i < vocab_size; i++) sorted_idx[i] = (uint32_t)i;
        g_work_logits = work;
        qsort(sorted_idx, vocab_size, sizeof(uint32_t), cmp_desc_idx);
        /* Keep top-k, set rest to -inf */
        for (i = cfg->top_k; i < vocab_size; i++) {
            work[sorted_idx[i]] = -INFINITY;
        }
        free(sorted_idx);
        sorted_idx = NULL;
    }

    /* ---- Step 5: Softmax for top-p and min-p filtering ---- */
    softmax_inplace(work, vocab_size);

    /* ---- Step 6: Top-p (nucleus) filtering ---- */
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
            work[sorted_idx[i]] = -INFINITY; /* tentatively mask out */
            kept_count++;
            if (accum >= cfg->top_p) break;
        }
        for (i = 0; i < kept_count; i++) {
            work[sorted_idx[i]] = logits[sorted_idx[i]]; /* restore original logit */
        }
        free(sorted_idx);
        sorted_idx = NULL;
    }

    /* ---- Step 7: Min-p filtering ---- */
    if (cfg->min_p > 0.0f && cfg->min_p < 1.0f) {
        /* Apply softmax on the remaining logits to get probabilities */
        softmax_inplace(work, vocab_size);
        max_prob = 0.0f;
        for (i = 0; i < vocab_size; i++) if (work[i] > max_prob) max_prob = work[i];
        threshold = cfg->min_p * max_prob;
        for (i = 0; i < vocab_size; i++) {
            if (work[i] < threshold) work[i] = -INFINITY;
        }
    }

    /* ---- Step 8: Final softmax ---- */
    softmax_inplace(work, vocab_size);

    /* ---- Step 9: Sample ---- */
    if (cfg->temperature == 0.0f) {
        /* Greedy: argmax */
        max_logit = work[0];
        rank = 0;
        for (i = 1; i < vocab_size; i++) if (work[i] > max_logit) { max_logit = work[i]; rank = i; }
        *out = (uint32_t)rank;
    } else {
        /* Gumbel-max sampling: add noise and take argmax */
        prng = (uint64_t)(uintptr_t)logits; /* seed from logits pointer for reproducibility */
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

    DBG_TRACE("sample_cfg vocab=%lu temp=%f top_k=%u top_p=%f min_p=%f -> token=%u",
              (unsigned long)vocab_size, cfg->temperature, cfg->top_k, cfg->top_p, cfg->min_p, *out);
    free(work);
    return VAIST_OK;
}

/* ======================================================================== */
/* Tokenizer                                                                */
/* Deprecated: use vaist_tokens.c for full tokenizer support.                */
/* The simple byte tokenizer is kept here for backwards compatibility.        */
/* ======================================================================== */

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
/* Paged KV Cache                                                           */
/* ======================================================================== */

#define VAIST_PAGED_SEQ_CAP_INIT 16
#define VAIST_PAGED_SEQ_GROW 16

/** \internal Hash-map entry mapping seq_id -> dynamic block array. */
typedef struct {
    uint32_t seq_id;
    uint32_t *blocks;   /* dynamic array of block IDs */
    size_t   count;     /* number of blocks assigned */
    size_t   capacity;  /* capacity of blocks array */
} PagedSeqEntry;

/** \internal Paged KV cache internal state. */
struct VaistKVCachePaged {
    VaistKVCachePagedConfig cfg;
    uint32_t  *bitmap;          /* free-block bitmap (1 = in-use) */
    uint32_t   max_blocks;
    uint8_t   *gpu_buf;         /* single large GPU allocation (or malloc'd fallback) */
    size_t     gpu_buf_size;
    uint8_t    is_vulkan;       /* 1 if gpu_buf is a real VkBuffer */
    void      *vk_buffer;       /* opaque VkBuffer handle when is_vulkan */
    void      *vk_memory;       /* VkDeviceMemory handle */
    VaistBuffer *vk_buf_obj;    /* owning VaistBuffer (freed on destroy) */
    VaistRuntime *rt;           /* non-owning runtime pointer */
    /* Sequence-to-block-table hash map (open addressing, power-of-2) */
    PagedSeqEntry *seq_table;
    size_t  seq_capacity;
    size_t  seq_count;
};

/** \internal Bitmap bit-scan: find index of lowest set bit in a uint32. */
static uint32_t bit_scan_low(uint32_t v) {
    uint32_t i = 0;
    while (i < 32 && (v & (1u << i)) == 0) i++;
    return i;
}

/** \internal Lookup or create a seq entry. Returns NULL if table full. */
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

/** \internal Grow a seq entry's block array. */
static VaistStatus paged_seq_grow(PagedSeqEntry *e, size_t min_cap) {
    size_t new_cap = e->capacity ? e->capacity : VAIST_PAGED_SEQ_CAP_INIT;
    while (new_cap < min_cap) new_cap *= 2;
    uint32_t *new_blocks = (uint32_t *)realloc(e->blocks, new_cap * sizeof(uint32_t));
    if (!new_blocks) {
        /* realloc may not have freed e->blocks */
        return VAIST_OUT_OF_MEMORY;
    }
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
        cfg->block_size == 0 || cfg->max_blocks == 0) {
        return VAIST_INVALID_ARGUMENT;
    }

    /* Validate kv_dtype: only VAIST_F32(0), VAIST_F16(5), VAIST_Q8_0(6) supported */
    if (cfg->kv_dtype != VAIST_F32 && cfg->kv_dtype != VAIST_F16 &&
        cfg->kv_dtype != VAIST_DTYPE_Q8_0) {
        return VAIST_UNSUPPORTED;
    }

    /* Element size */
    switch (cfg->kv_dtype) {
        case VAIST_F32:  elem_sz = sizeof(float);     break;
        case VAIST_F16:  elem_sz = sizeof(uint16_t);   break;
        case VAIST_DTYPE_Q8_0: elem_sz = 0; break; /* block-based, handled separately */
        default: return VAIST_UNSUPPORTED;
    }

    kvc = (VaistKVCachePaged *)calloc(1, sizeof(*kvc));
    if (!kvc) return VAIST_OUT_OF_MEMORY;

    kvc->cfg = *cfg;
    kvc->rt = rt;
    kvc->max_blocks = cfg->max_blocks;

    /* Allocate free-block bitmap */
    kvc->bitmap = (uint32_t *)calloc((cfg->max_blocks + 31) / 32, sizeof(uint32_t));
    if (!kvc->bitmap) { free(kvc); return VAIST_OUT_OF_MEMORY; }

    /* Allocate per-layer GPU buffer */
    /* For Q8_0: each token contributes ceil(num_kv_heads * head_dim / QK8_0) q8_0 blocks */
    /* Layout: [layer][block][kv: K | V], where K_size = V_size = block_size * tokens_per_head_q8_blocks */
    elems_per_token = (size_t)cfg->num_kv_heads * cfg->head_dim;
    if (cfg->kv_dtype == VAIST_Q8_0) {
        size_t q8_blocks_per_token = (elems_per_token + 31) / 32;
        per_block_bytes = (size_t)cfg->block_size * q8_blocks_per_token * sizeof(vaist_block_q8_0);
    } else {
        per_block_bytes = (size_t)cfg->block_size * elems_per_token * elem_sz;
    }

    /* K and V are contiguous per layer: 2 * per_block_bytes per block */
    per_layer_bytes = (size_t)cfg->max_blocks * per_block_bytes * 2;
    total_bytes = (size_t)cfg->num_layers * per_layer_bytes;

    kvc->gpu_buf_size = total_bytes;

    /* Try Vulkan allocation if runtime available */
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
                    DBG_TRACE("kv_paged_create: Vulkan buffer %p size=%lu", kvc->vk_buffer, (unsigned long)total_bytes);
                }
            }
        }
    }

    /* Fall back to host malloc if no Vulkan buffer was allocated */
    if (!kvc->is_vulkan) {
        kvc->gpu_buf = (uint8_t *)malloc(total_bytes);
        if (!kvc->gpu_buf) {
            free(kvc->bitmap);
            free(kvc);
            return VAIST_OUT_OF_MEMORY;
        }
        kvc->is_vulkan = 0;
        DBG_TRACE("kv_paged_create: host fallback size=%lu", (unsigned long)total_bytes);
    }

    /* Allocate sequence hash table (power of 2) */
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
    DBG_TRACE("kv_paged_create layers=%u heads=%u dim=%u block=%u dtype=%u max_blocks=%u -> OK",
              cfg->num_layers, cfg->num_kv_heads, cfg->head_dim, cfg->block_size,
              cfg->kv_dtype, cfg->max_blocks);
    return VAIST_OK;
}

VAIST_API void vaist_kv_paged_destroy(VaistKVCachePaged *kvc) {
    if (!kvc) return;
    /* Free per-seq block arrays */
    if (kvc->seq_table) {
        size_t i;
        for (i = 0; i < kvc->seq_capacity; i++) {
            if (kvc->seq_table[i].capacity > 0) {
                free(kvc->seq_table[i].blocks);
            }
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
    DBG_TRACE("kv_paged_destroy done");
}

VAIST_API VaistStatus vaist_kv_paged_alloc_block(
    VaistKVCachePaged *kvc, uint32_t num_blocks, uint32_t *block_ids) {

    uint32_t allocated = 0;
    uint32_t word_idx;

    if (!kvc || !block_ids || num_blocks == 0) return VAIST_INVALID_ARGUMENT;

    for (word_idx = 0; word_idx < (kvc->max_blocks + 31) / 32 && allocated < num_blocks; word_idx++) {
        uint32_t word = kvc->bitmap[word_idx];
        /* Find free bits (bit == 0) in this word */
        uint32_t free_mask = ~word;
        while (free_mask && allocated < num_blocks) {
            uint32_t bit = bit_scan_low(free_mask);
            if (bit >= 32) break;
            uint32_t block_id = word_idx * 32 + bit;
            if (block_id >= kvc->max_blocks) break;
            /* Mark as in-use */
            kvc->bitmap[word_idx] |= (1u << bit);
            block_ids[allocated] = block_id;
            allocated++;
            free_mask &= ~(1u << bit);
            kvc->cfg.total_blocks_used++;
        }
    }

    if (allocated < num_blocks) {
        /* Roll back allocated blocks */
        while (allocated > 0) {
            allocated--;
            uint32_t bid = block_ids[allocated];
            kvc->bitmap[bid / 32] &= ~(1u << (bid % 32));
            kvc->cfg.total_blocks_used--;
        }
        return VAIST_OUT_OF_MEMORY;
    }

    DBG_TRACE("kv_paged_alloc %u blocks -> OK", num_blocks);
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_free_block(
    VaistKVCachePaged *kvc, uint32_t block_id) {

    if (!kvc) return VAIST_INVALID_ARGUMENT;
    if (block_id >= kvc->max_blocks) return VAIST_INVALID_ARGUMENT;

    uint32_t word = block_id / 32;
    uint32_t bit = block_id % 32;
    if (!(kvc->bitmap[word] & (1u << bit))) {
        /* Already free — not an error */
        return VAIST_OK;
    }
    kvc->bitmap[word] &= ~(1u << bit);
    kvc->cfg.total_blocks_used--;
    DBG_TRACE("kv_paged_free block=%u -> OK", block_id);
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
    if (!e) return VAIST_OK; /* No blocks assigned yet */

    if (out_blocks) {
        for (i = 0; i < e->count && i < cap; i++) {
            out_blocks[i] = e->blocks[i];
        }
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

    /* Grow the hash table if needed (75% load factor) */
    if (kvc->seq_count * 4 >= kvc->seq_capacity * 3) {
        size_t new_cap = kvc->seq_capacity * 2;
        PagedSeqEntry *new_table = (PagedSeqEntry *)calloc(new_cap, sizeof(PagedSeqEntry));
        if (!new_table) return VAIST_OUT_OF_MEMORY;
        /* Rehash */
        size_t mask = new_cap - 1;
        for (i = 0; i < kvc->seq_capacity; i++) {
            if (kvc->seq_table[i].capacity > 0) {
                size_t idx = (size_t)(kvc->seq_table[i].seq_id & (uint32_t)mask);
                while (new_table[idx].capacity > 0) {
                    idx = (idx + 1) & mask;
                }
                new_table[idx] = kvc->seq_table[i];
            }
        }
        free(kvc->seq_table);
        kvc->seq_table = new_table;
        kvc->seq_capacity = new_cap;
    }

    e = paged_seq_lookup(kvc, seq_id, 1);
    if (!e) return VAIST_INTERNAL_ERROR;

    /* Grow block array if needed */
    if (e->count + count > e->capacity) {
        VaistStatus st = paged_seq_grow(e, e->count + count);
        if (st != VAIST_OK) return st;
    }

    /* Append blocks */
    for (i = 0; i < count; i++) {
        e->blocks[e->count + i] = blocks[i];
    }
    e->count += count;
    DBG_TRACE("kv_paged_assign seq=%u count=%lu total=%lu", seq_id, (unsigned long)count, (unsigned long)e->count);
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_write_tensor(
    VaistKVCachePaged *kvc, uint32_t block_id, uint32_t layer,
    uint32_t token_idx, const void *data, size_t element_size) {

    size_t per_block_bytes, per_layer_bytes, offset, elems_per_token;

    if (!kvc || !data || layer >= kvc->cfg.num_layers || block_id >= kvc->cfg.max_blocks)
        return VAIST_INVALID_ARGUMENT;
    if (token_idx >= kvc->cfg.block_size)
        return VAIST_INVALID_ARGUMENT;
    if (element_size == 0)
        return VAIST_INVALID_ARGUMENT;

    elems_per_token = (size_t)kvc->cfg.num_kv_heads * kvc->cfg.head_dim;

    /* Compute strides based on kv_dtype */
    if (kvc->cfg.kv_dtype == VAIST_Q8_0) {
        /* Q8_0: each token contributes ceil(elems / 32) q8_0 blocks */
        size_t blocks_per_token = (elems_per_token + 31) / 32;
        per_block_bytes = (size_t)kvc->cfg.block_size * blocks_per_token * sizeof(vaist_block_q8_0);
    } else {
        per_block_bytes = (size_t)kvc->cfg.block_size * elems_per_token * element_size;
    }

    /* Layout: [layer][block][kv: K_region | V_region]
     *   per_layer = max_blocks * per_block_bytes * 2  (K + V)
     *   block offset = layer * per_layer + block_id * per_block_bytes * 2
     */
    per_layer_bytes = (size_t)kvc->cfg.max_blocks * per_block_bytes * 2;
    offset = (size_t)layer * per_layer_bytes + (size_t)block_id * per_block_bytes * 2;

    if (kvc->cfg.kv_dtype == VAIST_Q8_0) {
        /* Q8_0: data is raw q8_0 blocks for one token's K heads */
        size_t blocks_per_token = (elems_per_token + 31) / 32;
        size_t tok_bytes = blocks_per_token * sizeof(vaist_block_q8_0);
        size_t tok_off_k = (size_t)token_idx * tok_bytes;
        size_t tok_off_v = per_block_bytes + tok_off_k;
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_off_k, data, tok_bytes);
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_off_v, data, tok_bytes);
    } else {
        /* F32/F16: data is raw element array for one token's heads */
        size_t tok_bytes = elems_per_token * element_size;
        size_t tok_off_k = (size_t)token_idx * elems_per_token * element_size;
        size_t tok_off_v = per_block_bytes + tok_off_k;
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_off_k, data, tok_bytes);
        memcpy((uint8_t *)kvc->gpu_buf + offset + tok_off_v, data, tok_bytes);
    }

    DBG_TRACE("kv_paged_write_tensor block=%u layer=%u token=%u elemsz=%lu -> OK",
              block_id, layer, token_idx, (unsigned long)element_size);
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_kv_paged_gpu_buffer(
    VaistKVCachePaged *kvc, uint32_t layer, void **handle) {

    if (!kvc || !handle) return VAIST_INVALID_ARGUMENT;
    if (layer >= kvc->cfg.num_layers) return VAIST_INVALID_ARGUMENT;

    *handle = kvc->vk_buffer;
    DBG_TRACE("kv_paged_gpu_buffer layer=%u handle=%p", layer, *handle);
    return VAIST_OK;
}

VAIST_API void vaist_kv_paged_clear_seq(VaistKVCachePaged *kvc, uint32_t seq_id) {
    if (!kvc) return;
    PagedSeqEntry *e = paged_seq_lookup(kvc, seq_id, 0);
    if (!e) return;
    /* Free all blocks in this sequence */
    if (e->blocks && e->count > 0) {
        size_t i;
        for (i = 0; i < e->count; i++) {
            vaist_kv_paged_free_block(kvc, e->blocks[i]);
        }
        free(e->blocks);
        e->blocks = NULL;
    }
    e->count = 0;
    e->capacity = 0;
    e->seq_id = 0;
    /* Remove from hash (mark as empty by clearing) */
    kvc->seq_count--;
    DBG_TRACE("kv_paged_clear_seq seq=%u", seq_id);
}
