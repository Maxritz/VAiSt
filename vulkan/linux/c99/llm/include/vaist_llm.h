/**
 * \file vaist_llm.h
 * \brief LLM inference primitives: KV cache, sampler, tokenizer.
 */
#ifndef VAIST_LLM_H
#define VAIST_LLM_H
#include "vaist_core.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for the simple ring KV cache */
typedef struct VaistKVCache VaistKVCache;

/* Opaque handle for the simple sampler */
typedef struct VaistSampler VaistSampler;

/* Opaque handle for the tokenizer */
typedef struct VaistTokenizer VaistTokenizer;

/* ---- Existing KV cache API (unchanged) ---- */
VAIST_API VaistStatus vaist_kv_create(size_t capacity, size_t width, VaistKVCache **out);
VAIST_API void vaist_kv_destroy(VaistKVCache *k);
VAIST_API VaistStatus vaist_kv_write(VaistKVCache *k, size_t pos, const float *data);
VAIST_API VaistStatus vaist_kv_read(const VaistKVCache *k, size_t pos, float *out);
VAIST_API size_t vaist_kv_length(const VaistKVCache *k);

/* ---- Existing sampler API (unchanged) ---- */
VAIST_API VaistStatus vaist_sampler_create(float temperature, uint32_t top_k, float top_p, VaistSampler **out);
VAIST_API void vaist_sampler_destroy(VaistSampler *s);
VAIST_API VaistStatus vaist_sample_greedy(const float *logits, size_t n, uint32_t *out);
VAIST_API VaistStatus vaist_sample(const float *logits, size_t n, VaistSampler *s, uint64_t *state, uint32_t *out);

/* ---- Tokenizer (unchanged) ---- */
VAIST_API VaistStatus vaist_tokenizer_create(VaistTokenizer **out);
VAIST_API void vaist_tokenizer_destroy(VaistTokenizer *t);
VAIST_API VaistStatus vaist_tokenize_bytes(const char *text, uint32_t *out, size_t cap, size_t *count);

/* ---- Extended sampler configuration ---- */

#pragma pack(push, 1)
typedef struct {
    float temperature;          /**< Sampling temperature; 0.0 forces greedy argmax */
    float top_p;                /**< Nucleus sampling threshold; 0 = disabled */
    uint32_t top_k;             /**< Top-K filtering; 0 = disabled */
    float min_p;                /**< Min-p (epsilon) filtering; 0 = disabled */
    float presence_penalty;     /**< Presence penalty; 0 = disabled */
    float frequency_penalty;    /**< Frequency penalty; 0 = disabled */
} VaistSamplerConfig;
#pragma pack(pop)

/**
 * \brief Initialize a sampler config with explicit parameters.
 * \param cfg        Destination config (must not be NULL).
 * \param temp       Sampling temperature (0 = greedy).
 * \param top_k      Top-K value (0 = disabled).
 * \param top_p      Nucleus threshold in (0, 1] (0 = disabled).
 * \param min_p      Min-p threshold in (0, 1] (0 = disabled).
 * \param presence_penalty   Repetition penalty for new tokens.
 * \param frequency_penalty  Per-occurrence penalty.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT if cfg is NULL.
 */
VAIST_API VaistStatus vaist_sampler_config_init(
    VaistSamplerConfig *cfg,
    float temp, uint32_t top_k, float top_p, float min_p,
    float presence_penalty, float frequency_penalty);

/**
 * \brief Full sampling: penalties -> stability -> temp -> top-k -> top-p -> min-p -> softmax -> Gumbel/argmax.
 * \param s          Sampler context (may be NULL for deterministic use).
 * \param cfg        Sampler config (must not be NULL).
 * \param logits     Input logits array of length vocab_size.
 * \param vocab_size Number of logits.
 * \param input_ids  Previous prompt token IDs (for repetition penalties).
 * \param input_len  Number of elements in input_ids.
 * \param out        Output sampled token ID.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL/empty inputs.
 * \retval VAIST_OUT_OF_MEMORY if internal allocation fails.
 */
VAIST_API VaistStatus vaist_sample_cfg(
    const VaistSampler *s,
    const VaistSamplerConfig *cfg,
    const float *logits, size_t vocab_size,
    const uint32_t *input_ids, size_t input_len,
    uint32_t *out);

/**
 * \brief Simple argmax alias (backwards compatibility).
 * \param logits   Input logits array.
 * \param n        Number of logits.
 * \param out      Output index of the maximum value.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT if logits or out is NULL, or n == 0.
 */
VAIST_API VaistStatus vaist_sample_logits(const float *logits, size_t n, uint32_t *out);

/* ---- Paged KV cache ---- */

/** Opaque handle for the paged KV cache */
typedef struct VaistKVCachePaged VaistKVCachePaged;

/** Forward declaration: runtime opaque handle */
struct VaistRuntime;

#pragma pack(push, 1)
typedef struct {
    uint32_t num_layers;      /**< Number of transformer layers */
    uint32_t num_kv_heads;    /**< Number of KV attention heads */
    uint32_t head_dim;        /**< Dimensionality per attention head */
    uint32_t block_size;      /**< Tokens per block (e.g. 16, 64) */
     uint16_t kv_dtype;        /**< KV data type: VAIST_F32, VAIST_F16, or VAIST_DTYPE_Q8_0 */
    uint32_t max_blocks;      /**< Total blocks available in the pool */
    uint32_t total_blocks_used; /**< Running counter of allocated blocks */
} VaistKVCachePagedConfig;
#pragma pack(pop)

/**
 * \brief Create a paged KV cache over a GPU-resident buffer.
 * \param rt   Runtime context (provides Vulkan device/queue).
 * \param cfg  Cache configuration (must not be NULL).
 * \param out  Receives the opaque cache handle.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL rt, cfg, or out.
 * \retval VAIST_DEVICE_ERROR if GPU allocation fails and no CPU fallback.
 * \retval VAIST_OUT_OF_MEMORY if host allocation fails.
 */
VAIST_API VaistStatus vaist_kv_paged_create(
    struct VaistRuntime *rt,
    const VaistKVCachePagedConfig *cfg,
    VaistKVCachePaged **out);

/**
 * \brief Destroy a paged KV cache, freeing all GPU and host resources.
 * \param kvc The cache to destroy (may be NULL).
 */
VAIST_API void vaist_kv_paged_destroy(VaistKVCachePaged *kvc);

/**
 * \brief Allocate N free blocks from the pool.
 * \param kvc        Cache handle (must not be NULL).
 * \param num_blocks Number of blocks to allocate.
 * \param block_ids  Output array of size num_blocks, receives allocated IDs.
 * \retval VAIST_OK on success (block_ids filled with IDs).
 * \retval VAIST_INVALID_ARGUMENT for NULL params.
 * \retval VAIST_OUT_OF_MEMORY if not enough free blocks.
 */
VAIST_API VaistStatus vaist_kv_paged_alloc_block(
    VaistKVCachePaged *kvc, uint32_t num_blocks, uint32_t *block_ids);

/**
 * \brief Return a single block to the free pool.
 * \param kvc      Cache handle (must not be NULL).
 * \param block_id Block ID to free.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL kvc or invalid block_id.
 */
VAIST_API VaistStatus vaist_kv_paged_free_block(
    VaistKVCachePaged *kvc, uint32_t block_id);

/**
 * \brief Retrieve the block table for a sequence.
 * \param kvc      Cache handle.
 * \param seq_id   Sequence identifier.
 * \param out_blocks Output buffer for block IDs.
 * \param cap      Capacity of out_blocks.
 * \param count    Receives number of blocks assigned.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL params.
 */
VAIST_API VaistStatus vaist_kv_paged_get_block_table(
    VaistKVCachePaged *kvc, uint32_t seq_id,
    uint32_t *out_blocks, size_t cap, size_t *count);

/**
 * \brief Assign a set of blocks to a sequence.
 * \param kvc    Cache handle.
 * \param seq_id Sequence identifier.
 * \param blocks Array of block IDs to assign.
 * \param count  Number of block IDs in blocks.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL params.
 * \retval VAIST_OUT_OF_MEMORY if the block table cannot grow.
 */
VAIST_API VaistStatus vaist_kv_paged_assign_blocks(
    VaistKVCachePaged *kvc, uint32_t seq_id,
    const uint32_t *blocks, size_t count);

/**
 * \brief Write K/V tensor data into a block at a token offset.
 * \param kvc         Cache handle.
 * \param block_id    Target block.
 * \param layer       Transformer layer index.
 * \param token_idx   Token index within the block (0..block_size-1).
 * \param data        Input tensor data.
 * \param element_size Size of one KV element in bytes (e.g. 2 for fp16, 1 for q8_0).
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL/invalid params.
 */
VAIST_API VaistStatus vaist_kv_paged_write_tensor(
    VaistKVCachePaged *kvc, uint32_t block_id, uint32_t layer,
    uint32_t token_idx, const void *data, size_t element_size);

/**
 * \brief Get the GPU buffer handle for a given layer's KV cache.
 * \param kvc    Cache handle.
 * \param layer  Layer index.
 * \param handle Receives the opaque VkBuffer handle.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL params or invalid layer.
 */
VAIST_API VaistStatus vaist_kv_paged_gpu_buffer(
    VaistKVCachePaged *kvc, uint32_t layer, void **handle);

/**
 * \brief Free all blocks assigned to a sequence.
 * \param kvc    Cache handle.
 * \param seq_id Sequence identifier.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL kvc.
 */
VAIST_API void vaist_kv_paged_clear_seq(VaistKVCachePaged *kvc, uint32_t seq_id);

#ifdef __cplusplus
}
#endif
#endif
