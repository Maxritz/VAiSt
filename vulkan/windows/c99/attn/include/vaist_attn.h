#ifndef VAIST_ATTN_H
#define VAIST_ATTN_H
#include "vaist_runtime.h"
#include "vaist_core.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vaist_attn_ctx vaist_attn_ctx;

typedef struct {
    float       scale;
    uint32_t    head_dim;
    uint32_t    num_q_heads;     // query heads
    uint32_t    num_kv_heads;     // KV heads (pre-armed GQA)
    uint32_t    max_seqlen;       // max context per page
    uint32_t    block_size;       // tokens per page (1, 16, 64...)
    uint32_t    max_blocks;       // max pages per sequence
    uint32_t    total_blocks;     // total cache pages in device memory
    uint32_t    batch;            // batch size (for spec_verify dispatch)
    /* --- Hierarchical sparse attention (CSA2) --- */
    uint32_t    sparse_ratio;     // 0-100: fraction of tokens to attend to (0 = dense)
    uint32_t    block_stride;     // block size for hierarchical scoring (default: 16)
} vaist_attn_cfg;

VAIST_API vaist_attn_ctx* vaist_attn_create(const VaistRuntime* rt, const vaist_attn_cfg* cfg);
VAIST_API VaistStatus vaist_attn_destroy(vaist_attn_ctx* ctx);

/* Dispatch flash-decode attention.
   q:          [num_q_heads * head_dim] f32 input query (host-visible)
   k_cache:    VkBuffer handle (fp16 K-cache, paged, via vaist_buffer_gpu_handle)
   v_cache:    VkBuffer handle (fp16 V-cache, paged)
   block_tables:[num_kv_heads * max_blocks] uint32 page indices (host-visible)
   seqlen:     current context length (≤ cfg.max_seqlen)
   out:        [num_q_heads * head_dim] f32 output (host-visible, written on success)

   K/V cache buffers must be pre-populated by the caller (via vaist_buffer_upload
   or GPU-side fill). Returns VAIST_DEVICE_ERROR if no Vulkan device available,
   so the caller falls back to CPU dequant+GEMM.
*/
VAIST_API VaistStatus vaist_attn_flash_decode(vaist_attn_ctx* ctx,
    const void* q,
    void* k_cache_gpu_buf,
    void* v_cache_gpu_buf,
    const uint32_t* block_tables,
    uint32_t seqlen,
    float* out);

/**
 * \brief Speculative-decoding verify attention.
 *
 * SpecForge pattern: draft model proposes B candidate tokens, then target
 * model verifies via attention. This shader computes the verify-phase
 * attention output for accepted positions only.
 *
 * q:              [batch * num_q_heads * head_dim] fp16 queries
 * k_cache/v_cache: paged fp16 KV cache (device buffers)
 * block_tables:    page indices into KV cache
 * draft_tokens:    [batch] draft model token IDs (for position masking)
 * seqlen:          context length
 * n_active:        [batch] verified token count per sequence
 * out:             [batch * num_q_heads * head_dim] fp16 outputs
 * verify_mask:     bitmask of positions to compute (0=skip, 1=verify)
 * spec_depth:      draft decode depth
 */
VAIST_API VaistStatus vaist_attn_spec_verify(vaist_attn_ctx* ctx,
    const void* q,
    void* k_cache_gpu_buf,
    void* v_cache_gpu_buf,
    const uint32_t* block_tables,
    const uint32_t* draft_tokens,
    uint32_t seqlen,
    const uint32_t* n_active,
    void* out,
    uint32_t verify_mask,
    uint32_t spec_depth);

/* ---- Hierarchical Sparse Attention (DeepSeek-V4.1-Flash CSA2 pattern) ---- */

/**
 * \brief Hierarchical sparse attention: block-level scoring then token-level
 *       top-K within selected blocks.
 *
 * Uses the HISA pattern (hierarchical indexing) to reduce long-context scoring:
 *   1. Score all blocks (coarse), select top-K blocks
 *   2. Score tokens only within selected blocks (fine)
 *   3. Apply softmax + weighted sum
 *
 * \note USAGE: GPU compute primitive — does NOT dispatch from model forward
 *       pass automatically. Requires cfg.sparse_ratio > 0.
 */
/**
 * Uses the HISA pattern (hierarchical indexing) to reduce long-context scoring:
 *   1. Score all blocks (coarse), select top-K blocks
 *   2. Score tokens only within selected blocks (fine)
 *   3. Apply softmax + weighted sum
 *
 * This replaces full O(N^2) attention with O(N * block_size + K_blocks * block_size)
 * for the scoring phase.
 *
 * q:              [num_q_heads * head_dim] f32 queries
 * k_cache/v_cache: paged KV cache
 * block_tables:   page indices
 * top_k_blocks:   output [num_heads] number of blocks to retrieve per head
 * selected_blocks: output [num_heads * top_k_blocks] block indices
 * seqlen:         context length
 * out:            [num_q_heads * head_dim] f32 output
 */
VAIST_API VaistStatus vaist_attn_sparse_hierarchical(vaist_attn_ctx* ctx,
    const void* q,
    void* k_cache_gpu_buf,
    void* v_cache_gpu_buf,
    const uint32_t* block_tables,
    uint32_t seqlen,
    uint32_t top_k_blocks,
    uint32_t* selected_blocks,
    float* out);

/**
 * \brief Causal Encoder-Decoder attention (DeepSeek-V4.1-Flash CED prefill).
 *
 * Lower layers act as causal encoder, producing compact global KV.
 * Upper layers derive from encoder output (reduces O(N*L) to O(N*L/2)).
 *
 * Same interface as vaist_attn_flashDecode but with encoder-side optimizations:
 * - encoder_output: [num_kv_heads * head_dim] global representation
 * - is_encoder_layer: if true, computes full encoder KV; if false, derives from encoder output
 */
VAIST_API VaistStatus vaist_attn_ced(vaist_attn_ctx* ctx,
    const void* q,
    void* k_cache_gpu_buf,
    void* v_cache_gpu_buf,
    const uint32_t* block_tables,
    uint32_t seqlen,
    const void* encoder_output,
    uint32_t is_encoder_layer,
    float* out);

#ifdef __cplusplus
}
#endif
#endif
