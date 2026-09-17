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

#ifdef __cplusplus
}
#endif
#endif
