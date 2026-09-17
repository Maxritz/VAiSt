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

/* Dispatch one prefill or decode step.
   Caller provides VkBuffer handles (from vaist_buffer_gpu_handle or direct).
   q_buf:   VkBuffer (host_dim * num_q_heads floats, host-visible for upload)
   k_cache: VkBuffer (fp16 K-cache, paged)
   v_cache: VkBuffer (fp16 V-cache, paged)
   bt_buf:  VkBuffer (num_kv_heads * max_blocks uint32 page indices)
   out_buf: VkBuffer (host_dim * num_q_heads floats, host-visible for readback)
   All buffers must be bound to memory and host-visible if staging is used.

   seqlen: current context length (≤ ctx->cfg.max_seqlen)
*/
VAIST_API VaistStatus vaist_attn_flash_decode(vaist_attn_ctx* ctx,
    const void* q,                    // host_dim * num_q_heads (f32)
    const uint32_t* block_tables,      // num_kv_heads * max_blocks
    uint32_t seqlen,                   // current context length
    float* out);                       // host_dim * num_q_heads (f32)

#ifdef __cplusplus
}
#endif
#endif
