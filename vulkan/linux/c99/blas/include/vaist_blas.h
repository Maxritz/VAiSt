#ifndef VAIST_BLAS_H
#define VAIST_BLAS_H
#include "vaist_core.h"
#include "vaist_runtime.h"
#include "vaist_compute.h"   /* VaistComputePath (needed by vaist_blas_best_path) */
#include "vaist_tensor.h"
#include "vaist_quant.h"     /* VaistQuantType (for vaist_blas_mul_mat_q) */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * vaist_blas: tiled + MatMul-free BLAS for weak GPUs (9070 XT / 6700 XT),
 * CPU (Zen3 AVX2 / Intel AMX), and iGPU (Intel Ultra 5 135H).
 *
 * Tiling contract (per C:/temp/SKILL.md):
 *   - logical tile is TILE_M x TILE_N, decoupled from subgroup size.
 *   - workgroup owns one tile; shared memory holds the kt-register tile.
 *   - edge tiles are bounds-checked in-shader.
 *   - dispatch ladder: VULKAN_DOT -> VULKAN_TILE -> MATMUL_FREE -> SIMD -> SCALAR.
 *
 * MatMul-free kernels remove matmul entirely:
 *   - ternary {-1,0,+1}: add/subtract per MAC (2406.02528, 2608.03142).
 *   - 1-bit xnor: popcount of xor + per-row scale (2608.01528, 2608.00860).
 */
#define VAIST_BLAS_TILE_M 16u
#define VAIST_BLAS_TILE_N 16u
#define VAIST_BLAS_TILE_K 16u   /* must divide K; edges handled */

/* Weight layout enums for the MatMul-free paths. */
typedef enum VaistWeightFormat {
    VAIST_W_FP16       = 0,  /* fp16/f32 baseline */
    VAIST_W_TERNARY_I8 = 1,  /* int8 signs {-1,0,+1}, one scale per row */
    VAIST_W_BINARY_I1  = 2,  /* 1-bit packed, one scale per row */
    VAIST_W_Q4_0       = 3,  /* ggml q4_0, dequantized on the fly */
    VAIST_W_Q8_0       = 4,  /* ggml q8_0 */
} VaistWeightFormat;

/*
 * Tiled GEMM: C += A(mxk) x B(kxn).
 *   A,B,C are row-major float32. tile_k must be a multiple of TILE_K for the
 *   Vulkan tile path; the scalar/simd/tile paths handle the general shape.
 *   'w' may be NULL for fp32, or a non-owning pointer to a quantized weight
 *   blob described by 'wfmt' (see vaist_pack_*).
 */
VAIST_API VaistStatus vaist_blas_gemm(const VaistRuntime*rt,
        const float*A,const float*B,float*C,size_t m,size_t k,size_t n,
        const void*w,VaistWeightFormat wfmt);

/* MatMul-free matvec: y(n) = scale .* (W * x) using sign/binary weights.
 *   wsign: int8 {-1,0,+1} of length k*n  (row-major).
 *   wbit : packed 1-bit of length ceil(k*n/8).
 *   scale: one float per output row (n). */
VAIST_API VaistStatus vaist_blas_matvec_ternary(const VaistRuntime*rt,
        const int8_t*wsign,const float*scale,size_t k,size_t n,const float*x,float*y);
VAIST_API VaistStatus vaist_blas_matvec_binary(const VaistRuntime*rt,
        const uint8_t*wbit,const float*scale,size_t k,size_t n,const float*x,float*y);

/* Quantized matmul: C(m,n) += A(m,k) * W_q(k,n)
 *   'w' is a GGUF quantized weight blob in 'qtype' format, row-major blocks.
 *   Dequantizes W_q to fp32 on the CPU (scalar fallback) then calls gemm_scalar.
 *   TODO: Vulkan fused dequant+gemm compute shader (vaist_blas_vk_mul_mat_q). */
VAIST_API VaistStatus vaist_blas_mul_mat_q(const VaistRuntime*rt,
        const float*A,float*C,size_t m,size_t k,size_t n,
        const void*w,VaistQuantType qtype);

/* MoE routing: top-k gate dispatch (mirrors FreeToken moe.cuh + ATOM
 * atom/attention/flash_deciding.py).  CPU fallback computes argmax(dot) per row;
 * the Vulkan fast path dispatches the moe_route.comp shader. */
VAIST_API VaistStatus vaist_blas_moe_route(const VaistRuntime*rt,
        const float*x,size_t m,size_t k,size_t k_gate,
        const float*gate,size_t num_experts,
        uint32_t*indices,float*scores);

/*
 * MoE (Mixture of Experts) dispatch — full top-k pipeline.
 *
 * Replaces: atom/model_ops/moe.py:FusedMoE (top-k routing + expert dispatch),
 *           atom/models/deepseek_v4.py:MoEBlock (routed experts + shared expert)
 *
 * Three-stage API:
 *   1. vaist_blas_moe_topk   — compute top-k indices+scores (softmax-normalized)
 *   2. vaist_blas_moe_dispatch — scatter token rows into expert buffers
 *   3. (caller runs per-expert GEMMs)
 *   4. vaist_blas_moe_combine  — gather weighted outputs back to per-token rows
 *
 * Data layouts:
 *   x:              (num_tokens, hidden_dim) row-major f32
 *   gate:           (num_experts, hidden_dim) row-major f32
 *   topk_indices:   (num_tokens, k) uint32 expert IDs
 *   topk_scores:    (num_tokens, k) raw f32 scores
 *   normalized_scores: (num_tokens, k) softmax-normalized f32
 *   blocks:         (num_experts) token count per expert
 *   offsets:        (num_experts) byte offset into dispatch_buf per expert
 *   dispatch_buf:   (sum_blocks * hidden_dim) scattered+weighted token inputs
 *   expert_out:     (sum_blocks * hidden_dim) per-expert GEMM outputs
 *   combine_buf:    (num_tokens * hidden_dim) final combined output
 */
#define VAIST_MOE_MAX_TOPK 8u

VAIST_API VaistStatus vaist_blas_moe_topk(
        const VaistRuntime* rt,
        const float* x,
        size_t num_tokens,
        size_t hidden_dim,
        const float* gate,
        size_t num_experts,
        uint32_t k,
        uint32_t* topk_indices,
        float* topk_scores,
        float* normalized_scores
);

VAIST_API VaistStatus vaist_blas_moe_dispatch(
        const VaistRuntime* rt,
        const float* x,
        const uint32_t* topk_indices,
        const float* topk_scores,
        size_t num_tokens,
        size_t hidden_dim,
        size_t num_experts,
        uint32_t k,
        uint32_t* blocks,
        uint32_t* offsets,
        float* dispatch_buf
);

VAIST_API VaistStatus vaist_blas_moe_combine(
        const VaistRuntime* rt,
        const float* expert_out,
        const uint32_t* topk_indices,
        const float* normalized_scores,
        const uint32_t* offsets,
        size_t num_tokens,
        size_t hidden_dim,
        size_t num_experts,
        uint32_t k,
        float* combine_buf
);

/* Pick the best path now that device caps are available. */
VAIST_API VaistComputePath vaist_blas_best_path(const VaistRuntime*rt,
        size_t m,size_t k,size_t n,VaistWeightFormat wfmt);
#ifdef __cplusplus
}
#endif
#endif
