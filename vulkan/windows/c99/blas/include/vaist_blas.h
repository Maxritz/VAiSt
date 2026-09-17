#ifndef VAIST_BLAS_H
#define VAIST_BLAS_H
#include "vaist_core.h"
#include "vaist_runtime.h"
#include "vaist_compute.h"   /* VaistComputePath (needed by vaist_blas_best_path) */
#include "vaist_tensor.h"
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

/* Pick the best path now that device caps are available. */
VAIST_API VaistComputePath vaist_blas_best_path(const VaistRuntime*rt,
        size_t m,size_t k,size_t n,VaistWeightFormat wfmt);
#ifdef __cplusplus
}
#endif
#endif
