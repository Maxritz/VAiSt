#ifndef VAIST_COMPUTE_H
#define VAIST_COMPUTE_H
#include "vaist_core.h"
#include "vaist_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute-path capability ladder (mirrors C:/temp/SKILL.md "capability ladder").
 * Selection is per-call from device caps + problem shape; the SDK never assumes
 * a fixed GPU capability. Order = fastest-available first:
 *
 *   VULKAN_DOT   — real OpSDotKHR (integer_dot_product_8bit_accelerated==1)
 *   VULKAN_TILE  — Vulkan tiled matmul but NO accelerated dot (scalar in-shader FMA)
 *   MATMUl_FREE  — zero-mul ternary {-1,0,+1} / 1-bit xnor matvec (2406.02528,
 *                  2608.03142, 2608.01528). Always available, weakest GPU friendly.
 *   SIMD         — CPU (Zen3 AVX2 / Intel AMX), host-visible
 *   SCALAR       — portable fallback, never fails to build
 *
 * MatMul-free is the *guaranteed* path: it removes matrix multiplication
 * entirely, so it is the fallback when integer-dot is absent / unaccelerated
 * and runs on any Vulkan 1.0 device + any CPU.
 */
typedef enum VaistComputePath {
    VAIST_PATH_SCALAR       = 0,
    VAIST_PATH_SIMD         = 1,
    VAIST_PATH_VULKAN_TILE  = 2,   /* new: tiled, no dot-product */
    VAIST_PATH_VULKAN_DOT   = 3,   /* existing meaning: accelerated dot */
    VAIST_PATH_MATMUL_FREE  = 4,   /* new: ternary + xnor, zero matmul */
} VaistComputePath;

/* Existing elementwise + reduce (unchanged ABI). CPU SIMD preferred. */
VAIST_API VaistStatus vaist_add_f32(const float*a,const float*b,float*out,size_t n);
VAIST_API VaistStatus vaist_mul_f32(const float*a,const float*b,float*out,size_t n);
VAIST_API VaistStatus vaist_reduce_sum_f32(const float*a,float*out,size_t n);
VAIST_API VaistStatus vaist_dot_i8(const int8_t*a,const int8_t*b,int32_t*out,size_t n);

/* Classic matmul (kept). Picks a path internally via vaist_compute_best_path. */
VAIST_API VaistStatus vaist_matmul_f32(const float*a,const float*b,float*out,size_t m,size_t k,size_t n);

/* Capability-ladder selector: pure function of caps + shape. */
VAIST_API VaistComputePath vaist_compute_best_path(const VaistRuntime*rt,size_t m,size_t k,size_t n,int prefer_matmul_free);

#ifdef __cplusplus
}
#endif
#endif
