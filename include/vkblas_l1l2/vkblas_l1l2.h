/**
 * \file vkblas_l1l2.h
 * \brief Vulkan-native BLAS Level 1 and Level 2 operations (rocBLAS-style).
 *
 * Companion library to VKBLAS (GEMM). vkblas_l1l2 REUSES the opaque
 * #VkBLASContext from include/vkblas/vkblas.h — it does not create its own
 * context. Every public function takes a `VkBLASContext*` and records compute
 * work into a caller-supplied `VkCommandBuffer`, exactly like vkblas_sgemm.
 *
 * Design (documented per task): option (a) — reuse VkBLASContext. The library
 * links against libvkblas and drives the shared context's descriptor pool,
 * pipeline cache, descriptor-set layout, pipeline layout and the internal
 * helpers in src/vkblas/vkblas_internal.h (vkblas_alloc_descriptor_set,
 * vkblas_write_descriptor_set, vkblas_push_pc). L1/L2 pipelines are stored in
 * the context's cache under hash keys that carry a distinct marker bit so they
 * can never collide with GEMM pipeline keys.
 *
 * Conventions:
 *   - Vectors/matrices are VkBuffer handles. All data is COLUMN-MAJOR like
 *     rocBLAS. incx/incy are element strides (must be positive).
 *   - The caller owns all buffers and synchronization; the library only records
 *     commands. No device memory is allocated.
 *   - f32 host scalars: alpha/beta are dereferenced from host pointers
 *     (VKBLAS_POINTER_MODE_HOST semantics; the context pointer-mode field is
 *     honored only for the GEMM family, not for these ops).
 *   - Zero/negative n is a no-op returning VK_SUCCESS, mirroring BLAS.
 *   - amax returns a 0-based index (deviation from 1-based BLAS convention;
 *     documented in \ref vkblas_l1_amax).
 *   - Result buffers for reductions must be large enough to hold the per-
 *     workgroup partials, see \ref vkblas_l1_dot.
 */
#ifndef VKBLAS_L1L2_H
#define VKBLAS_L1L2_H

#include <vulkan/vulkan.h>
#include <stdint.h>

#include "vkblas/vkblas.h" /* VkBLASContext, VkBLASOperation_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * BLAS Level 1
 * ========================================================================== */

/**
 * \brief axpy:  y = alpha * x + y   (in-place)
 *
 * \param ctx   Valid VkBLASContext (shared with VKBLAS).
 * \param cmd   Command buffer in the recording state.
 * \param n     Number of elements (no-op when n <= 0).
 * \param alpha Host pointer to the scalar alpha (f32).
 * \param x     Buffer holding vector x.
 * \param incx  Element stride for x (positive).
 * \param y     Buffer holding vector y (read and written in place).
 * \param incy  Element stride for y (positive).
 * \retval VK_SUCCESS
 */
VkResult vkblas_l1_axpy(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, const float* alpha,
                        VkBuffer x, int32_t incx,
                        VkBuffer y, int32_t incy);

/**
 * \brief axpy (f16):  y = alpha * x + y   (in-place)
 *
 * Halfs are stored one-per-uint32 (low 16 bits of each element) in the
 * buffers; alpha is an f32 host scalar. Implemented with integer bit-packing
 * in the shader so no shaderFloat16 device feature is required.
 */
VkResult vkblas_l1_axpy_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                            int32_t n, const float* alpha,
                            VkBuffer x, int32_t incx,
                            VkBuffer y, int32_t incy);

/**
 * \brief scal:  x = alpha * x   (in-place)
 *
 * \param ctx   Valid VkBLASContext (shared with VKBLAS).
 * \param cmd   Command buffer in the recording state.
 * \param n     Number of elements (no-op when n <= 0).
 * \param alpha Host pointer to the scalar alpha (f32).
 * \param x     Buffer holding vector x (read and written in place).
 * \param incx  Element stride for x (positive).
 * \retval VK_SUCCESS
 */
VkResult vkblas_l1_scal(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, const float* alpha,
                        VkBuffer x, int32_t incx);

/**
 * \brief scal (f16):  x = alpha * x   (in-place)
 *
 * Halfs stored one-per-uint32 (low 16 bits); alpha is an f32 host scalar.
 */
VkResult vkblas_l1_scal_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                            int32_t n, const float* alpha,
                            VkBuffer x, int32_t incx);

/**
 * \brief dot:  result[0] = dot(x, y) = sum_i x[i*incx] * y[i*incy]
 *
 * Performed as a two-stage reduction. The result buffer is used as scratch for
 * the per-workgroup partials, so it MUST be at least
 * `ceil(n / 256) * sizeof(float)` bytes (in practice allocate 64+ floats).
 * No-op for n <= 0 (result buffer is left unchanged).
 *
 * \param result VkBuffer receiving the f32 scalar; also holds partials
 *               during the two dispatches.
 */
VkResult vkblas_l1_dot(VkBLASContext* ctx, VkCommandBuffer cmd,
                       int32_t n, VkBuffer x, int32_t incx,
                       VkBuffer y, int32_t incy, VkBuffer result);

/**
 * \brief dot (f16):  result[0] = dot(x, y)  (f32 accumulate + f32 result)
 *
 * Halfs stored one-per-uint32 (low 16 bits); partials and the result are f32,
 * so the result buffer layout matches \ref vkblas_l1_dot.
 */
VkResult vkblas_l1_dot_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                           int32_t n, VkBuffer x, int32_t incx,
                           VkBuffer y, int32_t incy, VkBuffer result);

/**
 * \brief nrm2:  result[0] = ||x||_2 = sqrt(sum_i x[i*incx]^2)
 *
 * Same two-stage scratch convention as \ref vkblas_l1_dot.
 * No-op for n <= 0 (result buffer is left unchanged).
 */
VkResult vkblas_l1_nrm2(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, VkBuffer x, int32_t incx, VkBuffer result);

/**
 * \brief asum:  result[0] = sum_i |x[i*incx]|
 *
 * Same two-stage scratch convention as \ref vkblas_l1_dot.
 * No-op for n <= 0 (result buffer is left unchanged).
 */
VkResult vkblas_l1_asum(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, VkBuffer x, int32_t incx, VkBuffer result);

/**
 * \brief amax:  result[0] = index (0-based) of max |x[i*incx]|
 *
 * Ties resolve to the lowest index, matching rocBLAS iamax tie-breaking.
 * NOTE: the returned index is 0-based; BLAS/rocBLAS conventionally return a
 * 1-based index. This deviation is deliberate for native Vulkan indexing.
 *
 * The result buffer is used as scratch for per-workgroup (index, value) pairs,
 * so it MUST be at least `2 * ceil(n / 256)` uint32s (in practice allocate
 * 128+ floats). No-op for n <= 0 (result buffer is left unchanged).
 */
VkResult vkblas_l1_amax(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, VkBuffer x, int32_t incx, VkBuffer result);

/* ===========================================================================
 * BLAS Level 2
 * ========================================================================== */

/**
 * \brief gemv:  y = alpha * op(A) * x + beta * y   (in-place y)
 *
 * op(A) is m x n, column-major with leading dimension lda.
 *   VKBLAS_OP_N: A is m x n,       lda >= max(1, m).
 *   VKBLAS_OP_T: A is n x m stored, lda >= max(1, n);  op(A) = A^T.
 *   VKBLAS_OP_C: treated as transpose (real data).
 * x has length n (stride incx), y has length m (stride incy), both positive.
 *
 * \param alpha Host pointer to the f32 scalar alpha.
 * \param beta  Host pointer to the f32 scalar beta.
 * \retval VK_SUCCESS
 */
VkResult vkblas_l2_gemv(VkBLASContext* ctx, VkCommandBuffer cmd,
                        VkBLASOperation_t transA,
                        int32_t m, int32_t n,
                        const float* alpha,
                        VkBuffer A, int32_t lda,
                        VkBuffer x, int32_t incx,
                        const float* beta,
                        VkBuffer y, int32_t incy);

/**
 * \brief gemv (f16):  y = alpha * op(A) * x + beta * y   (in-place y)
 *
 * Halfs stored one-per-uint32 (low 16 bits); alpha/beta are f32 host scalars.
 */
VkResult vkblas_l2_gemv_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                            VkBLASOperation_t transA,
                            int32_t m, int32_t n,
                            const float* alpha,
                            VkBuffer A, int32_t lda,
                            VkBuffer x, int32_t incx,
                            const float* beta,
                            VkBuffer y, int32_t incy);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKBLAS_L1L2_H */
