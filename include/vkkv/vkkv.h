/**
 * \file vkkv.h
 * \brief Cross-model KV cache transfer (arXiv:2608.03893) via per-head
 *        closed-form ridge mappers.
 *
 * VKKV maps a source model's K/V cache to a target model's K/V cache so that
 * prefill can be skipped when swapping between same-family models. The module
 * implements the LINEAR-ALGEBRA layer only: it consumes and produces plain
 * float buffers (column-major convention per the API contract, implemented as
 * row-major [rows x cols] buffers where "rows" are the leading stride) and is
 * validated numerically against CPU linear algebra. It does NOT build an LLM
 * runtime.
 *
 * The three-step design of the paper is represented as follows:
 *   1. RoPE stripping is the CALLER's responsibility. The caller must present
 *      RoPE-free K/V vectors to vkkv_fit_cpu() and vkkv_apply().
 *   2. Per-head top-k source layer selection is represented by ONE fitted
 *      mapper per head (one Wh matrix per head); selecting which calibration
 *      data feeds each head is the caller's job.
 *   3. The per-head ridge fit + apply is this module.
 *
 * The mapper math: for a matched calibration set (X_h, Y_h) of n samples where
 * X_h is [n x src_dim] (source, RoPE-stripped) and Y_h is [n x tgt_dim]
 * (target), the per-head mapper solves
 *
 *     W_h = (X_h^T X_h + lambda * I)^-1 * X_h^T Y_h      (ridge regression)
 *
 * and apply maps a source block to a target block via
 *
 *     TARGET[r][j] = sum_k SOURCE[r][k] * W_h[k][j].
 *
 * The FIT runs on the HOST (calibration is an offline step): the closed-form
 * solve is done in C99 double precision inside vkkv_fit_cpu() and the fitted
 * W_h matrices are uploaded once to a device buffer. The APPLY runs on the GPU
 * (the production path): vkkv_apply() records one compute dispatch into a
 * caller-supplied command buffer.
 *
 * All buffers are Vulkan-native (VkBuffer). All matrix buffers are row-major
 * float32: element [r][c] of an [m x k] buffer lives at r*k + c.
 */
#ifndef VKKV_H
#define VKKV_H

#include <vulkan/vulkan.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Cross-model KV transfer context (owns the fitted mappers).
 *
 * Opaque. Owns an internal VKRuntime (used to allocate and upload the W
 * buffers), a lazily-created compute pipeline for the apply shader, and a
 * device buffer holding every fitted mapper W_h stacked as
 * [n_heads][src_dim * tgt_dim] row-major floats.
 *
 * Not thread-safe; callers must serialize concurrent vkkv_*() calls.
 */
typedef struct VkKVTransfer VkKVTransfer;

/* ===========================================================================
 * Context lifecycle
 * ========================================================================== */

/**
 * \brief Create a VkKVTransfer bound to a physical/logical device pair.
 *
 * The transfer owns an internal VKRuntime for W-buffer allocation and
 * uploads/downloads; it fetches a compute queue from queue family 0 of
 * \p dev (vkGetDeviceQueue(dev, 0, 0)). The device must therefore expose a
 * compute-capable queue on family 0 (this matches the test harnesses across
 * the stack). No W data is allocated yet beyond the device mapper buffer.
 *
 * \param pd          Physical device (used for capability queries / allocator).
 * \param dev         Logical device the transfer binds to.
 * \param n_heads     Number of heads to map (one Wh per head). Must be > 0.
 * \param src_dim     Source feature dimension (per K/V vector). Must be > 0.
 * \param tgt_dim     Target feature dimension (per K/V vector). Must be > 0.
 * \param ridge_lambda Ridge penalty on the diagonal of X^T X. Must be >= 0.
 * \param pT          Receives the new transfer on success.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument or no family-0 queue.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY W buffer allocation failed.
 */
VkResult vkkv_create_transfer(VkPhysicalDevice pd, VkDevice dev,
                              uint32_t n_heads, uint32_t src_dim,
                              uint32_t tgt_dim, float ridge_lambda,
                              VkKVTransfer **pT);

/**
 * \brief Destroy a transfer and release the W buffer, pipeline, and runtime.
 *
 * \param t Transfer to destroy. May be NULL.
 */
void vkkv_destroy_transfer(VkKVTransfer *t);

/* ===========================================================================
 * Fitting (host-side; calibration is offline)
 * ========================================================================== */

/**
 * \brief Fit all per-head ridge mappers from CPU calibration buffers.
 *
 * For each head h, solves W_h = (X_h^T X_h + lambda*I)^-1 X_h^T Y_h in C99
 * double precision (Gauss-Jordan with partial pivoting on the augmented
 * system [G | B]), then uploads the stacked W buffer to the device in one
 * copy. Calling this again re-fits and re-uploads (the device buffer is
 * overwritten).
 *
 * \param t         Valid transfer.
 * \param X         Array of n_heads pointers; X[h] is [n_samples x src_dim]
 *                  row-major (SOURCE keys/values, RoPE-stripped).
 * \param Y         Array of n_heads pointers; Y[h] is [n_samples x tgt_dim]
 *                  row-major (TARGET keys/values).
 * \param n_samples Number of matched calibration samples per head. Must be > 0.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument or singular system.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 */
VkResult vkkv_fit_cpu(VkKVTransfer *t, const float *const *X,
                      const float *const *Y, uint32_t n_samples);

/* ===========================================================================
 * Applying (GPU; the production path)
 * ========================================================================== */

/**
 * \brief Map a source KV block to a target KV block for head \p h on the GPU.
 *
 * Records a single compute dispatch into \p cmd:
 *
 *     dst[r][j] = sum_k src[r][k] * W_h[k][j]   (r in [0, n), j in [0, tgt_dim))
 *
 * with src [n x src_dim] row-major and dst [n x tgt_dim] row-major. The
 * caller owns \p cmd (recording state), synchronization, and buffer memory;
 * src/dst must be STORAGE buffers (created via vkr_malloc or equivalent).
 * A pipeline is created lazily on the first apply call and cached.
 *
 * \param t   Valid, fitted transfer (vkkv_fit_cpu must have been called).
 * \param cmd Command buffer in the recording state.
 * \param h   Head index in [0, n_heads).
 * \param src Source device buffer [n x src_dim] floats (RoPE-stripped).
 * \param n   Number of rows (sequence positions) in src/dst. Must be > 0.
 * \param dst Destination device buffer [n x tgt_dim] floats.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument, no fit yet, or no shader.
 */
VkResult vkkv_apply(VkKVTransfer *t, VkCommandBuffer cmd, uint32_t h,
                    VkBuffer src, uint32_t n, VkBuffer dst);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKKV_H */
