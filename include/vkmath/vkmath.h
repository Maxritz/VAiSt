/**
 * \file vkmath.h
 * \brief Vulkan-native math / activation / reduction interface.
 *
 * VKMath provides elementwise operations (ReLU, SiLU, GELU, tanh, sigmoid),
 * elementwise binary ops (add, mul, add_mul, scale), and dimension-wise
 * reductions (max_reduce, sum_reduce) as Vulkan compute dispatches.
 * All memory and command objects are standard Vulkan handles.
 *
 * The caller owns buffer memory and synchronization; VKMath only records
 * compute work into a caller-supplied VkCommandBuffer.
 */
#ifndef VKMATH_H
#define VKMATH_H

#include <vulkan/vulkan.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Vulkan-native math context.
 *
 * Wraps the VkDevice, a pipeline cache (lazily populated), a descriptor
 * set pool, and the detected GPU architecture. The caller creates this
 * once per VkDevice and reuses it for all math calls. It is thread-unsafe;
 * callers must serialize concurrent access.
 *
 * Internal layout is hidden; treat as opaque.
 */
typedef struct VkMathContext VkMathContext;

/* ===========================================================================
 * Context lifecycle
 * ========================================================================== */

/**
 * \brief Create a VkMathContext bound to a VkDevice.
 *
 * The context lazily creates Vulkan compute pipelines on first dispatch
 * and caches them for subsequent calls. No device memory is allocated
 * by this function; the caller retains full memory ownership.
 *
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the context will bind to.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Device queries or shader load failed.
 */
VkResult vkmath_create_context(VkPhysicalDevice physicalDevice,
                               VkDevice device,
                               VkMathContext** pContext);

/**
 * \brief Destroy a VkMathContext and release all cached pipelines / descriptors.
 *
 * \param context Pointer to the context to destroy. May be NULL.
 */
void vkmath_destroy_context(VkMathContext* context);

/**
 * \brief Query the GPU architecture index the context is bound to.
 *
 * 0 = baseline (vendor-agnostic), 1 = subgroup, 2 = coopmatrix.
 *
 * \param context Valid context.
 * \return Architecture index.
 */
uint32_t vkmath_get_arch_index(VkMathContext* context);

/**
 * \brief Get the human-readable GPU architecture string.
 *
 * \param context Valid context.
 * \return e.g. "baseline", "subgroup", or "coopmatrix".
 */
const char* vkmath_get_arch_name(VkMathContext* context);

/**
 * \brief Detect GPU capabilities and select the optimal shader tier.
 *
 * Called automatically by vkmath_create_context(). Can be called again
 * after context creation to re-detect on a different physical device.
 *
 * \param context Valid context.
 * \param physicalDevice Physical device to query.
 * \retval VK_SUCCESS
 */
VkResult vkmath_init_capabilities(VkMathContext* context,
                                  VkPhysicalDevice physicalDevice);

/**
 * \brief Flush cached pipelines associated with the context.
 *
 * Call before vkDeviceWaitIdle if the device will be reset.
 */
void vkmath_flush_pipelines(VkMathContext* context);

/* ===========================================================================
 * Elementwise unary ops (f32)
 * ========================================================================== */

/**
 * \brief ReLU: out = max(0, in)
 * \param pc    Push constant block (uses num_elements, alpha for scaling).
 */
VkResult vkmath_relu_f32(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief SiLU (Swish): out = in * sigmoid(in)
 */
VkResult vkmath_silu_f32(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief GELU (tanh approximation): out = 0.5 * in * (1 + tanh(sqrt(2/pi) * (in + 0.044715 * in^3)))
 */
VkResult vkmath_gelu_f32(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief Hyperbolic tangent: out = tanh(in)
 */
VkResult vkmath_tanh_f32(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief Sigmoid: out = 1 / (1 + exp(-in))
 */
VkResult vkmath_sigmoid_f32(VkMathContext* ctx,
                            VkCommandBuffer cmd,
                            uint32_t num_elements,
                            VkBuffer input,
                            VkBuffer output);

/* ===========================================================================
 * Elementwise unary ops (f16)
 * ========================================================================== */

VkResult vkmath_relu_f16(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

VkResult vkmath_silu_f16(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

VkResult vkmath_gelu_f16(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

VkResult vkmath_tanh_f16(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

VkResult vkmath_sigmoid_f16(VkMathContext* ctx,
                            VkCommandBuffer cmd,
                            uint32_t num_elements,
                            VkBuffer input,
                            VkBuffer output);

/* ===========================================================================
 * Elementwise binary ops (f32)
 * ========================================================================== */

/**
 * \brief Elementwise add: out = a + b
 */
VkResult vkmath_add_f32(VkMathContext* ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer a,
                        VkBuffer b,
                        VkBuffer output);

/**
 * \brief Elementwise multiply: out = a * b
 */
VkResult vkmath_mul_f32(VkMathContext* ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer a,
                        VkBuffer b,
                        VkBuffer output);

/**
 * \brief Fused add-multiply: out = (a + b) * alpha
 *
 * Uses pc.alpha as the scalar multiplier.
 */
VkResult vkmath_add_mul_f32(VkMathContext* ctx,
                            VkCommandBuffer cmd,
                            uint32_t num_elements,
                            VkBuffer a,
                            VkBuffer b,
                            float alpha,
                            VkBuffer output);

/**
 * \brief Scale: out = alpha * in
 */
VkResult vkmath_scale_f32(VkMathContext* ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          float alpha,
                          VkBuffer input,
                          VkBuffer output);

/* ===========================================================================
 * Elementwise binary ops (f16)
 * ========================================================================== */

VkResult vkmath_add_f16(VkMathContext* ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer a,
                        VkBuffer b,
                        VkBuffer output);

VkResult vkmath_mul_f16(VkMathContext* ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer a,
                        VkBuffer b,
                        VkBuffer output);

VkResult vkmath_add_mul_f16(VkMathContext* ctx,
                            VkCommandBuffer cmd,
                            uint32_t num_elements,
                            VkBuffer a,
                            VkBuffer b,
                            float alpha,
                            VkBuffer output);

VkResult vkmath_scale_f16(VkMathContext* ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          float alpha,
                          VkBuffer input,
                          VkBuffer output);

/* ===========================================================================
 * Elementwise binary ops (bf16)
 *
 * bf16 is stored as uint16_t (top 16 bits of the f32 value, truncated).
 * Compute runs in f32, results truncate back to bf16. Requires
 * storageBuffer16BitAccess + scalarBlockLayout.
 * ========================================================================== */

VkResult vkmath_add_bf16(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer a,
                         VkBuffer b,
                         VkBuffer output);

VkResult vkmath_mul_bf16(VkMathContext* ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer a,
                         VkBuffer b,
                         VkBuffer output);

VkResult vkmath_add_mul_bf16(VkMathContext* ctx,
                             VkCommandBuffer cmd,
                             uint32_t num_elements,
                             VkBuffer a,
                             VkBuffer b,
                             float alpha,
                             VkBuffer output);

VkResult vkmath_scale_bf16(VkMathContext* ctx,
                           VkCommandBuffer cmd,
                           uint32_t num_elements,
                           float alpha,
                           VkBuffer input,
                           VkBuffer output);

/* ===========================================================================
 * Reductions (f32 / f16 / bf16)
 * ========================================================================== */

/**
 * \brief Max reduction along rows: for each row of num_cols elements,
 *        produce the maximum value. Input is [num_rows * num_cols], output
 *        is [num_rows].
 */
VkResult vkmath_max_reduce_dim_f32(VkMathContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_rows,
                                   uint32_t num_cols,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Max reduction (bf16): same semantics as \ref vkmath_max_reduce_dim_f32.
 * Input/output are uint16_t bf16 (storageBuffer16BitAccess + scalarBlockLayout).
 * Compute runs in f32 for exact comparison; result truncates to bf16.
 */
VkResult vkmath_max_reduce_dim_bf16(VkMathContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_rows,
                                    uint32_t num_cols,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Max reduction (f16): same semantics as \ref vkmath_max_reduce_dim_f32.
 * Input/output are float16_t (shaderFloat16 + storageBuffer16BitAccess).
 */
VkResult vkmath_max_reduce_dim_f16(VkMathContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_rows,
                                   uint32_t num_cols,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Sum reduction along rows: for each row of num_cols elements,
 *        produce the sum. Input is [num_rows * num_cols], output is [num_rows].
 */
VkResult vkmath_sum_reduce_dim_f32(VkMathContext *ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_rows,
                                   uint32_t num_cols,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Sum reduction (bf16): same semantics as \ref vkmath_sum_reduce_dim_f32.
 * Input/output are uint16_t bf16 (storageBuffer16BitAccess + scalarBlockLayout).
 * Compute runs in f32; result truncates to bf16.
 */
VkResult vkmath_sum_reduce_dim_bf16(VkMathContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_rows,
                                    uint32_t num_cols,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Sum reduction (f16): same semantics as \ref vkmath_sum_reduce_dim_f32.
 * Input/output are float16_t (shaderFloat16 + storageBuffer16BitAccess).
 */
VkResult vkmath_sum_reduce_dim_f16(VkMathContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_rows,
                                   uint32_t num_cols,
                                   VkBuffer input,
                                   VkBuffer output);

/* ===========================================================================
 * Reductions / normalizations (f32)
 * Input is [num_rows x num_cols], row-major. Output shape noted per op.
 * ========================================================================== */

/**
 * \brief Softmax over each row (numerically stable: max-subtract then exp).
 *
 * Input is [num_rows * num_cols], output is [num_rows * num_cols].
 * Each workgroup processes one row.
 */
VkResult vkmath_softmax_f32(VkMathContext *ctx,
                            VkCommandBuffer cmd,
                            uint32_t num_rows,
                            uint32_t num_cols,
                            VkBuffer input,
                            VkBuffer output);

/**
 * \brief RMS normalization: out[r][c] = in[r][c] * rsqrt(mean_c(in[r][c]^2) + eps).
 *
 * Input is [num_rows * num_cols], output is [num_rows * num_cols].
 * Each workgroup processes one row.
 *
 * \param eps Epsilon added inside the square root.
 */
VkResult vkmath_rms_norm_f32(VkMathContext *ctx,
                             VkCommandBuffer cmd,
                             uint32_t num_rows,
                             uint32_t num_cols,
                             float eps,
                             VkBuffer input,
                             VkBuffer output);

/**
 * \brief Layer normalization: out[r][c] = (in[r][c] - mean_r) / sqrt(var_r + eps).
 *
 * Input is [num_rows * num_cols], output is [num_rows * num_cols].
 * Each workgroup processes one row.
 *
 * \param eps Epsilon added inside the square root.
 */
VkResult vkmath_layernorm_f32(VkMathContext *ctx,
                              VkCommandBuffer cmd,
                              uint32_t num_rows,
                              uint32_t num_cols,
                              float eps,
                              VkBuffer input,
                              VkBuffer output);

/* ===========================================================================
 * Reductions / normalizations (f16)
 * Input is [num_rows x num_cols] float16_t, output is [num_rows x num_cols]
 * float16_t. Compute runs in f32 internally. Requires shaderFloat16 +
 * storageBuffer16BitAccess + scalarBlockLayout.
 * ========================================================================== */

/**
 * \brief Softmax over each row (f16): same semantics as \ref vkmath_softmax_f32.
 */
VkResult vkmath_softmax_f16(VkMathContext *ctx,
                            VkCommandBuffer cmd,
                            uint32_t num_rows,
                            uint32_t num_cols,
                            VkBuffer input,
                            VkBuffer output);

/**
 * \brief RMS normalization (f16): same semantics as \ref vkmath_rms_norm_f32.
 */
VkResult vkmath_rms_norm_f16(VkMathContext *ctx,
                             VkCommandBuffer cmd,
                             uint32_t num_rows,
                             uint32_t num_cols,
                             float eps,
                             VkBuffer input,
                             VkBuffer output);

/**
 * \brief Layer normalization (f16): same semantics as \ref vkmath_layernorm_f32.
 */
VkResult vkmath_layernorm_f16(VkMathContext *ctx,
                              VkCommandBuffer cmd,
                              uint32_t num_rows,
                              uint32_t num_cols,
                              float eps,
                              VkBuffer input,
                              VkBuffer output);

/* ===========================================================================
 * Index reductions (f32)
 * Input is [num_rows x num_cols] floats; output is [num_rows] uint32_t.
 * Ties resolve to the lowest column index.
 * ========================================================================== */

/**
 * \brief Argmax along rows: output[r] = column index of the maximum value.
 *
 * Input is [num_rows * num_cols] (float), output is [num_rows] (uint32).
 */
VkResult vkmath_argmax_f32(VkMathContext *ctx,
                           VkCommandBuffer cmd,
                           uint32_t num_rows,
                           uint32_t num_cols,
                           VkBuffer input,
                           VkBuffer output);

/**
 * \brief Argmin along rows: output[r] = column index of the minimum value.
 *
 * Input is [num_rows * num_cols] (float), output is [num_rows] (uint32).
 */
VkResult vkmath_argmin_f32(VkMathContext *ctx,
                           VkCommandBuffer cmd,
                           uint32_t num_rows,
                           uint32_t num_cols,
                           VkBuffer input,
                           VkBuffer output);

/* ===========================================================================
 * Prefix sum (f32)
 * Input is [num_rows x num_cols] row-major; output has the same shape.
 * ========================================================================== */

/**
 * \brief Inclusive prefix sum along the column dimension.
 *
 * out[r][c] = sum_{i<=c} in[r][i]. Input is [num_rows * num_cols], output
 * is [num_rows * num_cols]. Each workgroup scans one row (Hillis-Steele
 * in shared memory; rows longer than 256 columns are handled by a stride
 * loop with a running block offset).
 */
VkResult vkmath_cumsum_f32(VkMathContext *ctx,
                           VkCommandBuffer cmd,
                           uint32_t num_rows,
                           uint32_t num_cols,
                           VkBuffer input,
                           VkBuffer output);

/* ===========================================================================
 * Elementwise math (f32)
 * Same signature shape as vkmath_relu_f32. f16 variants are not yet
 * provided for these ops.
 * ========================================================================== */

/**
 * \brief Clip: out = min(max(in, lo), hi)
 */
VkResult vkmath_clip_f32(VkMathContext *ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         float lo,
                         float hi,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief Absolute value: out = |in|
 */
VkResult vkmath_abs_f32(VkMathContext *ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer input,
                        VkBuffer output);

/**
 * \brief Sign: out = 1.0 if in > 0, -1.0 if in < 0, else 0.0
 */
VkResult vkmath_sign_f32(VkMathContext *ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief Natural exponential: out = exp(in)
 */
VkResult vkmath_exp_f32(VkMathContext *ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer input,
                        VkBuffer output);

/**
 * \brief Natural logarithm: out = log(in). Inputs must be positive.
 */
VkResult vkmath_log_f32(VkMathContext *ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        VkBuffer input,
                        VkBuffer output);

/**
 * \brief Square root: out = sqrt(in). Inputs must be non-negative.
 */
VkResult vkmath_sqrt_f32(VkMathContext *ctx,
                         VkCommandBuffer cmd,
                         uint32_t num_elements,
                         VkBuffer input,
                         VkBuffer output);

/**
 * \brief Reciprocal square root: out = rsqrt(in) = 1/sqrt(in).
 */
VkResult vkmath_rsqrt_f32(VkMathContext *ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          VkBuffer input,
                          VkBuffer output);

/**
 * \brief Elementwise power: out = pow(in, exponent).
 *
 * \param exponent Fixed exponent applied to every element.
 */
VkResult vkmath_pow_f32(VkMathContext *ctx,
                        VkCommandBuffer cmd,
                        uint32_t num_elements,
                        float exponent,
                        VkBuffer input,
                        VkBuffer output);

/* ===========================================================================
 * bf16 casts
 * Input/output buffers use bf16 as uint16_t in scalar-block layout (low
 * 16 bits hold the truncated f32 top bits), matching gemm_bf16.comp.
 * Requires storageBuffer16BitAccess + scalarBlockLayout device features.
 * ========================================================================== */

/**
 * \brief Cast f32 to bf16 (truncation): out[i] = top 16 bits of in[i].
 *
 * \param ctx Valid context.
 * \param cmd Command buffer in the recording state.
 * \param num_elements Element count.
 * \param input f32 input buffer.
 * \param output bf16 output buffer (uint16_t elements).
 */
VkResult vkmath_cast_f32_to_bf16(VkMathContext *ctx,
                                 VkCommandBuffer cmd,
                                 uint32_t num_elements,
                                 VkBuffer input,
                                 VkBuffer output);

/**
 * \brief Cast bf16 to f32 (exact): out[i] = float(bits(in[i]) << 16).
 *
 * \param ctx Valid context.
 * \param cmd Command buffer in the recording state.
 * \param num_elements Element count.
 * \param input bf16 input buffer (uint16_t elements).
 * \param output f32 output buffer.
 */
VkResult vkmath_cast_bf16_to_f32(VkMathContext *ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_elements,
                                  VkBuffer input,
                                  VkBuffer output);

/* ===========================================================================
 * bf16 activations
 * Input/output buffers use bf16 as uint16_t in scalar-block layout, same
 * convention as the cast ops. Compute runs in f32, results truncate back to
 * bf16 via floatBitsToUint(r) >> 16. Requires storageBuffer16BitAccess +
 * scalarBlockLayout. Same signature shape as vkmath_relu_f32.
 * ========================================================================== */

VkResult vkmath_relu_bf16(VkMathContext *ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          VkBuffer input,
                          VkBuffer output);

VkResult vkmath_silu_bf16(VkMathContext *ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          VkBuffer input,
                          VkBuffer output);

VkResult vkmath_gelu_bf16(VkMathContext *ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          VkBuffer input,
                          VkBuffer output);

VkResult vkmath_sigmoid_bf16(VkMathContext *ctx,
                             VkCommandBuffer cmd,
                             uint32_t num_elements,
                             VkBuffer input,
                             VkBuffer output);

VkResult vkmath_tanh_bf16(VkMathContext *ctx,
                          VkCommandBuffer cmd,
                          uint32_t num_elements,
                          VkBuffer input,
                          VkBuffer output);

/**
 * \brief Conv2D (NCHW, single batch, no bias in separate buffer).
 *
 * Performs a 2D cross-correlation (convolution with flipped kernel).
 * Input layout:  [C_in, H_in, W_in] (N=1)
 * Weight+bias:   buffer containing [C_out * C_in * kH * kW] weight floats
 *                 followed by [C_out] bias floats.
 * Output layout: [C_out, H_out, W_out]
 *
 * \param ctx Valid context.
 * \param cmd Command buffer in recording state.
 * \param in_w, in_h Input spatial dimensions.
 * \param kw, kh Kernel height/width.
 * \param stride_w, stride_h Stride.
 * \param pad_w, pad_h Padding.
 * \param in_c, out_c Input/output channels (groups=1).
 * \param w_stride Byte stride from binding 1 to weight data (elements, = 0 if
 *                 weights start at offset 0).
 * \param b_stride Byte stride from weight end to bias (elements; if 0, bias
 *                 follows weights immediately: b_offset = w_elements).
 * \param input Input buffer.
 * \param weights Bias+weights buffer.
 * \param output Output buffer.
 */
VkResult vkmath_conv2d_f32(VkMathContext *ctx,
                           VkCommandBuffer cmd,
                           uint32_t in_w, uint32_t in_h,
                           uint32_t kw, uint32_t kh,
                           uint32_t stride_w, uint32_t stride_h,
                           uint32_t pad_w, uint32_t pad_h,
                           uint32_t in_c, uint32_t out_c,
                           uint32_t w_offset, uint32_t b_offset,
                           VkBuffer input,
                           VkBuffer weights,
                           VkBuffer output);

/**
 * \brief 1D convolution (NCHW, batch=1, groups=1, with optional bias).
 *
 * Conv1D is a thin wrapper over the existing conv2d shader with kh=1.
 * Input:  [in_c, in_w]            (binding 0, SSBO float)
 * Weight: [out_c, in_c, 1, kw]    (binding 1, SSBO float)
 * Output: [out_c, in_w]           (binding 2, SSBO float)
 *
 * out_w computed from in_w, kw, stride_w, pad_w.
 *
 * \param in_w     Input width (sequence length).
 * \param kw       Kernel width (receptive field).
 * \param stride_w Stride in width dimension.
 * \param pad_w    Zero-padding on each side of width.
 * \param in_c     Number of input channels.
 * \param out_c    Number of output channels.
 * \param w_offset Offset (in floats) into weights buffer where weights start
 *                 (bias follows weights immediately: b_offset = w_elements).
 * \param b_offset Offset where bias starts.
 * \param input Input buffer.
 * \param weights Bias+weights buffer.
 * \param output Output buffer.
 */
VkResult vkmath_conv1d_f32(VkMathContext *ctx,
                           VkCommandBuffer cmd,
                           uint32_t in_w,
                           uint32_t kw,
                           uint32_t stride_w,
                           uint32_t pad_w,
                           uint32_t in_c, uint32_t out_c,
                           uint32_t w_offset, uint32_t b_offset,
                           VkBuffer input,
                           VkBuffer weights,
                           VkBuffer output);

/**
 * \brief 2D pooling (NCHW, batch=1).
 * Input:  [in_c, H_in, W_in]     (binding 0)
 * Output: [in_c, H_out, W_out]   (binding 1)
 *
 * \param pool_type 0 = max, 1 = average
 */
VkResult vkmath_pool2d_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                           uint32_t in_w, uint32_t in_h,
                           uint32_t kw, uint32_t kh,
                           uint32_t stride_w, uint32_t stride_h,
                           uint32_t pad_w, uint32_t pad_h,
                           uint32_t in_c,
                           uint32_t pool_type,
                           VkBuffer input, VkBuffer output);

/**
 * \brief Batch normalization inference (NCHW, batch=1, per-channel params).
 *
 * out = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * scale/params buffer layout: [gamma[C], beta[C], mean[C], var[C]]
 *
 * \param h, w Spatial dimensions.
 * \param channels Number of channels C.
 * \param eps Epsilon added to variance.
 * \param scale_offset Offset (in floats) into params buffer where gamma starts.
 * \param bias_offset  Offset where beta starts.
 * \param mean_offset  Offset where mean starts.
 * \param var_offset   Offset where var starts.
 * \param input Input buffer [C, H, W].
 * \param params Scale/bias/mean/var buffer.
 * \param output Output buffer [C, H, W].
 */
VkResult vkmath_batchnorm_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                              uint32_t in_w, uint32_t in_h,
                              uint32_t channels, float eps,
                              uint32_t scale_offset, uint32_t bias_offset,
                              uint32_t mean_offset, uint32_t var_offset,
                              VkBuffer input, VkBuffer params, VkBuffer output);

/**
 * \brief 2D tensor transpose: out[j][i] = in[i][j] (row-major both sides).
 * Input:  [rows * cols] (binding 0)
 * Output: [cols * rows] (binding 1)
 */
VkResult vkmath_transpose_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                              uint32_t rows, uint32_t cols,
                              VkBuffer input, VkBuffer output);

/* ===========================================================================
 * Utility
 * ========================================================================== */

/**
 * \brief Get the internal pipeline layout handle for binding external sets.
 *
 * \param context Valid context.
 * \return VkPipelineLayout handle.
 */
VkPipelineLayout vkmath_get_pipeline_layout(VkMathContext* context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKMATH_H */
