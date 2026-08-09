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
 * Reductions (f32)
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
 * \brief Sum reduction along rows: for each row of num_cols elements,
 *        produce the sum. Input is [num_rows * num_cols], output is [num_rows].
 */
VkResult vkmath_sum_reduce_dim_f32(VkMathContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_rows,
                                   uint32_t num_cols,
                                   VkBuffer input,
                                   VkBuffer output);

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
