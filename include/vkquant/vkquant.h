/**
 * \file vkquant.h
 * \brief Vulkan-native block-quantized tensor dequantization interface.
 *
 * VKQuant dequantizes GGML-style Q4_0 / Q8_0 block-quantized weight tensors
 * to f32 as Vulkan compute dispatches. Every handle is a standard Vulkan
 * object (VkDevice, VkCommandBuffer, VkBuffer); the caller owns buffer memory
 * and synchronization, and VKQuant only records compute work into a
 * caller-supplied VkCommandBuffer.
 *
 * Block formats (authoritative byte layouts, matching GGML):
 *  - Q8_0: 36 bytes/block. bytes 0..3 = f32 scale d, bytes 4..35 = 32 x int8
 *    qs. Dequant: out[i] = d * qs[i] for i in [0, 32).
 *  - Q4_0: 20 bytes/block. bytes 0..3 = f32 scale d, bytes 4..19 = 16 x uint8
 *    packed nibbles. For element i in [0, 32): xi = qs[i>>1],
 *    nibble = (i&1) ? (xi>>4) : (xi&0xF); v = (int)nibble - 8; out[i] = d * v.
 */
#ifndef VKQUANT_H
#define VKQUANT_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Vulkan-native quantized-tensor context.
 *
 * Wraps the VkDevice, a pipeline cache (lazily populated), a descriptor
 * set pool, and the detected GPU architecture. The caller creates this
 * once per VkDevice and reuses it for all dequantization calls. It is
 * thread-unsafe; callers must serialize concurrent access.
 *
 * Internal layout is hidden; treat as opaque.
 */
typedef struct VkQuantContext VkQuantContext;

/* ===========================================================================
 * Context lifecycle
 * ========================================================================== */

/**
 * \brief Create a VkQuantContext bound to a VkDevice.
 *
 * The context lazily creates Vulkan compute pipelines on first dispatch
 * and caches them for subsequent calls. No device memory is allocated by
 * this function; the caller retains full memory ownership.
 *
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the context will bind to.
 * \param pContext        Receives the new context on success.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Device queries or shader load failed.
 */
VkResult vkquant_create_context(VkPhysicalDevice physicalDevice,
                                VkDevice device,
                                VkQuantContext** pContext);

/**
 * \brief Destroy a VkQuantContext and release all cached pipelines / descriptors.
 *
 * \param context Pointer to the context to destroy. May be NULL.
 */
void vkquant_destroy_context(VkQuantContext* context);

/**
 * \brief Query the GPU architecture index the context is bound to.
 *
 * 0 = baseline (vendor-agnostic). Only baseline shaders exist today, so the
 * index is always 0; the field is kept for API consistency with the other
 * stack libraries.
 *
 * \param context Valid context.
 * \return Architecture index.
 */
uint32_t vkquant_get_arch_index(VkQuantContext* context);

/**
 * \brief Get the human-readable GPU architecture string.
 *
 * \param context Valid context.
 * \return e.g. "baseline".
 */
const char* vkquant_get_arch_name(VkQuantContext* context);

/**
 * \brief Detect GPU capabilities and select the optimal shader tier.
 *
 * Called automatically by vkquant_create_context(). Only baseline shaders
 * exist, so the active tier always resolves to baseline regardless of the
 * detected subgroup / cooperative-matrix capabilities.
 *
 * \param context Valid context.
 * \param physicalDevice Physical device to query.
 * \retval VK_SUCCESS
 */
VkResult vkquant_init_capabilities(VkQuantContext* context,
                                   VkPhysicalDevice physicalDevice);

/**
 * \brief Flush cached pipelines associated with the context.
 *
 * Call before vkDeviceWaitIdle if the device will be reset.
 */
void vkquant_flush_pipelines(VkQuantContext* context);

/* ===========================================================================
 * Dequantization ops (block -> f32)
 * ========================================================================== */

/**
 * \brief Dequantize Q8_0 blocks to f32.
 *
 * Input buffer: num_blocks * 36 bytes (f32 scale + 32 x int8 qs per block).
 * Output buffer: num_blocks * 32 floats. out[i] = d * qs[i].
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of Q8_0 blocks to dequantize.
 * \param input      SSBO holding the raw quantized bytes.
 * \param output     SSBO receiving num_blocks * 32 f32 values.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q8_0_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Dequantize Q4_0 blocks to f32.
 *
 * Input buffer: num_blocks * 20 bytes (f32 scale + 16 x uint8 packed nibbles
 * per block). Output buffer: num_blocks * 32 floats.
 * out[i] = d * ((int)nibble - 8).
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of Q4_0 blocks to dequantize.
 * \param input      SSBO holding the raw quantized bytes.
 * \param output     SSBO receiving num_blocks * 32 f32 values.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q4_0_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
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
VkPipelineLayout vkquant_get_pipeline_layout(VkQuantContext* context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKQUANT_H */
