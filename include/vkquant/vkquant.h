/**
 * \file vkquant.h
 * \brief Vulkan-native block-quantized tensor dequantization interface.
 *
 * VKQuant dequantizes GGML-style Q4_0 / Q8_0 block-quantized weight tensors
 * (plus Q4_K / Q6_K / IQ4_XS super-block formats) to f32 and forward-quantizes
 * f32 to the f32-scale Q8_0 / Q4_0 formats, all as Vulkan compute dispatches.
 * Every handle is a standard Vulkan
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
 *  - Q4_K: 144 bytes/block, 256 elements. ggml block_q4_K (f16 d,dmin +
 *    12 scale bytes + 128 nibble bytes); dequant matches ggml
 *    dequantize_row_q4_K (get_scale_min_k4).
 *  - Q6_K: 210 bytes/block, 256 elements. ggml block_q6_K (ql[128] + qh[64] +
 *    16 int8 scales + f16 d); dequant matches ggml dequantize_row_q6_K.
 *  - IQ4_XS: 136 bytes/block, 256 elements. ggml block_iq4_xs (f16 d + u16
 *    scales_h + scales_l[4] + 128 nibble bytes); dequant matches ggml
 *    dequantize_row_iq4_xs using the kvalues_iq4nl lookup table.
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

/**
 * \brief Dequantize Q4_K super-blocks to f32.
 *
 * Input buffer: num_blocks * 144 bytes (ggml block_q4_K: f16 d + f16 dmin +
 * 12 scale bytes + 128 packed-nibble qs). Output: num_blocks * 256 floats.
 * Bit-exact with ggml dequantize_row_q4_K / get_scale_min_k4.
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of Q4_K blocks to dequantize.
 * \param input      SSBO holding the raw quantized bytes.
 * \param output     SSBO receiving num_blocks * 256 f32 values.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q4k_f32(VkQuantContext* ctx,
                                 VkCommandBuffer cmd,
                                 uint32_t num_blocks,
                                 VkBuffer input,
                                 VkBuffer output);

/**
 * \brief Dequantize Q6_K super-blocks to f32.
 *
 * Input buffer: num_blocks * 210 bytes (ggml block_q6_K: ql[128] + qh[64] +
 * 16 int8 scales + f16 d). Output: num_blocks * 256 floats.
 * Bit-exact with ggml dequantize_row_q6_K.
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of Q6_K blocks to dequantize.
 * \param input      SSBO holding the raw quantized bytes.
 * \param output     SSBO receiving num_blocks * 256 f32 values.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q6k_f32(VkQuantContext* ctx,
                                 VkCommandBuffer cmd,
                                 uint32_t num_blocks,
                                 VkBuffer input,
                                 VkBuffer output);

/**
 * \brief Dequantize IQ4_XS super-blocks to f32.
 *
 * Input buffer: num_blocks * 136 bytes (ggml block_iq4_xs: f16 d + u16
 * scales_h + scales_l[4] + 128 packed-nibble qs). Output: num_blocks * 256
 * floats. Bit-exact with ggml dequantize_row_iq4_xs (kvalues_iq4nl table).
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of IQ4_XS blocks to dequantize.
 * \param input      SSBO holding the raw quantized bytes.
 * \param output     SSBO receiving num_blocks * 256 f32 values.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq4xs_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/* ===========================================================================
 * Forward quantization ops (f32 -> block)
 * ========================================================================== */

/**
 * \brief Quantize f32 data to Q8_0 blocks (f32-scale variant).
 *
 * Input buffer: num_blocks * 32 floats. Output: num_blocks * 36 bytes
 * (f32 scale d + 32 x int8 qs). d = max(|x|)/127; q = round(x/d) clamped to
 * [-127,127]. Round-trips through vkquant_dequant_q8_0_f32().
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of 32-element blocks to quantize.
 * \param input      SSBO holding the f32 source data.
 * \param output     SSBO receiving num_blocks * 36 quantized bytes.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q8_0_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Quantize f32 data to Q4_0 blocks (f32-scale variant).
 *
 * Input buffer: num_blocks * 32 floats. Output: num_blocks * 20 bytes
 * (f32 scale d + 16 x uint8 packed nibbles). d = max(|x|)/8;
 * q = round(x/d) + 8 clamped to [0,15]; low nibble first.
 * Round-trips through vkquant_dequant_q4_0_f32().
 *
 * \param ctx        Valid context.
 * \param cmd        Command buffer in the recording state.
 * \param num_blocks Number of 32-element blocks to quantize.
 * \param input      SSBO holding the f32 source data.
 * \param output     SSBO receiving num_blocks * 20 quantized bytes.
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q4_0_f32(VkQuantContext* ctx,
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
