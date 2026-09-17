/**
 * \file vkquant.h
 * \brief Vulkan-native block-quantized tensor dequantization interface.
 *
 * VKQuant dequantizes every GGML block-quantized weight format (legacy
 * Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/IQ4_NL 32-elem blocks; Q2_K/Q3_K/Q4_K/Q5_K/
 * Q6_K K-quant super-blocks; IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/
 * IQ4_XS and TQ1_0/TQ2_0 super-blocks) to f32 and forward-quantizes f32 to
 * Q8_0/Q4_0 (f32-scale) and Q4_1/Q5_0/Q5_1/Q8_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
 * (ggml scale-selection math), all as Vulkan compute dispatches.
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
 *  - All remaining formats (Q4_1/Q5_0/Q5_1/Q8_1/IQ4_NL 32-elem; Q2_K/Q3_K/
 *    Q5_K K-quants; IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S; TQ1_0/
 *    TQ2_0) use the exact ggml-common.h byte layouts and dequant rules (see
 *    the individual op docs below and src/vkquant/AGENTS.md).
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

/**
 * \brief Dequantize Q4_1 blocks to f32 (ggml block_q4_1, 20 B/block, 32 elems).
 *
 * d (f16) @0, m (f16) @2, qs[16] @4. out = d * nibble + m (nibble 0..15).
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q4_1_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Dequantize Q5_0 blocks to f32 (ggml block_q5_0, 22 B/block, 32 elems).
 *
 * d (f16) @0, qh[4] @2, qs[16] @6. out = d * ((nibble | xh) - 16).
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q5_0_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Dequantize Q5_1 blocks to f32 (ggml block_q5_1, 24 B/block, 32 elems).
 *
 * d (f16) @0, m (f16) @2, qh[4] @4, qs[16] @8. out = d * (nibble | xh) + m.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q5_1_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Dequantize Q8_1 blocks to f32 (ggml block_q8_1, 36 B/block, 32 elems).
 *
 * d (f16) @0, s (f16) @2, qs[32] int8 @4. out = d * qs[i]. The s field
 * (d * sum(qs)) is stored for layout fidelity but unused in dequantization.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q8_1_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Dequantize Q2_K super-blocks to f32 (ggml block_q2_K, 84 B, 256 elems).
 *
 * scales[16] @0, qs[64] @16, d (f16) @80, dmin (f16) @82.
 * out = d*(sc&0xF)*level - dmin*(sc>>4)*level, level 0..3.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q2k_f32(VkQuantContext* ctx,
                                 VkCommandBuffer cmd,
                                 uint32_t num_blocks,
                                 VkBuffer input,
                                 VkBuffer output);

/**
 * \brief Dequantize Q3_K super-blocks to f32 (ggml block_q3_K, 110 B, 256 elems).
 *
 * hmask[32] @0, qs[64] @32, scales[12] @96, d (f16) @108.
 * 16 scales decoded from 12 bytes (6 bits each); out = d*(sc-32)*level.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q3k_f32(VkQuantContext* ctx,
                                 VkCommandBuffer cmd,
                                 uint32_t num_blocks,
                                 VkBuffer input,
                                 VkBuffer output);

/**
 * \brief Dequantize Q5_K super-blocks to f32 (ggml block_q5_K, 176 B, 256 elems).
 *
 * d (f16) @0, dmin (f16) @2, scales[12] @4, qh[32] @16, qs[128] @48.
 * out = d*sc*(nib + qh_bit*16) - dmin*mn (get_scale_min_k4 scale decode).
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_q5k_f32(VkQuantContext* ctx,
                                 VkCommandBuffer cmd,
                                 uint32_t num_blocks,
                                 VkBuffer input,
                                 VkBuffer output);

/**
 * \brief Dequantize IQ4_NL blocks to f32 (ggml block_iq4_nl, 18 B, 32 elems).
 *
 * d (f16) @0, qs[16] @2. out = d * kvalues_iq4nl[nibble].
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq4_nl_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Dequantize IQ1_S super-blocks to f32 (ggml block_iq1_s, 50 B, 256 elems).
 *
 * d (f16) @0, qs[32] @2, qh[16 u16] @34. Uses the iq1s grid + IQ1S_DELTA.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq1_s_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Dequantize IQ1_M super-blocks to f32 (ggml block_iq1_m, 56 B, 256 elems).
 *
 * qs[32] @0, qh[16] @32, scales[8] @48 (block scale assembled from 12 bits).
 * Uses the iq1s grid + IQ1S_DELTA.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq1_m_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Dequantize IQ2_XS super-blocks to f32 (ggml block_iq2_xs, 74 B, 256 elems).
 *
 * d (f16) @0, qs[32 u16] @2, scales[8] @66. Uses iq2xs_grid + ksigns_iq2xs.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq2_xs_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Dequantize IQ2_S super-blocks to f32 (ggml block_iq2_s, 82 B, 256 elems).
 *
 * d (f16) @0, qs[64] @2, qh[8] @66, scales[8] @74. Uses iq2s_grid.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq2_s_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Dequantize IQ2_XXS super-blocks to f32 (ggml block_iq2_xxs, 66 B, 256 elems).
 *
 * d (f16) @0, qs[32 u16] @2. Uses iq2xxs_grid + ksigns_iq2xs.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq2_xxs_f32(VkQuantContext* ctx,
                                     VkCommandBuffer cmd,
                                     uint32_t num_blocks,
                                     VkBuffer input,
                                     VkBuffer output);

/**
 * \brief Dequantize IQ3_S super-blocks to f32 (ggml block_iq3_s, 110 B, 256 elems).
 *
 * d (f16) @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106.
 * Uses iq3s_grid + ksigns_iq2xs.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq3_s_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Dequantize IQ3_XXS super-blocks to f32 (ggml block_iq3_xxs, 98 B, 256 elems).
 *
 * d (f16) @0, qs[96] @2. Uses iq3xxs_grid + ksigns_iq2xs.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_iq3_xxs_f32(VkQuantContext* ctx,
                                     VkCommandBuffer cmd,
                                     uint32_t num_blocks,
                                     VkBuffer input,
                                     VkBuffer output);

/**
 * \brief Dequantize TQ1_0 super-blocks to f32 (ggml block_tq1_0, 54 B, 256 elems).
 *
 * qs[48] @0, qh[4] @48, d (f16) @52. Ternary 5-in-1 grouping.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_tq1_0_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Dequantize TQ2_0 super-blocks to f32 (ggml block_tq2_0, 66 B, 256 elems).
 *
 * qs[64] @0, d (f16) @64. Ternary 2-bits-per-element.
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_dequant_tq2_0_f32(VkQuantContext* ctx,
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

/**
 * \brief Quantize f32 data to Q4_1 blocks (ggml block_q4_1, 20 B/block).
 *
 * d = (max-min)/15, m = min, level = round((x-min)/d) clamped [0,15].
 * Round-trips through vkquant_dequant_q4_1_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q4_1_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Quantize f32 data to Q5_0 blocks (ggml block_q5_0, 22 B/block).
 *
 * d = max/-16 (signed max-|x| element), xi clamped [0,31]; 5th bit to qh.
 * Round-trips through vkquant_dequant_q5_0_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q5_0_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Quantize f32 data to Q5_1 blocks (ggml block_q5_1, 24 B/block).
 *
 * d = (max-min)/31, m = min, xi = (uint8)((x-min)/d + 0.5); 5th bit to qh.
 * Round-trips through vkquant_dequant_q5_1_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q5_1_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Quantize f32 data to Q8_1 blocks (ggml block_q8_1, 36 B/block).
 *
 * d = max(|x|)/127, q = round(x/d), s = d*sum(qs) stored as f16.
 * Round-trips through vkquant_dequant_q8_1_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q8_1_f32(VkQuantContext* ctx,
                                   VkCommandBuffer cmd,
                                   uint32_t num_blocks,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Quantize f32 data to Q2_K blocks (ggml block_q2_K, 84 B/block, 256 elems).
 *
 * make_qkx1 per-16-group scale/min + 4-bit block-relative scale/min encoding.
 * Round-trips through vkquant_dequant_q2k_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q2k_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Quantize f32 data to Q3_K blocks (ggml block_q3_K, 110 B/block, 256 elems).
 *
 * make_q3-style per-16-group signed scale + 12-byte 6-bit scale encoding.
 * Round-trips through vkquant_dequant_q3k_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q3k_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Quantize f32 data to Q4_K blocks (ggml block_q4_K, 144 B/block, 256 elems).
 *
 * make_qkx1 per-32-group scale/min + get_scale_min_k4-packed 12-byte encoding.
 * Round-trips through vkquant_dequant_q4k_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q4k_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Quantize f32 data to Q5_K blocks (ggml block_q5_K, 176 B/block, 256 elems).
 *
 * make_qkx1 per-32-group scale/min (nmax=31) + qh bit plane.
 * Round-trips through vkquant_dequant_q5k_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q5k_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Quantize f32 data to Q6_K blocks (ggml block_q6_K, 210 B/block, 256 elems).
 *
 * make_qx-style per-16-group signed scale + int8 block-relative scales.
 * Round-trips through vkquant_dequant_q6k_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_q6k_f32(VkQuantContext* ctx,
                                  VkCommandBuffer cmd,
                                  uint32_t num_blocks,
                                  VkBuffer input,
                                  VkBuffer output);

/**
 * \brief Quantize f32 data to IQ1_S blocks (ggml block_iq1_s, 50 B/block, 256 elems).
 *
 * Direct grid search over the iq1s grid (+ IQ1S_DELTA), replacing ggml's
 * kmap/neighbours optimisation with an exact exhaustive codebook search.
 * Round-trips through vkquant_dequant_iq1_s_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq1_s_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Quantize f32 data to IQ1_M blocks (ggml block_iq1_m, 56 B/block, 256 elems).
 *
 * Direct grid search over the iq1s grid; 3-bit per-16-block scales + shift bits.
 * Round-trips through vkquant_dequant_iq1_m_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq1_m_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Quantize f32 data to IQ2_XXS blocks (ggml block_iq2_xxs, 66 B/block, 256 elems).
 *
 * Direct grid search over the iq2xxs grid; 8-bit grid indices + 7-bit signs.
 * Round-trips through vkquant_dequant_iq2_xxs_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq2_xxs_f32(VkQuantContext* ctx,
                                      VkCommandBuffer cmd,
                                      uint32_t num_blocks,
                                      VkBuffer input,
                                      VkBuffer output);

/**
 * \brief Quantize f32 data to IQ2_XS blocks (ggml block_iq2_xs, 74 B/block, 256 elems).
 *
 * Direct grid search over the iq2xs grid; 9-bit grid indices + 7-bit signs.
 * Round-trips through vkquant_dequant_iq2_xs_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq2_xs_f32(VkQuantContext* ctx,
                                     VkCommandBuffer cmd,
                                     uint32_t num_blocks,
                                     VkBuffer input,
                                     VkBuffer output);

/**
 * \brief Quantize f32 data to IQ2_S blocks (ggml block_iq2_s, 82 B/block, 256 elems).
 *
 * Direct grid search over the iq2s grid; 10-bit grid indices + full 8-bit signs.
 * Round-trips through vkquant_dequant_iq2_s_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq2_s_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Quantize f32 data to IQ3_XXS blocks (ggml block_iq3_xxs, 98 B/block, 256 elems).
 *
 * Direct grid search over the iq3xxs grid; 8-bit grid indices + 7-bit signs.
 * Round-trips through vkquant_dequant_iq3_xxs_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq3_xxs_f32(VkQuantContext* ctx,
                                      VkCommandBuffer cmd,
                                      uint32_t num_blocks,
                                      VkBuffer input,
                                      VkBuffer output);

/**
 * \brief Quantize f32 data to IQ3_S blocks (ggml block_iq3_s, 110 B/block, 256 elems).
 *
 * Direct grid search over the iq3s grid; 9-bit grid indices + full 8-bit signs.
 * Round-trips through vkquant_dequant_iq3_s_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq3_s_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Quantize f32 data to IQ4_NL blocks (ggml block_iq4_nl, 18 B/block, 32 elems).
 *
 * kvalues_iq4nl best-index search (ggml quantize_row_iq4_nl_ref, ntry=-1 path).
 * Round-trips through vkquant_dequant_iq4_nl_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq4_nl_f32(VkQuantContext* ctx,
                                     VkCommandBuffer cmd,
                                     uint32_t num_blocks,
                                     VkBuffer input,
                                     VkBuffer output);

/**
 * \brief Quantize f32 data to IQ4_XS blocks (ggml block_iq4_xs, 136 B/block, 256 elems).
 *
 * kvalues_iq4nl best-index search + per-32-group scale (ggml
 * quantize_row_iq4_xs_ref, ntry=7 path). Round-trips through
 * vkquant_dequant_iq4xs_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_iq4xs_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Quantize f32 data to TQ1_0 blocks (ggml block_tq1_0, 54 B/block, 256 elems).
 *
 * Ternary 5-in-1 byte grouping (ggml quantize_row_tq1_0_ref).
 * Round-trips through vkquant_dequant_tq1_0_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_tq1_0_f32(VkQuantContext* ctx,
                                    VkCommandBuffer cmd,
                                    uint32_t num_blocks,
                                    VkBuffer input,
                                    VkBuffer output);

/**
 * \brief Quantize f32 data to TQ2_0 blocks (ggml block_tq2_0, 66 B/block, 256 elems).
 *
 * Ternary 2-bits-per-element (ggml quantize_row_tq2_0_ref).
 * Round-trips through vkquant_dequant_tq2_0_f32().
 *
 * \retval VK_SUCCESS On success (recorded into cmd).
 */
VkResult vkquant_quantize_tq2_0_f32(VkQuantContext* ctx,
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
