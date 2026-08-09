/**
 * \file vkrand.h
 * \brief Vulkan-native counter-based PRNG / uniform distribution interface.
 *
 * VKRAND provides stateless, deterministic random number generation as
 * Vulkan compute dispatches. The first op is Philox4x32-10 (10-round
 * counter-based PRNG) mapped to uniform f32 in [0,1).
 *
 * The generator is stateless: the counter is derived from the global
 * thread index and the caller-supplied seed, so the same (seed, count)
 * pair always produces identical output. No internal state is kept
 * between calls.
 *
 * The caller owns buffer memory and synchronization; VKRAND only records
 * compute work into a caller-supplied VkCommandBuffer.
 */
#ifndef VKRAND_H
#define VKRAND_H

#include <vulkan/vulkan.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Vulkan-native random number context.
 *
 * Wraps the VkDevice, a pipeline cache (lazily populated), a descriptor
 * set pool, and the detected GPU architecture. The caller creates this
 * once per VkDevice and reuses it for all random number calls. It is
 * thread-unsafe; callers must serialize concurrent access.
 *
 * Internal layout is hidden; treat as opaque.
 */
typedef struct VkRandContext VkRandContext;

/* ===========================================================================
 * Context lifecycle
 * ========================================================================== */

/**
 * \brief Create a VkRandContext bound to a VkDevice.
 *
 * The context lazily creates Vulkan compute pipelines on first dispatch
 * and caches them for subsequent calls. No device memory is allocated
 * by this function; the caller retains full memory ownership.
 *
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the context will bind to.
 * \param pContext        Receives the created context (or NULL on failure).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Device queries or shader load failed.
 */
VkResult vkrand_create_context(VkPhysicalDevice physicalDevice,
                               VkDevice device,
                               VkRandContext** pContext);

/**
 * \brief Destroy a VkRandContext and release all cached pipelines / descriptors.
 *
 * \param context Pointer to the context to destroy. May be NULL.
 */
void vkrand_destroy_context(VkRandContext* context);

/**
 * \brief Query the GPU architecture index the context is bound to.
 *
 * 0 = baseline (vendor-agnostic). Only a baseline shader exists today.
 *
 * \param context Valid context.
 * \return Architecture index.
 */
uint32_t vkrand_get_arch_index(VkRandContext* context);

/**
 * \brief Get the human-readable GPU architecture string.
 *
 * \param context Valid context.
 * \return e.g. "baseline".
 */
const char* vkrand_get_arch_name(VkRandContext* context);

/**
 * \brief Flush cached pipelines associated with the context.
 *
 * Call before vkDeviceWaitIdle if the device will be reset.
 */
void vkrand_flush_pipelines(VkRandContext* context);

/* ===========================================================================
 * Ops
 * ========================================================================== */

/**
 * \brief Fill a buffer with count uniform f32 values in [0,1).
 *
 * Stateless Philox4x32-10 counter-based PRNG. Thread i derives its counter
 * from the global invocation index and the seed, and writes
 * float(philox_out & 0xFFFFFF) / 16777216.0f to output[i]. Deterministic:
 * the same (seed, count) always yields identical output.
 *
 * The output buffer must be large enough for count floats (4 * count bytes)
 * and must be bound as a storage buffer.
 *
 * \param ctx    Valid context.
 * \param cmd    Command buffer in the recording state.
 * \param seed   Deterministic seed for the generator.
 * \param count  Number of f32 values to generate.
 * \param output Storage buffer receiving count floats.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob for the requested op.
 * \retval Other pipeline creation / dispatch failures.
 */
VkResult vkrand_uniform_f32(VkRandContext* ctx,
                            VkCommandBuffer cmd,
                            uint32_t seed,
                            uint32_t count,
                            VkBuffer output);

/**
 * \brief Fill a buffer with count uniform f32 values in [0,1) using the
 *        ThreeFry2x32-20 counter-based PRNG.
 *
 * Stateless ThreeFry2x32-20 (Random123) with the same thread/seed counter
 * derivation as the Philox op: X0 = global invocation index, X1 = seed hash,
 * key = (seed, 0x9E3779B9 ^ seed). 20 rounds using the Threefish rotation
 * schedule {13, 15, 26, 6, 17, 29, 16, 24} with key injection every 4
 * rounds. Maps float(X0 & 0xFFFFFF) / 16777216.0f into [0,1). Validated
 * against the Random123 threefry2x32x20 known-answer vectors. Deterministic:
 * the same (seed, count) always yields identical output.
 *
 * The output buffer must be large enough for count floats (4 * count bytes)
 * and must be bound as a storage buffer.
 *
 * \param ctx    Valid context.
 * \param cmd    Command buffer in the recording state.
 * \param seed   Deterministic seed for the generator.
 * \param count  Number of f32 values to generate.
 * \param output Storage buffer receiving count floats.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob for the requested op.
 * \retval Other pipeline creation / dispatch failures.
 */
VkResult vkrand_threefry_uniform_f32(VkRandContext* ctx,
                                     VkCommandBuffer cmd,
                                     uint32_t seed,
                                     uint32_t count,
                                     VkBuffer output);

/**
 * \brief Fill a buffer with count standard-normal N(0,1) f32 values.
 *
 * Philox4x32-10 based. Thread i draws two consecutive uniform values from
 * counters 2*i and 2*i+1 and applies the Box-Muller transform
 * z = sqrt(-2 ln u0) * cos(2 pi u1). u0 is clamped away from 0 so every
 * output is finite. Deterministic: the same (seed, count) always yields
 * identical output.
 *
 * The output buffer must be large enough for count floats (4 * count bytes)
 * and must be bound as a storage buffer.
 *
 * \param ctx    Valid context.
 * \param cmd    Command buffer in the recording state.
 * \param seed   Deterministic seed for the generator.
 * \param count  Number of f32 values to generate.
 * \param output Storage buffer receiving count floats.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob for the requested op.
 * \retval Other pipeline creation / dispatch failures.
 */
VkResult vkrand_normal_f32(VkRandContext* ctx,
                           VkCommandBuffer cmd,
                           uint32_t seed,
                           uint32_t count,
                           VkBuffer output);

/**
 * \brief Fill a buffer with count raw uint32 values uniform in [0, 2^32).
 *
 * Stateless Philox4x32-10 with the same counter/key derivation as
 * vkrand_uniform_f32, but writes the full 32-bit c0 counter word directly
 * (no float mapping). Mirrors rocrand_generate(). Deterministic: the same
 * (seed, count) always yields identical output.
 *
 * The output buffer must be large enough for count uint32s (4 * count bytes)
 * and must be bound as a storage buffer.
 *
 * \param ctx    Valid context.
 * \param cmd    Command buffer in the recording state.
 * \param seed   Deterministic seed for the generator.
 * \param count  Number of uint32 values to generate.
 * \param output Storage buffer receiving count uint32 values.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob for the requested op.
 * \retval Other pipeline creation / dispatch failures.
 */
VkResult vkrand_uniform_uint32(VkRandContext* ctx,
                               VkCommandBuffer cmd,
                               uint32_t seed,
                               uint32_t count,
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
VkPipelineLayout vkrand_get_pipeline_layout(VkRandContext* context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKRAND_H */
