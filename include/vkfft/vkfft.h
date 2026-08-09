/**
 * \file vkfft.h
 * \brief Vulkan-native forward complex FFT interface.
 *
 * VKFFT provides a forward radix-2 complex FFT (f32) as a Vulkan compute
 * dispatch. Each VkFFTPlan is bound to a VkDevice and caches the compute
 * pipeline it needs (created lazily on first execute). Buffer data is
 * interleaved (Re, Im) float pairs: element j lives at in[2j], in[2j+1].
 *
 * The caller owns buffer memory, descriptor bindings, and synchronization;
 * VKFFT only records compute work into a caller-supplied VkCommandBuffer.
 */
#ifndef VKFFT_H
#define VKFFT_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Errors
 * ========================================================================== */

/**
 * \brief Invalid argument to vkfft_create_plan().
 *
 * VK_ERROR_INVALID_ARGUMENT is not part of the core Vulkan VkResult enum.
 * VKFFT defines it as a distinct negative VkResult in the unallocated gap
 * between the core error range (-1..-13) and the extension range (<= -1000000)
 * so callers can branch on it unambiguously.
 */
#define VKFFT_ERROR_INVALID_ARGUMENT ((VkResult)-10001)

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Vulkan-native FFT plan.
 *
 * Owns the VkDevice, a lazily-populated pipeline cache, a descriptor set
 * layout, a pipeline layout, an optional descriptor pool (push-descriptor
 * fallback), and the FFT size n. Mirrors rocfft plan semantics: the plan is
 * the context. The caller creates one plan per FFT size and reuses it. It is
 * thread-unsafe; callers must serialize concurrent access.
 *
 * Internal layout is hidden; treat as opaque.
 */
typedef struct VkFFTPlan VkFFTPlan;

/* ===========================================================================
 * Plan lifecycle
 * ========================================================================== */

/**
 * \brief Create an FFT plan for a forward radix-2 complex FFT of size n.
 *
 * n must be a power of two with 2 <= n <= 256. The plan lazily creates its
 * Vulkan compute pipeline on first vkfft_execute_f32 and caches it for
 * subsequent calls. No device memory is allocated by this function; the caller
 * retains full memory ownership.
 *
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the plan will bind to.
 * \param n               FFT size (power of two, 2..256).
 * \param pPlan           Receives the plan handle on success.
 * \retval VK_SUCCESS On success.
 * \retval VKFFT_ERROR_INVALID_ARGUMENT n is not a power of two in [2, 256].
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Device queries or shader load failed.
 */
VkResult vkfft_create_plan(VkPhysicalDevice physicalDevice,
                           VkDevice device,
                           uint32_t n,
                           VkFFTPlan** pPlan);

/**
 * \brief Destroy a VkFFTPlan and release all cached pipelines / descriptors.
 *
 * \param plan Pointer to the plan to destroy. May be NULL.
 */
void vkfft_destroy_plan(VkFFTPlan* plan);

/**
 * \brief Record a forward f32 FFT dispatch into a command buffer.
 *
 * Reads 2*n interleaved floats from input, computes the forward transform
 * X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n), and writes 2*n interleaved floats to
 * output. Records one workgroup of n active threads (256-thread workgroup,
 * idle threads above n). The command buffer must already be in the recording
 * state; the caller must order memory barriers as required.
 *
 * \param plan   Valid plan (created for the same n as the buffers hold).
 * \param cmd    Command buffer in the recording state.
 * \param input  Storage buffer of 2*n floats (interleaved Re/Im).
 * \param output Storage buffer of 2*n floats (interleaved Re/Im).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_INITIALIZATION_FAILED plan is NULL.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob available for the tier.
 */
VkResult vkfft_execute_f32(VkFFTPlan* plan,
                           VkCommandBuffer cmd,
                           VkBuffer input,
                           VkBuffer output);

/* ===========================================================================
 * Queries
 * ========================================================================== */

/**
 * \brief Return the FFT size n this plan was created with.
 *
 * \param plan Valid plan.
 * \return The plan's FFT size n.
 */
uint32_t vkfft_get_size(VkFFTPlan* plan);

/**
 * \brief Get the human-readable GPU architecture string.
 *
 * \param plan Valid plan.
 * \return e.g. "baseline", "subgroup", or "coopmatrix".
 */
const char* vkfft_get_arch_name(VkFFTPlan* plan);

/**
 * \brief Query the GPU architecture index the plan is bound to.
 *
 * 0 = baseline (vendor-agnostic), 1 = subgroup, 2 = coopmatrix.
 *
 * \param plan Valid plan.
 * \return Architecture index.
 */
uint32_t vkfft_get_arch_index(VkFFTPlan* plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKFFT_H */
