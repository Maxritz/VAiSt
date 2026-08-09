/**
 * \file vkfft.h
 * \brief Vulkan-native complex FFT interface (forward + inverse, f32 + f16).
 *
 * VKFFT provides a radix-2 complex FFT (f32 or f16 I/O, forward or inverse)
 * as a Vulkan compute dispatch. Each VkFFTPlan is bound to a VkDevice and
 * caches the compute pipelines it needs (created lazily on first execute).
 * The direction is selected per-execute; the inverse is unnormalized (rocfft
 * default). Buffer data is interleaved (Re, Im) pairs: element j lives at
 * in[2j], in[2j+1].
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
 * \brief Transform direction for a VKFFT plan.
 *
 * Mirrors rocfft_transform_type (complex_forward / complex_inverse). The
 * direction is passed per-execute via the vkfft_execute_* family; the plan
 * itself is direction-agnostic. The inverse is unnormalized, matching the
 * rocfft default, so forward-then-inverse recovers the original signal scaled
 * by n.
 */
typedef enum VkFFTDirection {
    VKFFT_DIR_FORWARD = 0, /**< X[k] = sum_j x[j] exp(-2*pi*i*j*k/n). */
    VKFFT_DIR_INVERSE = 1, /**< y[m] = sum_k X[k] exp(+2*pi*i*k*m/n). */
} VkFFTDirection_t;

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
 * \brief Create an FFT plan for a radix-2 complex FFT of size n.
 *
 * n must be a power of two with 2 <= n <= 1024. The plan is direction- and
 * precision-agnostic: the same plan drives forward/inverse and f32/f16 via the
 * vkfft_execute_* functions. It lazily creates its Vulkan compute pipelines on
 * first execute and caches them for subsequent calls. No device memory is
 * allocated by this function; the caller retains full memory ownership.
 *
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the plan will bind to.
 * \param n               FFT size (power of two, 2..1024).
 * \param pPlan           Receives the plan handle on success.
 * \retval VK_SUCCESS On success.
 * \retval VKFFT_ERROR_INVALID_ARGUMENT n is not a power of two in [2, 1024].
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
 * output. Records one 256-thread workgroup (n active elements, strided loop
 * over the FFT). The command buffer must already be in the recording state;
 * the caller must order memory barriers as required.
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

/**
 * \brief Record an inverse f32 FFT dispatch into a command buffer.
 *
 * Reads 2*n interleaved floats from input, computes the inverse transform
 * y[m] = sum_k X[k] * exp(+2*pi*i*k*m/n), and writes 2*n interleaved floats to
 * output. The inverse is unnormalized (matching the rocfft default): a
 * forward-then-inverse round trip returns the original signal scaled by n.
 * Same workgroup/dispatch shape as vkfft_execute_f32.
 *
 * \param plan   Valid plan (created for the same n as the buffers hold).
 * \param cmd    Command buffer in the recording state.
 * \param input  Storage buffer of 2*n floats (interleaved Re/Im).
 * \param output Storage buffer of 2*n floats (interleaved Re/Im).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_INITIALIZATION_FAILED plan is NULL.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob available for the tier.
 */
VkResult vkfft_execute_inverse_f32(VkFFTPlan* plan,
                                   VkCommandBuffer cmd,
                                   VkBuffer input,
                                   VkBuffer output);

/**
 * \brief Record a forward f16 FFT dispatch into a command buffer.
 *
 * Same transform and dispatch shape as vkfft_execute_f32, but input/output are
 * interleaved float16_t (Re, Im) pairs: element j lives at in[2j], in[2j+1].
 * Each buffer holds 2*n float16_t values (n*4 bytes). Internal compute and
 * shared memory are f32; only the I/O is f16, so results match the f32 path to
 * within f16 rounding.
 *
 * \param plan   Valid plan (created for the same n as the buffers hold).
 * \param cmd    Command buffer in the recording state.
 * \param input  Storage buffer of 2*n float16_t values (interleaved Re/Im).
 * \param output Storage buffer of 2*n float16_t values (interleaved Re/Im).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_INITIALIZATION_FAILED plan is NULL.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob available for the tier.
 */
VkResult vkfft_execute_f16(VkFFTPlan* plan,
                           VkCommandBuffer cmd,
                           VkBuffer input,
                           VkBuffer output);

/**
 * \brief Record an inverse f16 FFT dispatch into a command buffer.
 *
 * Inverse of vkfft_execute_f16, with the same unnormalized semantics as
 * vkfft_execute_inverse_f32: forward-then-inverse returns the original signal
 * scaled by n.
 *
 * \param plan   Valid plan (created for the same n as the buffers hold).
 * \param cmd    Command buffer in the recording state.
 * \param input  Storage buffer of 2*n float16_t values (interleaved Re/Im).
 * \param output Storage buffer of 2*n float16_t values (interleaved Re/Im).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_INITIALIZATION_FAILED plan is NULL.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No shader blob available for the tier.
 */
VkResult vkfft_execute_inverse_f16(VkFFTPlan* plan,
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
