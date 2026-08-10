/**
 * \file vkruntime.h
 * \brief Vulkan-native runtime foundation (hipRuntime-equivalent base layer).
 *
 * VKRuntime is the bottom of the Vulkan AI stack. Every higher library
 * (vkmath, vkblas, vkquant, vkrand, vkfft) sits on it: it owns the
 * device/context abstraction, capability detection, a pooled buffer
 * allocator (hipMalloc/hipFree equivalent), staging upload/download, and
 * command/descriptor pool helpers.
 *
 * Every handle in this library is a native Vulkan object (\c VkDevice,
 * \c VkBuffer, \c VkDeviceMemory, ...). There is no ROCm / HIP / CUDA
 * dependency anywhere.
 *
 * Public API names mirror the ROCm \c hip*_api.h surface (create/destroy,
 * malloc/free, memcpy-as-upload/download, wait-idle) so porting existing
 * model code between backends is mechanical, not conceptual.
 */
#ifndef VKRUNTIME_H
#define VKRUNTIME_H

#include <vulkan/vulkan.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Vulkan-native runtime context.
 *
 * Binds one physical/logical device pair and a compute (or transfer) queue,
 * caches the device capability set, and owns the pooled memory allocator.
 * Create once per VkDevice; reuse for the device's whole lifetime.
 *
 * Internal layout is hidden; treat as opaque.
 */
typedef struct VkRuntime VkRuntime;

/**
 * \brief Device capability set shared by every library's context.
 *
 * Filled by vkr_detect_capabilities(). This is the single de-duplicated
 * description of what the five per-library contexts (vkmath, vkblas, vkquant,
 * vkrand, vkfft) used to detect inline: the core 1.1-1.4 feature set, the
 * subgroup / cooperative-matrix features, AMD shader-core geometry, push
 * descriptors, and the derived shader-tier architecture index/name.
 *
 * arch_index follows the tier ladder: 2 = coopmatrix, 1 = subgroup,
 * 0 = baseline. arch_name is one of the static strings "coopmatrix",
 * "subgroup", "baseline".
 */
typedef struct {
    /* core scalar type features (VkPhysicalDeviceFeatures) */
    VkBool32 has_shader_int64;      /**< VkPhysicalDeviceFeatures::shaderInt64.  */
    VkBool32 has_shader_float64;    /**< shaderFloat64 (dgemm).                  */
    VkBool32 has_shader_int16;      /**< shaderInt16.                            */

    /* Vulkan 1.1/1.2 feature set */
    VkBool32 has_shader_float16;    /**< Vulkan12Features::shaderFloat16.        */
    VkBool32 has_shader_int8;       /**< Vulkan12Features::shaderInt8.           */
    VkBool32 has_storage_buffer16;  /**< Vulkan11Features::storageBuffer16BitAccess. */
    VkBool32 has_storage_buffer8;   /**< Vulkan12Features::storageBuffer8BitAccess.  */
    VkBool32 has_scalar_block_layout;   /**< tight quantized packing (std430/scalar). */
    VkBool32 has_buffer_device_address; /**< GPU-side pointers for fused kernels.     */
    VkBool32 has_subgroup_extended_types; /**< fp16/int8 subgroup reduce/shuffle.    */
    VkBool32 has_timeline_semaphore;     /**< multi-queue sync without wait-idle.    */
    VkBool32 has_vulkan_memory_model;    /**< device-scope atomics ordering.         */

    /* Vulkan 1.3/1.4 feature set */
    VkBool32 has_subgroup;          /**< Compute-stage subgroup ops available.   */
    VkBool32 has_subgroup_size_control; /**< can force a compute subgroup size.  */
    VkBool32 has_sync2;             /**< synchronization2 / vkCmdPipelineBarrier2. */
    VkBool32 has_pipeline_creation_cache_control; /**< async/immediate compile. */
    VkBool32 has_maintenance4;      /**< maxBufferSize = 2 GiB cap query.        */
    VkBool32 has_shader_integer_dot_product; /**< dp4a quantized GEMM.           */
    VkBool32 has_shader_expect_assume;      /**< Vulkan14Features::shaderExpectAssume. */

    /* cooperative matrix (RISKY: gated on VAIT_COOPMATRIX) */
    VkBool32 has_coop_matrix;       /**< VK_KHR_cooperative_matrix feature.      */

    /* named extensions */
    VkBool32 has_push_descriptor;   /**< vkCmdPushDescriptorSetKHR available.    */
    VkBool32 has_pipeline_binary;   /**< VK_KHR_pipeline_binary (startup cache). */
    VkBool32 has_atomic_float;      /**< VK_EXT_shader_atomic_float (f32 atomics). */
    VkBool32 has_shader_bfloat16;   /**< VK_KHR_shader_bfloat16 (native bf16).   */

    /* ReBAR / SAM zero-copy (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT) */
    VkBool32 has_zero_copy_memory;      /**< A full-heap CPU-writable VRAM type. */
    uint32_t zero_copy_memory_type;     /**< Memory type index or VK_MAX_MEMORY_TYPES. */

    /* subgroup / wave geometry */
    uint32_t subgroup_size;         /**< Device subgroup size (e.g. 64 RDNA2).   */
    uint32_t min_subgroup_size;     /**< minSubgroupSize (Vulkan 1.3).           */
    uint32_t max_subgroup_size;     /**< maxSubgroupSize (Vulkan 1.3).           */
    uint32_t required_subgroup_size;/**< Compute-stage required size (32 on RDNA4). */
    uint32_t wavefront_size;        /**< VK_AMD_shader_core_properties (64 RDNA2 / 32 RDNA4). */
    uint32_t active_compute_units;  /**< VK_AMD_shader_core_properties2 (40 RDNA2 / 32 RDNA4). */
    uint32_t max_push_descriptors;  /**< VkPhysicalDevicePushDescriptorProperties. */

    uint32_t max_workgroup_size[3]; /**< maxComputeWorkGroupSize x/y/z limits.   */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn; /**< Loaded device fn (may be NULL). */
    uint32_t arch_index;            /**< 2=coopmatrix, 1=subgroup, 0=baseline.   */
    const char *arch_name;          /**< Static name for arch_index.             */
} VkRuntimeCaps;

/* ===========================================================================
 * Capability detection
 * ========================================================================== */

/**
 * \brief Detect the GPU capability set for a physical/logical device pair.
 *
 * Single implementation of the capability detection every higher library used
 * to duplicate at context creation: a VkPhysicalDeviceFeatures2 pNext chain
 * (with VkPhysicalDeviceCooperativeMatrixFeaturesKHR) for shaderInt64 and
 * cooperative matrix, a VkPhysicalDeviceProperties2 pNext chain (with
 * VkPhysicalDeviceSubgroupProperties) for subgroup size/stages and the
 * maxComputeWorkGroupSize limits, and a vkGetDeviceProcAddr lookup for
 * vkCmdPushDescriptorSetKHR.
 *
 * \param pd     Physical device to query.
 * \param device Logical device (used for the push-descriptor fn lookup).
 * \param caps   Receives the detected capability set.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 */
VkResult vkr_detect_capabilities(VkPhysicalDevice pd, VkDevice device,
                                 VkRuntimeCaps* caps);

/* ===========================================================================
 * Canonical device creation (full feature enablement)
 * ========================================================================== */

/**
 * \brief Create a logical device with the stack's canonical full feature set.
 *
 * This is the single, harness-first way to create a VkDevice for this stack:
 * it queries what the physical device supports and enables the full
 * Vulkan 1.1-1.4 compute feature set in one pNext chain:
 *
 *   - base: shaderInt64 / shaderFloat64 / shaderInt16 / 64-bit buffer+shared atomics
 *   - 1.1:  storageBuffer16BitAccess, uniformAndStorageBuffer16BitAccess
 *   - 1.2:  storageBuffer8BitAccess, shaderFloat16, shaderInt8, scalarBlockLayout,
 *           bufferDeviceAddress, shaderSubgroupExtendedTypes, timelineSemaphore,
 *           vulkanMemoryModel
 *   - 1.3:  subgroupSizeControl, computeFullSubgroups, synchronization2,
 *           pipelineCreationCacheControl, maintenance4, shaderIntegerDotProduct
 *   - 1.4:  shaderExpectAssume
 *   - named: VK_KHR_push_descriptor, VK_AMD_shader_core_properties(2),
 *            VK_KHR_pipeline_binary, VK_EXT_shader_atomic_float,
 *            VK_KHR_shader_bfloat16 — enabled when advertised
 *   - VK_KHR_cooperative_matrix — only when VAIT_COOPMATRIX is set (driver-buggy)
 *
 * Only features the device actually supports are enabled; nothing is forced.
 * The device is created with one queue of the given queue family.
 *
 * \param pd           Physical device to create from.
 * \param queue_family Queue family index for the created queue.
 * \param out_device   Receives the VkDevice handle.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument or queue family invalid.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Extension enumeration failed.
 * \retval Other VkResult from vkCreateDevice.
 */
VkResult vkr_create_device(VkPhysicalDevice pd, uint32_t queue_family,
                           VkDevice* out_device);

/* ===========================================================================
 * Runtime lifecycle
 * ========================================================================== */

/**
 * \brief Create a VkRuntime bound to a physical/logical device and a queue.
 *
 * The runtime performs capability detection (shaderInt64, subgroup support,
 * cooperative matrix, push-descriptor availability) using the same
 * pNext-chain technique as vkmath/vkblas, and prepares the pooled allocator
 * for two memory classes (device-local and host-visible/coherent).
 *
 * \param physicalDevice Physical device (used for capability queries and
 *                       memory-type selection).
 * \param device          Logical device the runtime binds to. All buffers,
 *                        memory, and pools created through the runtime are
 *                        created on this device.
 * \param compute_queue   A queue on \p device used for staging uploads /
 *                        downloads and vkr_wait_idle().
 * \param pRuntime        Receives the new runtime on success.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host-side bookkeeping allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid arguments or device query failed.
 */
VkResult vkr_create_runtime(VkPhysicalDevice physicalDevice,
                            VkDevice device,
                            VkQueue compute_queue,
                            VkRuntime** pRuntime);

/**
 * \brief Destroy a runtime and release every pooled memory block.
 *
 * Frees all VkDeviceMemory blocks the allocator grew, plus host-side
 * bookkeeping. Buffers previously returned by vkr_malloc() that were not
 * passed to vkr_free() are invalidated; callers must not use them after
 * destroying the runtime.
 *
 * \param runtime Runtime to destroy. May be NULL.
 */
void vkr_destroy_runtime(VkRuntime* runtime);

/* ===========================================================================
 * Capability queries
 * ========================================================================== */

/**
 * \brief Query the detected GPU architecture index.
 *
 * 0 = baseline (vendor-agnostic), 1 = subgroup, 2 = cooperative matrix.
 * The index is the highest shader tier the physical device supports, mirroring
 * vkmath/vkblas tier selection.
 *
 * \param rt Valid runtime.
 * \return Architecture index (0, 1, or 2).
 */
uint32_t vkr_get_arch_index(VkRuntime* rt);

/**
 * \brief Get the human-readable GPU architecture string.
 *
 * \param rt Valid runtime.
 * \return e.g. "baseline", "subgroup", or "coopmatrix".
 */
const char* vkr_get_arch_name(VkRuntime* rt);

/**
 * \brief Query whether compute shaders can use subgroup operations.
 *
 * Subgroup is a core Vulkan 1.3+ feature; this reports whether the
 * VkPhysicalDeviceSubgroupProperties::supportedStages includes the compute
 * stage.
 *
 * \param rt Valid runtime.
 * \return VK_TRUE if compute-stage subgroup operations are supported.
 */
VkBool32 vkr_has_subgroup(VkRuntime* rt);

/**
 * \brief Query whether cooperative matrix is available (VK_KHR_cooperative_matrix).
 *
 * \param rt Valid runtime.
 * \return VK_TRUE if the cooperativeMatrix feature is supported.
 */
VkBool32 vkr_has_coop_matrix(VkRuntime* rt);

/**
 * \brief Query the device's fixed subgroup size.
 *
 * \param rt Valid runtime.
 * \return The device subgroup size (e.g. 64 on RDNA2, 32 on RDNA4).
 */
uint32_t vkr_get_subgroup_size(VkRuntime* rt);

/**
 * \brief Get the logical device the runtime is bound to.
 *
 * \param rt Valid runtime.
 * \return VkDevice handle.
 */
VkDevice vkr_get_device(VkRuntime* rt);

/* ===========================================================================
 * Pooled memory allocator (hipMalloc / hipFree equivalent)
 * ========================================================================== */

/**
 * \brief Allocate a buffer from the runtime's memory pool.
 *
 * Equivalent of hipMalloc(). Memory is sub-allocated from a small set of large
 * VkDeviceMemory blocks that grow on demand; each block owns one Vulkan memory
 * allocation, so allocation churn does not cause per-free vkAllocateMemory /
 * vkFreeMemory syscalls. vkr_free() returns the region to the pool's free list
 * for reuse.
 *
 * Memory class selection from \p usage:
 * - If \p usage consists purely of transfer bits (TRANSFER_SRC and/or
 *   TRANSFER_DST) the buffer is backed by host-visible + host-coherent memory
 *   (staging / readback style).
 * - Otherwise the buffer is backed by device-local memory.
 *
 * Every buffer is additionally created with
 * VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT so that
 * any pooled buffer can be used with vkr_upload() / vkr_download() (mirroring
 * hipMemcpy working on any hipMalloc pointer).
 *
 * The region size is rounded up to the buffer's required alignment.
 *
 * \param rt       Valid runtime.
 * \param size     Requested size in bytes (0 is allowed: returns null handles).
 * \param usage    VkBufferUsageFlags selecting the memory class (see above).
 * \param pBuffer  Receives the VkBuffer handle.
 * \param pMemory  Receives the backing VkDeviceMemory handle (block-owned).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY No block could be grown for the size.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host-side bookkeeping failed.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No matching memory type found.
 */
VkResult vkr_malloc(VkRuntime* rt, VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkBuffer* pBuffer, VkDeviceMemory* pMemory);

/**
 * \brief Return a pooled buffer to the runtime's allocator.
 *
 * Equivalent of hipFree(). The backing VkDeviceMemory block is retained by the
 * pool; only the buffer handle is destroyed and the region is made available
 * for reuse. Both \p buffer and \p memory must be the exact values returned by
 * vkr_malloc().
 *
 * \param rt     Valid runtime.
 * \param buffer VkBuffer previously returned by vkr_malloc().
 * \param memory VkDeviceMemory previously returned by vkr_malloc().
 */
void vkr_free(VkRuntime* rt, VkBuffer buffer, VkDeviceMemory memory);

/* ===========================================================================
 * Staging upload / download (hipMemcpy H2D / D2H equivalent)
 * ========================================================================== */

/**
 * \brief Upload host memory into a pooled device buffer.
 *
 * Equivalent of hipMemcpy(kind = H2D) followed by a device synchronize.
 *
 * The caller supplies a command buffer (from a pool created on the same queue
 * family as \p queue) and the queue to submit on. This function takes the
 * command buffer over for exactly one submission: it begins, records a single
 * vkCmdCopyBuffer (host-visible staging -> \p dev), ends, submits to \p queue,
 * waits for the queue to go idle, resets the command buffer (returning it to
 * the initial state), and frees the transient staging buffer. The command
 * buffer must be in the initial (unbegun) state on entry and is reusable
 * afterwards.
 *
 * The staging buffer is allocated from the runtime's host-visible pool and
 * freed before returning; no persistent staging memory is retained.
 *
 * \param rt      Valid runtime.
 * \param cmd     Command buffer to record into (must be unbegun).
 * \param queue   Queue to submit the copy on.
 * \param host    Host source pointer (must be readable for \p size bytes).
 * \param dev     Device destination buffer (created by vkr_malloc()).
 * \param offset  Byte offset into \p dev.
 * \param size    Number of bytes to copy. 0 is a no-op that returns VK_SUCCESS.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY Staging allocation failed.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT Staging could not be mapped.
 */
VkResult vkr_upload(VkRuntime* rt, VkCommandBuffer cmd, VkQueue queue,
                    const void* host, VkBuffer dev,
                    VkDeviceSize offset, VkDeviceSize size);

/**
 * \brief Download a pooled device buffer into host memory.
 *
 * Equivalent of hipMemcpy(kind = D2H) followed by a device synchronize.
 *
 * Same ownership contract as vkr_upload(): \p cmd is taken over for one
 * submission and returned reset. Data is copied \p dev -> host-visible
 * staging with a single vkCmdCopyBuffer, submitted and waited on, then read
 * back into \p host. The staging buffer is freed before returning.
 *
 * \param rt      Valid runtime.
 * \param cmd     Command buffer to record into (must be unbegun).
 * \param queue   Queue to submit the copy on.
 * \param dev     Device source buffer (created by vkr_malloc()).
 * \param offset  Byte offset into \p dev.
 * \param host    Host destination pointer (must be writable for \p size bytes).
 * \param size    Number of bytes to copy. 0 is a no-op that returns VK_SUCCESS.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY Staging allocation failed.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT Staging could not be mapped.
 */
VkResult vkr_download(VkRuntime* rt, VkCommandBuffer cmd, VkQueue queue,
                      VkBuffer dev, VkDeviceSize offset,
                      void* host, VkDeviceSize size);

/* ===========================================================================
 * Command / descriptor pool helpers
 * ========================================================================== */

/**
 * \brief Create a command pool on a queue family.
 *
 * The pool is created with VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT so
 * command buffers allocated from it can be reset and reused (as vkr_upload() /
 * vkr_download() require). The caller owns the returned pool and destroys it
 * with vkDestroyCommandPool().
 *
 * \param rt           Valid runtime.
 * \param queue_family Queue family index the pool must be compatible with.
 * \param pPool        Receives the VkCommandPool.
 * \retval VK_SUCCESS
 */
VkResult vkr_create_command_pool(VkRuntime* rt, uint32_t queue_family,
                                 VkCommandPool* pPool);

/**
 * \brief Create a descriptor pool sized for storage-buffer descriptor sets.
 *
 * Allocates \p max_sets sets, each able to consume up to \p ssbo_count storage
 * buffer descriptors (the descriptor count is the pooled total, not per-set).
 * The pool is created with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
 * and a single VK_DESCRIPTOR_TYPE_STORAGE_BUFFER pool size. This is the shared
 * helper every library's context uses for its push-descriptor fallback pool.
 * The caller owns the returned pool and destroys it with
 * vkDestroyDescriptorPool().
 *
 * \param device     Logical device to create the pool on.
 * \param max_sets   Maximum number of descriptor sets that can be allocated.
 * \param ssbo_count Pooled count of VK_DESCRIPTOR_TYPE_STORAGE_BUFFER descriptors.
 * \param pPool      Receives the VkDescriptorPool.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 */
VkResult vkr_create_descriptor_pool(VkDevice device, uint32_t max_sets,
                                    uint32_t ssbo_count,
                                    VkDescriptorPool* pPool);

/**
 * \brief Create a pipeline layout for one descriptor set + push constants.
 *
 * Shared helper for the per-library pipeline layouts: binds exactly \p set_layout
 * as set 0 and declares \p push_range_count compute-stage push-constant ranges.
 * The caller owns the returned layout and destroys it with
 * vkDestroyPipelineLayout().
 *
 * \param device           Logical device to create the layout on.
 * \param set_layout       Descriptor set layout for set 0 (may be VK_NULL_HANDLE).
 * \param push_range_count Number of push-constant ranges (0 allowed).
 * \param ranges           Array of \p push_range_count VkPushConstantRange.
 * \param pLayout          Receives the VkPipelineLayout.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 */
VkResult vkr_create_pipeline_layout(VkDevice device,
                                    VkDescriptorSetLayout set_layout,
                                    uint32_t push_range_count,
                                    const VkPushConstantRange* ranges,
                                    VkPipelineLayout* pLayout);

/**
 * \brief Create a driver-level pipeline cache.
 *
 * Shared helper for the per-library pipeline caches: creates an empty
 * VkPipelineCache (no initial data) that accelerates lazy compute pipeline
 * creation. The caller owns the returned cache and destroys it with
 * vkDestroyPipelineCache().
 *
 * \param device Logical device to create the cache on.
 * \param pCache Receives the VkPipelineCache.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 */
VkResult vkr_create_pipeline_cache(VkDevice device, VkPipelineCache* pCache);

/**
 * \brief Wait until all submitted work on the runtime's queue is complete.
 *
 * Equivalent of hipDeviceSynchronize() scoped to the runtime's queue.
 *
 * \param rt Valid runtime.
 */
void vkr_wait_idle(VkRuntime* rt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKRUNTIME_H */
