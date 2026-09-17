/**
 * \file vkruntime.c
 * \brief VKRuntime implementation: context lifecycle, capability detection,
 *        pooled allocator, staging upload/download, pool helpers.
 *
 * Design notes (mirroring hip_runtime_api.h semantics with Vulkan handles):
 *  - vkr_create_runtime() runs capability detection exactly like
 *    vkmath_init_capabilities() (VkPhysicalDeviceFeatures2 with a
 *    VkPhysicalDeviceCooperativeMatrixFeaturesKHR pNext + a
 *    VkPhysicalDeviceSubgroupProperties property chain) and detects push
 *    descriptors via vkGetDeviceProcAddr("vkCmdPushDescriptorSetKHR").
 *  - vkr_malloc()/vkr_free() are the hipMalloc()/hipFree() equivalents: one
 *    large VkDeviceMemory block per memory class, sub-allocated with a bump
 *    pointer + coalescing free list. Free returns the region to the pool; only
 *    vkr_destroy_runtime() frees the underlying blocks.
 *  - vkr_upload()/vkr_download() are the hipMemcpy H2D/D2H equivalents. The
 *    caller supplies a command buffer and queue; each call records one
 *    vkCmdCopyBuffer through a transient host-visible staging buffer, submits,
 *    waits, and resets the command buffer for reuse.
 */
#include "vkruntime_internal.h"

#include <string.h>
#include <stdlib.h>

/* ===========================================================================
 * Small helpers
 * ========================================================================== */

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize align)
{
    if (align == 0) return value;
    return (value + align - 1) & ~(align - 1);
}

static uint32_t pool_index_for_usage(VkBufferUsageFlags usage)
{
    /* A buffer used purely as a transfer source/destination is a host-visible
       staging buffer. Any compute/graphics usage selects device-local memory. */
    if (usage != 0 &&
        (usage & ~(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT)) == 0) {
        return VKR_POOL_HOST_VISIBLE;
    }
    return VKR_POOL_DEVICE_LOCAL;
}

/* ===========================================================================
 * Capability detection
 * ========================================================================== */

static VkBool32 vkr_device_extension_supported(VkPhysicalDevice pd,
                                               const char *name)
{
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL) != VK_SUCCESS)
        return VK_FALSE;
    if (count == 0) return VK_FALSE;

    VkExtensionProperties *props =
        (VkExtensionProperties *)malloc(count * sizeof(*props));
    if (!props) return VK_FALSE;

    VkResult r = vkEnumerateDeviceExtensionProperties(pd, NULL, &count, props);
    VkBool32 found = VK_FALSE;
    if (r == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            if (strcmp(props[i].extensionName, name) == 0) { found = VK_TRUE; break; }
        }
    }
    free(props);
    return found;
}

/**
 * \brief Query the full stack feature set through one VkPhysicalDeviceFeatures2
 *        pNext chain (Vulkan 1.1-1.4 + cooperative matrix + pipeline binary).
 *
 * \param pd   Physical device to query.
 * \param caps Receives the feature flags.
 */
static void vkr_query_features(VkPhysicalDevice pd, VkRuntimeCaps *caps)
{
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop;
    VkPhysicalDeviceVulkan11Features vk11;
    VkPhysicalDeviceVulkan12Features vk12;
    VkPhysicalDeviceVulkan13Features vk13;
    VkPhysicalDeviceVulkan14Features vk14;
    VkPhysicalDeviceShaderFloat16Int8Features f16i8;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot;
    VkPhysicalDevicePipelineBinaryFeaturesKHR pbin;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16;

    memset(&coop, 0, sizeof(coop));
    coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk11, 0, sizeof(vk11));
    vk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    memset(&vk12, 0, sizeof(vk12));
    vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk13, 0, sizeof(vk13));
    vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    memset(&vk14, 0, sizeof(vk14));
    vk14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    memset(&f16i8, 0, sizeof(f16i8));
    f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    memset(&dot, 0, sizeof(dot));
    dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    memset(&pbin, 0, sizeof(pbin));
    pbin.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
    memset(&af, 0, sizeof(af));
    af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    memset(&bf16, 0, sizeof(bf16));
    bf16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;

    VkPhysicalDeviceFeatures2 f2;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &vk11;
    vk11.pNext = &vk12;
    vk12.pNext = &vk13;
    vk13.pNext = &vk14;
    vk14.pNext = &f16i8;
    f16i8.pNext = &dot;
    dot.pNext = &pbin;
    pbin.pNext = &af;
    af.pNext = &bf16;
    bf16.pNext = &coop;
    vkGetPhysicalDeviceFeatures2(pd, &f2);

    caps->has_shader_int64   = f2.features.shaderInt64;
    caps->has_shader_float64 = f2.features.shaderFloat64;
    caps->has_shader_int16   = f2.features.shaderInt16;

    caps->has_storage_buffer16 = vk11.storageBuffer16BitAccess;
    caps->has_storage_buffer8  = vk12.storageBuffer8BitAccess;

    caps->has_shader_float16  = vk12.shaderFloat16;
    caps->has_shader_int8     = vk12.shaderInt8;
    caps->has_scalar_block_layout   = vk12.scalarBlockLayout;
    caps->has_buffer_device_address = vk12.bufferDeviceAddress;
    caps->has_subgroup_extended_types = vk12.shaderSubgroupExtendedTypes;
    caps->has_timeline_semaphore  = vk12.timelineSemaphore;
    caps->has_vulkan_memory_model = vk12.vulkanMemoryModel;

    caps->has_subgroup                = vk13.subgroupSizeControl ? VK_TRUE : VK_FALSE;
    caps->has_subgroup_size_control   = vk13.subgroupSizeControl;
    caps->has_sync2                   = vk13.synchronization2;
    caps->has_pipeline_creation_cache_control = vk13.pipelineCreationCacheControl;
    caps->has_maintenance4            = vk13.maintenance4;
    caps->has_shader_integer_dot_product = vk13.shaderIntegerDotProduct;

    caps->has_shader_expect_assume = vk14.shaderExpectAssume;

    /* Co-op matrix / WMMA path is stubbed: no cooperatice-matrix-tier SPIR-V
       kernels are registered, and RDNA4's WMMA is broken (crashes drivers).
       Force the subgroup/baseline fallback on all devices so the tier ladder
       never advertises VKMATH_TIER_COOPMATRIX. Remove this line once a real
       coopmatrix kernel + RDNA4 WMMA driver fix exist. */
    caps->has_coop_matrix = VK_FALSE;
    caps->has_pipeline_binary = pbin.pipelineBinaries;
    caps->has_atomic_float =
        (af.shaderBufferFloat32Atomics && af.shaderBufferFloat32AtomicAdd) ? VK_TRUE : VK_FALSE;
    caps->has_shader_bfloat16 = bf16.shaderBFloat16Type;
}

/**
 * \brief Query subgroup / wave geometry through one VkPhysicalDeviceProperties2
 *        pNext chain (subgroup, subgroup-size control, AMD shader core 1+2,
 *        push-descriptor properties).
 *
 * \param pd   Physical device to query.
 * \param caps Receives the geometry fields.
 */
static void vkr_query_properties(VkPhysicalDevice pd, VkRuntimeCaps *caps)
{
    VkPhysicalDeviceSubgroupProperties subgroup;
    VkPhysicalDeviceSubgroupSizeControlProperties ssc;
    VkPhysicalDeviceShaderCorePropertiesAMD amd1;
    VkPhysicalDeviceShaderCoreProperties2AMD amd2;
    VkPhysicalDevicePushDescriptorPropertiesKHR pdprops;

    memset(&subgroup, 0, sizeof(subgroup));
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    memset(&ssc, 0, sizeof(ssc));
    ssc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    memset(&amd1, 0, sizeof(amd1));
    amd1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
    memset(&amd2, 0, sizeof(amd2));
    amd2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD;
    memset(&pdprops, 0, sizeof(pdprops));
    pdprops.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 p2;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &subgroup;
    subgroup.pNext = &ssc;
    ssc.pNext = &amd1;
    amd1.pNext = &amd2;
    amd2.pNext = &pdprops;
    vkGetPhysicalDeviceProperties2(pd, &p2);

    caps->subgroup_size = subgroup.subgroupSize;
    caps->has_subgroup = (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
                         ? VK_TRUE : VK_FALSE;
    caps->min_subgroup_size = ssc.minSubgroupSize;
    caps->max_subgroup_size = ssc.maxSubgroupSize;
    caps->required_subgroup_size = 0u; /* stage mask exposed, value is not */
    caps->wavefront_size = amd1.wavefrontSize;
    caps->active_compute_units = amd2.activeComputeUnitCount;
    caps->max_push_descriptors = pdprops.maxPushDescriptors;

    caps->max_workgroup_size[0] = p2.properties.limits.maxComputeWorkGroupSize[0];
    caps->max_workgroup_size[1] = p2.properties.limits.maxComputeWorkGroupSize[1];
    caps->max_workgroup_size[2] = p2.properties.limits.maxComputeWorkGroupSize[2];
}

/**
 * \brief Detect the ReBAR / SAM zero-copy memory type, if any.
 *
 * A zero-copy type is DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT on the
 * device-local heap (CPU-writable VRAM through the PCIe BAR). Found on the
 * RX 9070 XT (RDNA4, ReBAR) as memory type 2 / heap 1. Absent on non-ReBAR
 * discrete GPUs; the caller then falls back to staged uploads.
 *
 * \param pd   Physical device to query.
 * \param caps Receives has_zero_copy_memory / zero_copy_memory_type.
 */
static void vkr_query_zero_copy(VkPhysicalDevice pd, VkRuntimeCaps *caps)
{
    caps->zero_copy_memory_type = VK_MAX_MEMORY_TYPES;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryType *t = &mp.memoryTypes[i];
        const VkMemoryHeap *h = &mp.memoryHeaps[t->heapIndex];
        if ((t->propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            (void)h; /* full-heap BAR visibility is a driver guarantee on ReBAR */
            caps->zero_copy_memory_type = i;
            caps->has_zero_copy_memory = VK_TRUE;
            return;
        }
    }
    caps->has_zero_copy_memory = VK_FALSE;
}

/**
 * \brief Detect the full stack capability set for a physical/logical device pair.
 *
 * Single implementation of the capability detection every higher library used
 * to duplicate at context creation. Queries the full Vulkan 1.1-1.4 feature
 * chain, subgroup / wave geometry, AMD shader-core properties, push-descriptor
 * limits, the named-extension availability (pipeline binary, atomic float,
 * shader bfloat16), and the ReBAR zero-copy memory type. Fills push_desc_fn
 * from the logical device.
 *
 * \param pd     Physical device to query.
 * \param device Logical device (used for the push-descriptor fn lookup).
 * \param caps   Receives the detected capability set.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 */
VkResult vkr_detect_capabilities(VkPhysicalDevice pd, VkDevice device,
                                 VkRuntimeCaps *caps)
{
    if (!pd || !device || !caps) return VK_ERROR_INITIALIZATION_FAILED;
    memset(caps, 0, sizeof(*caps));

    vkr_query_features(pd, caps);
    vkr_query_properties(pd, caps);
    vkr_query_zero_copy(pd, caps);

    /* Named-extension availability (device-level, independent of creation) */
    caps->has_pipeline_binary =
        caps->has_pipeline_binary &&
        vkr_device_extension_supported(pd, VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
    caps->has_atomic_float =
        caps->has_atomic_float &&
        vkr_device_extension_supported(pd, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    caps->has_shader_bfloat16 =
        caps->has_shader_bfloat16 &&
        vkr_device_extension_supported(pd, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);

    /* Subgroup support from the subgroup properties (compute-stage gate above) */

    /* Highest supported shader tier (mirrors vkmath tier selection) */
    if (caps->has_coop_matrix) {
        caps->arch_index = 2;
        caps->arch_name = "coopmatrix";
    } else if (caps->has_subgroup) {
        caps->arch_index = 1;
        caps->arch_name = "subgroup";
    } else {
        caps->arch_index = 0;
        caps->arch_name = "baseline";
    }

    /* Push descriptor availability via device function pointer */
    caps->push_desc_fn = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    caps->has_push_descriptor = caps->push_desc_fn ? VK_TRUE : VK_FALSE;

    return VK_SUCCESS;
}

/**
 * \brief Populate a runtime's cached capability set.
 *
 * \param rt Runtime whose caps are populated.
 * \param pd Physical device to query.
 */
static void vkr_init_capabilities(VkRuntime *rt, VkPhysicalDevice pd)
{
    (void)vkr_detect_capabilities(pd, rt->device, &rt->caps);
    rt->push_desc_fn = rt->caps.push_desc_fn;
}

/* ===========================================================================
 * Allocation-record table (open addressing, linear probing)
 * ========================================================================== */

static uint32_t alloc_table_hash(VkBuffer buffer)
{
    uint64_t key = (uint64_t)(uintptr_t)buffer;
    return (uint32_t)((key >> 32) ^ key);
}

static void alloc_table_rehash(VkRuntime *rt, vkr_alloc_slot_t *new_table,
                               uint32_t new_capacity)
{
    for (uint32_t i = 0; i < rt->alloc_capacity; i++) {
        vkr_alloc_slot_t *s = &rt->alloc_table[i];
        if (!s->buffer) continue;
        uint32_t idx = alloc_table_hash(s->buffer) & (new_capacity - 1);
        while (new_table[idx].buffer) idx = (idx + 1) & (new_capacity - 1);
        new_table[idx] = *s;
    }
}

static VkResult alloc_table_grow(VkRuntime *rt)
{
    uint32_t new_capacity = rt->alloc_capacity * 2;
    vkr_alloc_slot_t *new_table =
        (vkr_alloc_slot_t *)calloc(new_capacity, sizeof(vkr_alloc_slot_t));
    if (!new_table) return VK_ERROR_OUT_OF_HOST_MEMORY;
    alloc_table_rehash(rt, new_table, new_capacity);
    free(rt->alloc_table);
    rt->alloc_table = new_table;
    rt->alloc_capacity = new_capacity;
    return VK_SUCCESS;
}

static VkResult alloc_table_insert(VkRuntime *rt, const vkr_alloc_slot_t *rec)
{
    if (rt->alloc_count * 10 >= rt->alloc_capacity * 6) {
        VkResult r = alloc_table_grow(rt);
        if (r != VK_SUCCESS) return r;
    }
    uint32_t idx = alloc_table_hash(rec->buffer) & (rt->alloc_capacity - 1);
    while (rt->alloc_table[idx].buffer) idx = (idx + 1) & (rt->alloc_capacity - 1);
    rt->alloc_table[idx] = *rec;
    rt->alloc_count++;
    return VK_SUCCESS;
}

static vkr_alloc_slot_t *alloc_table_find(VkRuntime *rt, VkBuffer buffer)
{
    if (!rt->alloc_table) return NULL;
    uint32_t idx = alloc_table_hash(buffer) & (rt->alloc_capacity - 1);
    for (uint32_t probe = 0; probe < rt->alloc_capacity; probe++) {
        vkr_alloc_slot_t *s = &rt->alloc_table[idx];
        if (!s->buffer) return NULL;
        if (s->buffer == buffer) return s;
        idx = (idx + 1) & (rt->alloc_capacity - 1);
    }
    return NULL;
}

static void alloc_table_remove(VkRuntime *rt, VkBuffer buffer)
{
    if (!rt->alloc_table) return;
    uint32_t idx = alloc_table_hash(buffer) & (rt->alloc_capacity - 1);
    for (uint32_t probe = 0; probe < rt->alloc_capacity; probe++) {
        vkr_alloc_slot_t *s = &rt->alloc_table[idx];
        if (!s->buffer) return;
        if (s->buffer == buffer) {
            memset(s, 0, sizeof(*s));
            rt->alloc_count--;
            return;
        }
        idx = (idx + 1) & (rt->alloc_capacity - 1);
    }
}

/* ===========================================================================
 * Pooled allocator
 * ========================================================================== */

static vkr_block_t *pool_find_block(VkRuntime *rt, VkDeviceMemory memory)
{
    for (uint32_t p = 0; p < VKR_POOL_COUNT; p++) {
        for (vkr_block_t *b = rt->pools[p].blocks; b; b = b->next) {
            if (b->memory == memory) return b;
        }
    }
    return NULL;
}

static void free_region_remove(vkr_block_t *b, uint32_t idx)
{
    b->free_offs[idx] = b->free_offs[b->free_count - 1];
    b->free_sz[idx]   = b->free_sz[b->free_count - 1];
    b->free_count--;
}

/**
 * \brief Add a free region to a block, coalescing with neighbours.
 *
 * A region that ends exactly at the bump cursor rolls the cursor back (the
 * tail is freed for free). Otherwise the region is inserted, merged with any
 * immediately preceding/following free region, or appended.
 *
 * \param b    Block to update.
 * \param off  Region offset (relative to block memory).
 * \param size Region size.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Free-list array grow failed.
 */
static VkResult free_region_add(vkr_block_t *b, VkDeviceSize off, VkDeviceSize size)
{
    if (off + size == b->cursor) {
        b->cursor = off;
        return VK_SUCCESS;
    }

    /* try to extend an existing region */
    for (uint32_t i = 0; i < b->free_count; i++) {
        if (b->free_offs[i] + b->free_sz[i] == off) {
            b->free_sz[i] += size;
            /* now we may abut a following region — merge it */
            for (uint32_t j = 0; j < b->free_count; j++) {
                if (j == i) continue;
                if (b->free_offs[i] + b->free_sz[i] == b->free_offs[j]) {
                    b->free_sz[i] += b->free_sz[j];
                    free_region_remove(b, j);
                    break;
                }
            }
            return VK_SUCCESS;
        }
        if (off + size == b->free_offs[i]) {
            b->free_offs[i] = off;
            b->free_sz[i] += size;
            return VK_SUCCESS;
        }
    }

    /* append */
    if (b->free_count == b->free_cap) {
        uint32_t new_cap = b->free_cap ? b->free_cap * 2 : 16u;
        VkDeviceSize *no = (VkDeviceSize *)realloc(b->free_offs, new_cap * sizeof(VkDeviceSize));
        if (!no) return VK_ERROR_OUT_OF_HOST_MEMORY;
        b->free_offs = no;
        VkDeviceSize *ns = (VkDeviceSize *)realloc(b->free_sz, new_cap * sizeof(VkDeviceSize));
        if (!ns) return VK_ERROR_OUT_OF_HOST_MEMORY;
        b->free_sz = ns;
        b->free_cap = new_cap;
    }
    b->free_offs[b->free_count] = off;
    b->free_sz[b->free_count]   = size;
    b->free_count++;
    return VK_SUCCESS;
}

/**
 * \brief Carve \p size bytes from a block, first-fit through the free list
 *        then the bump cursor.
 *
 * \param b       Block to carve from.
 * \param align   Alignment for the returned offset.
 * \param size    Size of the region (already aligned up).
 * \param out_off Receives the region offset.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Free-list grow failed.
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY No room in the block.
 */
static VkResult block_alloc_region(vkr_block_t *b, VkDeviceSize align,
                                   VkDeviceSize size, VkDeviceSize *out_off)
{
    for (uint32_t i = 0; i < b->free_count; i++) {
        VkDeviceSize foff = b->free_offs[i];
        VkDeviceSize fsz  = b->free_sz[i];
        VkDeviceSize s    = align_up(foff, align);
        if (s + size <= foff + fsz) {
            VkDeviceSize rem_lo = s - foff;
            VkDeviceSize rem_hi = fsz - rem_lo - size;
            if (rem_hi > 0) {
                VkResult r = free_region_add(b, s + size, rem_hi);
                if (r != VK_SUCCESS) return r;
            }
            if (rem_lo > 0) {
                b->free_sz[i] = rem_lo;
            } else {
                free_region_remove(b, i);
            }
            *out_off = s;
            return VK_SUCCESS;
        }
    }
    /* bump allocate from the unused tail */
    VkDeviceSize s = align_up(b->cursor, align);
    if (s + size <= b->size) {
        b->cursor = s + size;
        *out_off = s;
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

/**
 * \brief Create a new block of \p size bytes and prepend it to a pool.
 *
 * Host-visible blocks are mapped immediately so upload/download staging can
 * memcpy without per-call vkMapMemory.
 *
 * \param rt        Runtime (owns the device).
 * \param pool      Pool to attach the block to.
 * \param type_idx  VkMemoryType index to allocate from.
 * \param size      Block size in bytes.
 * \param out_block Receives the new block.
 * \retval VK_SUCCESS
 */
static VkResult vkr_create_block(VkRuntime *rt, vkr_pool_t *pool,
                                 uint32_t type_idx, VkDeviceSize size,
                                 vkr_block_t **out_block)
{
    vkr_block_t *b = (vkr_block_t *)calloc(1, sizeof(*b));
    if (!b) return VK_ERROR_OUT_OF_HOST_MEMORY;

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = size;
    mai.memoryTypeIndex = type_idx;

    VkResult r = vkAllocateMemory(rt->device, &mai, NULL, &b->memory);
    if (r != VK_SUCCESS) {
        free(b);
        return r;
    }
    b->size = size;

    if (pool == &rt->pools[VKR_POOL_HOST_VISIBLE]) {
        r = vkMapMemory(rt->device, b->memory, 0, size, 0, &b->mapped);
        if (r != VK_SUCCESS) {
            vkFreeMemory(rt->device, b->memory, NULL);
            free(b);
            return r;
        }
    }

    b->free_offs = NULL;
    b->free_sz = NULL;
    b->free_cap = 0;
    b->free_count = 0;
    b->next = pool->blocks;
    pool->blocks = b;
    pool->memory_type_index = type_idx;

    *out_block = b;
    return VK_SUCCESS;
}

/**
 * \brief Pick a VkMemoryType index for a pool class from a buffer's
 *        memoryTypeBits.
 *
 * Device-local pools prefer non-host-visible device-local types; host-visible
 * pools prefer HOST_VISIBLE | HOST_COHERENT. Falls back to looser matches.
 *
 * \param rt        Runtime (cached memory properties).
 * \param pool_idx  Pool class.
 * \param type_bits Bitmask of acceptable memory types for the buffer.
 * \param out_type  Receives the memory type index.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_FEATURE_NOT_PRESENT No matching type.
 */
static VkResult vkr_pick_memory_type(VkRuntime *rt, uint32_t pool_idx,
                                     uint32_t type_bits, uint32_t *out_type)
{
    const VkMemoryType *types = rt->mem_props.memoryTypes;
    uint32_t count = rt->mem_props.memoryTypeCount;

    if (pool_idx == VKR_POOL_DEVICE_LOCAL) {
        for (uint32_t i = 0; i < count; i++) {
            if (!(type_bits & (1u << i))) continue;
            VkMemoryPropertyFlags f = types[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                *out_type = i;
                return VK_SUCCESS;
            }
        }
        for (uint32_t i = 0; i < count; i++) {
            if (!(type_bits & (1u << i))) continue;
            if (types[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                *out_type = i;
                return VK_SUCCESS;
            }
        }
    } else {
        for (uint32_t i = 0; i < count; i++) {
            if (!(type_bits & (1u << i))) continue;
            VkMemoryPropertyFlags f = types[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                *out_type = i;
                return VK_SUCCESS;
            }
        }
        for (uint32_t i = 0; i < count; i++) {
            if (!(type_bits & (1u << i))) continue;
            if (types[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                *out_type = i;
                return VK_SUCCESS;
            }
        }
    }
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

/**
 * \brief Get the host mapping for a pooled buffer region.
 *
 * \param rt     Runtime.
 * \param buffer Pooled buffer handle.
 * \param memory Block memory returned by vkr_malloc().
 * \param offset Byte offset within the buffer.
 * \return Host pointer, or NULL if the block is not host-visible.
 */
static void *vkr_host_ptr(VkRuntime *rt, VkBuffer buffer,
                          VkDeviceMemory memory, VkDeviceSize offset)
{
    vkr_alloc_slot_t *s = alloc_table_find(rt, buffer);
    if (!s) return NULL;
    for (uint32_t p = 0; p < VKR_POOL_COUNT; p++) {
        for (vkr_block_t *b = rt->pools[p].blocks; b; b = b->next) {
            if (b->memory == memory) {
                if (!b->mapped) return NULL;
                return (char *)b->mapped + s->offset + offset;
            }
        }
    }
    return NULL;
}

/* ===========================================================================
 * Staging copy helper (upload/download share this)
 * ========================================================================== */

/**
 * \brief Record one vkCmdCopyBuffer, submit, wait, reset the command buffer.
 *
 * Takes ownership of \p cmd for exactly one submission. On return \p cmd is
 * reset to the initial state and reusable. Requires \p cmd to be allocated
 * from a pool with VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT on the
 * same queue family as \p queue.
 *
 * \param rt       Runtime.
 * \param cmd      Command buffer (must be unbegun).
 * \param queue    Queue to submit on.
 * \param src      Source buffer.
 * \param src_off  Source byte offset.
 * \param dst      Destination buffer.
 * \param dst_off  Destination byte offset.
 * \param size     Copy size.
 * \retval VK_SUCCESS
 */
static VkResult vkr_copy_and_sync(VkRuntime *rt, VkCommandBuffer cmd,
                                  VkQueue queue, VkBuffer src, VkDeviceSize src_off,
                                  VkBuffer dst, VkDeviceSize dst_off,
                                  VkDeviceSize size)
{
    (void)rt;
    VkCommandBufferBeginInfo bbi;
    memset(&bbi, 0, sizeof(bbi));
    bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult r = vkBeginCommandBuffer(cmd, &bbi);
    if (r != VK_SUCCESS) return r;

    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = src_off;
    region.dstOffset = dst_off;
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    r = vkEndCommandBuffer(cmd);
    if (r != VK_SUCCESS) return r;

    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) return r;
    r = vkQueueWaitIdle(queue);
    if (r != VK_SUCCESS) return r;

    return vkResetCommandBuffer(cmd, 0);
}

/* ===========================================================================
 * Public API: runtime lifecycle
 * ========================================================================== */

/**
 * \brief Create a logical device with the stack's canonical full feature set.
 *
 * Enables exactly what the physical device supports, across the full
 * Vulkan 1.1-1.4 compute feature chain plus the named EXPLOIT extensions
 * (push descriptor, AMD shader core properties, pipeline binary, atomic float,
 * shader bfloat16) and VAIT_COOPMATRIX-gated cooperative matrix.
 *
 * \param pd           Physical device to create from.
 * \param queue_family Queue family index for the created queue.
 * \param out_device   Receives the VkDevice handle.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument or queue family invalid.
 */
VkResult vkr_create_device(VkPhysicalDevice pd, uint32_t queue_family,
                           VkDevice *out_device)
{
    if (!pd || !out_device) return VK_ERROR_INITIALIZATION_FAILED;

    /* Validate the queue family exists and is usable */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, NULL);
    if (qf_count == 0) return VK_ERROR_INITIALIZATION_FAILED;
    VkQueueFamilyProperties *qf =
        (VkQueueFamilyProperties *)malloc(qf_count * sizeof(*qf));
    if (!qf) return VK_ERROR_OUT_OF_HOST_MEMORY;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qf);
    if (queue_family >= qf_count || qf[queue_family].queueCount == 0) {
        free(qf);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    float priority = 1.0f;

    VkDeviceQueueCreateInfo qci;
    memset(&qci, 0, sizeof(qci));
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    /* ---- supported-feature mirror (enable only what exists) ---- */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop;
    VkPhysicalDeviceVulkan11Features vk11;
    VkPhysicalDeviceVulkan12Features vk12;
    VkPhysicalDeviceVulkan13Features vk13;
    VkPhysicalDeviceVulkan14Features vk14;
    VkPhysicalDeviceShaderFloat16Int8Features f16i8;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot;
    VkPhysicalDevicePipelineBinaryFeaturesKHR pbin;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16;

    memset(&coop, 0, sizeof(coop));
    coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk11, 0, sizeof(vk11));
    vk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    memset(&vk12, 0, sizeof(vk12));
    vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk13, 0, sizeof(vk13));
    vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    memset(&vk14, 0, sizeof(vk14));
    vk14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    memset(&f16i8, 0, sizeof(f16i8));
    f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    memset(&dot, 0, sizeof(dot));
    dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    memset(&pbin, 0, sizeof(pbin));
    pbin.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
    memset(&af, 0, sizeof(af));
    af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    memset(&bf16, 0, sizeof(bf16));
    bf16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;

    VkPhysicalDeviceFeatures2 supported;
    memset(&supported, 0, sizeof(supported));
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported.pNext = &vk11;
    vk11.pNext = &vk12;
    vk12.pNext = &vk13;
    vk13.pNext = &vk14;
    vk14.pNext = &f16i8;
    f16i8.pNext = &dot;
    dot.pNext = &pbin;
    pbin.pNext = &af;
    af.pNext = &bf16;
    bf16.pNext = &coop;
    vkGetPhysicalDeviceFeatures2(pd, &supported);

    /* ---- build the enable chain ---- */
    VkBool32 coop_gated = (getenv("VAIT_COOPMATRIX") != NULL) && coop.cooperativeMatrix;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_en;
    VkPhysicalDeviceVulkan11Features vk11_en;
    VkPhysicalDeviceVulkan12Features vk12_en;
    VkPhysicalDeviceVulkan13Features vk13_en;
    VkPhysicalDeviceVulkan14Features vk14_en;
    VkPhysicalDevicePipelineBinaryFeaturesKHR pbin_en;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af_en;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_en;

    memset(&coop_en, 0, sizeof(coop_en));
    coop_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk11_en, 0, sizeof(vk11_en));
    vk11_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    memset(&vk12_en, 0, sizeof(vk12_en));
    vk12_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk13_en, 0, sizeof(vk13_en));
    vk13_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    memset(&vk14_en, 0, sizeof(vk14_en));
    vk14_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    vk14_en.pushDescriptor = (vkr_device_extension_supported(pd, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) ? VK_TRUE : VK_FALSE;
    memset(&pbin_en, 0, sizeof(pbin_en));
    pbin_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
    memset(&af_en, 0, sizeof(af_en));
    af_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    memset(&bf16_en, 0, sizeof(bf16_en));
    bf16_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;

    VkPhysicalDeviceFeatures2 f2_en;
    memset(&f2_en, 0, sizeof(f2_en));
    f2_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    f2_en.features.shaderInt64 = VK_TRUE;
    f2_en.features.shaderFloat64 = supported.features.shaderFloat64;
    f2_en.features.shaderInt16 = supported.features.shaderInt16;

    vk11_en.storageBuffer16BitAccess = vk11.storageBuffer16BitAccess;
    vk11_en.uniformAndStorageBuffer16BitAccess = vk11.uniformAndStorageBuffer16BitAccess;

    vk12_en.storageBuffer8BitAccess = vk12.storageBuffer8BitAccess;
    vk12_en.uniformAndStorageBuffer8BitAccess = vk12.uniformAndStorageBuffer8BitAccess;
    vk12_en.shaderBufferInt64Atomics = vk12.shaderBufferInt64Atomics;
    vk12_en.shaderSharedInt64Atomics = vk12.shaderSharedInt64Atomics;
    vk12_en.shaderFloat16 = vk12.shaderFloat16;
    vk12_en.shaderInt8 = vk12.shaderInt8;
    vk12_en.scalarBlockLayout = vk12.scalarBlockLayout;
    vk12_en.bufferDeviceAddress = vk12.bufferDeviceAddress;
    vk12_en.shaderSubgroupExtendedTypes = vk12.shaderSubgroupExtendedTypes;
    vk12_en.timelineSemaphore = vk12.timelineSemaphore;
    vk12_en.vulkanMemoryModel = vk12.vulkanMemoryModel;
    vk12_en.descriptorBindingPartiallyBound = VK_TRUE;

    vk13_en.subgroupSizeControl = vk13.subgroupSizeControl;
    vk13_en.computeFullSubgroups = vk13.computeFullSubgroups;
    vk13_en.synchronization2 = vk13.synchronization2;
    vk13_en.pipelineCreationCacheControl = vk13.pipelineCreationCacheControl;
    vk13_en.maintenance4 = vk13.maintenance4;
    vk13_en.shaderIntegerDotProduct = vk13.shaderIntegerDotProduct;

    vk14_en.shaderExpectAssume = vk14.shaderExpectAssume;

    pbin_en.pipelineBinaries = pbin.pipelineBinaries;
    af_en.shaderBufferFloat32Atomics = af.shaderBufferFloat32Atomics;
    af_en.shaderBufferFloat32AtomicAdd = af.shaderBufferFloat32AtomicAdd;
    bf16_en.shaderBFloat16Type = bf16.shaderBFloat16Type;
    coop_en.cooperativeMatrix = coop_gated;
    coop_en.cooperativeMatrixRobustBufferAccess =
        coop_gated && coop.cooperativeMatrixRobustBufferAccess;

    f2_en.pNext = &vk11_en;
    vk11_en.pNext = &vk12_en;
    vk12_en.pNext = &vk13_en;
    vk13_en.pNext = &vk14_en;
    vk14_en.pNext = &pbin_en;
    pbin_en.pNext = &af_en;
    af_en.pNext = &bf16_en;
    bf16_en.pNext = &coop_en;

    /* ---- build the extension list (only available ones) ---- */
    const char *exts[8];
    uint32_t ext_count = 0;
    if (vkr_device_extension_supported(pd, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
        exts[ext_count++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    if (vkr_device_extension_supported(pd, VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME))
        exts[ext_count++] = VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME;
    if (vkr_device_extension_supported(pd, VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME))
        exts[ext_count++] = VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME;
    if (vkr_device_extension_supported(pd, VK_KHR_PIPELINE_BINARY_EXTENSION_NAME) &&
        pbin.pipelineBinaries)
        exts[ext_count++] = VK_KHR_PIPELINE_BINARY_EXTENSION_NAME;
    if (vkr_device_extension_supported(pd, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) &&
        af.shaderBufferFloat32Atomics)
        exts[ext_count++] = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
    if (vkr_device_extension_supported(pd, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) &&
        bf16.shaderBFloat16Type)
        exts[ext_count++] = VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;
    if (coop_gated)
        exts[ext_count++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;

    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2_en;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = ext_count;
    dci.ppEnabledExtensionNames = exts;

    VkResult r = vkCreateDevice(pd, &dci, NULL, out_device);
    free(qf);
    return r;
}

VkResult vkr_create_runtime(VkPhysicalDevice physicalDevice,
                            VkDevice device, VkQueue compute_queue,
                            VkRuntime **pRuntime)
{
    if (!pRuntime || !device) return VK_ERROR_INITIALIZATION_FAILED;

    VkRuntime *rt = (VkRuntime *)calloc(1, sizeof(VkRuntime));
    if (!rt) return VK_ERROR_OUT_OF_HOST_MEMORY;

    rt->device = device;
    rt->physical_device = physicalDevice;
    rt->queue = compute_queue;

    /* cache physical device memory properties for allocator use */
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &rt->mem_props);

    vkr_init_capabilities(rt, physicalDevice);

    /* allocation record table */
    rt->alloc_capacity = 4096u;
    rt->alloc_count = 0;
    rt->alloc_table = (vkr_alloc_slot_t *)
        calloc(rt->alloc_capacity, sizeof(vkr_alloc_slot_t));
    if (!rt->alloc_table) {
        free(rt);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    *pRuntime = rt;
    return VK_SUCCESS;
}

void vkr_destroy_runtime(VkRuntime *rt)
{
    if (!rt) return;
    for (uint32_t p = 0; p < VKR_POOL_COUNT; p++) {
        vkr_block_t *b = rt->pools[p].blocks;
        while (b) {
            vkr_block_t *next = b->next;
            if (b->mapped) vkUnmapMemory(rt->device, b->memory);
            vkFreeMemory(rt->device, b->memory, NULL);
            free(b->free_offs);
            free(b->free_sz);
            free(b);
            b = next;
        }
        rt->pools[p].blocks = NULL;
    }
    free(rt->alloc_table);
    free(rt);
}

/* ===========================================================================
 * Public API: capability queries
 * ========================================================================== */

uint32_t vkr_get_arch_index(VkRuntime *rt)
{
    return rt ? rt->caps.arch_index : 0;
}

const char *vkr_get_arch_name(VkRuntime *rt)
{
    return rt ? rt->caps.arch_name : "unknown";
}

VkBool32 vkr_has_subgroup(VkRuntime *rt)
{
    return rt ? rt->caps.has_subgroup : VK_FALSE;
}

VkBool32 vkr_has_coop_matrix(VkRuntime *rt)
{
    return rt ? rt->caps.has_coop_matrix : VK_FALSE;
}

uint32_t vkr_get_subgroup_size(VkRuntime *rt)
{
    return rt ? rt->caps.subgroup_size : 0;
}

VkDevice vkr_get_device(VkRuntime *rt)
{
    return rt ? rt->device : VK_NULL_HANDLE;
}

/* ===========================================================================
 * Public API: pooled allocator
 * ========================================================================== */

VkResult vkr_malloc(VkRuntime *rt, VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkBuffer *pBuffer, VkDeviceMemory *pMemory)
{
    if (!rt || !pBuffer || !pMemory) return VK_ERROR_INITIALIZATION_FAILED;
    *pBuffer = VK_NULL_HANDLE;
    *pMemory = VK_NULL_HANDLE;
    if (size == 0) return VK_SUCCESS;

    uint32_t pool_idx = pool_index_for_usage(usage);
    vkr_pool_t *pool = &rt->pools[pool_idx];

    VkBuffer buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(rt->device, &bci, NULL, &buffer);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(rt->device, buffer, &req);

    VkDeviceSize aligned = align_up(size, req.alignment);
    if (pool->alignment < req.alignment) pool->alignment = req.alignment;

    uint32_t mtype = 0;
    r = vkr_pick_memory_type(rt, pool_idx, req.memoryTypeBits, &mtype);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(rt->device, buffer, NULL);
        return r;
    }

    /* find a block with room, or grow one */
    VkDeviceSize offset = 0;
    vkr_block_t *block = pool->blocks;
    while (block) {
        if (block_alloc_region(block, pool->alignment, aligned, &offset) == VK_SUCCESS) break;
        block = block->next;
    }
    if (!block) {
        VkDeviceSize bsize = aligned > VKR_DEFAULT_BLOCK_SIZE
                             ? align_up(aligned, pool->alignment)
                             : VKR_DEFAULT_BLOCK_SIZE;
        r = vkr_create_block(rt, pool, mtype, bsize, &block);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(rt->device, buffer, NULL);
            return r;
        }
        r = block_alloc_region(block, pool->alignment, aligned, &offset);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(rt->device, buffer, NULL);
            return r;
        }
    }

    r = vkBindBufferMemory(rt->device, buffer, block->memory, offset);
    if (r != VK_SUCCESS) {
        (void)free_region_add(block, offset, aligned);
        vkDestroyBuffer(rt->device, buffer, NULL);
        return r;
    }

    vkr_alloc_slot_t rec;
    memset(&rec, 0, sizeof(rec));
    rec.buffer = buffer;
    rec.memory = block->memory;
    rec.offset = offset;
    rec.size = aligned;
    r = alloc_table_insert(rt, &rec);
    if (r != VK_SUCCESS) {
        (void)free_region_add(block, offset, aligned);
        vkDestroyBuffer(rt->device, buffer, NULL);
        return r;
    }

    *pBuffer = buffer;
    *pMemory = block->memory;
    return VK_SUCCESS;
}

void vkr_free(VkRuntime *rt, VkBuffer buffer, VkDeviceMemory memory)
{
    if (!rt || !buffer) return;
    vkr_alloc_slot_t *s = alloc_table_find(rt, buffer);
    if (!s) return;

    vkr_block_t *block = pool_find_block(rt, memory);
    if (block) {
        (void)free_region_add(block, s->offset, s->size);
    }
    alloc_table_remove(rt, buffer);
    vkDestroyBuffer(rt->device, buffer, NULL);
}

/* ===========================================================================
 * Public API: staging upload / download
 * ========================================================================== */

VkResult vkr_upload(VkRuntime *rt, VkCommandBuffer cmd, VkQueue queue,
                    const void *host, VkBuffer dev,
                    VkDeviceSize offset, VkDeviceSize size)
{
    if (!rt) return VK_ERROR_INITIALIZATION_FAILED;
    if (size == 0) return VK_SUCCESS;
    if (!cmd || !queue || !host || !dev) return VK_ERROR_INITIALIZATION_FAILED;

    /* transient host-visible staging buffer */
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkResult r = vkr_malloc(rt, size,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            &staging, &staging_mem);
    if (r != VK_SUCCESS) return r;

    void *dst = vkr_host_ptr(rt, staging, staging_mem, 0);
    if (!dst) {
        vkr_free(rt, staging, staging_mem);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    memcpy(dst, host, (size_t)size);

    r = vkr_copy_and_sync(rt, cmd, queue, staging, 0, dev, offset, size);
    vkr_free(rt, staging, staging_mem);
    return r;
}

VkResult vkr_download(VkRuntime *rt, VkCommandBuffer cmd, VkQueue queue,
                      VkBuffer dev, VkDeviceSize offset,
                      void *host, VkDeviceSize size)
{
    if (!rt) return VK_ERROR_INITIALIZATION_FAILED;
    if (size == 0) return VK_SUCCESS;
    if (!cmd || !queue || !host || !dev) return VK_ERROR_INITIALIZATION_FAILED;

    /* transient host-visible staging buffer */
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkResult r = vkr_malloc(rt, size,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            &staging, &staging_mem);
    if (r != VK_SUCCESS) return r;

    r = vkr_copy_and_sync(rt, cmd, queue, dev, offset, staging, 0, size);
    if (r == VK_SUCCESS) {
        void *src = vkr_host_ptr(rt, staging, staging_mem, 0);
        if (!src) {
            vkr_free(rt, staging, staging_mem);
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        memcpy(host, src, (size_t)size);
    }
    vkr_free(rt, staging, staging_mem);
    return r;
}

/* ===========================================================================
 * Public API: pool helpers
 * ========================================================================== */

VkResult vkr_create_command_pool(VkRuntime *rt, uint32_t queue_family,
                                 VkCommandPool *pPool)
{
    if (!rt || !pPool) return VK_ERROR_INITIALIZATION_FAILED;

    VkCommandPoolCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queue_family;
    return vkCreateCommandPool(rt->device, &cpci, NULL, pPool);
}

VkResult vkr_create_descriptor_pool(VkDevice device, uint32_t max_sets,
                                    uint32_t ssbo_count,
                                    VkDescriptorPool *pPool)
{
    if (!device || !pPool) return VK_ERROR_INITIALIZATION_FAILED;

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = ssbo_count;

    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = max_sets;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    return vkCreateDescriptorPool(device, &dpci, NULL, pPool);
}

VkResult vkr_create_pipeline_layout(VkDevice device,
                                    VkDescriptorSetLayout set_layout,
                                    uint32_t push_range_count,
                                    const VkPushConstantRange *ranges,
                                    VkPipelineLayout *pLayout)
{
    if (!device || !pLayout) return VK_ERROR_INITIALIZATION_FAILED;
    if (push_range_count > 0 && !ranges) return VK_ERROR_INITIALIZATION_FAILED;

    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = push_range_count;
    plci.pPushConstantRanges = ranges;
    return vkCreatePipelineLayout(device, &plci, NULL, pLayout);
}

VkResult vkr_create_pipeline_cache(VkDevice device, VkPipelineCache *pCache)
{
    if (!device || !pCache) return VK_ERROR_INITIALIZATION_FAILED;

    VkPipelineCacheCreateInfo pcci;
    memset(&pcci, 0, sizeof(pcci));
    pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    return vkCreatePipelineCache(device, &pcci, NULL, pCache);
}

void vkr_wait_idle(VkRuntime *rt)
{
    if (!rt || !rt->queue) return;
    vkQueueWaitIdle(rt->queue);
}
