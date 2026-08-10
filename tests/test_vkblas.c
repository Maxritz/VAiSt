/**
 * \file test_vkblas.c
 * \brief Public-API test harness for the VKBLAS library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkBLASContext via vkblas_create_context(), records f32/f16/f64 GEMM
 * dispatches into one command buffer, submits once, and validates each output
 * D against a CPU-computed reference (column-major BLAS layout).
 *
 * - sgemm (f32): tolerance 1e-3.
 * - hgemm (f16): f16 storage with f32 accumulation; reference rounded to f16;
 *   compared as f32 with a tolerance that absorbs f16 ULP quantization.
 * - dgemm (f64): double-precision reference, tolerance 1e-10.
 *
 * The device enables shaderInt64, shaderFloat16, shaderFloat64,
 * storageBuffer16BitAccess, uniformAndStorageBuffer16BitAccess,
 * scalarBlockLayout, and (when supported) VK_KHR_cooperative_matrix so the
 * real cooperative-matrix path can be exercised. Tests for a missing feature
 * are reported as SKIP, not failure.
 *
 * Build (Windows):
 *   gcc -std=c99 -IC:/VulkanSDK/1.4.357.0/Include -IF:/VAiT/include \
 *       F:/VAiT/tests/test_vkblas.c vkblas.o C:/VulkanSDK/1.4.357.0/Lib/vulkan-1.lib \
 *       -o test_vkblas.exe
 *
 * Exit status: 0 when every run test passes, 1 on any real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkblas/vkblas.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_GEMM_M  8    /**< Rows of op(A) and D.                           */
#define TEST_GEMM_N  8    /**< Cols of op(B) and D.                           */
#define TEST_GEMM_K  8    /**< Cols of op(A), rows of op(B).                  */
#define TEST_GEMM_LDA 8   /**< Leading dimension of A (>= m).                 */
#define TEST_GEMM_LDB 8   /**< Leading dimension of B (>= k).                 */
#define TEST_GEMM_LDD 8   /**< Leading dimension of D (>= m).                 */
#define TEST_BATCH 2      /**< Batch count for the strided-batched tests.     */
#define TEST_STAGING_SIZE ((VkDeviceSize)(1u << 20))  /**< 1 MiB host buffer. */
#define TEST_F32_TOLERANCE 1e-3f  /**< GEMM f32 comparison tolerance.         */
#define TEST_F64_TOLERANCE 1e-10  /**< GEMM f64 comparison tolerance.         */
#define TEST_F16_TOLERANCE 1.0f   /**< Half-ULP-safe f16 comparison tol.      */
#define TEST_BF16_TOLERANCE 1e-2f /**< bf16 truncation-safe comparison tol.   */

/* ===========================================================================
 * f16 <-> f32 helpers (test-side; not part of the library)
 * ========================================================================== */

static float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x03FFu;
    uint32_t fbits;

    if (exp == 0) {
        if (man == 0) {
            fbits = sign;
        } else {
            int e = 1;
            while ((man & 0x0400u) == 0u) { man <<= 1; --e; }
            man &= 0x03FFu;
            fbits = sign | ((uint32_t)(e + 112) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        fbits = sign | 0x7F800000u | (man << 13);
    } else {
        fbits = sign | ((exp + 112u) << 23) | (man << 13);
    }

    float r;
    memcpy(&r, &fbits, sizeof(r));
    return r;
}

static uint16_t f32_to_f16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u & 0x80000000u) >> 16;
    uint32_t exp  = (u & 0x7F800000u) >> 23;
    uint32_t man  = u & 0x007FFFFFu;

    if (exp == 0xFF) {
        return (uint16_t)(sign | 0x7C00u | ((man >> 13) & 0x03FFu));
    }
    if (exp == 0) {
        return (uint16_t)sign;
    }
    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (e <= 0) {
        man |= 0x00800000u;
        uint32_t shift = (uint32_t)(1 - e);
        if (shift > 24) return (uint16_t)sign;
        return (uint16_t)(sign | (man >> shift));
    }
    uint32_t h = sign | ((uint32_t)e << 10) | (man >> 13);
    uint32_t rem = man & 0x00001FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (man & 0x00002000u)))
        h += 1;
    return (uint16_t)h;
}

/* ===========================================================================
 * bf16 <-> f32 helpers (test-side; not part of the library)
 * ========================================================================== */

/* bfloat16 = top 16 bits of an f32. */
static uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    /* round-to-nearest-even on bit 15 before truncating (matches hardware
       bf16 conversion semantics closely enough for the 1e-2 tolerance). */
    uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
    u += rounding_bias;
    return (uint16_t)(u >> 16);
}

static float bf16_to_f32(uint16_t b)
{
    uint32_t fbits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &fbits, sizeof(f));
    return f;
}

/* ===========================================================================
 * Harness state
 * ========================================================================== */

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkDeviceMemory mem;
    VkBuffer staging;
    void *mapped;
    VkDeviceSize align;
    VkDeviceSize cursor;
    VkFence fence;
    uint32_t subgroup_size;
    VkBool32 test_f16;
    VkBool32 test_f64;
    VkBool32 test_bf16;
    VkBLASContext *blas_ctx;
} harness_t;

/** Describes the buffers + regions backing one GEMM test case. */
typedef struct {
    VkBuffer A;
    VkBuffer B;
    VkBuffer D;
    VkDeviceSize off_A;
    VkDeviceSize off_B;
    VkDeviceSize off_D;
    VkDeviceSize off_readback;
} gemm_buffers_t;

/** Describes the buffers + regions backing one fused quantized-GEMM case. */
typedef struct {
    VkBuffer Wq;
    VkBuffer x;
    VkBuffer y;
    VkDeviceSize off_Wq;
    VkDeviceSize off_x;
    VkDeviceSize off_y;
    VkDeviceSize off_readback;
} qgemm_buffers_t;

/** FP16-output qgemm case: a distinct y16 buffer (fp16 bits) + readback.
    Reuses the matching f32 case's Wq/x buffers; only the output buffer is
    separate because the f32 and f16 dispatches all run in one command buffer. */
typedef struct {
    VkBuffer y16;
    VkDeviceSize off_y16;
    VkDeviceSize off_readback16;
} qgemm_f16_buffers_t;

/* ===========================================================================
 * Bootstrap helpers
 * ========================================================================== */

static VkResult create_instance(const char *app_name, VkInstance *out_instance)
{
    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = app_name;
    app_info.applicationVersion = 1;
    app_info.pEngineName = "vait";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = NULL;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = NULL;
    return vkCreateInstance(&create_info, NULL, out_instance);
}

static VkResult find_physical_device(VkInstance instance,
                                     VkPhysicalDevice *out_physical_device)
{
    uint32_t count = 0;
    VkResult r = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (r != VK_SUCCESS || count == 0) return VK_ERROR_INITIALIZATION_FAILED;

    VkPhysicalDevice *devices =
        (VkPhysicalDevice *)malloc(count * sizeof(VkPhysicalDevice));
    if (!devices) return VK_ERROR_OUT_OF_HOST_MEMORY;

    r = vkEnumeratePhysicalDevices(instance, &count, devices);
    if (r == VK_SUCCESS) *out_physical_device = devices[0];
    free(devices);
    return r;
}

static VkBool32 query_shader_int64(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features2.features.shaderInt64;
}

static uint32_t query_subgroup_size(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceSubgroupProperties subgroup;
    memset(&subgroup, 0, sizeof(subgroup));
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2;
    memset(&props2, 0, sizeof(props2));
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);
    return subgroup.subgroupSize;
}

static VkBool32 queue_family_supports_compute(VkPhysicalDevice physical_device,
                                              uint32_t family)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    if (count == 0) return VK_FALSE;

    VkQueueFamilyProperties *props =
        (VkQueueFamilyProperties *)malloc(count * sizeof(*props));
    if (!props) return VK_FALSE;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props);

    VkBool32 supported = VK_FALSE;
    if (family < count) {
        supported =
            (props[family].queueFlags & VK_QUEUE_COMPUTE_BIT) ? VK_TRUE : VK_FALSE;
    }
    free(props);
    return supported;
}

static VkBool32 device_extension_available(VkPhysicalDevice physical_device,
                                           const char *name)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, NULL);
    if (count == 0) return VK_FALSE;

    VkExtensionProperties *props =
        (VkExtensionProperties *)malloc(count * sizeof(*props));
    if (!props) return VK_FALSE;
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, props);

    VkBool32 found = VK_FALSE;
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(props[i].extensionName, name) == 0) { found = VK_TRUE; break; }
    }
    free(props);
    return found;
}

/**
 * \brief Create a logical device on queue family 0.
 *
 * Enables the features the VKBLAS GEMM family needs: shaderInt64 (all shaders),
 * shaderFloat16 + storageBuffer16BitAccess (hgemm), shaderFloat64 (dgemm),
 * scalarBlockLayout (tight f16 SSBO packing), and VK_KHR_cooperative_matrix
 * (real coop-matrix path) when the device supports them.
 *
 * \param physical_device Device to create from.
 * \param out_device Receives the logical device handle.
 * \param out_test_f16 Set VK_TRUE when hgemm features were enabled.
 * \param out_test_f64 Set VK_TRUE when dgemm features were enabled.
 * \retval VK_SUCCESS
 */
static VkResult create_device(VkPhysicalDevice physical_device,
                              VkDevice *out_device,
                              VkBool32 *out_test_f16,
                              VkBool32 *out_test_f64,
                              VkBool32 *out_test_bf16)
{
    float priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    /* Query which features the device actually supports. */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_supported;
    VkPhysicalDeviceVulkan12Features vk12_supported;
    VkPhysicalDeviceVulkan11Features vk11_supported;
    VkPhysicalDevice8BitStorageFeatures eightbit_supported;
    memset(&coop_supported, 0, sizeof(coop_supported));
    coop_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk12_supported, 0, sizeof(vk12_supported));
    vk12_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk11_supported, 0, sizeof(vk11_supported));
    vk11_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    memset(&eightbit_supported, 0, sizeof(eightbit_supported));
    eightbit_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;

    VkPhysicalDeviceFeatures2 supported;
    memset(&supported, 0, sizeof(supported));
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    eightbit_supported.pNext = &coop_supported;
    vk12_supported.pNext = &eightbit_supported;
    vk11_supported.pNext = &vk12_supported;
    supported.pNext = &vk11_supported;
    vkGetPhysicalDeviceFeatures2(physical_device, &supported);

    /* Build the enable chain (base -> vk11 -> vk12 -> 8bit -> coop matrix). */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_enable;
    VkPhysicalDeviceVulkan12Features vk12_enable;
    VkPhysicalDeviceVulkan11Features vk11_enable;
    VkPhysicalDevice8BitStorageFeatures eightbit_enable;
    VkPhysicalDeviceFeatures2 features2;
    memset(&coop_enable, 0, sizeof(coop_enable));
    coop_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk12_enable, 0, sizeof(vk12_enable));
    vk12_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk11_enable, 0, sizeof(vk11_enable));
    vk11_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    memset(&eightbit_enable, 0, sizeof(eightbit_enable));
    eightbit_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    features2.features.shaderInt64 = VK_TRUE;
    features2.features.shaderFloat64 = supported.features.shaderFloat64;
    features2.features.shaderInt16 = supported.features.shaderInt16;
    vk11_enable.storageBuffer16BitAccess = vk11_supported.storageBuffer16BitAccess;
    vk11_enable.uniformAndStorageBuffer16BitAccess =
        vk11_supported.uniformAndStorageBuffer16BitAccess;
    vk12_enable.shaderFloat16 = vk12_supported.shaderFloat16;
    vk12_enable.shaderInt8 = vk12_supported.shaderInt8;
    vk12_enable.scalarBlockLayout = vk12_supported.scalarBlockLayout;
    eightbit_enable.storageBuffer8BitAccess = eightbit_supported.storageBuffer8BitAccess;
    eightbit_enable.uniformAndStorageBuffer8BitAccess =
        eightbit_supported.uniformAndStorageBuffer8BitAccess;
    if (coop_supported.cooperativeMatrix) {
        coop_enable.cooperativeMatrix = VK_TRUE;
        coop_enable.cooperativeMatrixRobustBufferAccess =
            coop_supported.cooperativeMatrixRobustBufferAccess;
    }

    void *pNext = NULL;
    if (coop_supported.cooperativeMatrix) {
        coop_enable.pNext = pNext;
        pNext = &coop_enable;
    }
    eightbit_enable.pNext = pNext;
    pNext = &eightbit_enable;
    vk12_enable.pNext = pNext;
    pNext = &vk12_enable;
    vk11_enable.pNext = pNext;
    pNext = &vk11_enable;
    features2.pNext = pNext;

    /* Device extensions. */
    const char *extensions[2];
    uint32_t ext_count = 0;
    if (coop_supported.cooperativeMatrix &&
        device_extension_available(physical_device,
                                   VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        extensions[ext_count++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }

    VkDeviceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &features2;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount = ext_count;
    create_info.ppEnabledExtensionNames = extensions;
    create_info.pEnabledFeatures = NULL;

    VkResult r = vkCreateDevice(physical_device, &create_info, NULL, out_device);
    if (r == VK_SUCCESS) {
        *out_test_f16 = vk12_supported.shaderFloat16 &&
                        vk11_supported.storageBuffer16BitAccess &&
                        vk11_supported.uniformAndStorageBuffer16BitAccess;
        *out_test_f64 = supported.features.shaderFloat64;
        *out_test_bf16 = supported.features.shaderInt16 &&
                         vk11_supported.storageBuffer16BitAccess;
    }
    return r;
}

static VkResult create_command_pool_and_buffer(VkDevice device,
                                               VkCommandPool *out_pool,
                                               VkCommandBuffer *out_cmd)
{
    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = 0;

    VkResult r = vkCreateCommandPool(device, &pool_info, NULL, out_pool);
    if (r != VK_SUCCESS) return r;

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = *out_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(device, &alloc_info, out_cmd);
    if (r != VK_SUCCESS) {
        vkDestroyCommandPool(device, *out_pool, NULL);
        *out_pool = VK_NULL_HANDLE;
    }
    return r;
}

static VkResult create_sub_buffer(VkDevice device, VkDeviceMemory mem,
                                  VkDeviceSize offset, VkDeviceSize size,
                                  VkBuffer *out_buffer)
{
    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &buffer_info, NULL, out_buffer);
    if (r != VK_SUCCESS) return r;
    return vkBindBufferMemory(device, *out_buffer, mem, offset);
}

static VkResult allocate_staging_memory(VkPhysicalDevice physical_device,
                                        VkDevice device, VkDeviceSize size,
                                        VkDeviceMemory *out_memory,
                                        VkBuffer *out_staging,
                                        VkDeviceSize *out_align)
{
    VkBufferCreateInfo probe_info;
    memset(&probe_info, 0, sizeof(probe_info));
    probe_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    probe_info.size = size;
    probe_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer probe = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(device, &probe_info, NULL, &probe);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, probe, &requirements);
    vkDestroyBuffer(device, probe, NULL);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    uint32_t memory_index = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((requirements.memoryTypeBits & (1u << i)) == 0u) continue;
        VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
            memory_index = i;
            break;
        }
    }
    if (memory_index == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = memory_index;

    r = vkAllocateMemory(device, &alloc_info, NULL, out_memory);
    if (r != VK_SUCCESS) return r;

    r = create_sub_buffer(device, *out_memory, 0, size, out_staging);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device, *out_memory, NULL);
        *out_memory = VK_NULL_HANDLE;
        return r;
    }

    *out_align = requirements.alignment;
    return VK_SUCCESS;
}

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize align)
{
    return (value + align - 1) & ~(align - 1);
}

static VkDeviceSize take_region(VkDeviceSize *cursor, VkDeviceSize align,
                                VkDeviceSize size)
{
    VkDeviceSize offset = align_up(*cursor, align);
    *cursor = offset + size;
    return offset;
}

/* ===========================================================================
 * Command recording helpers
 * ========================================================================== */

static void record_compute_to_transfer_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}

static void record_copy_readback(VkCommandBuffer cmd, VkBuffer src,
                                 VkBuffer dst, VkDeviceSize dst_offset,
                                 VkDeviceSize size)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = 0;
    region.dstOffset = dst_offset;
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

static int record_dispatch(VkResult result, const char *name)
{
    if (result == VK_SUCCESS) {
        printf("  %-18s : recorded\n", name);
        return 1;
    }
    printf("  %-18s : FAIL (record, VkResult=%d)\n", name, (int)result);
    return 0;
}

/* ===========================================================================
 * Reference computation + comparison
 * ========================================================================== */

/**
 * \brief Compute the CPU reference D = alpha * A * B + beta * C (f32 path).
 * Column-major storage. Accumulates in double for the reference.
 */
static void compute_reference_f32(const float *A, int32_t lda,
                                  const float *B, int32_t ldb,
                                  float alpha, float beta,
                                  float *D, int32_t ldd,
                                  int32_t m, int32_t n, int32_t k)
{
    for (int32_t row = 0; row < m; row++) {
        for (int32_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (int32_t t = 0; t < k; t++) {
                acc += (double)A[row + t * lda] * (double)B[t + c * ldb];
            }
            D[row + c * ldd] = (float)(alpha * acc + beta * 0.0);
        }
    }
}

static int check_output_f32(const char *name, const void *mapped,
                            VkDeviceSize off_readback, const float *expected,
                            uint32_t count, float tolerance)
{
    const float *got = (const float *)((const char *)mapped + off_readback);
    int pass = 1;
    uint32_t mismatches = 0;

    for (uint32_t i = 0; i < count; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.6f expected %.6f (diff %.3e)\n",
                       i, got[i], expected[i], diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-18s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

static int check_output_f64(const char *name, const void *mapped,
                            VkDeviceSize off_readback, const double *expected,
                            uint32_t count, double tolerance)
{
    const double *got = (const double *)((const char *)mapped + off_readback);
    int pass = 1;
    uint32_t mismatches = 0;

    for (uint32_t i = 0; i < count; i++) {
        double diff = fabs(got[i] - expected[i]);
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.15g expected %.15g (diff %.3g)\n",
                       i, got[i], expected[i], diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-18s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/**
 * \brief f16 check: readback is uint16_t f16; expected is f16 rounded from the
 * double reference. Both are converted to f32 and compared with a tolerance
 * that absorbs f16 ULP quantization.
 */
static int check_output_f16(const char *name, const void *mapped,
                            VkDeviceSize off_readback,
                            const uint16_t *expected, uint32_t count,
                            float tolerance)
{
    const uint16_t *got = (const uint16_t *)((const char *)mapped + off_readback);
    int pass = 1;
    uint32_t mismatches = 0;

    for (uint32_t i = 0; i < count; i++) {
        float g = f16_to_f32(got[i]);
        float e = f16_to_f32(expected[i]);
        float diff = fabsf(g - e);
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.4f expected %.4f (diff %.3e)\n",
                       i, g, e, diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-18s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/**
 * \brief bf16 check: readback is uint16_t bf16; expected is bf16 rounded from
 * the double reference. Both are converted to f32 and compared with the
 * TEST_BF16_TOLERANCE (bf16 truncation loses ~8 mantissa bits; 1e-2 absorbs
 * the resulting quantization at the small magnitudes used here).
 */
static int check_output_bf16(const char *name, const void *mapped,
                             VkDeviceSize off_readback,
                             const uint16_t *expected, uint32_t count,
                             float tolerance)
{
    const uint16_t *got = (const uint16_t *)((const char *)mapped + off_readback);
    int pass = 1;
    uint32_t mismatches = 0;

    for (uint32_t i = 0; i < count; i++) {
        float g = bf16_to_f32(got[i]);
        float e = bf16_to_f32(expected[i]);
        float diff = fabsf(g - e);
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.5f expected %.5f (diff %.3e)\n",
                       i, g, e, diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-18s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/* ===========================================================================
 * Fused quantized-GEMM helpers (test-side; mirror the vkquant/vkblas formats)
 * ========================================================================== */

/** Quantize `num_blocks` blocks of 32 f32 values into Q8_0 (36 B/block).
    d = amax/127, q = round(x/d), clamped to [-127, 127]. */
static void quantize_q8_0(const float *src, int32_t num_blocks, uint8_t *dst)
{
    for (int32_t b = 0; b < num_blocks; ++b) {
        const float *xs = src + 32 * b;
        uint8_t *blk = dst + 36 * b;

        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float a = fabsf(xs[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        memcpy(blk, &d, sizeof(d));  /* f32 scale at bytes 0..3 (LE host) */

        int8_t *qs = (int8_t *)(blk + 4);
        for (int i = 0; i < 32; ++i) {
            int q = (d > 0.0f) ? (int)lroundf(xs[i] / d) : 0;
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[i] = (int8_t)q;
        }
    }
}

/** Dequantize Q8_0 element `idx` (0..31) of block `blk` to f32 (reference). */
static float dequant_q8_0(const uint8_t *blk, int idx)
{
    float d;
    memcpy(&d, blk, sizeof(d));
    return d * (float)((const int8_t *)(blk + 4))[idx];
}

/* ggml get_scale_min_k4: decode the 6-bit scale/min pair for 32-group j. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Port of ggml make_qkx1_quants: least-squares scale + min fit for a group.
   Returns scale s, sets *the_min = -min, with x ~= s*L - the_min. */
static float make_qkx1_quants(int n, int nmax, const float *x,
                              uint8_t *L, float *the_min,
                              int ntry, float alpha)
{
    float min = x[0], max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.0f;
    }
    if (min > 0) min = 0;
    float iscale = (float)nmax / (max - min);
    float scale = 1.0f / iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0; int did_change = 0;
        for (int i = 0; i < n; ++i) {
            int l = (int)lroundf(iscale * (x[i] - min));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            if (l != L[i]) { L[i] = (uint8_t)l; did_change = 1; }
            sumlx += (x[i] - min) * (float)l;
            suml2 += l * l;
        }
        scale = suml2 ? sumlx / (float)suml2 : 0.0f;
        float sum = 0;
        for (int i = 0; i < n; ++i) sum += x[i] - scale * (float)L[i];
        min = alpha * min + (1.0f - alpha) * sum / (float)n;
        if (min > 0) min = 0;
        iscale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

/** Quantize 256 f32 values into one Q4_K block (144 B, ggml packing).
    d, dmin as f16; scales[12] holds 6-bit scale/min pairs; qs[128] nibbles.
    Dequant must round-trip through dequant_q4k_f32 (and the shader). */
static void quantize_q4k_f32(const float *src, uint8_t *dst)
{
    enum { QK = 256 };
    uint8_t L[QK];
    float   mins[8], scales[8];

    float max_scale = 0, max_min = 0;
    for (int j = 0; j < 8; ++j) {
        scales[j] = make_qkx1_quants(32, 15, src + 32 * j, L + 32 * j,
                                     &mins[j], 9, 0.5f);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min)     max_min = mins[j];
    }

    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min   = max_min   > 0.0f ? 63.0f / max_min   : 0.0f;

    uint8_t *sc = dst + 4;  /* scales[12] at bytes 4..15 */
    memset(sc, 0, 12);
    for (int j = 0; j < 8; ++j) {
        uint8_t ls = (uint8_t)(int)(inv_scale * scales[j] + 0.5f);
        uint8_t lm = (uint8_t)(int)(inv_min   * mins[j]   + 0.5f);
        if (ls > 63) ls = 63;
        if (lm > 63) lm = 63;
        if (j < 4) {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            sc[j - 4] |= (uint8_t)(((ls >> 4) & 0x3) << 6);
            sc[j]     |= (uint8_t)(((lm >> 4) & 0x3) << 6);
        }
    }

    uint16_t d16  = f32_to_f16(max_scale / 63.0f);
    uint16_t dm16 = f32_to_f16(max_min   / 63.0f);
    dst[0] = (uint8_t)(d16 & 0xFF);  dst[1] = (uint8_t)(d16 >> 8);
    dst[2] = (uint8_t)(dm16 & 0xFF); dst[3] = (uint8_t)(dm16 >> 8);

    /* Re-derive nibbles from the DECODED (sc, m) so the dequant round-trips. */
    for (int j = 0; j < 8; ++j) {
        uint8_t scs, mns;
        get_scale_min_k4(j, sc, &scs, &mns);
        float d = f16_to_f32(d16) * (float)scs;
        if (d == 0.0f) continue;
        float dm = f16_to_f32(dm16) * (float)mns;
        for (int ii = 0; ii < 32; ++ii) {
            int l = (int)lroundf((src[32 * j + ii] + dm) / d);
            if (l < 0) l = 0;
            if (l > 15) l = 15;
            L[32 * j + ii] = (uint8_t)l;
        }
    }

    /* Pack qs[128]: super-block s -> qs[32s..32s+31], low nibble first. */
    uint8_t *q = dst + 16;
    for (int j = 0; j < QK; j += 64) {
        for (int l = 0; l < 32; ++l)
            q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
        q += 32;
    }
}

/** Dequantize Q4_K element `idx` (0..255) of block `blk` to f32 (reference).
    Mirrors the dequant math in shaders/vkquant/baseline/dequant_q4k_f32.comp. */
static float dequant_q4k_f32(const uint8_t *blk, int idx)
{
    uint16_t d16  = (uint16_t)(blk[0] | (blk[1] << 8));
    uint16_t dm16 = (uint16_t)(blk[2] | (blk[3] << 8));
    float d  = f16_to_f32(d16);
    float dm = f16_to_f32(dm16);

    uint32_t super = (uint32_t)idx >> 6;
    uint32_t off   = (uint32_t)idx & 63u;
    uint32_t hi    = off >> 5;
    uint32_t l     = off & 31u;
    uint32_t is    = super * 2u + hi;

    uint8_t scs, mns;
    get_scale_min_k4((int)is, blk + 4, &scs, &mns);

    uint8_t qbyte = blk[16 + super * 32 + l];
    uint32_t nib  = (hi != 0u) ? (qbyte >> 4u) : (qbyte & 0xFu);

    return d * (float)scs * (float)nib - dm * (float)mns;
}

/* ===========================================================================
 * Reference quantizers/dequantizers for the new fused qgemm formats.
 *
 * Every quantizer produces a block in the canonical ggml (or VAIT) byte
 * layout that the matching dequantizer AND the shader decode identically, so
 * the CPU reference and the GPU qgemm consume the same Wq bytes.
 * ========================================================================== */

/* ── Q4_0: 20 B/block of 32 elems; f32 d + 16 packed nibbles ─────────────── */
static void quantize_q4_0(const float *src, int32_t num_blocks, uint8_t *dst)
{
    for (int32_t b = 0; b < num_blocks; ++b) {
        const float *xs = src + 32 * b;
        uint8_t *blk = dst + 20 * b;

        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float a = fabsf(xs[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 8.0f;  /* nib-8 in [-8,7] */
        memcpy(blk, &d, sizeof(d));

        for (int i = 0; i < 32; ++i) {
            int nib = (d > 0.0f) ? (int)lroundf(xs[i] / d) + 8 : 8;
            if (nib > 15) nib = 15;
            if (nib < 0)  nib = 0;
            if ((i & 1) == 0) blk[4 + (i >> 1)]  = (uint8_t)nib;
            else              blk[4 + (i >> 1)] |= (uint8_t)(nib << 4);
        }
    }
}

static float dequant_q4_0(const uint8_t *blk, int idx)
{
    float d;
    memcpy(&d, blk, sizeof(d));
    uint8_t xi = blk[4 + (idx >> 1)];
    uint32_t nib = (idx & 1) ? (xi >> 4) : (xi & 0xFu);
    return d * (float)((int)nib - 8);
}

/* ── Q5_K: 176 B/block of 256 elems; ggml block_q5_K ─────────────────────── */
static void quantize_q5k_f32(const float *src, uint8_t *dst)
{
    enum { QK = 256 };
    uint8_t L[QK];
    float   mins[8], scales[8];

    float max_scale = 0, max_min = 0;
    for (int j = 0; j < 8; ++j) {
        scales[j] = make_qkx1_quants(32, 31, src + 32 * j, L + 32 * j,
                                     &mins[j], 9, 0.5f);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min)     max_min = mins[j];
    }

    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min   = max_min   > 0.0f ? 63.0f / max_min   : 0.0f;

    uint8_t *sc = dst + 4;
    memset(sc, 0, 12);
    for (int j = 0; j < 8; ++j) {
        uint8_t ls = (uint8_t)(int)(inv_scale * scales[j] + 0.5f);
        uint8_t lm = (uint8_t)(int)(inv_min   * mins[j]   + 0.5f);
        if (ls > 63) ls = 63;
        if (lm > 63) lm = 63;
        if (j < 4) {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            sc[j - 4] |= (uint8_t)(((ls >> 4) & 0x3) << 6);
            sc[j]     |= (uint8_t)(((lm >> 4) & 0x3) << 6);
        }
    }

    uint16_t d16  = f32_to_f16(max_scale / 63.0f);
    uint16_t dm16 = f32_to_f16(max_min   / 63.0f);
    dst[0] = (uint8_t)(d16 & 0xFF);  dst[1] = (uint8_t)(d16 >> 8);
    dst[2] = (uint8_t)(dm16 & 0xFF); dst[3] = (uint8_t)(dm16 >> 8);

    for (int j = 0; j < 8; ++j) {
        uint8_t scs, mns;
        get_scale_min_k4(j, sc, &scs, &mns);
        float d = f16_to_f32(d16) * (float)scs;
        if (d == 0.0f) continue;
        float dm = f16_to_f32(dm16) * (float)mns;
        for (int ii = 0; ii < 32; ++ii) {
            int l = (int)lroundf((src[32 * j + ii] + dm) / d);
            if (l < 0) l = 0;
            if (l > 31) l = 31;
            L[32 * j + ii] = (uint8_t)l;
        }
    }

    uint8_t *qh = dst + 16;
    uint8_t *ql = dst + 48;
    memset(qh, 0, 32);
    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j];
            if (l1 > 15) { l1 -= 16; qh[j] |= m1; }
            int l2 = L[n + j + 32];
            if (l2 > 15) { l2 -= 16; qh[j] |= m2; }
            ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 <<= 2; m2 <<= 2;
        ql += 32;
    }
}

static float dequant_q5k_f32(const uint8_t *blk, int idx)
{
    uint16_t d16  = (uint16_t)(blk[0] | (blk[1] << 8));
    uint16_t dm16 = (uint16_t)(blk[2] | (blk[3] << 8));
    float d  = f16_to_f32(d16);
    float dm = f16_to_f32(dm16);

    uint32_t super = (uint32_t)idx >> 6;
    uint32_t off   = (uint32_t)idx & 63u;
    uint32_t hi    = off >> 5;
    uint32_t l     = off & 31u;
    uint32_t is    = super * 2u + hi;

    uint8_t scs, mns;
    get_scale_min_k4((int)is, blk + 4, &scs, &mns);

    uint8_t  qbyte = blk[48 + super * 32 + l];
    uint32_t nib   = (hi != 0u) ? (qbyte >> 4u) : (qbyte & 0xFu);
    uint32_t hbit  = (blk[16 + l] >> (2u * super + hi)) & 1u;
    uint32_t level = nib + (hbit ? 16u : 0u);

    return d * (float)scs * (float)level - dm * (float)mns;
}

/* ── Q6_K: 210 B/block of 256 elems; ggml block_q6_K ─────────────────────── */
static void quantize_q6k_f32(const float *src, uint8_t *dst)
{
    enum { QK = 256 };
    int8_t  L[QK];
    float   scales[16];

    float max_scale = 0.0f;
    for (int ib = 0; ib < 16; ++ib) {
        float mx = 0.0f;
        for (int j = 0; j < 16; ++j) {
            float a = fabsf(src[16 * ib + j]);
            if (a > mx) mx = a;
        }
        scales[ib] = mx / 31.0f;
        if (scales[ib] > max_scale) max_scale = scales[ib];
    }

    float d = (max_scale > 0.0f) ? max_scale / 127.0f : 0.0f;
    uint16_t d16 = f32_to_f16(d);
    dst[208] = (uint8_t)(d16 & 0xFF); dst[209] = (uint8_t)(d16 >> 8);

    int8_t *sc = (int8_t *)(dst + 192);
    for (int ib = 0; ib < 16; ++ib) {
        int s = (d > 0.0f) ? (int)lroundf(scales[ib] / d) : 0;
        if (s > 127)  s = 127;
        if (s < -127) s = -127;
        sc[ib] = (int8_t)s;
    }

    for (int j = 0; j < QK; ++j) {
        int g = j >> 4;
        float de = f16_to_f32(d16) * (float)sc[g];
        int l;
        if (de == 0.0f) l = 0;
        else {
            l = (int)lroundf(src[j] / de);
            if (l < -32) l = -32;
            if (l > 31)  l = 31;
        }
        L[j] = (int8_t)(l + 32);
    }

    uint8_t *ql = dst;
    uint8_t *qh = dst + 128;
    memset(dst, 0, 192);  /* ql + qh */
    for (int j = 0; j < QK; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)(L[j + l +  0] & 0xF);
            const uint8_t q2 = (uint8_t)(L[j + l + 32] & 0xF);
            const uint8_t q3 = (uint8_t)(L[j + l + 64] & 0xF);
            const uint8_t q4 = (uint8_t)(L[j + l + 96] & 0xF);
            ql[l +  0] = (uint8_t)(q1 | (q3 << 4));
            ql[l + 32] = (uint8_t)(q2 | (q4 << 4));
            qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                              ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

static float dequant_q6k_f32(const uint8_t *blk, int idx)
{
    uint16_t d16 = (uint16_t)(blk[208] | (blk[209] << 8));
    float d = f16_to_f32(d16);

    uint32_t chunk = (uint32_t)idx >> 7;
    uint32_t sub   = ((uint32_t)idx >> 5) & 3u;
    uint32_t l     = (uint32_t)idx & 31u;
    uint32_t is    = l >> 4;

    uint32_t ql_off = chunk * 64u + l + (sub & 1u) * 32u;
    uint32_t qh_off = chunk * 32u + l;
    uint32_t sc_off = chunk * 8u + is + sub * 2u;

    uint8_t  ql = blk[ql_off];
    uint8_t  qh = blk[128 + qh_off];
    uint32_t ql4 = (sub == 0u || sub == 1u) ? (ql & 0xFu) : (ql >> 4u);
    uint32_t qh2 = (qh >> (sub * 2u)) & 3u;
    int level = (int)(ql4 | (qh2 << 4u)) - 32;
    int sc = (int)((int8_t)blk[192 + sc_off]);
    return d * (float)sc * (float)level;
}

/* ── Q3_K: 110 B/block of 256 elems; ggml block_q3_K ─────────────────────── */
static void quantize_q3k_f32(const float *src, uint8_t *dst)
{
    enum { QK = 256 };
    int8_t  L[QK];
    float   scales[16];

    float max_scale = 0.0f;
    for (int ib = 0; ib < 16; ++ib) {
        float mx = 0.0f;
        for (int j = 0; j < 16; ++j) {
            float a = fabsf(src[16 * ib + j]);
            if (a > mx) mx = a;
        }
        scales[ib] = mx / 3.0f;
        if (scales[ib] > max_scale) max_scale = scales[ib];
    }

    float d = (max_scale > 0.0f) ? max_scale / 95.0f : 0.0f;
    uint16_t d16 = f32_to_f16(d);
    dst[108] = (uint8_t)(d16 & 0xFF); dst[109] = (uint8_t)(d16 >> 8);

    int8_t sc8[16];
    for (int ib = 0; ib < 16; ++ib) {
        int s = (d > 0.0f) ? (int)lroundf(scales[ib] / d) + 32 : 32;
        if (s < 33)  s = 33;   /* keep the group multiplier >= 1 */
        if (s > 127) s = 127;
        sc8[ib] = (int8_t)s;
    }

    for (int j = 0; j < QK; ++j) {
        int g = j >> 4;
        float de = f16_to_f32(d16) * (float)(sc8[g] - 32);
        int l;
        if (de == 0.0f) l = 0;
        else {
            l = (int)lroundf(src[j] / de);
            if (l < -4) l = -4;
            if (l > 3)  l = 3;
        }
        L[j] = (int8_t)l;
    }

    /* Pack the 16 int8 scales into the 12 raw bytes s[0..11]. */
    uint8_t *s = dst + 96;
    memset(s, 0, 12);
    for (int is = 0; is < 16; ++is) {
        uint8_t v = (uint8_t)sc8[is];
        if (is < 4) {
            s[is]     |= (v & 0xF);
            s[8 + is] |= ((v >> 4) & 3);
        } else if (is < 8) {
            s[is]     |= (v & 0xF);
            s[is + 4] |= ((v >> 4) & 3) << 2;
        } else if (is < 12) {
            s[is - 8] |= (v & 0xF) << 4;
            s[is]     |= ((v >> 4) & 3) << 4;
        } else {
            s[is - 8] |= (v & 0xF) << 4;
            s[is - 4] |= ((v >> 4) & 3) << 6;
        }
    }

    /* Pack 2-bit levels into qs[64] and the sign/high bits into hmask[32]. */
    uint8_t *hm = dst;
    uint8_t *qs = dst + 32;
    memset(hm, 0, 32);
    memset(qs, 0, 64);
    for (int half = 0; half < 2; ++half) {
        for (int j = 0; j < 4; ++j) {
            int shift = 2 * j;
            int mbit  = j + half * 4;
            for (int hi = 0; hi < 2; ++hi) {
                for (int ll = 0; ll < 16; ++ll) {
                    int idx = half * 128 + j * 32 + hi * 16 + ll;
                    int lvl = L[idx];
                    int q2  = lvl & 3;
                    int q_off = half * 32 + ll + hi * 16;
                    int h_off = ll + hi * 16;
                    qs[q_off] |= (uint8_t)(q2 << shift);
                    if (lvl >= 0) hm[h_off] |= (uint8_t)(1u << mbit);
                }
            }
        }
    }
}

static float dequant_q3k_f32(const uint8_t *blk, int idx)
{
    uint16_t d16 = (uint16_t)(blk[108] | (blk[109] << 8));
    float d = f16_to_f32(d16);

    uint32_t half = (uint32_t)idx >> 7;
    uint32_t pos  = (uint32_t)idx & 127u;
    uint32_t j    = pos >> 5;
    uint32_t l    = pos & 31u;
    uint32_t hi   = l >> 4;
    uint32_t ll   = l & 15u;

    uint32_t is    = half * 8u + j * 2u + hi;
    uint32_t shift = 2u * j;
    uint32_t mbit  = j + (half ? 4u : 0u);
    uint32_t q_off = half * 32u + ll + (hi ? 16u : 0u);
    uint32_t h_off = ll + (hi ? 16u : 0u);

    uint32_t q2 = (blk[32 + q_off] >> shift) & 3u;
    uint32_t hb = (blk[h_off] >> mbit) & 1u;
    int level = (int)q2 - (hb ? 0 : 4);

    const uint8_t *s = blk + 96;
    uint32_t s0, s1;
    if (is < 4)       { s0 = s[is] & 0xFu;        s1 = s[8 + is] & 3u; }
    else if (is < 8)  { s0 = s[is] & 0xFu;        s1 = (s[is + 4] >> 2) & 3u; }
    else if (is < 12) { s0 = s[is - 8] >> 4u;     s1 = (s[is] >> 4) & 3u; }
    else              { s0 = s[is - 8] >> 4u;     s1 = (s[is - 4] >> 6) & 3u; }
    int sc = (int)(s0 | (s1 << 4u)) - 32;
    return d * (float)sc * (float)level;
}

/* ── IQ4_XS: 136 B/block of 256 elems; ggml block_iq4_xs ─────────────────── */
static const int iq4nl_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
      1,   13,  25,  38,  53,  69,  89, 113
};

static void quantize_iq4xs_f32(const float *src, uint8_t *dst)
{
    enum { QK = 256 };
    float dls[8];
    float max_dl = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        float mx = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float a = fabsf(src[32 * ib + j]);
            if (a > mx) mx = a;
        }
        dls[ib] = mx / 113.0f;   /* 113 = largest |iq4nl| value */
        if (dls[ib] > max_dl) max_dl = dls[ib];
    }

    float d = (max_dl > 0.0f) ? max_dl / 31.0f : 0.0f;
    uint16_t d16 = f32_to_f16(d);
    dst[0] = (uint8_t)(d16 & 0xFF); dst[1] = (uint8_t)(d16 >> 8);

    uint16_t scales_h = 0;
    uint8_t  scales_l[4];
    memset(scales_l, 0, 4);
    for (int ib = 0; ib < 8; ++ib) {
        int ls = (d > 0.0f) ? (int)lroundf(dls[ib] / d) + 32 : 32;
        if (ls < 32) ls = 32;
        if (ls > 63) ls = 63;
        scales_l[ib >> 1] |= (uint8_t)((ls & 0xF) << (4 * (ib & 1)));
        scales_h |= (uint16_t)(((ls >> 4) & 3) << (2 * ib));
    }
    dst[2] = (uint8_t)(scales_h & 0xFF); dst[3] = (uint8_t)(scales_h >> 8);
    dst[4] = scales_l[0]; dst[5] = scales_l[1];
    dst[6] = scales_l[2]; dst[7] = scales_l[3];

    uint8_t *qs = dst + 8;
    memset(qs, 0, 128);
    for (int ib = 0; ib < 8; ++ib) {
        int ls = (int)(scales_l[ib >> 1] >> (4 * (ib & 1)) & 0xF) |
                 ((int)((scales_h >> (2 * ib)) & 3) << 4);
        float dl = f16_to_f32(d16) * (float)(ls - 32);
        if (dl == 0.0f) continue;
        for (int j = 0; j < 32; ++j) {
            float x = src[32 * ib + j];
            int best = 0;
            float best_err = fabsf(x - dl * (float)iq4nl_lut[0]);
            for (int n = 1; n < 16; ++n) {
                float e = fabsf(x - dl * (float)iq4nl_lut[n]);
                if (e < best_err) { best_err = e; best = n; }
            }
            if (j < 16) qs[ib * 16 + j] |= (uint8_t)best;
            else        qs[ib * 16 + (j - 16)] |= (uint8_t)(best << 4);
        }
    }
}

static float dequant_iq4xs_f32(const uint8_t *blk, int idx)
{
    uint16_t d16 = (uint16_t)(blk[0] | (blk[1] << 8));
    float d = f16_to_f32(d16);
    uint16_t scales_h = (uint16_t)(blk[2] | (blk[3] << 8));

    uint32_t ib = (uint32_t)idx >> 5;
    uint32_t j  = (uint32_t)idx & 15u;
    uint32_t hi = ((uint32_t)idx >> 4) & 1u;

    uint32_t ls_l = (blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xFu;
    uint32_t ls_h = (scales_h >> (2 * ib)) & 3u;
    int ls = (int)(ls_l | (ls_h << 4));
    float dl = d * (float)(ls - 32);

    uint8_t qbyte = blk[8 + ib * 16 + j];
    uint32_t nib = (hi != 0u) ? (qbyte >> 4u) : (qbyte & 0xFu);
    return dl * (float)iq4nl_lut[nib];
}

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    harness_t h;
    memset(&h, 0, sizeof(h));
    h.mem = VK_NULL_HANDLE;
    h.staging = VK_NULL_HANDLE;
    h.fence = VK_NULL_HANDLE;

    const uint32_t m = TEST_GEMM_M;
    const uint32_t n = TEST_GEMM_N;
    const uint32_t k = TEST_GEMM_K;
    const uint32_t elem_count = m * n;

    /* f32 case buffers */
    gemm_buffers_t f32b;
    float A[TEST_GEMM_M * TEST_GEMM_K];
    float B[TEST_GEMM_K * TEST_GEMM_N];
    float D_expected[TEST_GEMM_M * TEST_GEMM_N];
    /* f16 case buffers */
    gemm_buffers_t f16b;
    uint16_t A16[TEST_GEMM_M * TEST_GEMM_K];
    uint16_t B16[TEST_GEMM_K * TEST_GEMM_N];
    uint16_t D16_expected[TEST_GEMM_M * TEST_GEMM_N];
    /* f64 case buffers */
    gemm_buffers_t f64b;
    double A64[TEST_GEMM_M * TEST_GEMM_K];
    double B64[TEST_GEMM_K * TEST_GEMM_N];
    double D64_expected[TEST_GEMM_M * TEST_GEMM_N];
    /* bf16 case buffers */
    gemm_buffers_t bf16b;
    uint16_t A16b[TEST_GEMM_M * TEST_GEMM_K];
    uint16_t B16b[TEST_GEMM_K * TEST_GEMM_N];
    uint16_t D16b_expected[TEST_GEMM_M * TEST_GEMM_N];
    /* strided-batched case buffers (TEST_BATCH batches per dtype) */
    gemm_buffers_t sb32b, sb16b, sb64b;
    float  SB32_expected[TEST_BATCH][TEST_GEMM_M * TEST_GEMM_N];
    uint16_t SB16_expected[TEST_BATCH][TEST_GEMM_M * TEST_GEMM_N];
    double SB64_expected[TEST_BATCH][TEST_GEMM_M * TEST_GEMM_N];
    /* gemm_ex buffers (reuse the single-gemm A/B; own D + readback) */
    gemm_buffers_t gex32b, gex16b, gex16bb;
    /* f64 alpha/beta packing check (non-trivial alpha/beta + C read) */
    gemm_buffers_t bt64b;
    double C64in[TEST_GEMM_M * TEST_GEMM_N];
    double Dbt_expected[TEST_GEMM_M * TEST_GEMM_N];
    /* fused quantized GEMM cases: Q8_0 (k=32, k=64), Q4_K (k=256), plus
       Q4_0 (k=32, k=64), Q5_K, Q6_K, Q3_K, IQ4_XS (k=256) */
    qgemm_buffers_t q8_32, q8_64, q4k;
    qgemm_buffers_t q40_32, q40_64, q5k, q6k, q3k, iq4xs;
    /* fp16-output qgemm cases (index mirrors qg_yexp[] / the f32 dispatch set). */
    qgemm_f16_buffers_t qg_f16[9];
    /* CPU references for the nine qgemm cases, filled in section 10b. */
    static float qg_yexp[9][8 * 8];

    memset(&f32b, 0, sizeof(f32b));
    memset(&f16b, 0, sizeof(f16b));
    memset(&f64b, 0, sizeof(f64b));
    memset(&bf16b, 0, sizeof(bf16b));
    memset(&sb32b, 0, sizeof(sb32b));
    memset(&sb16b, 0, sizeof(sb16b));
    memset(&sb64b, 0, sizeof(sb64b));
    memset(&gex32b, 0, sizeof(gex32b));
    memset(&gex16b, 0, sizeof(gex16b));
    memset(&gex16bb, 0, sizeof(gex16bb));
    memset(&bt64b, 0, sizeof(bt64b));
    memset(&q8_32, 0, sizeof(q8_32));
    memset(&q8_64, 0, sizeof(q8_64));
    memset(&q4k, 0, sizeof(q4k));
    memset(&q40_32, 0, sizeof(q40_32));
    memset(&q40_64, 0, sizeof(q40_64));
    memset(&q5k, 0, sizeof(q5k));
    memset(&q6k, 0, sizeof(q6k));
    memset(&q3k, 0, sizeof(q3k));
    memset(&iq4xs, 0, sizeof(iq4xs));
    memset(&qg_f16, 0, sizeof(qg_f16));

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkblas", &h.instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(h.instance, &h.physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkblas: SKIP (no physical device found)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 3. shaderInt64 gate ────────────────────────────────────────────── */
    if (query_shader_int64(h.physical_device) == VK_FALSE) {
        printf("test_vkblas: SKIP (shaderInt64 not supported)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }
    h.subgroup_size = query_subgroup_size(h.physical_device);

    /* ── 4. Queue family gate ───────────────────────────────────────────── */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkblas: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device, &h.test_f16, &h.test_f64,
                      &h.test_bf16);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(h.device, 0, 0, &h.queue);

    /* ── 6. Command pool + one command buffer ───────────────────────────── */
    r = create_command_pool_and_buffer(h.device, &h.cmd_pool, &h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. 1 MiB host-visible/host-coherent staging memory ─────────────── */
    r = allocate_staging_memory(h.physical_device, h.device, TEST_STAGING_SIZE,
                                &h.mem, &h.staging, &h.align);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: staging allocation failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkMapMemory(h.device, h.mem, 0, VK_WHOLE_SIZE, 0, &h.mapped);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkMapMemory failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. Context ─────────────────────────────────────────────────────── */
    r = vkblas_create_context(h.instance, h.physical_device, h.device, &h.blas_ctx);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkblas_create_context failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkblas: device ready (arch=%s, tier=%u, subgroup=%u, staging=%u)\n",
           vkblas_get_arch_name(h.blas_ctx), vkblas_get_arch_index(h.blas_ctx),
           (unsigned)h.subgroup_size, (unsigned)TEST_STAGING_SIZE);
    printf("test_vkblas: features f16=%s f64=%s bf16=%s\n",
           h.test_f16 ? "on" : "off", h.test_f64 ? "on" : "off",
           h.test_bf16 ? "on" : "off");

    /* ── 8b. Cooperative-matrix probe (DRIVER-GUARDED, default safe) ───────
     *
     * The real GL_KHR_cooperative_matrix GEMM (coopMatMulAdd) is compiled and
     * embedded (vkblas_spv_coopmatrix_gemm_f32) but is DORMANT by default: the
     * AMD Windows driver 26.7.1 hard-crashes the process (0xE06D7363) inside
     * vkCreateComputePipelines for any module containing coopMatMulAdd, even
     * though it advertises VK_KHR_cooperative_matrix rev 2. vkblas_init_
     * capabilities only activates the coopmatrix tier when VAIT_COOPMATRIX is
     * set. NEVER run this test with VAIT_COOPMATRIX set on a 26.7.1-era
     * driver; it will take down the process. All GEMMs below use the
     * shared-memory subgroup/baseline path. */
    if (getenv("VAIT_COOPMATRIX") != NULL) {
        printf("test_vkblas: WARNING VAIT_COOPMATRIX set — coop-matrix tier "
               "requested. Not exercised here (AMD 26.7.1 driver crash on "
               "coopMatMulAdd pipeline creation); using shared-memory path.\n");
    } else {
        printf("test_vkblas: coop-matrix tier dormant (VAIT_COOPMATRIX unset) — "
               "safe shared-memory path active.\n");
    }

    /* ── 9. Region + buffer layout for each dtype ───────────────────────── */
    /* f32: A (m*k), B (k*n), D (m*n), readback, expected */
    f32b.off_A = take_region(&h.cursor, h.align, m * k * sizeof(float));
    f32b.off_B = take_region(&h.cursor, h.align, k * n * sizeof(float));
    f32b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(float));
    f32b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(float));
    take_region(&h.cursor, h.align, m * n * sizeof(float)); /* expected */

    /* f16: same layout, 2-byte elements */
    if (h.test_f16) {
        f16b.off_A = take_region(&h.cursor, h.align, m * k * sizeof(uint16_t));
        f16b.off_B = take_region(&h.cursor, h.align, k * n * sizeof(uint16_t));
        f16b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
        f16b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
        take_region(&h.cursor, h.align, m * n * sizeof(uint16_t)); /* expected */
    }

    /* f64: same layout, 8-byte elements */
    if (h.test_f64) {
        f64b.off_A = take_region(&h.cursor, h.align, m * k * sizeof(double));
        f64b.off_B = take_region(&h.cursor, h.align, k * n * sizeof(double));
        f64b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(double));
        f64b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(double));
        take_region(&h.cursor, h.align, m * n * sizeof(double)); /* expected */
    }

    /* bf16: same layout, 2-byte elements (uint16 storage) */
    if (h.test_bf16) {
        bf16b.off_A = take_region(&h.cursor, h.align, m * k * sizeof(uint16_t));
        bf16b.off_B = take_region(&h.cursor, h.align, k * n * sizeof(uint16_t));
        bf16b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
        bf16b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
        take_region(&h.cursor, h.align, m * n * sizeof(uint16_t)); /* expected */
    }

    r = create_sub_buffer(h.device, h.mem, f32b.off_A,
                          m * k * sizeof(float), &f32b.A);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, f32b.off_B,
                          k * n * sizeof(float), &f32b.B);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, f32b.off_D,
                          m * n * sizeof(float), &f32b.D);
    if (r != VK_SUCCESS) goto cleanup;

    if (h.test_f16) {
        r = create_sub_buffer(h.device, h.mem, f16b.off_A,
                              m * k * sizeof(uint16_t), &f16b.A);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, f16b.off_B,
                              k * n * sizeof(uint16_t), &f16b.B);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, f16b.off_D,
                              m * n * sizeof(uint16_t), &f16b.D);
        if (r != VK_SUCCESS) goto cleanup;
    }

    if (h.test_f64) {
        r = create_sub_buffer(h.device, h.mem, f64b.off_A,
                              m * k * sizeof(double), &f64b.A);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, f64b.off_B,
                              k * n * sizeof(double), &f64b.B);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, f64b.off_D,
                              m * n * sizeof(double), &f64b.D);
        if (r != VK_SUCCESS) goto cleanup;
    }

    if (h.test_bf16) {
        r = create_sub_buffer(h.device, h.mem, bf16b.off_A,
                              m * k * sizeof(uint16_t), &bf16b.A);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, bf16b.off_B,
                              k * n * sizeof(uint16_t), &bf16b.B);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, bf16b.off_D,
                              m * n * sizeof(uint16_t), &bf16b.D);
        if (r != VK_SUCCESS) goto cleanup;
    }

    /* strided-batched: 2 matrices per dtype, packed back-to-back */
    sb32b.off_A = take_region(&h.cursor, h.align, TEST_BATCH * m * k * sizeof(float));
    sb32b.off_B = take_region(&h.cursor, h.align, TEST_BATCH * k * n * sizeof(float));
    sb32b.off_D = take_region(&h.cursor, h.align, TEST_BATCH * m * n * sizeof(float));
    sb32b.off_readback = take_region(&h.cursor, h.align, TEST_BATCH * m * n * sizeof(float));
    if (h.test_f16) {
        sb16b.off_A = take_region(&h.cursor, h.align, TEST_BATCH * m * k * sizeof(uint16_t));
        sb16b.off_B = take_region(&h.cursor, h.align, TEST_BATCH * k * n * sizeof(uint16_t));
        sb16b.off_D = take_region(&h.cursor, h.align, TEST_BATCH * m * n * sizeof(uint16_t));
        sb16b.off_readback = take_region(&h.cursor, h.align, TEST_BATCH * m * n * sizeof(uint16_t));
    }
    if (h.test_f64) {
        sb64b.off_A = take_region(&h.cursor, h.align, TEST_BATCH * m * k * sizeof(double));
        sb64b.off_B = take_region(&h.cursor, h.align, TEST_BATCH * k * n * sizeof(double));
        sb64b.off_D = take_region(&h.cursor, h.align, TEST_BATCH * m * n * sizeof(double));
        sb64b.off_readback = take_region(&h.cursor, h.align, TEST_BATCH * m * n * sizeof(double));
    }

    r = create_sub_buffer(h.device, h.mem, sb32b.off_A,
                          TEST_BATCH * m * k * sizeof(float), &sb32b.A);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, sb32b.off_B,
                          TEST_BATCH * k * n * sizeof(float), &sb32b.B);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, sb32b.off_D,
                          TEST_BATCH * m * n * sizeof(float), &sb32b.D);
    if (r != VK_SUCCESS) goto cleanup;
    if (h.test_f16) {
        r = create_sub_buffer(h.device, h.mem, sb16b.off_A,
                              TEST_BATCH * m * k * sizeof(uint16_t), &sb16b.A);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, sb16b.off_B,
                              TEST_BATCH * k * n * sizeof(uint16_t), &sb16b.B);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, sb16b.off_D,
                              TEST_BATCH * m * n * sizeof(uint16_t), &sb16b.D);
        if (r != VK_SUCCESS) goto cleanup;
    }
    if (h.test_f64) {
        r = create_sub_buffer(h.device, h.mem, sb64b.off_A,
                              TEST_BATCH * m * k * sizeof(double), &sb64b.A);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, sb64b.off_B,
                              TEST_BATCH * k * n * sizeof(double), &sb64b.B);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, sb64b.off_D,
                              TEST_BATCH * m * n * sizeof(double), &sb64b.D);
        if (r != VK_SUCCESS) goto cleanup;
    }

    /* gemm_ex: own D + readback regions (A/B reused from the single-gemm). */
    gex32b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(float));
    gex32b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(float));
    if (h.test_f16) {
        gex16b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
        gex16b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
    }
    if (h.test_bf16) {
        gex16bb.off_D = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
        gex16bb.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(uint16_t));
    }
    r = create_sub_buffer(h.device, h.mem, gex32b.off_D,
                          m * n * sizeof(float), &gex32b.D);
    if (r != VK_SUCCESS) goto cleanup;
    if (h.test_f16) {
        r = create_sub_buffer(h.device, h.mem, gex16b.off_D,
                              m * n * sizeof(uint16_t), &gex16b.D);
        if (r != VK_SUCCESS) goto cleanup;
    }
    if (h.test_bf16) {
        r = create_sub_buffer(h.device, h.mem, gex16bb.off_D,
                              m * n * sizeof(uint16_t), &gex16bb.D);
        if (r != VK_SUCCESS) goto cleanup;
    }

    /* f64 alpha/beta packing check: own C + D + readback. */
    bt64b.off_A = take_region(&h.cursor, h.align, m * n * sizeof(double));
    bt64b.off_D = take_region(&h.cursor, h.align, m * n * sizeof(double));
    bt64b.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(double));
    r = create_sub_buffer(h.device, h.mem, bt64b.off_A,
                          m * n * sizeof(double), &bt64b.A);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, bt64b.off_D,
                          m * n * sizeof(double), &bt64b.D);
    if (r != VK_SUCCESS) goto cleanup;

    /* ── 9b. Fused quantized-GEMM regions + buffers ─────────────────────── */
    /* Weight row count = output rows = 8; activation cols = 8. */
    const uint32_t qgn = 8;
    const uint32_t qgm = 8;
    {
        const VkDeviceSize q8_32_k = 32;  /* 1 Q8_0 block per row   */
        const VkDeviceSize q8_64_k = 64;  /* 2 Q8_0 blocks per row  */
        const VkDeviceSize q4k_k   = 256; /* 1 Q4_K block per row   */

        /* Q8_0 k=32: ldw = 36, Wq = 8*36 B. */
        q8_32.off_Wq = take_region(&h.cursor, h.align, qgn * 36);
        q8_32.off_x  = take_region(&h.cursor, h.align, q8_32_k * qgm * sizeof(float));
        q8_32.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q8_32.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* Q8_0 k=64: ldw = 72, Wq = 8*72 B. */
        q8_64.off_Wq = take_region(&h.cursor, h.align, qgn * 72);
        q8_64.off_x  = take_region(&h.cursor, h.align, q8_64_k * qgm * sizeof(float));
        q8_64.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q8_64.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* Q4_K k=256: ldw = 144, Wq = 8*144 B. */
        q4k.off_Wq = take_region(&h.cursor, h.align, qgn * 144);
        q4k.off_x  = take_region(&h.cursor, h.align, q4k_k * qgm * sizeof(float));
        q4k.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q4k.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));

        r = create_sub_buffer(h.device, h.mem, q8_32.off_Wq, qgn * 36, &q8_32.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q8_32.off_x, q8_32_k * qgm * sizeof(float), &q8_32.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q8_32.off_y, qgn * qgm * sizeof(float), &q8_32.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, q8_64.off_Wq, qgn * 72, &q8_64.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q8_64.off_x, q8_64_k * qgm * sizeof(float), &q8_64.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q8_64.off_y, qgn * qgm * sizeof(float), &q8_64.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, q4k.off_Wq, qgn * 144, &q4k.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q4k.off_x, q4k_k * qgm * sizeof(float), &q4k.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q4k.off_y, qgn * qgm * sizeof(float), &q4k.y);
        if (r != VK_SUCCESS) goto cleanup;

        /* Q4_0 k=32: ldw = 20, Wq = 8*20 B. */
        q40_32.off_Wq = take_region(&h.cursor, h.align, qgn * 20);
        q40_32.off_x  = take_region(&h.cursor, h.align, 32 * qgm * sizeof(float));
        q40_32.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q40_32.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* Q4_0 k=64: ldw = 40, Wq = 8*40 B. */
        q40_64.off_Wq = take_region(&h.cursor, h.align, qgn * 40);
        q40_64.off_x  = take_region(&h.cursor, h.align, 64 * qgm * sizeof(float));
        q40_64.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q40_64.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* Q5_K k=256: ldw = 176, Wq = 8*176 B. */
        q5k.off_Wq = take_region(&h.cursor, h.align, qgn * 176);
        q5k.off_x  = take_region(&h.cursor, h.align, q4k_k * qgm * sizeof(float));
        q5k.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q5k.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* Q6_K k=256: ldw = 210, Wq = 8*210 B. */
        q6k.off_Wq = take_region(&h.cursor, h.align, qgn * 210);
        q6k.off_x  = take_region(&h.cursor, h.align, q4k_k * qgm * sizeof(float));
        q6k.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q6k.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* Q3_K k=256: ldw = 110, Wq = 8*110 B. */
        q3k.off_Wq = take_region(&h.cursor, h.align, qgn * 110);
        q3k.off_x  = take_region(&h.cursor, h.align, q4k_k * qgm * sizeof(float));
        q3k.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        q3k.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        /* IQ4_XS k=256: ldw = 136, Wq = 8*136 B. */
        iq4xs.off_Wq = take_region(&h.cursor, h.align, qgn * 136);
        iq4xs.off_x  = take_region(&h.cursor, h.align, q4k_k * qgm * sizeof(float));
        iq4xs.off_y  = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));
        iq4xs.off_readback = take_region(&h.cursor, h.align, qgn * qgm * sizeof(float));

        r = create_sub_buffer(h.device, h.mem, q40_32.off_Wq, qgn * 20, &q40_32.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q40_32.off_x, 32 * qgm * sizeof(float), &q40_32.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q40_32.off_y, qgn * qgm * sizeof(float), &q40_32.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, q40_64.off_Wq, qgn * 40, &q40_64.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q40_64.off_x, 64 * qgm * sizeof(float), &q40_64.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q40_64.off_y, qgn * qgm * sizeof(float), &q40_64.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, q5k.off_Wq, qgn * 176, &q5k.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q5k.off_x, q4k_k * qgm * sizeof(float), &q5k.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q5k.off_y, qgn * qgm * sizeof(float), &q5k.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, q6k.off_Wq, qgn * 210, &q6k.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q6k.off_x, q4k_k * qgm * sizeof(float), &q6k.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q6k.off_y, qgn * qgm * sizeof(float), &q6k.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, q3k.off_Wq, qgn * 110, &q3k.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q3k.off_x, q4k_k * qgm * sizeof(float), &q3k.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, q3k.off_y, qgn * qgm * sizeof(float), &q3k.y);
        if (r != VK_SUCCESS) goto cleanup;

        r = create_sub_buffer(h.device, h.mem, iq4xs.off_Wq, qgn * 136, &iq4xs.Wq);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, iq4xs.off_x, q4k_k * qgm * sizeof(float), &iq4xs.x);
        if (r != VK_SUCCESS) goto cleanup;
        r = create_sub_buffer(h.device, h.mem, iq4xs.off_y, qgn * qgm * sizeof(float), &iq4xs.y);
        if (r != VK_SUCCESS) goto cleanup;
    }

    /* fp16-output qgemm cases: one distinct y16 buffer + readback per case
       (Wq/x are shared with the f32 cases above). Gated on f16 features. */
    if (h.test_f16) {
        for (int i = 0; i < 9; ++i) {
            qg_f16[i].off_y16        = take_region(&h.cursor, h.align,
                                                   qgn * qgm * sizeof(uint16_t));
            qg_f16[i].off_readback16 = take_region(&h.cursor, h.align,
                                                   qgn * qgm * sizeof(uint16_t));
        }
        for (int i = 0; i < 9; ++i) {
            r = create_sub_buffer(h.device, h.mem, qg_f16[i].off_y16,
                                  qgn * qgm * sizeof(uint16_t), &qg_f16[i].y16);
            if (r != VK_SUCCESS) goto cleanup;
        }
    }

    /* ── 10. Fill inputs + references (column-major) ────────────────────── */
    /* Shared shape: A = all ones, B = ramp 1..64, so D = alpha * sum_k B.   */
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t t = 0; t < k; t++) {
            A[row + t * TEST_GEMM_LDA] = 1.0f;
            A16[row + t * TEST_GEMM_LDA] = f32_to_f16(1.0f);
            A64[row + t * TEST_GEMM_LDA] = 1.0;
        }
    }
    for (uint32_t t = 0; t < k; t++) {
        for (uint32_t c = 0; c < n; c++) {
            float v = (float)(1 + t + 8 * c);
            B[t + c * TEST_GEMM_LDB] = v;
            B16[t + c * TEST_GEMM_LDB] = f32_to_f16(v);
            B64[t + c * TEST_GEMM_LDB] = (double)v;
        }
    }

    /* f32 reference (double-accumulated, stored f32). */
    compute_reference_f32(A, TEST_GEMM_LDA, B, TEST_GEMM_LDB,
                          1.0f, 0.0f, D_expected, TEST_GEMM_LDD,
                          (int32_t)m, (int32_t)n, (int32_t)k);

    /* f16 reference: accumulate in double, round to f16 (GPU accumulates in
       f32 then rounds to f16; values are small exact integers). */
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < k; t++) {
                acc += (double)A64[row + t * TEST_GEMM_LDA]
                     * (double)B64[t + c * TEST_GEMM_LDB];
            }
            D16_expected[row + c * TEST_GEMM_LDD] = f32_to_f16((float)acc);
        }
    }

    /* f64 reference: pure double. */
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < k; t++) {
                acc += A64[row + t * TEST_GEMM_LDA] * B64[t + c * TEST_GEMM_LDB];
            }
            D64_expected[row + c * TEST_GEMM_LDD] = acc;
        }
    }

    /* bf16 reference: A/B stored as bf16 (converted to f32 for the compute),
       accumulate in double, round result back to bf16. Mirrors the shader
       (bf16 in -> f32 accumulate -> bf16 out). */
    if (h.test_bf16) {
        for (uint32_t row = 0; row < m; row++) {
            for (uint32_t t = 0; t < k; t++) {
                A16b[row + t * TEST_GEMM_LDA] = f32_to_bf16(1.0f);
            }
        }
        for (uint32_t t = 0; t < k; t++) {
            for (uint32_t c = 0; c < n; c++) {
                float v = (float)(1 + t + 8 * c);
                B16b[t + c * TEST_GEMM_LDB] = f32_to_bf16(v);
            }
        }
        for (uint32_t row = 0; row < m; row++) {
            for (uint32_t c = 0; c < n; c++) {
                double acc = 0.0;
                for (uint32_t t = 0; t < k; t++) {
                    double av = (double)bf16_to_f32(A16b[row + t * TEST_GEMM_LDA]);
                    double bv = (double)bf16_to_f32(B16b[t + c * TEST_GEMM_LDB]);
                    acc += av * bv;
                }
                D16b_expected[row + c * TEST_GEMM_LDD] = f32_to_bf16((float)acc);
            }
        }
    }

    /* strided-batched fill + references: batch b scales A by (b+1). */
    for (int b = 0; b < TEST_BATCH; b++) {
        float scale = (float)(b + 1);
        for (uint32_t row = 0; row < m; row++) {
            for (uint32_t t = 0; t < k; t++) {
                ((float *)((char *)h.mapped + sb32b.off_A))
                    [b * m * k + row + t * TEST_GEMM_LDA] = scale;
            }
        }
        for (uint32_t t = 0; t < k; t++) {
            for (uint32_t c = 0; c < n; c++) {
                ((float *)((char *)h.mapped + sb32b.off_B))
                    [b * k * n + t + c * TEST_GEMM_LDB] = (float)(1 + t + 8 * c);
            }
        }
        for (uint32_t row = 0; row < m; row++) {
            for (uint32_t c = 0; c < n; c++) {
                double acc = 0.0;
                for (uint32_t t = 0; t < k; t++) {
                    acc += (double)scale * (double)(1 + t + 8 * c);
                }
                SB32_expected[b][row + c * TEST_GEMM_LDD] = (float)acc;
            }
        }
        if (h.test_f16) {
            for (uint32_t row = 0; row < m; row++) {
                for (uint32_t t = 0; t < k; t++) {
                    ((uint16_t *)((char *)h.mapped + sb16b.off_A))
                        [b * m * k + row + t * TEST_GEMM_LDA] = f32_to_f16(scale);
                }
            }
            for (uint32_t t = 0; t < k; t++) {
                for (uint32_t c = 0; c < n; c++) {
                    ((uint16_t *)((char *)h.mapped + sb16b.off_B))
                        [b * k * n + t + c * TEST_GEMM_LDB] =
                            f32_to_f16((float)(1 + t + 8 * c));
                }
            }
            for (uint32_t row = 0; row < m; row++) {
                for (uint32_t c = 0; c < n; c++) {
                    double acc = 0.0;
                    for (uint32_t t = 0; t < k; t++) {
                        acc += (double)scale * (double)(1 + t + 8 * c);
                    }
                    SB16_expected[b][row + c * TEST_GEMM_LDD] = f32_to_f16((float)acc);
                }
            }
        }
        if (h.test_f64) {
            double dscale = (double)scale;
            for (uint32_t row = 0; row < m; row++) {
                for (uint32_t t = 0; t < k; t++) {
                    ((double *)((char *)h.mapped + sb64b.off_A))
                        [b * m * k + row + t * TEST_GEMM_LDA] = dscale;
                }
            }
            for (uint32_t t = 0; t < k; t++) {
                for (uint32_t c = 0; c < n; c++) {
                    ((double *)((char *)h.mapped + sb64b.off_B))
                        [b * k * n + t + c * TEST_GEMM_LDB] = (double)(1 + t + 8 * c);
                }
            }
            for (uint32_t row = 0; row < m; row++) {
                for (uint32_t c = 0; c < n; c++) {
                    double acc = 0.0;
                    for (uint32_t t = 0; t < k; t++) {
                        acc += dscale * (double)(1 + t + 8 * c);
                    }
                    SB64_expected[b][row + c * TEST_GEMM_LDD] = acc;
                }
            }
        }
    }

    memcpy((char *)h.mapped + f32b.off_A, A, m * k * sizeof(float));
    memcpy((char *)h.mapped + f32b.off_B, B, k * n * sizeof(float));
    if (h.test_f16) {
        memcpy((char *)h.mapped + f16b.off_A, A16, m * k * sizeof(uint16_t));
        memcpy((char *)h.mapped + f16b.off_B, B16, k * n * sizeof(uint16_t));
    }
    if (h.test_f64) {
        memcpy((char *)h.mapped + f64b.off_A, A64, m * k * sizeof(double));
        memcpy((char *)h.mapped + f64b.off_B, B64, k * n * sizeof(double));
    }
    if (h.test_bf16) {
        memcpy((char *)h.mapped + bf16b.off_A, A16b, m * k * sizeof(uint16_t));
        memcpy((char *)h.mapped + bf16b.off_B, B16b, k * n * sizeof(uint16_t));
    }

    /* f64 alpha/beta packing check: D = 1.5 * A*B + (-0.5) * C. */
    if (h.test_f64) {
        for (uint32_t row = 0; row < m; row++) {
            for (uint32_t c = 0; c < n; c++) {
                C64in[row + c * TEST_GEMM_LDD] = 0.5 + (double)row + (double)c;
            }
        }
        for (uint32_t row = 0; row < m; row++) {
            for (uint32_t c = 0; c < n; c++) {
                double acc = 0.0;
                for (uint32_t t = 0; t < k; t++) {
                    acc += A64[row + t * TEST_GEMM_LDA] * B64[t + c * TEST_GEMM_LDB];
                }
                Dbt_expected[row + c * TEST_GEMM_LDD] =
                    1.5 * acc + (-0.5) * C64in[row + c * TEST_GEMM_LDD];
            }
        }
        memcpy((char *)h.mapped + bt64b.off_A, C64in, m * n * sizeof(double));
    }

    /* ── 10b. Fused quantized-GEMM inputs + references ──────────────────── */
    {
        /* Deterministic weight/activation data with mixed signs. */
        static float W8_32[8 * 32], W8_64[8 * 64], W4k[8 * 256];
        static float x32[32 * 8], x64[64 * 8], x256[256 * 8];
        static float yinit[8 * 8];
        static float yexp8_32[8 * 8], yexp8_64[8 * 8], yexp4k[8 * 8];
        uint8_t Wq32[8 * 36], Wq64[8 * 72], Wq4k[8 * 144];

        /* qgemm reference: y = alpha*(dequant(W)*x) + beta*y. Column-major. */
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t t = 0; t < 32; ++t)
                W8_32[r * 32 + t] = ((float)((r * 3 + t) % 17) - 8.0f) * 0.25f;
            for (uint32_t t = 0; t < 64; ++t)
                W8_64[r * 64 + t] = ((float)((r * 5 + t) % 29) - 14.0f) * 0.2f;
            for (uint32_t t = 0; t < 256; ++t)
                W4k[r * 256 + t] = ((float)((r * 7 + t * 3) % 37) - 18.0f) * 0.25f;
        }
        for (uint32_t t = 0; t < 32; ++t)
            for (uint32_t c = 0; c < qgm; ++c)
                x32[t + c * 32] = ((float)((t * 5 + c * 3) % 11) - 5.0f) * 0.5f;
        for (uint32_t t = 0; t < 64; ++t)
            for (uint32_t c = 0; c < qgm; ++c)
                x64[t + c * 64] = ((float)((t * 5 + c * 3) % 11) - 5.0f) * 0.5f;
        for (uint32_t t = 0; t < 256; ++t)
            for (uint32_t c = 0; c < qgm; ++c)
                x256[t + c * 256] = ((float)((t * 5 + c * 3) % 11) - 5.0f) * 0.5f;
        for (uint32_t r = 0; r < qgn; ++r)
            for (uint32_t c = 0; c < qgm; ++c)
                yinit[r + c * 8] = 1.0f + 0.25f * (float)(r + 2 * (int)c);

        /* Q8_0 k=32, alpha=1, beta=0. */
        quantize_q8_0(W8_32, 8, Wq32);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 32; ++t) {
                    float w = dequant_q8_0(Wq32 + r * 36 + ((t >> 5) * 36), (int)(t & 31u));
                    acc += (double)w * (double)x32[t + c * 32];
                }
                yexp8_32[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + q8_32.off_Wq, Wq32, sizeof(Wq32));
        memcpy((char *)h.mapped + q8_32.off_x, x32, sizeof(x32));
        memset((char *)h.mapped + q8_32.off_y, 0, qgn * qgm * sizeof(float));

        /* Q8_0 k=64, alpha=0.75, beta=0.5 (exercises the beta/C read path). */
        quantize_q8_0(W8_64, 16, Wq64);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 64; ++t) {
                    uint32_t blk = t >> 5;
                    float w = dequant_q8_0(Wq64 + r * 72 + blk * 36, (int)(t & 31u));
                    acc += (double)w * (double)x64[t + c * 64];
                }
                yexp8_64[r + c * 8] =
                    (float)(0.75 * acc + 0.5 * (double)yinit[r + c * 8]);
            }
        }
        memcpy((char *)h.mapped + q8_64.off_Wq, Wq64, sizeof(Wq64));
        memcpy((char *)h.mapped + q8_64.off_x, x64, sizeof(x64));
        memcpy((char *)h.mapped + q8_64.off_y, yinit, sizeof(yinit));

        /* Q4_K k=256, alpha=1, beta=0. */
        for (uint32_t r = 0; r < qgn; ++r)
            quantize_q4k_f32(W4k + r * 256, Wq4k + r * 144);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 256; ++t) {
                    float w = dequant_q4k_f32(Wq4k + r * 144, (int)t);
                    acc += (double)w * (double)x256[t + c * 256];
                }
                yexp4k[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + q4k.off_Wq, Wq4k, sizeof(Wq4k));
        memcpy((char *)h.mapped + q4k.off_x, x256, sizeof(x256));
        memset((char *)h.mapped + q4k.off_y, 0, qgn * qgm * sizeof(float));

        /* Store the references for the check phase. */
        memcpy(qg_yexp[0], yexp8_32, sizeof(yexp8_32));
        memcpy(qg_yexp[1], yexp8_64, sizeof(yexp8_64));
        memcpy(qg_yexp[2], yexp4k,   sizeof(yexp4k));

        /* ── Q4_0 (k=32, k=64) + Q5_K / Q6_K / Q3_K / IQ4_XS (k=256) ──── */
        static float W40_32[8 * 32], W40_64[8 * 64];
        static float W5k[8 * 256], W6k[8 * 256], W3k[8 * 256], Wiq[8 * 256];
        static float yexp40_32[8 * 8], yexp40_64[8 * 8];
        static float yexp5k[8 * 8], yexp6k[8 * 8], yexp3k[8 * 8], yexpiq[8 * 8];
        uint8_t Wq40_32[8 * 20], Wq40_64[8 * 40];
        uint8_t Wq5k[8 * 176], Wq6k[8 * 210], Wq3k[8 * 110], Wqiq[8 * 136];

        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t t = 0; t < 32; ++t)
                W40_32[r * 32 + t] = ((float)((r * 11 + t * 7) % 21) - 10.0f) * 0.3f;
            for (uint32_t t = 0; t < 64; ++t)
                W40_64[r * 64 + t] = ((float)((r * 13 + t * 5) % 31) - 15.0f) * 0.25f;
            for (uint32_t t = 0; t < 256; ++t)
                W5k[r * 256 + t] = ((float)((r * 17 + t * 11) % 43) - 21.0f) * 0.2f;
            for (uint32_t t = 0; t < 256; ++t)
                W6k[r * 256 + t] = ((float)((r * 19 + t * 13) % 61) - 30.0f) * 0.15f;
            for (uint32_t t = 0; t < 256; ++t)
                W3k[r * 256 + t] = ((float)((r * 23 + t * 17) % 41) - 20.0f) * 0.25f;
            for (uint32_t t = 0; t < 256; ++t)
                Wiq[r * 256 + t] = ((float)((r * 29 + t * 19) % 53) - 26.0f) * 0.2f;
        }

        /* Q4_0 k=32, alpha=1, beta=0. */
        quantize_q4_0(W40_32, 8, Wq40_32);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 32; ++t)
                    acc += (double)dequant_q4_0(Wq40_32 + r * 20 + ((t >> 5) * 20), (int)(t & 31u))
                         * (double)x32[t + c * 32];
                yexp40_32[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + q40_32.off_Wq, Wq40_32, sizeof(Wq40_32));
        memcpy((char *)h.mapped + q40_32.off_x, x32, sizeof(x32));
        memset((char *)h.mapped + q40_32.off_y, 0, qgn * qgm * sizeof(float));

        /* Q4_0 k=64, alpha=0.75, beta=0.5 (beta/C read path). */
        quantize_q4_0(W40_64, 16, Wq40_64);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 64; ++t)
                    acc += (double)dequant_q4_0(Wq40_64 + r * 40 + ((t >> 5) * 20), (int)(t & 31u))
                         * (double)x64[t + c * 64];
                yexp40_64[r + c * 8] = (float)(0.75 * acc + 0.5 * (double)yinit[r + c * 8]);
            }
        }
        memcpy((char *)h.mapped + q40_64.off_Wq, Wq40_64, sizeof(Wq40_64));
        memcpy((char *)h.mapped + q40_64.off_x, x64, sizeof(x64));
        memcpy((char *)h.mapped + q40_64.off_y, yinit, sizeof(yinit));

        /* Q5_K k=256, alpha=1, beta=0. */
        for (uint32_t r = 0; r < qgn; ++r)
            quantize_q5k_f32(W5k + r * 256, Wq5k + r * 176);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 256; ++t)
                    acc += (double)dequant_q5k_f32(Wq5k + r * 176, (int)t) * (double)x256[t + c * 256];
                yexp5k[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + q5k.off_Wq, Wq5k, sizeof(Wq5k));
        memcpy((char *)h.mapped + q5k.off_x, x256, sizeof(x256));
        memset((char *)h.mapped + q5k.off_y, 0, qgn * qgm * sizeof(float));

        /* Q6_K k=256, alpha=1, beta=0. */
        for (uint32_t r = 0; r < qgn; ++r)
            quantize_q6k_f32(W6k + r * 256, Wq6k + r * 210);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 256; ++t)
                    acc += (double)dequant_q6k_f32(Wq6k + r * 210, (int)t) * (double)x256[t + c * 256];
                yexp6k[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + q6k.off_Wq, Wq6k, sizeof(Wq6k));
        memcpy((char *)h.mapped + q6k.off_x, x256, sizeof(x256));
        memset((char *)h.mapped + q6k.off_y, 0, qgn * qgm * sizeof(float));

        /* Q3_K k=256, alpha=1, beta=0. */
        for (uint32_t r = 0; r < qgn; ++r)
            quantize_q3k_f32(W3k + r * 256, Wq3k + r * 110);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 256; ++t)
                    acc += (double)dequant_q3k_f32(Wq3k + r * 110, (int)t) * (double)x256[t + c * 256];
                yexp3k[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + q3k.off_Wq, Wq3k, sizeof(Wq3k));
        memcpy((char *)h.mapped + q3k.off_x, x256, sizeof(x256));
        memset((char *)h.mapped + q3k.off_y, 0, qgn * qgm * sizeof(float));

        /* IQ4_XS k=256, alpha=1, beta=0. */
        for (uint32_t r = 0; r < qgn; ++r)
            quantize_iq4xs_f32(Wiq + r * 256, Wqiq + r * 136);
        for (uint32_t r = 0; r < qgn; ++r) {
            for (uint32_t c = 0; c < qgm; ++c) {
                double acc = 0.0;
                for (uint32_t t = 0; t < 256; ++t)
                    acc += (double)dequant_iq4xs_f32(Wqiq + r * 136, (int)t) * (double)x256[t + c * 256];
                yexpiq[r + c * 8] = (float)acc;
            }
        }
        memcpy((char *)h.mapped + iq4xs.off_Wq, Wqiq, sizeof(Wqiq));
        memcpy((char *)h.mapped + iq4xs.off_x, x256, sizeof(x256));
        memset((char *)h.mapped + iq4xs.off_y, 0, qgn * qgm * sizeof(float));

        /* Store the references for the check phase. */
        memcpy(qg_yexp[3], yexp40_32, sizeof(yexp40_32));
        memcpy(qg_yexp[4], yexp40_64, sizeof(yexp40_64));
        memcpy(qg_yexp[5], yexp5k,    sizeof(yexp5k));
        memcpy(qg_yexp[6], yexp6k,    sizeof(yexp6k));
        memcpy(qg_yexp[7], yexp3k,    sizeof(yexp3k));
        memcpy(qg_yexp[8], yexpiq,    sizeof(yexpiq));

        /* fp16-output cases: expected[i] = f16(f32 reference); the two beta
           cases (q8_0 k=64, q4_0 k=64) get y16 prefilled with f16(yinit). */
        if (h.test_f16) {
            uint16_t yinit16[8 * 8];
            for (uint32_t r = 0; r < qgn; ++r)
                for (uint32_t c = 0; c < qgm; ++c)
                    yinit16[r + c * 8] = f32_to_f16(yinit[r + c * 8]);

            for (int i = 0; i < 9; ++i)
                memset((char *)h.mapped + qg_f16[i].off_y16, 0,
                       qgn * qgm * sizeof(uint16_t));
            memcpy((char *)h.mapped + qg_f16[1].off_y16, yinit16, sizeof(yinit16));
            memcpy((char *)h.mapped + qg_f16[4].off_y16, yinit16, sizeof(yinit16));
        }
    }

    /* ── 11. Record all GEMM dispatches into one command buffer ─────────── */
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    r = vkBeginCommandBuffer(h.cmd, &begin_info);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkBeginCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    overall_pass &= record_dispatch(
        vkblas_sgemm(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                     (int32_t)m, (int32_t)n, (int32_t)k,
                     &alpha, f32b.A, TEST_GEMM_LDA,
                     f32b.B, TEST_GEMM_LDB,
                     &beta, VK_NULL_HANDLE, 0,
                     f32b.D, TEST_GEMM_LDD),
        "sgemm m=n=k=8");

    if (h.test_f16) {
        const uint16_t alpha16 = f32_to_f16(1.0f);
        const uint16_t beta16  = f32_to_f16(0.0f);
        overall_pass &= record_dispatch(
            vkblas_hgemm(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                         (int32_t)m, (int32_t)n, (int32_t)k,
                         &alpha16, f16b.A, TEST_GEMM_LDA,
                         f16b.B, TEST_GEMM_LDB,
                         &beta16, VK_NULL_HANDLE, 0,
                         f16b.D, TEST_GEMM_LDD),
            "hgemm m=n=k=8");
    }

    if (h.test_f64) {
        const double alpha64 = 1.0;
        const double beta64  = 0.0;
        overall_pass &= record_dispatch(
            vkblas_dgemm(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                         (int32_t)m, (int32_t)n, (int32_t)k,
                         &alpha64, f64b.A, TEST_GEMM_LDA,
                         f64b.B, TEST_GEMM_LDB,
                         &beta64, VK_NULL_HANDLE, 0,
                         f64b.D, TEST_GEMM_LDD),
            "dgemm m=n=k=8");
    }

    /* bgemm (bf16): storage is uint16 bf16; alpha/beta are bf16 bits too. */
    if (h.test_bf16) {
        const uint16_t alpha16b = f32_to_bf16(1.0f);
        const uint16_t beta16b  = f32_to_bf16(0.0f);
        overall_pass &= record_dispatch(
            vkblas_bgemm(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                         (int32_t)m, (int32_t)n, (int32_t)k,
                         &alpha16b, bf16b.A, TEST_GEMM_LDA,
                         bf16b.B, TEST_GEMM_LDB,
                         &beta16b, VK_NULL_HANDLE, 0,
                         bf16b.D, TEST_GEMM_LDD),
            "bgemm m=n=k=8");
    }

    /* ── 11b. Strided-batched dispatches (TEST_BATCH batches) ───────────── */
    /* API strides are in bytes; each dtype uses its own element size. */
    const int64_t sbStrideA32 = (int64_t)(m * k * sizeof(float));
    const int64_t sbStrideB32 = (int64_t)(k * n * sizeof(float));
    const int64_t sbStrideD32 = (int64_t)(m * n * sizeof(float));
    const int64_t sbStrideA16 = (int64_t)(m * k * sizeof(uint16_t));
    const int64_t sbStrideB16 = (int64_t)(k * n * sizeof(uint16_t));
    const int64_t sbStrideD16 = (int64_t)(m * n * sizeof(uint16_t));
    const int64_t sbStrideA64 = (int64_t)(m * k * sizeof(double));
    const int64_t sbStrideB64 = (int64_t)(k * n * sizeof(double));
    const int64_t sbStrideD64 = (int64_t)(m * n * sizeof(double));
    overall_pass &= record_dispatch(
        vkblas_sgemm_strided_batched(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                                     (int32_t)m, (int32_t)n, (int32_t)k,
                                     &alpha, sb32b.A, TEST_GEMM_LDA, sbStrideA32,
                                     sb32b.B, TEST_GEMM_LDB, sbStrideB32,
                                     &beta, VK_NULL_HANDLE, 0, sbStrideD32,
                                     sb32b.D, TEST_GEMM_LDD, sbStrideD32,
                                     TEST_BATCH),
        "sgemm_strided x2");
    if (h.test_f16) {
        const uint16_t alpha16 = f32_to_f16(1.0f);
        const uint16_t beta16  = f32_to_f16(0.0f);
        overall_pass &= record_dispatch(
            vkblas_hgemm_strided_batched(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                                         (int32_t)m, (int32_t)n, (int32_t)k,
                                         &alpha16, sb16b.A, TEST_GEMM_LDA, sbStrideA16,
                                         sb16b.B, TEST_GEMM_LDB, sbStrideB16,
                                         &beta16, VK_NULL_HANDLE, 0, sbStrideD16,
                                         sb16b.D, TEST_GEMM_LDD, sbStrideD16,
                                         TEST_BATCH),
            "hgemm_strided x2");
    }
    if (h.test_f64) {
        const double alpha64 = 1.0;
        const double beta64  = 0.0;
        overall_pass &= record_dispatch(
            vkblas_dgemm_strided_batched(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                                         (int32_t)m, (int32_t)n, (int32_t)k,
                                         &alpha64, sb64b.A, TEST_GEMM_LDA, sbStrideA64,
                                         sb64b.B, TEST_GEMM_LDB, sbStrideB64,
                                         &beta64, VK_NULL_HANDLE, 0, sbStrideD64,
                                         sb64b.D, TEST_GEMM_LDD, sbStrideD64,
                                         TEST_BATCH),
            "dgemm_strided x2");
    }

    /* ── 11c. gemm_ex dispatches ─────────────────────────────────────────── */
    {
        const float gex_alpha = 1.0f;
        const float gex_beta  = 0.0f;
        overall_pass &= record_dispatch(
            vkblas_gemm_ex(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                           (int32_t)m, (int32_t)n, (int32_t)k,
                           &gex_alpha, f32b.A, TEST_GEMM_LDA, 0,
                           f32b.B, TEST_GEMM_LDB, 0,
                           &gex_beta, VK_NULL_HANDLE, 0, 0,
                           gex32b.D, TEST_GEMM_LDD, 0,
                           VKBLAS_COMPUTE_32F, VKBLAS_GEMM_FLAGS_NONE),
            "gemm_ex 32F");
    }
    if (h.test_f16) {
        const uint16_t gex_alpha16 = f32_to_f16(1.0f);
        const uint16_t gex_beta16  = f32_to_f16(0.0f);
        overall_pass &= record_dispatch(
            vkblas_gemm_ex(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                           (int32_t)m, (int32_t)n, (int32_t)k,
                           &gex_alpha16, f16b.A, TEST_GEMM_LDA, 0,
                           f16b.B, TEST_GEMM_LDB, 0,
                           &gex_beta16, VK_NULL_HANDLE, 0, 0,
                           gex16b.D, TEST_GEMM_LDD, 0,
                           VKBLAS_COMPUTE_16F, VKBLAS_GEMM_FLAGS_NONE),
            "gemm_ex 16F");
    }
    if (h.test_bf16) {
        const uint16_t gex_alpha16b = f32_to_bf16(1.0f);
        const uint16_t gex_beta16b  = f32_to_bf16(0.0f);
        overall_pass &= record_dispatch(
            vkblas_gemm_ex(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                           (int32_t)m, (int32_t)n, (int32_t)k,
                           &gex_alpha16b, bf16b.A, TEST_GEMM_LDA, 0,
                           bf16b.B, TEST_GEMM_LDB, 0,
                           &gex_beta16b, VK_NULL_HANDLE, 0, 0,
                           gex16bb.D, TEST_GEMM_LDD, 0,
                           VKBLAS_COMPUTE_16B, VKBLAS_GEMM_FLAGS_NONE),
            "gemm_ex BF16");
    }

    /* ── 11d. f64 alpha/beta packing check ───────────────────────────────── */
    if (h.test_f64) {
        const double bt_alpha = 1.5;
        const double bt_beta  = -0.5;
        overall_pass &= record_dispatch(
            vkblas_dgemm(h.blas_ctx, h.cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                         (int32_t)m, (int32_t)n, (int32_t)k,
                         &bt_alpha, f64b.A, TEST_GEMM_LDA,
                         f64b.B, TEST_GEMM_LDB,
                         &bt_beta, bt64b.A, TEST_GEMM_LDD,
                         bt64b.D, TEST_GEMM_LDD),
            "dgemm a/b pack");
    }

    /* ── 11e. Fused quantized-GEMM dispatches ────────────────────────────── */
    {
        const float qa1 = 1.0f, qb0 = 0.0f;
        const float qa2 = 0.75f, qb2 = 0.5f;

        overall_pass &= record_dispatch(
            vkblas_qgemm_q8_0_f32(h.blas_ctx, h.cmd,
                                  (int32_t)qgm, (int32_t)qgn, 32,
                                  &qa1, q8_32.Wq, 36, q8_32.x, 32,
                                  &qb0, q8_32.y, 8),
            "qgemm q8_0 k=32");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q8_0_f32(h.blas_ctx, h.cmd,
                                  (int32_t)qgm, (int32_t)qgn, 64,
                                  &qa2, q8_64.Wq, 72, q8_64.x, 64,
                                  &qb2, q8_64.y, 8),
            "qgemm q8_0 k=64");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q4k_f32(h.blas_ctx, h.cmd,
                                 (int32_t)qgm, (int32_t)qgn, 256,
                                 &qa1, q4k.Wq, 144, q4k.x, 256,
                                 &qb0, q4k.y, 8),
            "qgemm q4k k=256");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q4_0_f32(h.blas_ctx, h.cmd,
                                  (int32_t)qgm, (int32_t)qgn, 32,
                                  &qa1, q40_32.Wq, 20, q40_32.x, 32,
                                  &qb0, q40_32.y, 8),
            "qgemm q4_0 k=32");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q4_0_f32(h.blas_ctx, h.cmd,
                                  (int32_t)qgm, (int32_t)qgn, 64,
                                  &qa2, q40_64.Wq, 40, q40_64.x, 64,
                                  &qb2, q40_64.y, 8),
            "qgemm q4_0 k=64");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q5k_f32(h.blas_ctx, h.cmd,
                                 (int32_t)qgm, (int32_t)qgn, 256,
                                 &qa1, q5k.Wq, 176, q5k.x, 256,
                                 &qb0, q5k.y, 8),
            "qgemm q5k k=256");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q6k_f32(h.blas_ctx, h.cmd,
                                 (int32_t)qgm, (int32_t)qgn, 256,
                                 &qa1, q6k.Wq, 210, q6k.x, 256,
                                 &qb0, q6k.y, 8),
            "qgemm q6k k=256");

        overall_pass &= record_dispatch(
            vkblas_qgemm_q3k_f32(h.blas_ctx, h.cmd,
                                 (int32_t)qgm, (int32_t)qgn, 256,
                                 &qa1, q3k.Wq, 110, q3k.x, 256,
                                 &qb0, q3k.y, 8),
            "qgemm q3k k=256");

        overall_pass &= record_dispatch(
            vkblas_qgemm_iq4xs_f32(h.blas_ctx, h.cmd,
                                   (int32_t)qgm, (int32_t)qgn, 256,
                                   &qa1, iq4xs.Wq, 136, iq4xs.x, 256,
                                   &qb0, iq4xs.y, 8),
            "qgemm iq4xs k=256");

        /* fp16-output storage path: same Wq/x/alpha/beta as the f32 cases,
           but y16 (fp16 bits) is written instead of y (f32). Exercises the
           float16_t BufY/BufZ bindings and the f32->f16 rounding at the store.
           Runs only when the harness enabled the 16-bit storage features. */
        if (h.test_f16) {
            overall_pass &= record_dispatch(
                vkblas_qgemm_q8_0_f16(h.blas_ctx, h.cmd,
                                      (int32_t)qgm, (int32_t)qgn, 32,
                                      &qa1, q8_32.Wq, 36, q8_32.x, 32,
                                      &qb0, qg_f16[0].y16, 8),
                "qgemm q8_0 k=32 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q8_0_f16(h.blas_ctx, h.cmd,
                                      (int32_t)qgm, (int32_t)qgn, 64,
                                      &qa2, q8_64.Wq, 72, q8_64.x, 64,
                                      &qb2, qg_f16[1].y16, 8),
                "qgemm q8_0 k=64 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q4k_f16(h.blas_ctx, h.cmd,
                                     (int32_t)qgm, (int32_t)qgn, 256,
                                     &qa1, q4k.Wq, 144, q4k.x, 256,
                                     &qb0, qg_f16[2].y16, 8),
                "qgemm q4k k=256 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q4_0_f16(h.blas_ctx, h.cmd,
                                      (int32_t)qgm, (int32_t)qgn, 32,
                                      &qa1, q40_32.Wq, 20, q40_32.x, 32,
                                      &qb0, qg_f16[3].y16, 8),
                "qgemm q4_0 k=32 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q4_0_f16(h.blas_ctx, h.cmd,
                                      (int32_t)qgm, (int32_t)qgn, 64,
                                      &qa2, q40_64.Wq, 40, q40_64.x, 64,
                                      &qb2, qg_f16[4].y16, 8),
                "qgemm q4_0 k=64 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q5k_f16(h.blas_ctx, h.cmd,
                                     (int32_t)qgm, (int32_t)qgn, 256,
                                     &qa1, q5k.Wq, 176, q5k.x, 256,
                                     &qb0, qg_f16[5].y16, 8),
                "qgemm q5k k=256 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q6k_f16(h.blas_ctx, h.cmd,
                                     (int32_t)qgm, (int32_t)qgn, 256,
                                     &qa1, q6k.Wq, 210, q6k.x, 256,
                                     &qb0, qg_f16[6].y16, 8),
                "qgemm q6k k=256 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_q3k_f16(h.blas_ctx, h.cmd,
                                     (int32_t)qgm, (int32_t)qgn, 256,
                                     &qa1, q3k.Wq, 110, q3k.x, 256,
                                     &qb0, qg_f16[7].y16, 8),
                "qgemm q3k k=256 f16");
            overall_pass &= record_dispatch(
                vkblas_qgemm_iq4xs_f16(h.blas_ctx, h.cmd,
                                       (int32_t)qgm, (int32_t)qgn, 256,
                                       &qa1, iq4xs.Wq, 136, iq4xs.x, 256,
                                       &qb0, qg_f16[8].y16, 8),
                "qgemm iq4xs k=256 f16");
        }
    }

    /* ── 11f. Fused qgemm execution-tier routing ───────────────────────────
       Verifies the subgroup tier is actually dispatched for every quantized
       weight format (decode hot path) on subgroup-capable devices. On
       non-subgroup hardware all formats must resolve to the baseline tier.
       arch index == active tier (0/1/2). */
    {
        uint32_t qtier = 0;
        uint32_t arch  = vkblas_get_arch_index(h.blas_ctx);
        uint32_t expect_tier = arch >= 1 ? 1u : 0u;

        const uint32_t all_fmts[] = {
            (uint32_t)VKBLAS_QGEMM_Q8_0,  (uint32_t)VKBLAS_QGEMM_Q4K,
            (uint32_t)VKBLAS_QGEMM_Q4_0,  (uint32_t)VKBLAS_QGEMM_Q5K,
            (uint32_t)VKBLAS_QGEMM_Q6K,   (uint32_t)VKBLAS_QGEMM_Q3K,
            (uint32_t)VKBLAS_QGEMM_IQ4XS,
        };
        const char* all_names[] = {
            "qgemm q8_0", "qgemm q4k", "qgemm q4_0", "qgemm q5k",
            "qgemm q6k", "qgemm q3k", "qgemm iq4xs",
        };
        uint32_t q8_tier = 0;
        for (int i = 0; i < 7; ++i) {
            r = vkblas_qgemm_get_tier(h.blas_ctx,
                                      (VkBLASQGemmFormat_t)all_fmts[i], &qtier);
            if (r != VK_SUCCESS || qtier != expect_tier) {
                fprintf(stderr, "FAIL: %s resolved tier=%u expected=%u "
                                "(arch=%u)\n", all_names[i], qtier,
                        expect_tier, arch);
                overall_pass = 0;
            }
            if (i == 0)
                q8_tier = qtier;
        }
        printf("test_vkblas: qgemm tier routing checked (all 7 formats=%s)\n",
               expect_tier == 1 ? "subgroup" : "baseline");
        printf("test_vkblas: qgemm q8_0 tier=%u (%s)\n", q8_tier,
               q8_tier == 1 ? "subgroup" : "baseline");
    }

    /* ── 12. Make shader writes visible to the transfer readbacks ───────── */
    record_compute_to_transfer_barrier(h.cmd);
    record_copy_readback(h.cmd, f32b.D, h.staging, f32b.off_readback,
                         m * n * sizeof(float));
    if (h.test_f16)
        record_copy_readback(h.cmd, f16b.D, h.staging, f16b.off_readback,
                             m * n * sizeof(uint16_t));
    if (h.test_f64)
        record_copy_readback(h.cmd, f64b.D, h.staging, f64b.off_readback,
                             m * n * sizeof(double));
    if (h.test_bf16)
        record_copy_readback(h.cmd, bf16b.D, h.staging, bf16b.off_readback,
                             m * n * sizeof(uint16_t));
    record_copy_readback(h.cmd, sb32b.D, h.staging, sb32b.off_readback,
                         TEST_BATCH * m * n * sizeof(float));
    if (h.test_f16)
        record_copy_readback(h.cmd, sb16b.D, h.staging, sb16b.off_readback,
                             TEST_BATCH * m * n * sizeof(uint16_t));
    if (h.test_f64)
        record_copy_readback(h.cmd, sb64b.D, h.staging, sb64b.off_readback,
                             TEST_BATCH * m * n * sizeof(double));
    record_copy_readback(h.cmd, gex32b.D, h.staging, gex32b.off_readback,
                         m * n * sizeof(float));
    if (h.test_f16)
        record_copy_readback(h.cmd, gex16b.D, h.staging, gex16b.off_readback,
                             m * n * sizeof(uint16_t));
    if (h.test_bf16)
        record_copy_readback(h.cmd, gex16bb.D, h.staging, gex16bb.off_readback,
                             m * n * sizeof(uint16_t));
    if (h.test_f64)
        record_copy_readback(h.cmd, bt64b.D, h.staging, bt64b.off_readback,
                             m * n * sizeof(double));
    record_copy_readback(h.cmd, q8_32.y, h.staging, q8_32.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q8_64.y, h.staging, q8_64.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q4k.y, h.staging, q4k.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q40_32.y, h.staging, q40_32.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q40_64.y, h.staging, q40_64.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q5k.y, h.staging, q5k.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q6k.y, h.staging, q6k.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, q3k.y, h.staging, q3k.off_readback,
                         8 * 8 * sizeof(float));
    record_copy_readback(h.cmd, iq4xs.y, h.staging, iq4xs.off_readback,
                         8 * 8 * sizeof(float));
    if (h.test_f16) {
        for (int i = 0; i < 9; ++i)
            record_copy_readback(h.cmd, qg_f16[i].y16, h.staging,
                                 qg_f16[i].off_readback16,
                                 8 * 8 * sizeof(uint16_t));
    }

    r = vkEndCommandBuffer(h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkEndCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    /* ── 13. One submit, one fence, device idle ─────────────────────────── */
    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(h.device, &fence_info, NULL, &h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkCreateFence failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &h.cmd;

    r = vkQueueSubmit(h.queue, 1, &submit_info, h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas: vkQueueSubmit failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }
    vkWaitForFences(h.device, 1, &h.fence, VK_TRUE, UINT64_MAX);
    vkQueueWaitIdle(h.queue);

    /* ── 14. Compare each D against its CPU reference ───────────────────── */
    overall_pass &= check_output_f32("sgemm m=n=k=8", h.mapped, f32b.off_readback,
                                     D_expected, elem_count, TEST_F32_TOLERANCE);
    if (h.test_f16)
        overall_pass &= check_output_f16("hgemm m=n=k=8", h.mapped,
                                         f16b.off_readback, D16_expected,
                                         elem_count, TEST_F16_TOLERANCE);
    else
        printf("  %-18s : SKIP (f16 features absent)\n", "hgemm m=n=k=8");
    if (h.test_f64)
        overall_pass &= check_output_f64("dgemm m=n=k=8", h.mapped,
                                         f64b.off_readback, D64_expected,
                                         elem_count, TEST_F64_TOLERANCE);
    else
        printf("  %-18s : SKIP (f64 features absent)\n", "dgemm m=n=k=8");

    /* bgemm (bf16) — must be a real PASS line, not FEATURE_NOT_PRESENT. */
    if (h.test_bf16)
        overall_pass &= check_output_bf16("bgemm m=n=k=8", h.mapped,
                                          bf16b.off_readback, D16b_expected,
                                          elem_count, TEST_BF16_TOLERANCE);
    else
        printf("  %-18s : SKIP (bf16 16-bit features absent)\n", "bgemm m=n=k=8");

    /* strided-batched: expected array is [batch][m*n]; the readback is one
       contiguous [batch][m*n] block, so pass the flattened batch pointer. */
    overall_pass &= check_output_f32("sgemm_strided x2", h.mapped,
                                     sb32b.off_readback, &SB32_expected[0][0],
                                     TEST_BATCH * elem_count, TEST_F32_TOLERANCE);
    if (h.test_f16)
        overall_pass &= check_output_f16("hgemm_strided x2", h.mapped,
                                         sb16b.off_readback, &SB16_expected[0][0],
                                         TEST_BATCH * elem_count,
                                         TEST_F16_TOLERANCE);
    else
        printf("  %-18s : SKIP (f16 features absent)\n", "hgemm_strided x2");
    if (h.test_f64)
        overall_pass &= check_output_f64("dgemm_strided x2", h.mapped,
                                         sb64b.off_readback, &SB64_expected[0][0],
                                         TEST_BATCH * elem_count,
                                         TEST_F64_TOLERANCE);
    else
        printf("  %-18s : SKIP (f64 features absent)\n", "dgemm_strided x2");

    overall_pass &= check_output_f32("gemm_ex 32F", h.mapped, gex32b.off_readback,
                                     D_expected, elem_count, TEST_F32_TOLERANCE);
    if (h.test_f16)
        overall_pass &= check_output_f16("gemm_ex 16F", h.mapped,
                                         gex16b.off_readback, D16_expected,
                                         elem_count, TEST_F16_TOLERANCE);
    else
        printf("  %-18s : SKIP (f16 features absent)\n", "gemm_ex 16F");
    if (h.test_bf16)
        overall_pass &= check_output_bf16("gemm_ex BF16", h.mapped,
                                          gex16bb.off_readback, D16b_expected,
                                          elem_count, TEST_BF16_TOLERANCE);
    else
        printf("  %-18s : SKIP (bf16 16-bit features absent)\n", "gemm_ex BF16");

    if (h.test_f64)
        overall_pass &= check_output_f64("dgemm a/b pack", h.mapped,
                                         bt64b.off_readback, Dbt_expected,
                                         elem_count, TEST_F64_TOLERANCE);
    else
        printf("  %-18s : SKIP (f64 features absent)\n", "dgemm a/b pack");

    /* fused quantized GEMM checks (64 outputs each). */
    overall_pass &= check_output_f32("qgemm q8_0 k=32", h.mapped,
                                     q8_32.off_readback, qg_yexp[0],
                                     8 * 8, TEST_F32_TOLERANCE);
    overall_pass &= check_output_f32("qgemm q8_0 k=64", h.mapped,
                                     q8_64.off_readback, qg_yexp[1],
                                     8 * 8, TEST_F32_TOLERANCE);
    overall_pass &= check_output_f32("qgemm q4k k=256", h.mapped,
                                     q4k.off_readback, qg_yexp[2],
                                     8 * 8, 1e-2f);
    overall_pass &= check_output_f32("qgemm q4_0 k=32", h.mapped,
                                     q40_32.off_readback, qg_yexp[3],
                                     8 * 8, TEST_F32_TOLERANCE);
    overall_pass &= check_output_f32("qgemm q4_0 k=64", h.mapped,
                                     q40_64.off_readback, qg_yexp[4],
                                     8 * 8, TEST_F32_TOLERANCE);
    overall_pass &= check_output_f32("qgemm q5k k=256", h.mapped,
                                     q5k.off_readback, qg_yexp[5],
                                     8 * 8, 1e-2f);
    overall_pass &= check_output_f32("qgemm q6k k=256", h.mapped,
                                     q6k.off_readback, qg_yexp[6],
                                     8 * 8, 1e-2f);
    overall_pass &= check_output_f32("qgemm q3k k=256", h.mapped,
                                     q3k.off_readback, qg_yexp[7],
                                     8 * 8, 1e-2f);
    overall_pass &= check_output_f32("qgemm iq4xs k=256", h.mapped,
                                     iq4xs.off_readback, qg_yexp[8],
                                     8 * 8, 1e-2f);

    /* fp16-output qgemm checks: expected = f16(f32 reference). The tolerance
       absorbs the f32 reference vs f32 shader diff (1e-3..1e-2 for the
       quantized formats) plus one f16 ULP of rounding at the store and the
       f16-rounded beta input on the two beta cases. */
    if (h.test_f16) {
        static const char* qg_f16_names[9] = {
            "qgemm q8_0 k=32 f16", "qgemm q8_0 k=64 f16", "qgemm q4k k=256 f16",
            "qgemm q4_0 k=32 f16", "qgemm q4_0 k=64 f16", "qgemm q5k k=256 f16",
            "qgemm q6k k=256 f16", "qgemm q3k k=256 f16", "qgemm iq4xs k=256 f16",
        };
        uint16_t exp16[8 * 8];
        for (int i = 0; i < 9; ++i) {
            for (uint32_t e = 0; e < 8 * 8; ++e)
                exp16[e] = f32_to_f16(qg_yexp[i][e]);
            overall_pass &= check_output_f16(qg_f16_names[i], h.mapped,
                                             qg_f16[i].off_readback16, exp16,
                                             8 * 8, 1e-2f);
        }
    }

cleanup:
    if (h.blas_ctx) vkblas_destroy_context(h.blas_ctx);
    if (f32b.A) vkDestroyBuffer(h.device, f32b.A, NULL);
    if (f32b.B) vkDestroyBuffer(h.device, f32b.B, NULL);
    if (f32b.D) vkDestroyBuffer(h.device, f32b.D, NULL);
    if (f16b.A) vkDestroyBuffer(h.device, f16b.A, NULL);
    if (f16b.B) vkDestroyBuffer(h.device, f16b.B, NULL);
    if (f16b.D) vkDestroyBuffer(h.device, f16b.D, NULL);
    if (f64b.A) vkDestroyBuffer(h.device, f64b.A, NULL);
    if (f64b.B) vkDestroyBuffer(h.device, f64b.B, NULL);
    if (f64b.D) vkDestroyBuffer(h.device, f64b.D, NULL);
    if (bf16b.A) vkDestroyBuffer(h.device, bf16b.A, NULL);
    if (bf16b.B) vkDestroyBuffer(h.device, bf16b.B, NULL);
    if (bf16b.D) vkDestroyBuffer(h.device, bf16b.D, NULL);
    if (sb32b.A) vkDestroyBuffer(h.device, sb32b.A, NULL);
    if (sb32b.B) vkDestroyBuffer(h.device, sb32b.B, NULL);
    if (sb32b.D) vkDestroyBuffer(h.device, sb32b.D, NULL);
    if (sb16b.A) vkDestroyBuffer(h.device, sb16b.A, NULL);
    if (sb16b.B) vkDestroyBuffer(h.device, sb16b.B, NULL);
    if (sb16b.D) vkDestroyBuffer(h.device, sb16b.D, NULL);
    if (sb64b.A) vkDestroyBuffer(h.device, sb64b.A, NULL);
    if (sb64b.B) vkDestroyBuffer(h.device, sb64b.B, NULL);
    if (sb64b.D) vkDestroyBuffer(h.device, sb64b.D, NULL);
    if (gex32b.D) vkDestroyBuffer(h.device, gex32b.D, NULL);
    if (gex16b.D) vkDestroyBuffer(h.device, gex16b.D, NULL);
    if (gex16bb.D) vkDestroyBuffer(h.device, gex16bb.D, NULL);
    if (bt64b.A) vkDestroyBuffer(h.device, bt64b.A, NULL);
    if (bt64b.D) vkDestroyBuffer(h.device, bt64b.D, NULL);
    if (q8_32.Wq) vkDestroyBuffer(h.device, q8_32.Wq, NULL);
    if (q8_32.x) vkDestroyBuffer(h.device, q8_32.x, NULL);
    if (q8_32.y) vkDestroyBuffer(h.device, q8_32.y, NULL);
    if (q8_64.Wq) vkDestroyBuffer(h.device, q8_64.Wq, NULL);
    if (q8_64.x) vkDestroyBuffer(h.device, q8_64.x, NULL);
    if (q8_64.y) vkDestroyBuffer(h.device, q8_64.y, NULL);
    if (q4k.Wq) vkDestroyBuffer(h.device, q4k.Wq, NULL);
    if (q4k.x) vkDestroyBuffer(h.device, q4k.x, NULL);
    if (q4k.y) vkDestroyBuffer(h.device, q4k.y, NULL);
    if (q40_32.Wq) vkDestroyBuffer(h.device, q40_32.Wq, NULL);
    if (q40_32.x) vkDestroyBuffer(h.device, q40_32.x, NULL);
    if (q40_32.y) vkDestroyBuffer(h.device, q40_32.y, NULL);
    if (q40_64.Wq) vkDestroyBuffer(h.device, q40_64.Wq, NULL);
    if (q40_64.x) vkDestroyBuffer(h.device, q40_64.x, NULL);
    if (q40_64.y) vkDestroyBuffer(h.device, q40_64.y, NULL);
    if (q5k.Wq) vkDestroyBuffer(h.device, q5k.Wq, NULL);
    if (q5k.x) vkDestroyBuffer(h.device, q5k.x, NULL);
    if (q5k.y) vkDestroyBuffer(h.device, q5k.y, NULL);
    if (q6k.Wq) vkDestroyBuffer(h.device, q6k.Wq, NULL);
    if (q6k.x) vkDestroyBuffer(h.device, q6k.x, NULL);
    if (q6k.y) vkDestroyBuffer(h.device, q6k.y, NULL);
    if (q3k.Wq) vkDestroyBuffer(h.device, q3k.Wq, NULL);
    if (q3k.x) vkDestroyBuffer(h.device, q3k.x, NULL);
    if (q3k.y) vkDestroyBuffer(h.device, q3k.y, NULL);
    if (iq4xs.Wq) vkDestroyBuffer(h.device, iq4xs.Wq, NULL);
    if (iq4xs.x) vkDestroyBuffer(h.device, iq4xs.x, NULL);
    if (iq4xs.y) vkDestroyBuffer(h.device, iq4xs.y, NULL);
    for (int i = 0; i < 9; ++i)
        if (qg_f16[i].y16) vkDestroyBuffer(h.device, qg_f16[i].y16, NULL);
    if (h.staging) vkDestroyBuffer(h.device, h.staging, NULL);
    if (h.mapped)  vkUnmapMemory(h.device, h.mem);
    if (h.mem != VK_NULL_HANDLE) vkFreeMemory(h.device, h.mem, NULL);
    if (h.fence != VK_NULL_HANDLE) vkDestroyFence(h.device, h.fence, NULL);
    if (h.cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(h.device, h.cmd_pool, 1, &h.cmd);
    if (h.cmd_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(h.device, h.cmd_pool, NULL);
    if (h.device != VK_NULL_HANDLE) vkDestroyDevice(h.device, NULL);
    if (h.instance != VK_NULL_HANDLE) vkDestroyInstance(h.instance, NULL);

    printf("test_vkblas: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
