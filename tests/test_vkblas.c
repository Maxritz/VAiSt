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
    memset(&coop_supported, 0, sizeof(coop_supported));
    coop_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk12_supported, 0, sizeof(vk12_supported));
    vk12_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk11_supported, 0, sizeof(vk11_supported));
    vk11_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    VkPhysicalDeviceFeatures2 supported;
    memset(&supported, 0, sizeof(supported));
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vk12_supported.pNext = &coop_supported;
    vk11_supported.pNext = &vk12_supported;
    supported.pNext = &vk11_supported;
    vkGetPhysicalDeviceFeatures2(physical_device, &supported);

    /* Build the enable chain (base -> vk11 -> vk12 -> coop matrix). */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_enable;
    VkPhysicalDeviceVulkan12Features vk12_enable;
    VkPhysicalDeviceVulkan11Features vk11_enable;
    VkPhysicalDeviceFeatures2 features2;
    memset(&coop_enable, 0, sizeof(coop_enable));
    coop_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memset(&vk12_enable, 0, sizeof(vk12_enable));
    vk12_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&vk11_enable, 0, sizeof(vk11_enable));
    vk11_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    features2.features.shaderInt64 = VK_TRUE;
    features2.features.shaderFloat64 = supported.features.shaderFloat64;
    features2.features.shaderInt16 = supported.features.shaderInt16;
    vk11_enable.storageBuffer16BitAccess = vk11_supported.storageBuffer16BitAccess;
    vk11_enable.uniformAndStorageBuffer16BitAccess =
        vk11_supported.uniformAndStorageBuffer16BitAccess;
    vk12_enable.shaderFloat16 = vk12_supported.shaderFloat16;
    vk12_enable.scalarBlockLayout = vk12_supported.scalarBlockLayout;
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
    gemm_buffers_t gex32b, gex16b;
    /* f64 alpha/beta packing check (non-trivial alpha/beta + C read) */
    gemm_buffers_t bt64b;
    double C64in[TEST_GEMM_M * TEST_GEMM_N];
    double Dbt_expected[TEST_GEMM_M * TEST_GEMM_N];

    memset(&f32b, 0, sizeof(f32b));
    memset(&f16b, 0, sizeof(f16b));
    memset(&f64b, 0, sizeof(f64b));
    memset(&bf16b, 0, sizeof(bf16b));
    memset(&sb32b, 0, sizeof(sb32b));
    memset(&sb16b, 0, sizeof(sb16b));
    memset(&sb64b, 0, sizeof(sb64b));
    memset(&gex32b, 0, sizeof(gex32b));
    memset(&gex16b, 0, sizeof(gex16b));
    memset(&bt64b, 0, sizeof(bt64b));

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
    r = vkblas_create_context(h.physical_device, h.device, &h.blas_ctx);
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
    r = create_sub_buffer(h.device, h.mem, gex32b.off_D,
                          m * n * sizeof(float), &gex32b.D);
    if (r != VK_SUCCESS) goto cleanup;
    if (h.test_f16) {
        r = create_sub_buffer(h.device, h.mem, gex16b.off_D,
                              m * n * sizeof(uint16_t), &gex16b.D);
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
    if (h.test_f64)
        record_copy_readback(h.cmd, bt64b.D, h.staging, bt64b.off_readback,
                             m * n * sizeof(double));

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

    if (h.test_f64)
        overall_pass &= check_output_f64("dgemm a/b pack", h.mapped,
                                         bt64b.off_readback, Dbt_expected,
                                         elem_count, TEST_F64_TOLERANCE);
    else
        printf("  %-18s : SKIP (f64 features absent)\n", "dgemm a/b pack");

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
    if (bt64b.A) vkDestroyBuffer(h.device, bt64b.A, NULL);
    if (bt64b.D) vkDestroyBuffer(h.device, bt64b.D, NULL);
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
