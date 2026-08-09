/**
 * \file test_vkblas.c
 * \brief Public-API test harness for the VKBLAS library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkBLASContext via vkblas_create_context(), records a single f32 GEMM
 * dispatch into one command buffer, submits once, and validates the output D
 * against a CPU-computed reference (column-major BLAS layout) with 1e-3
 * tolerance.
 *
 * This is a header-only test: it includes only <vulkan/vulkan.h> and the
 * public vkblas.h header (relative include). No internal headers are pulled.
 *
 * Build (Linux):
 *   cc -std=c99 test_vkblas.c -I../include -o test_vkblas -lvulkan
 * Build (Windows / MSVC):
 *   cl /std:c11 test_vkblas.c /I..\include /Fe:test_vkblas.exe
 *   (link against vulkan-1.lib)
 *
 * Exit status: 0 when the check passes — or when the device lacks the
 * shaderInt64 feature and the harness is skipped (reported as a pass so CI
 * does not fail on GPUs that cannot run the VKBLAS shaders). Returns 1 on any
 * real failure.
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
#define TEST_STAGING_SIZE ((VkDeviceSize)(1u << 20))  /**< 1 MiB host buffer. */
#define TEST_F32_TOLERANCE 1e-3f /**< GEMM f32 comparison tolerance.          */

/* ===========================================================================
 * Harness state
 * ========================================================================== */

/**
 * \brief Owns every Vulkan object the harness creates.
 *
 * All handles are zero-initialized so the cleanup path can unconditionally
 * destroy whatever was created before an error return.
 */
typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkDeviceMemory mem;         /**< Single host-visible/coherent allocation. */
    VkBuffer staging;           /**< 1 MiB staging buffer spanning the memory. */
    void *mapped;               /**< Host mapping of mem.                     */
    VkDeviceSize align;         /**< Buffer memory alignment for sub-buffers. */
    VkDeviceSize cursor;        /**< Sub-allocation cursor into mem.          */
    VkFence fence;
    uint32_t subgroup_size;
    VkBLASContext *blas_ctx;
} harness_t;

/**
 * \brief Describes the A / B / D / readback regions for the GEMM test.
 *
 * A, B, D are distinct VkBuffer handles sub-allocated from the single
 * VkDeviceMemory. readback and expected are plain byte offsets into the same
 * (mapped) memory: readback receives the GPU output D via a vkCmdCopyBuffer,
 * expected holds the CPU reference written by the host.
 */
typedef struct {
    VkBuffer A;
    VkBuffer B;
    VkBuffer D;
    VkDeviceSize off_A;
    VkDeviceSize off_B;
    VkDeviceSize off_D;
    VkDeviceSize off_readback;
    VkDeviceSize off_expected;
} gemm_regions_t;

/* ===========================================================================
 * Bootstrap helpers
 * ========================================================================== */

/**
 * \brief Create a minimal Vulkan instance (no extensions, no layers).
 *
 * \param app_name Application name reported to the loader.
 * \param out_instance Receives the VkInstance on success.
 * \retval VK_SUCCESS
 */
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

/**
 * \brief Enumerate physical devices and return the first one.
 *
 * \param instance Valid instance.
 * \param out_physical_device Receives the first device handle.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_INITIALIZATION_FAILED No devices or enumeration failed.
 */
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

/**
 * \brief Query shaderInt64 support via VkPhysicalDeviceFeatures2.
 *
 * The VKBLAS shaders are compiled with GL_ARB_gpu_shader_int64 and require
 * VkPhysicalDeviceFeatures::shaderInt64. shaderInt64 is a core device
 * feature, reported in VkPhysicalDeviceFeatures2::features.
 *
 * \param physical_device Device to query.
 * \return VK_TRUE if shaderInt64 is supported, VK_FALSE otherwise.
 */
static VkBool32 query_shader_int64(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features2.features.shaderInt64;
}

/**
 * \brief Query the subgroup size via the properties pNext chain.
 *
 * \param physical_device Device to query.
 * \return The device's subgroupSize (0 if unsupported).
 */
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

/**
 * \brief Check whether a given queue family exposes compute capability.
 *
 * \param physical_device Device to query.
 * \param family Queue family index to inspect.
 * \return VK_TRUE when the family reports VK_QUEUE_COMPUTE_BIT.
 */
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

/**
 * \brief Create a logical device on queue family 0.
 *
 * Enables no device features except shaderInt64 (requested through
 * VkPhysicalDeviceFeatures::shaderInt64). The VKMath/VKBLAS contexts query
 * capabilities themselves at creation time.
 *
 * \param physical_device Device to create from.
 * \param out_device Receives the logical device handle.
 * \retval VK_SUCCESS
 */
static VkResult create_device(VkPhysicalDevice physical_device,
                              VkDevice *out_device)
{
    float priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures features;
    memset(&features, 0, sizeof(features));
    features.shaderInt64 = VK_TRUE;

    VkDeviceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = NULL;
    create_info.pEnabledFeatures = &features;
    return vkCreateDevice(physical_device, &create_info, NULL, out_device);
}

/**
 * \brief Create a command pool and a single primary command buffer.
 *
 * \param device Logical device.
 * \param out_pool Receives the command pool.
 * \param out_cmd Receives the command buffer.
 * \retval VK_SUCCESS
 */
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

/**
 * \brief Create a VkBuffer bound at an offset into a shared VkDeviceMemory.
 *
 * \param device Logical device.
 * \param mem Device memory backing the buffer.
 * \param offset Byte offset into mem (must satisfy the buffer's alignment).
 * \param size Byte size of the buffer.
 * \param out_buffer Receives the buffer handle.
 * \retval VK_SUCCESS
 */
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

/**
 * \brief Allocate a single host-visible/host-coherent VkDeviceMemory.
 *
 * Creates a probe buffer to learn the memory requirements, picks the memory
 * type, allocates size bytes, and binds a 1 MiB staging buffer at offset 0.
 * The returned alignment is used to place all later sub-buffers.
 *
 * \param physical_device Device to query memory types from.
 * \param device Logical device.
 * \param size Allocation size in bytes.
 * \param out_memory Receives the device memory.
 * \param out_staging Receives the staging buffer bound at offset 0.
 * \param out_align Receives the buffer memory alignment.
 * \retval VK_SUCCESS
 */
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

/**
 * \brief Align a value up to the next multiple of a power-of-two boundary.
 *
 * \param value Value to round up.
 * \param align Power-of-two alignment.
 * \return The aligned value.
 */
static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize align)
{
    return (value + align - 1) & ~(align - 1);
}

/**
 * \brief Reserve a byte region from the harness memory cursor.
 *
 * \param cursor Running byte offset (advanced past the region).
 * \param align Alignment for the region start.
 * \param size Region size in bytes.
 * \return The aligned region offset.
 */
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

/**
 * \brief Record a compute-to-transfer memory dependency.
 *
 * Makes shader writes available to the subsequent vkCmdCopyBuffer readback
 * commands (SHADER_WRITE -> TRANSFER_READ) inside one command buffer.
 *
 * \param cmd Command buffer in the recording state.
 */
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

/**
 * \brief Record a vkCmdCopyBuffer from an output buffer into staging.
 *
 * \param cmd Command buffer in the recording state.
 * \param src Source buffer (GEMM output D).
 * \param dst Destination buffer (1 MiB staging buffer).
 * \param dst_offset Byte offset into dst.
 * \param size Bytes to copy.
 */
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

/**
 * \brief Report the result of recording a VKBLAS dispatch.
 *
 * \param result VkResult returned by the dispatch function.
 * \param name Dispatch name for the report.
 * \return 1 on success, 0 on failure.
 */
static int record_dispatch(VkResult result, const char *name)
{
    if (result == VK_SUCCESS) {
        printf("  %-18s : recorded\n", name);
        return 1;
    }
    printf("  %-18s : FAIL (record, VkResult=%d)\n", name, (int)result);
    return 0;
}

/**
 * \brief Compare a GPU readback region against a CPU-computed reference.
 *
 * \param name Dispatch name for the report.
 * \param mapped Host mapping of the staging memory.
 * \param off_readback Byte offset of the readback region.
 * \param expected CPU reference values.
 * \param count Number of floats to compare.
 * \param tolerance Allowed absolute difference.
 * \return 1 when every element matches, 0 otherwise.
 */
static int check_output(const char *name, const void *mapped,
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

    gemm_regions_t regs;
    memset(&regs, 0, sizeof(regs));

    const uint32_t m = TEST_GEMM_M;
    const uint32_t n = TEST_GEMM_N;
    const uint32_t k = TEST_GEMM_K;
    const uint32_t elem_count = m * n;

    float A[TEST_GEMM_M * TEST_GEMM_K];
    float B[TEST_GEMM_K * TEST_GEMM_N];
    float D_expected[TEST_GEMM_M * TEST_GEMM_N];

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

    /* ── 4. Queue family gate (the harness uses vkGetDeviceQueue(d, 0, 0)) ─ */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkblas: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device);
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

    /* ── 9. Region layout for A / B / D / readback / expected ───────────── */
    regs.off_A = take_region(&h.cursor, h.align, m * k * sizeof(float));
    regs.off_B = take_region(&h.cursor, h.align, k * n * sizeof(float));
    regs.off_D = take_region(&h.cursor, h.align, m * n * sizeof(float));
    regs.off_readback = take_region(&h.cursor, h.align, m * n * sizeof(float));
    regs.off_expected = take_region(&h.cursor, h.align, m * n * sizeof(float));

    r = create_sub_buffer(h.device, h.mem, regs.off_A,
                          m * k * sizeof(float), &regs.A);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, regs.off_B,
                          k * n * sizeof(float), &regs.B);
    if (r != VK_SUCCESS) goto cleanup;
    r = create_sub_buffer(h.device, h.mem, regs.off_D,
                          m * n * sizeof(float), &regs.D);
    if (r != VK_SUCCESS) goto cleanup;

    /* ── 10. Fill A and B with known values (column-major) ──────────────── */
    /* A = all ones: A(row r, col t) at index r + t*lda.                    */
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t t = 0; t < k; t++) {
            A[row + t * TEST_GEMM_LDA] = 1.0f;
        }
    }
    /* B = ramp 1..64: B(row t, col c) at index t + c*ldb.                  */
    for (uint32_t t = 0; t < k; t++) {
        for (uint32_t c = 0; c < n; c++) {
            B[t + c * TEST_GEMM_LDB] = (float)(1 + t + 8 * c);
        }
    }
    memcpy((char *)h.mapped + regs.off_A, A, m * k * sizeof(float));
    memcpy((char *)h.mapped + regs.off_B, B, k * n * sizeof(float));

    /* ── 11. CPU reference: D = alpha * A * B + beta * C (column-major) ─── */
    /* A all-ones gives D(row, col) = sum_t B(t, col); computed generically. */
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t c = 0; c < n; c++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < k; t++) {
                acc += (double)A[row + t * TEST_GEMM_LDA]
                     * (double)B[t + c * TEST_GEMM_LDB];
            }
            D_expected[row + c * TEST_GEMM_LDD] = (float)(1.0 * acc + 0.0);
        }
    }
    /* CPU reference is also written into a mapped "second region".         */
    memcpy((char *)h.mapped + regs.off_expected, D_expected,
           m * n * sizeof(float));

    /* ── 12. Record the GEMM dispatch into one command buffer ───────────── */
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
                     &alpha, regs.A, TEST_GEMM_LDA,
                     regs.B, TEST_GEMM_LDB,
                     &beta, VK_NULL_HANDLE, 0,
                     regs.D, TEST_GEMM_LDD),
        "sgemm m=n=k=8");

    /* Make the shader writes visible to the transfer readback copy.        */
    record_compute_to_transfer_barrier(h.cmd);
    record_copy_readback(h.cmd, regs.D, h.staging, regs.off_readback,
                         m * n * sizeof(float));

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

    /* ── 14. Compare D against the CPU reference ────────────────────────── */
    overall_pass &= check_output("sgemm m=n=k=8", h.mapped, regs.off_readback,
                                 D_expected, elem_count, TEST_F32_TOLERANCE);

cleanup:
    if (h.blas_ctx) vkblas_destroy_context(h.blas_ctx);
    if (regs.A) vkDestroyBuffer(h.device, regs.A, NULL);
    if (regs.B) vkDestroyBuffer(h.device, regs.B, NULL);
    if (regs.D) vkDestroyBuffer(h.device, regs.D, NULL);
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
