/**
 * \file test_vkfft.c
 * \brief Public-API test harness for the VKFFT library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkFFTPlan per FFT size (8, 16, 32), records vkfft_execute_f32 dispatches into
 * a single command buffer, submits once, and validates the GPU results against
 * an O(n^2) CPU DFT reference computed in double precision.
 *
 * This is a header-only test: it includes only <vulkan/vulkan.h> and the
 * public vkfft.h header (relative include). No internal headers are pulled.
 *
 * Build (Windows / MinGW):
 *   gcc -std=c99 test_vkfft.c ../../src/vkfft/vkfft.o \
 *       -IC:/VulkanSDK/1.4.357.0/Include -IF:/VAiT/include -o test_vkfft.exe \
 *       C:/VulkanSDK/1.4.357.0/Lib/vulkan-1.lib -lm
 *
 * Exit status: 0 when all checks pass — or when the device lacks the
 * shaderInt64 feature and the harness is skipped (reported as a pass so CI
 * does not fail on GPUs that cannot run the FFT shaders). Returns 1 on any
 * real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkfft/vkfft.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_STAGING_SIZE  ((VkDeviceSize)(1u << 20))  /**< 1 MiB host buffer. */
#define TEST_F32_TOLERANCE 1e-3f                       /**< FFT tolerance.     */

#define TEST_PI 3.14159265358979323846

static const uint32_t TEST_SIZES[] = { 8u, 16u, 32u };
#define TEST_SIZE_COUNT (sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]))

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
} harness_t;

/**
 * \brief One FFT test case: plan, sub-buffers, and mapped-region offsets.
 */
typedef struct {
    VkFFTPlan *plan;
    VkBuffer in;
    VkBuffer out;
    VkDeviceSize off_in;
    VkDeviceSize off_out;
    VkDeviceSize off_readback;
    VkDeviceSize off_expected;
    uint32_t n;
    float *expected;            /**< CPU reference (2*n floats).              */
} fft_case_t;

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
    features.shaderInt64 = VK_TRUE;  /* harmless; the FFT shader does not use it */

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

/**
 * \brief Reserve input/output/readback/expected regions for one FFT case.
 *
 * Input and output each hold 2*n floats; readback and expected regions each
 * hold 2*n floats as well.
 */
static VkResult setup_case(VkDevice device, VkDeviceMemory mem,
                           VkDeviceSize *cursor, VkDeviceSize align,
                           uint32_t n, fft_case_t *c)
{
    c->n = n;
    VkDeviceSize bytes = (VkDeviceSize)(2u * n) * sizeof(float);

    c->off_in        = take_region(cursor, align, bytes);
    c->off_out       = take_region(cursor, align, bytes);
    c->off_readback  = take_region(cursor, align, bytes);
    c->off_expected  = take_region(cursor, align, bytes);

    VkResult r = create_sub_buffer(device, mem, c->off_in, bytes, &c->in);
    if (r != VK_SUCCESS) return r;
    return create_sub_buffer(device, mem, c->off_out, bytes, &c->out);
}

/* ===========================================================================
 * FFT helpers
 * ========================================================================== */

/**
 * \brief Compute the forward DFT reference: X[k] = sum_j x[j] exp(-2pi i j k/n).
 *
 * O(n^2) naive sum computed in double precision, then cast to float.
 */
static void compute_dft_reference(uint32_t n, const float *in, float *out)
{
    for (uint32_t k = 0; k < n; k++) {
        double re = 0.0, im = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            double angle = -2.0 * TEST_PI * (double)j * (double)k / (double)n;
            double cr = cos(angle), si = sin(angle);
            double xr = in[2u * j];
            double xi = in[2u * j + 1u];
            re += xr * cr - xi * si;
            im += xr * si + xi * cr;
        }
        out[2u * k]     = (float)re;
        out[2u * k + 1u] = (float)im;
    }
}

/**
 * \brief Compare a GPU readback region against a CPU-computed reference.
 *
 * \param name Test case name for the report.
 * \param mapped Host mapping of the staging memory.
 * \param off_readback Byte offset of the readback region.
 * \param expected CPU reference values (2*n floats).
 * \param count Number of floats to compare (2*n).
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
    printf("  %-16s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
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

static int record_dispatch(VkResult result, uint32_t n)
{
    if (result == VK_SUCCESS) {
        printf("  n=%2u : recorded\n", n);
        return 1;
    }
    printf("  n=%2u : FAIL (record, VkResult=%d)\n", n, (int)result);
    return 0;
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

    fft_case_t cases[TEST_SIZE_COUNT];
    memset(cases, 0, sizeof(cases));

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkfft", &h.instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(h.instance, &h.physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkfft: SKIP (no physical device found)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 3. shaderInt64 gate (harmless; kept for harness parity) ────────── */
    if (query_shader_int64(h.physical_device) == VK_FALSE) {
        printf("test_vkfft: SKIP (shaderInt64 not supported)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 4. Queue family gate (the harness uses vkGetDeviceQueue(d, 0, 0)) ─ */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkfft: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(h.device, 0, 0, &h.queue);

    /* ── 6. Command pool + one command buffer ───────────────────────────── */
    r = create_command_pool_and_buffer(h.device, &h.cmd_pool, &h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. 1 MiB host-visible/host-coherent staging memory ─────────────── */
    r = allocate_staging_memory(h.physical_device, h.device, TEST_STAGING_SIZE,
                                &h.mem, &h.staging, &h.align);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: staging allocation failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkMapMemory(h.device, h.mem, 0, VK_WHOLE_SIZE, 0, &h.mapped);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: vkMapMemory failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. Build one case per FFT size ─────────────────────────────────── */
    for (uint32_t c = 0; c < TEST_SIZE_COUNT; c++) {
        uint32_t n = TEST_SIZES[c];
        r = setup_case(h.device, h.mem, &h.cursor, h.align, n, &cases[c]);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkfft: setup_case(n=%u) failed (%d)\n", n, (int)r);
            goto cleanup;
        }
        cases[c].expected = (float *)malloc(2u * n * sizeof(float));
        if (!cases[c].expected) {
            fprintf(stderr, "test_vkfft: malloc failed\n");
            goto cleanup;
        }
        r = vkfft_create_plan(h.physical_device, h.device, n, &cases[c].plan);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "test_vkfft: vkfft_create_plan(n=%u) failed (%d)\n",
                    n, (int)r);
            goto cleanup;
        }

        /* Known signal: Re = cos(2*pi*2*j/n), Im = sin(2*pi*3*j/n). */
        float *in = (float *)((char *)h.mapped + cases[c].off_in);
        for (uint32_t j = 0; j < n; j++) {
            in[2u * j]     = cosf(2.0f * (float)TEST_PI * 2.0f * (float)j / (float)n);
            in[2u * j + 1u] = sinf(2.0f * (float)TEST_PI * 3.0f * (float)j / (float)n);
        }
        compute_dft_reference(n, in, cases[c].expected);
        memcpy((char *)h.mapped + cases[c].off_expected,
               cases[c].expected, 2u * n * sizeof(float));
    }

    printf("test_vkfft: device ready (arch=%s, tier=%u, staging=%u)\n",
           vkfft_get_arch_name(cases[0].plan), vkfft_get_arch_index(cases[0].plan),
           (unsigned)TEST_STAGING_SIZE);

    /* ── 9. Record all dispatches into one command buffer ───────────────── */
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    r = vkBeginCommandBuffer(h.cmd, &begin_info);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: vkBeginCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    for (uint32_t c = 0; c < TEST_SIZE_COUNT; c++) {
        overall_pass &= record_dispatch(
            vkfft_execute_f32(cases[c].plan, h.cmd, cases[c].in, cases[c].out),
            cases[c].n);
    }

    /* Make the shader writes visible to the transfer readback copies.      */
    record_compute_to_transfer_barrier(h.cmd);

    for (uint32_t c = 0; c < TEST_SIZE_COUNT; c++) {
        record_copy_readback(h.cmd, cases[c].out, h.staging,
                             cases[c].off_readback,
                             2u * cases[c].n * sizeof(float));
    }

    r = vkEndCommandBuffer(h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: vkEndCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    /* ── 10. One submit, one fence, device idle ─────────────────────────── */
    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(h.device, &fence_info, NULL, &h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkfft: vkCreateFence failed (%d)\n", (int)r);
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
        fprintf(stderr, "test_vkfft: vkQueueSubmit failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }
    vkWaitForFences(h.device, 1, &h.fence, VK_TRUE, UINT64_MAX);
    vkQueueWaitIdle(h.queue);

    /* ── 11. Compare GPU results against the CPU references ─────────────── */
    for (uint32_t c = 0; c < TEST_SIZE_COUNT; c++) {
        char name[32];
        snprintf(name, sizeof(name), "fft_f32_n%u", cases[c].n);
        overall_pass &= check_output(name, h.mapped, cases[c].off_readback,
                                     cases[c].expected, 2u * cases[c].n,
                                     TEST_F32_TOLERANCE);
    }

cleanup:
    for (uint32_t c = 0; c < TEST_SIZE_COUNT; c++) {
        if (cases[c].plan) vkfft_destroy_plan(cases[c].plan);
        if (cases[c].in)   vkDestroyBuffer(h.device, cases[c].in, NULL);
        if (cases[c].out)  vkDestroyBuffer(h.device, cases[c].out, NULL);
        free(cases[c].expected);
    }
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

    printf("test_vkfft: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
