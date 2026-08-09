/**
 * \file test_vkquant.c
 * \brief Public-API test harness for the VKQuant library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkQuantContext via vkquant_create_context(), records Q8_0 and Q4_0 block
 * dequantization dispatches into a single command buffer, submits once, and
 * validates the GPU results against a CPU reference implementing the exact
 * block byte formats.
 *
 * This is a header-only test: it includes only <vulkan/vulkan.h> and the
 * public vkquant.h header (relative include). No internal headers are pulled.
 *
 * Exit status: 0 when all checks pass. Returns 1 on any real failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkquant/vkquant.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_NUM_BLOCKS    8u    /**< Blocks per quant format.              */
#define TEST_Q8_BLOCK_SIZE 36u   /**< Bytes per Q8_0 block (4 + 32 int8).   */
#define TEST_Q4_BLOCK_SIZE 20u   /**< Bytes per Q4_0 block (4 + 16 nibbles).*/
#define TEST_ELEMS_PER_BLOCK 32u /**< f32 elements per block.               */
#define TEST_Q8_SCALE       0.5f /**< Q8_0 scale d.                          */
#define TEST_Q4_SCALE       1.5f /**< Q4_0 scale d.                          */
#define TEST_STAGING_SIZE  ((VkDeviceSize)(1u << 20))  /**< 1 MiB host buffer. */
#define TEST_F32_TOLERANCE 1e-4f /**< f32 comparison tolerance.              */

/* ===========================================================================
 * Harness state
 * ========================================================================== */

/**
 * \brief Owns every Vulkan object the harness creates.
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
    VkQuantContext *quant_ctx;
} harness_t;

/**
 * \brief Describes the input/output regions for one dequantization op.
 *
 * in / out are distinct VkBuffer handles sub-allocated from the single
 * VkDeviceMemory. readback and expected are plain byte offsets into the same
 * (mapped) memory.
 */
typedef struct {
    VkBuffer in;
    VkBuffer out;
    VkDeviceSize off_in;
    VkDeviceSize off_out;
    VkDeviceSize off_readback;
    VkDeviceSize off_expected;
} op_t;

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

/**
 * \brief Query shaderInt64 support via VkPhysicalDeviceFeatures2.
 * Enabled for API parity with the other stack libraries; the VKQuant shaders
 * do not use 64-bit integers.
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
 * \brief Query shaderInt8 support via VkPhysicalDeviceVulkan12Features.
 * The VKQuant dequant shaders use 8-bit arithmetic types
 * (GL_EXT_shader_explicit_arithmetic_types), which require this feature.
 */
static VkBool32 query_shader_int8(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceVulkan12Features vulkan12;
    memset(&vulkan12, 0, sizeof(vulkan12));
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vulkan12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return vulkan12.shaderInt8;
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

/**
 * \brief Create a logical device on queue family 0.
 * Enables shaderInt64 (parity with test_vkmath) and shaderInt8 (required by
 * the 8-bit arithmetic types in the dequant shaders).
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

    VkPhysicalDeviceVulkan12Features vulkan12;
    memset(&vulkan12, 0, sizeof(vulkan12));
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12.shaderInt8 = VK_TRUE;

    VkDeviceCreateInfo create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &vulkan12;
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
 * \brief Allocate in/out/readback/expected regions for one dequant op.
 *
 * \param device Logical device.
 * \param mem Device memory backing all sub-buffers.
 * \param cursor Sub-allocation cursor into mem.
 * \param align Buffer memory alignment.
 * \param in_bytes Raw quantized input bytes.
 * \param out_count f32 output element count.
 * \param op Receives the region layout and buffer handles.
 */
static VkResult setup_quant_op(VkDevice device, VkDeviceMemory mem,
                               VkDeviceSize *cursor, VkDeviceSize align,
                               VkDeviceSize in_bytes, uint32_t out_count,
                               op_t *op)
{
    VkDeviceSize out_bytes = (VkDeviceSize)out_count * sizeof(float);

    op->off_in        = take_region(cursor, align, in_bytes);
    op->off_out       = take_region(cursor, align, out_bytes);
    op->off_readback  = take_region(cursor, align, out_bytes);
    op->off_expected  = take_region(cursor, align, out_bytes);

    VkResult r = create_sub_buffer(device, mem, op->off_in, in_bytes, &op->in);
    if (r != VK_SUCCESS) return r;
    return create_sub_buffer(device, mem, op->off_out, out_bytes, &op->out);
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

    op_t op_q8, op_q4;
    memset(&op_q8, 0, sizeof(op_q8));
    memset(&op_q4, 0, sizeof(op_q4));

    const uint32_t total = TEST_NUM_BLOCKS * TEST_ELEMS_PER_BLOCK;

    int overall_pass = 1;
    VkResult r;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkquant", &h.instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(h.instance, &h.physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkquant: SKIP (no physical device found)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 3. shaderInt8 gate (the dequant shaders use 8-bit types) ──────── */
    if (query_shader_int8(h.physical_device) == VK_FALSE) {
        printf("test_vkquant: SKIP (shaderInt8 not supported)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }
    /* shaderInt64 is enabled for API parity; not used by these shaders.    */
    (void)query_shader_int64(h.physical_device);

    /* ── 4. Queue family gate ───────────────────────────────────────────── */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkquant: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(h.device, 0, 0, &h.queue);

    /* ── 6. Command pool + one command buffer ───────────────────────────── */
    r = create_command_pool_and_buffer(h.device, &h.cmd_pool, &h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. 1 MiB host-visible/host-coherent staging memory ─────────────── */
    r = allocate_staging_memory(h.physical_device, h.device, TEST_STAGING_SIZE,
                                &h.mem, &h.staging, &h.align);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: staging allocation failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkMapMemory(h.device, h.mem, 0, VK_WHOLE_SIZE, 0, &h.mapped);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkMapMemory failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. Context ─────────────────────────────────────────────────────── */
    r = vkquant_create_context(h.physical_device, h.device, &h.quant_ctx);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkquant_create_context failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkquant: device ready (arch=%s, tier=%u, staging=%u)\n",
           vkquant_get_arch_name(h.quant_ctx), vkquant_get_arch_index(h.quant_ctx),
           (unsigned)TEST_STAGING_SIZE);

    /* ── 9. Region layout for each op ───────────────────────────────────── */
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_NUM_BLOCKS * TEST_Q8_BLOCK_SIZE,
                       total, &op_q8);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_NUM_BLOCKS * TEST_Q4_BLOCK_SIZE,
                       total, &op_q4);
    if (r != VK_SUCCESS) goto cleanup;

    /* ── 10. Fill raw quantized bytes with known values ─────────────────── */
    unsigned char *q8_bytes = (unsigned char *)h.mapped + op_q8.off_in;
    unsigned char *q4_bytes = (unsigned char *)h.mapped + op_q4.off_in;

    for (uint32_t b = 0; b < TEST_NUM_BLOCKS; b++) {
        uint32_t scale_off = b * TEST_Q8_BLOCK_SIZE;
        float scale = TEST_Q8_SCALE;
        memcpy(q8_bytes + scale_off, &scale, 4);
        for (uint32_t i = 0; i < TEST_ELEMS_PER_BLOCK; i++) {
            /* int8 qs in [-3, 3], repeating: (i%7)-3 */
            int8_t q = (int8_t)((i % 7) - 3);
            q8_bytes[scale_off + 4 + i] = (unsigned char)q;
        }
    }

    for (uint32_t b = 0; b < TEST_NUM_BLOCKS; b++) {
        uint32_t scale_off = b * TEST_Q4_BLOCK_SIZE;
        float scale = TEST_Q4_SCALE;
        memcpy(q4_bytes + scale_off, &scale, 4);
        for (uint32_t j = 0; j < 16; j++) {
            /* two nibbles per byte: low = element 2j, high = element 2j+1.
               nib = (elem % 16) -> v in [-8, 7]. */
            uint32_t e0 = 2u * j;
            uint32_t e1 = 2u * j + 1u;
            unsigned char nib0 = (unsigned char)(e0 % 16);
            unsigned char nib1 = (unsigned char)(e1 % 16);
            q4_bytes[scale_off + 4 + j] = (unsigned char)((nib1 << 4) | nib0);
        }
    }

    /* ── 11. CPU reference dequant (exact block formats) ────────────────── */
    float *exp_q8 = (float *)((char *)h.mapped + op_q8.off_expected);
    float *exp_q4 = (float *)((char *)h.mapped + op_q4.off_expected);

    for (uint32_t idx = 0; idx < total; idx++) {
        uint32_t b    = idx / TEST_ELEMS_PER_BLOCK;
        uint32_t lane = idx % TEST_ELEMS_PER_BLOCK;
        (void)b;

        /* Q8_0: out = d * qs[i], qs[i] = (i%7)-3.
           (int) cast BEFORE float: lane%7 is uint32_t, subtract would wrap. */
        float q8_v = (float)(int)((lane % 7) - 3);
        exp_q8[idx] = TEST_Q8_SCALE * q8_v;

        /* Q4_0: out = d * ((int)nibble - 8), nibble = (elem%16) */
        float q4_v = (float)(int)((lane % 16) - 8);
        exp_q4[idx] = TEST_Q4_SCALE * q4_v;
    }

    /* ── 12. Record both dispatches into one command buffer ─────────────── */
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    r = vkBeginCommandBuffer(h.cmd, &begin_info);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkBeginCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    overall_pass &= record_dispatch(
        vkquant_dequant_q8_0_f32(h.quant_ctx, h.cmd, TEST_NUM_BLOCKS,
                                 op_q8.in, op_q8.out),
        "dequant_q8_0_f32");

    overall_pass &= record_dispatch(
        vkquant_dequant_q4_0_f32(h.quant_ctx, h.cmd, TEST_NUM_BLOCKS,
                                 op_q4.in, op_q4.out),
        "dequant_q4_0_f32");

    /* Make the shader writes visible to the transfer readback copies.      */
    record_compute_to_transfer_barrier(h.cmd);

    record_copy_readback(h.cmd, op_q8.out, h.staging,
                         op_q8.off_readback, total * sizeof(float));
    record_copy_readback(h.cmd, op_q4.out, h.staging,
                         op_q4.off_readback, total * sizeof(float));

    r = vkEndCommandBuffer(h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkEndCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    /* ── 13. One submit, one fence, device idle ─────────────────────────── */
    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(h.device, &fence_info, NULL, &h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkquant: vkCreateFence failed (%d)\n", (int)r);
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
        fprintf(stderr, "test_vkquant: vkQueueSubmit failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }
    vkWaitForFences(h.device, 1, &h.fence, VK_TRUE, UINT64_MAX);
    vkQueueWaitIdle(h.queue);

    /* ── 14. Compare GPU results against the CPU references ─────────────── */
    overall_pass &= check_output("dequant_q8_0_f32", h.mapped, op_q8.off_readback,
                                 exp_q8, total, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_q4_0_f32", h.mapped, op_q4.off_readback,
                                 exp_q4, total, TEST_F32_TOLERANCE);

cleanup:
    if (h.quant_ctx) vkquant_destroy_context(h.quant_ctx);
    if (op_q8.in)   vkDestroyBuffer(h.device, op_q8.in, NULL);
    if (op_q8.out)  vkDestroyBuffer(h.device, op_q8.out, NULL);
    if (op_q4.in)   vkDestroyBuffer(h.device, op_q4.in, NULL);
    if (op_q4.out)  vkDestroyBuffer(h.device, op_q4.out, NULL);
    if (h.staging)  vkDestroyBuffer(h.device, h.staging, NULL);
    if (h.mapped)   vkUnmapMemory(h.device, h.mem);
    if (h.mem != VK_NULL_HANDLE) vkFreeMemory(h.device, h.mem, NULL);
    if (h.fence != VK_NULL_HANDLE) vkDestroyFence(h.device, h.fence, NULL);
    if (h.cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(h.device, h.cmd_pool, 1, &h.cmd);
    if (h.cmd_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(h.device, h.cmd_pool, NULL);
    if (h.device != VK_NULL_HANDLE) vkDestroyDevice(h.device, NULL);
    if (h.instance != VK_NULL_HANDLE) vkDestroyInstance(h.instance, NULL);

    printf("test_vkquant: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
