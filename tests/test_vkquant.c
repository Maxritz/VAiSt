/**
 * \file test_vkquant.c
 * \brief Public-API test harness for the VKQuant library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device, creates a
 * VkQuantContext via vkquant_create_context(), records Q8_0 / Q4_0 / Q4_K /
 * Q6_K / IQ4_XS block dequantization dispatches and Q8_0 / Q4_0 forward
 * quantization dispatches into a single command buffer, submits once, and
 * validates the GPU results against CPU references implementing the exact
 * ggml block byte formats (dequantize_row_q4_K / q6_K / iq4_xs from
 * ggml-common.h).
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

#define TEST_NUM_BLOCKS    8u    /**< Blocks per legacy quant format.      */
#define TEST_Q8_BLOCK_SIZE 36u   /**< Bytes per Q8_0 block (4 + 32 int8).  */
#define TEST_Q4_BLOCK_SIZE 20u   /**< Bytes per Q4_0 block (4 + 16 nibbles).*/
#define TEST_ELEMS_PER_BLOCK 32u /**< f32 elements per block.              */
#define TEST_Q8_SCALE       0.5f /**< Q8_0 scale d.                         */
#define TEST_Q4_SCALE       1.5f /**< Q4_0 scale d.                         */
#define TEST_STAGING_SIZE  ((VkDeviceSize)(2u << 20))  /**< 2 MiB host buffer. */

#define TEST_Q4K_BLOCK_SIZE 144u  /**< Bytes per Q4_K block (ggml).        */
#define TEST_Q6K_BLOCK_SIZE 210u  /**< Bytes per Q6_K block (ggml).        */
#define TEST_IQ4XS_BLOCK_SIZE 136u/**< Bytes per IQ4_XS block (ggml).      */
#define TEST_K_BLOCKS       2u    /**< Super-blocks per K-format test.     */
#define TEST_K_ELEMS        (TEST_K_BLOCKS * 256u) /**< 512 f32 each.      */

#define TEST_QUANT_BLOCKS   4u    /**< 32-element blocks for quant tests.  */
#define TEST_QUANT_ELEMS    (TEST_QUANT_BLOCKS * 32u) /**< 128 f32 each.   */

#define TEST_F32_TOLERANCE 1e-4f /**< f32 comparison tolerance.            */
#define TEST_QUANT_TOLERANCE 0.1f /**< quant round-trip error bound.       */

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
    VkBuffer staging;           /**< Staging buffer spanning the memory.      */
    void *mapped;               /**< Host mapping of mem.                     */
    VkDeviceSize align;         /**< Buffer memory alignment for sub-buffers. */
    VkDeviceSize cursor;        /**< Sub-allocation cursor into mem.          */
    VkFence fence;
    VkQuantContext *quant_ctx;
} harness_t;

/**
 * \brief Describes the input/output regions for one dequantization op.
 */
typedef struct {
    VkBuffer in;
    VkBuffer out;
    VkDeviceSize off_in;
    VkDeviceSize off_out;
    VkDeviceSize off_readback;
    VkDeviceSize off_expected;
} op_t;

/**
 * \brief Describes a quantize -> dequantize round-trip op.
 *
 * in = f32 source, qbytes = quantized bytes (quantize output / dequant input),
 * out = dequant f32 output. off_readback receives the final f32 values.
 */
typedef struct {
    VkBuffer in;
    VkBuffer qbytes;
    VkBuffer out;
    VkDeviceSize off_in;
    VkDeviceSize off_q;
    VkDeviceSize off_out;
    VkDeviceSize off_readback;
} rtop_t;

/* ===========================================================================
 * f16 helpers (IEEE half precision)
 * ========================================================================== */

static float f16_to_f32(uint16_t h)
{
    uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            f = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp   = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    uint32_t expbits = (x >> 23) & 0xFFu;

    if (expbits == 0xFFu) { /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); /* overflow -> inf */
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* underflow -> 0 */
        mant |= 0x800000u;
        uint32_t shift = 14u - (uint32_t)exp;
        uint16_t half = (uint16_t)(mant >> shift);
        if (mant & (1u << (shift - 1u))) half++;
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)(((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) half++;
    return (uint16_t)(sign | half);
}

/* ===========================================================================
 * CPU references (bit-exact ggml dequant, ggml-common.h)
 * ========================================================================== */

static const int8_t iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void ref_dequant_q4k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 144;
        const uint8_t *scales = b + 4;
        const uint8_t *qs     = b + 16;
        float d  = f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
        float dm = f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
        int is = 0;
        uint8_t sc, mn;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &mn);
            float d1 = d * sc; float m1 = dm * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn);
            float d2 = d * sc; float m2 = dm * mn;
            const uint8_t *q = qs + (j / 64) * 32;
            for (int l = 0; l < 32; ++l) out[i * 256 + j + l]     = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) out[i * 256 + j + l + 32] = d2 * (q[l] >> 4) - m2;
            is += 2;
        }
    }
}

static void ref_dequant_q6k(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 210;
        const uint8_t *ql = b + 0;
        const uint8_t *qh = b + 128;
        const int8_t  *sc = (const int8_t *)(b + 192);
        float d = f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
        float *y = out + (size_t)i * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void ref_dequant_iq4xs(const uint8_t *bytes, float *out, int nb)
{
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = bytes + (size_t)i * 136;
        const uint8_t *qs = b + 8;
        float d = f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
        uint16_t scales_h = (uint16_t)(b[2] | (b[3] << 8));
        const uint8_t *scales_l = b + 4;
        for (int ib = 0; ib < 8; ++ib) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                out[i * 256 + ib * 32 + j]      = dl * iq4nl[qs[ib * 16 + j] & 0xF];
                out[i * 256 + ib * 32 + j + 16] = dl * iq4nl[qs[ib * 16 + j] >> 4];
            }
        }
    }
}

/* ===========================================================================
 * Block encoders (inverse of the ggml decode rules above)
 * ========================================================================== */

static void encode_q4k(uint8_t *dst, float d, float dm,
                       const uint8_t sc[8], const uint8_t mn[8],
                       const uint8_t nib[256])
{
    uint16_t dh = f32_to_f16(d), mh = f32_to_f16(dm);
    dst[0] = (uint8_t)(dh & 0xFF); dst[1] = (uint8_t)(dh >> 8);
    dst[2] = (uint8_t)(mh & 0xFF); dst[3] = (uint8_t)(mh >> 8);
    uint8_t *scales = dst + 4;
    for (int j = 0; j < 4; ++j) {
        scales[j]     = (uint8_t)((sc[j] & 0x3F) | ((sc[j + 4] >> 4) << 6));
        scales[j + 4] = (uint8_t)((mn[j] & 0x3F) | ((mn[j + 4] >> 4) << 6));
        scales[j + 8] = (uint8_t)((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
    }
    uint8_t *qs = dst + 16;
    for (int j = 0; j < 256; j += 64) {
        for (int l = 0; l < 32; ++l) {
            qs[(j / 64) * 32 + l] = (uint8_t)(nib[j + l] | (nib[j + l + 32] << 4));
        }
    }
}

static void encode_q6k(uint8_t *dst, float d, const int8_t sc[16],
                       const uint8_t level[256])
{
    uint16_t dh = f32_to_f16(d);
    dst[208] = (uint8_t)(dh & 0xFF); dst[209] = (uint8_t)(dh >> 8);
    memset(dst + 0, 0, 192); /* ql + qh zero first */
    for (int i = 0; i < 16; ++i) dst[192 + i] = (uint8_t)sc[i];
    for (int i = 0; i < 256; ++i) {
        int chunk = i / 128;
        int rem   = i % 128;
        int sub   = rem / 32;
        int l     = rem % 32;
        uint8_t ql4 = level[i] & 0xF;
        uint8_t qh2 = (level[i] >> 4) & 3;
        uint8_t *ql = dst + chunk * 64;
        uint8_t *qh = dst + 128 + chunk * 32;
        if (sub == 0) { ql[l] |= ql4;         qh[l] |= (uint8_t)(qh2 << 0); }
        else if (sub == 1) { ql[l + 32] |= ql4;       qh[l] |= (uint8_t)(qh2 << 2); }
        else if (sub == 2) { ql[l] |= (uint8_t)(ql4 << 4); qh[l] |= (uint8_t)(qh2 << 4); }
        else { ql[l + 32] |= (uint8_t)(ql4 << 4); qh[l] |= (uint8_t)(qh2 << 6); }
    }
}

static void encode_iq4xs(uint8_t *dst, float d, const uint8_t ls[8],
                         const uint8_t nib[256])
{
    uint16_t dh = f32_to_f16(d);
    dst[0] = (uint8_t)(dh & 0xFF); dst[1] = (uint8_t)(dh >> 8);
    memset(dst + 2, 0, 6); /* scales_h + scales_l */
    uint16_t sh = 0;
    for (int ib = 0; ib < 8; ++ib) {
        sh |= (uint16_t)((ls[ib] >> 4) & 3) << (2 * ib);
    }
    dst[2] = (uint8_t)(sh & 0xFF); dst[3] = (uint8_t)(sh >> 8);
    for (int ib = 0; ib < 8; ++ib) {
        uint8_t low = ls[ib] & 0xF;
        if (ib & 1) dst[4 + ib / 2] |= (uint8_t)(low << 4);
        else        dst[4 + ib / 2] |= low;
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int j = 0; j < 16; ++j) {
            dst[8 + ib * 16 + j] = (uint8_t)((nib[ib * 32 + j + 16] << 4) | nib[ib * 32 + j]);
        }
    }
}

/* Deterministic f32 source for the quant round-trip tests (|x| <= ~0.56).
 * Q4_0 uses qmax=8 so d=max/8 and the worst round-trip error ~ d/2; keeping
 * the amplitude modest keeps max error well under the 0.1 tolerance. */
static float gen_quant_src(uint32_t i)
{
    return 0.50f * sinf((float)(i * 7u) * 0.17f) + 0.03f * (float)(i % 7);
}

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

static VkResult setup_roundtrip_op(VkDevice device, VkDeviceMemory mem,
                                   VkDeviceSize *cursor, VkDeviceSize align,
                                   uint32_t num_blocks, uint32_t block_bytes,
                                   rtop_t *op)
{
    VkDeviceSize in_bytes = (VkDeviceSize)num_blocks * 32u * sizeof(float);
    VkDeviceSize q_bytes  = (VkDeviceSize)num_blocks * block_bytes;
    VkDeviceSize out_bytes = (VkDeviceSize)num_blocks * 32u * sizeof(float);

    op->off_in       = take_region(cursor, align, in_bytes);
    op->off_q        = take_region(cursor, align, q_bytes);
    op->off_out      = take_region(cursor, align, out_bytes);
    op->off_readback = take_region(cursor, align, out_bytes);

    VkResult r = create_sub_buffer(device, mem, op->off_in, in_bytes, &op->in);
    if (r != VK_SUCCESS) return r;
    r = create_sub_buffer(device, mem, op->off_q, q_bytes, &op->qbytes);
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

static void record_compute_to_compute_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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

/* Round-trip check: dequant output vs the original f32 source (max abs err). */
static int check_roundtrip(const char *name, const void *mapped,
                           VkDeviceSize off_readback, uint32_t count,
                           float tolerance)
{
    const float *got = (const float *)((const char *)mapped + off_readback);
    int pass = 1;
    float max_err = 0.0f;
    uint32_t mismatches = 0;

    for (uint32_t i = 0; i < count; i++) {
        float expected = gen_quant_src(i);
        float diff = fabsf(got[i] - expected);
        if (diff > max_err) max_err = diff;
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.6f expected %.6f (diff %.3e)\n",
                       i, got[i], expected, diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-18s : %s (max abs err %.5f)\n", name, pass ? "PASS" : "FAIL", max_err);
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

    op_t op_q8, op_q4, op_q4k, op_q6k, op_iq4xs;
    rtop_t op_q8rt, op_q4rt;
    memset(&op_q8, 0, sizeof(op_q8));
    memset(&op_q4, 0, sizeof(op_q4));
    memset(&op_q4k, 0, sizeof(op_q4k));
    memset(&op_q6k, 0, sizeof(op_q6k));
    memset(&op_iq4xs, 0, sizeof(op_iq4xs));
    memset(&op_q8rt, 0, sizeof(op_q8rt));
    memset(&op_q4rt, 0, sizeof(op_q4rt));

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

    /* ── 7. Staging memory ──────────────────────────────────────────────── */
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

    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_K_BLOCKS * TEST_Q4K_BLOCK_SIZE,
                       TEST_K_ELEMS, &op_q4k);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_K_BLOCKS * TEST_Q6K_BLOCK_SIZE,
                       TEST_K_ELEMS, &op_q6k);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_quant_op(h.device, h.mem, &h.cursor, h.align,
                       (VkDeviceSize)TEST_K_BLOCKS * TEST_IQ4XS_BLOCK_SIZE,
                       TEST_K_ELEMS, &op_iq4xs);
    if (r != VK_SUCCESS) goto cleanup;

    r = setup_roundtrip_op(h.device, h.mem, &h.cursor, h.align,
                           TEST_QUANT_BLOCKS, TEST_Q8_BLOCK_SIZE, &op_q8rt);
    if (r != VK_SUCCESS) goto cleanup;
    r = setup_roundtrip_op(h.device, h.mem, &h.cursor, h.align,
                           TEST_QUANT_BLOCKS, TEST_Q4_BLOCK_SIZE, &op_q4rt);
    if (r != VK_SUCCESS) goto cleanup;

    /* ── 10. Fill raw quantized bytes with known values ─────────────────── */
    unsigned char *q8_bytes = (unsigned char *)h.mapped + op_q8.off_in;
    unsigned char *q4_bytes = (unsigned char *)h.mapped + op_q4.off_in;

    for (uint32_t b = 0; b < TEST_NUM_BLOCKS; b++) {
        uint32_t scale_off = b * TEST_Q8_BLOCK_SIZE;
        float scale = TEST_Q8_SCALE;
        memcpy(q8_bytes + scale_off, &scale, 4);
        for (uint32_t i = 0; i < TEST_ELEMS_PER_BLOCK; i++) {
            int8_t q = (int8_t)((i % 7) - 3);
            q8_bytes[scale_off + 4 + i] = (unsigned char)q;
        }
    }

    for (uint32_t b = 0; b < TEST_NUM_BLOCKS; b++) {
        uint32_t scale_off = b * TEST_Q4_BLOCK_SIZE;
        float scale = TEST_Q4_SCALE;
        memcpy(q4_bytes + scale_off, &scale, 4);
        for (uint32_t j = 0; j < 16; j++) {
            uint32_t e0 = 2u * j;
            uint32_t e1 = 2u * j + 1u;
            unsigned char nib0 = (unsigned char)(e0 % 16);
            unsigned char nib1 = (unsigned char)(e1 % 16);
            q4_bytes[scale_off + 4 + j] = (unsigned char)((nib1 << 4) | nib0);
        }
    }

    /* Encode Q4_K / Q6_K / IQ4_XS blocks from the same rules the CPU
     * reference and the shaders decode. */
    unsigned char *q4k_bytes = (unsigned char *)h.mapped + op_q4k.off_in;
    unsigned char *q6k_bytes = (unsigned char *)h.mapped + op_q6k.off_in;
    unsigned char *iq4xs_bytes = (unsigned char *)h.mapped + op_iq4xs.off_in;

    {
        const float q4k_d[2] = { 0.50f, 0.75f };
        const float q4k_dm[2] = { 0.25f, -0.125f };
        for (int b = 0; b < (int)TEST_K_BLOCKS; b++) {
            uint8_t sc[8], mn[8], nib[256];
            for (int j = 0; j < 8; ++j) {
                sc[j] = (uint8_t)((j * 7 + b * 3) & 0x3F);
                mn[j] = (uint8_t)((j * 5 + b * 11 + 3) & 0x3F);
            }
            for (int i = 0; i < 256; ++i) nib[i] = (uint8_t)((i * 7 + b * 13) & 0xF);
            encode_q4k(q4k_bytes + (size_t)b * TEST_Q4K_BLOCK_SIZE,
                       q4k_d[b], q4k_dm[b], sc, mn, nib);
        }
    }
    {
        const float q6k_d[2] = { 0.50f, 0.125f };
        const int8_t sc_b[2][16] = {
            { 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16 },
            { 16, -15, 14, -13, 12, -11, 10, -9, 8, -7, 6, -5, 4, -3, 2, -1 },
        };
        for (int b = 0; b < (int)TEST_K_BLOCKS; b++) {
            uint8_t level[256];
            for (int i = 0; i < 256; ++i) level[i] = (uint8_t)((i * 5 + b * 17) & 0x3F);
            encode_q6k(q6k_bytes + (size_t)b * TEST_Q6K_BLOCK_SIZE,
                       q6k_d[b], sc_b[b], level);
        }
    }
    {
        const float iq4_d[2] = { 1.0f, 0.25f };
        const uint8_t ls_b[2][8] = {
            { 16, 33, 48, 63, 0, 8, 40, 55 },
            { 63, 0, 31, 32, 24, 17, 50, 7 },
        };
        for (int b = 0; b < (int)TEST_K_BLOCKS; b++) {
            uint8_t nib[256];
            for (int i = 0; i < 256; ++i) nib[i] = (uint8_t)((i * 3 + b * 5) & 0xF);
            encode_iq4xs(iq4xs_bytes + (size_t)b * TEST_IQ4XS_BLOCK_SIZE,
                         iq4_d[b], ls_b[b], nib);
        }
    }

    /* Fill f32 quant sources. */
    float *q8_src = (float *)((char *)h.mapped + op_q8rt.off_in);
    float *q4_src = (float *)((char *)h.mapped + op_q4rt.off_in);
    for (uint32_t i = 0; i < TEST_QUANT_ELEMS; i++) {
        q8_src[i] = gen_quant_src(i);
        q4_src[i] = gen_quant_src(i);
    }

    /* ── 11. CPU reference dequant (exact block formats) ────────────────── */
    float *exp_q8 = (float *)((char *)h.mapped + op_q8.off_expected);
    float *exp_q4 = (float *)((char *)h.mapped + op_q4.off_expected);

    for (uint32_t idx = 0; idx < total; idx++) {
        uint32_t lane = idx % TEST_ELEMS_PER_BLOCK;
        float q8_v = (float)(int)((lane % 7) - 3);
        exp_q8[idx] = TEST_Q8_SCALE * q8_v;
        float q4_v = (float)(int)((lane % 16) - 8);
        exp_q4[idx] = TEST_Q4_SCALE * q4_v;
    }

    float *exp_q4k  = (float *)((char *)h.mapped + op_q4k.off_expected);
    float *exp_q6k  = (float *)((char *)h.mapped + op_q6k.off_expected);
    float *exp_iq4  = (float *)((char *)h.mapped + op_iq4xs.off_expected);
    ref_dequant_q4k(q4k_bytes, exp_q4k, TEST_K_BLOCKS);
    ref_dequant_q6k(q6k_bytes, exp_q6k, TEST_K_BLOCKS);
    ref_dequant_iq4xs(iq4xs_bytes, exp_iq4, TEST_K_BLOCKS);

    /* ── 12. Record all dispatches into one command buffer ──────────────── */
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
    overall_pass &= record_dispatch(
        vkquant_dequant_q4k_f32(h.quant_ctx, h.cmd, TEST_K_BLOCKS,
                                op_q4k.in, op_q4k.out),
        "dequant_q4k_f32");
    overall_pass &= record_dispatch(
        vkquant_dequant_q6k_f32(h.quant_ctx, h.cmd, TEST_K_BLOCKS,
                                op_q6k.in, op_q6k.out),
        "dequant_q6k_f32");
    overall_pass &= record_dispatch(
        vkquant_dequant_iq4xs_f32(h.quant_ctx, h.cmd, TEST_K_BLOCKS,
                                  op_iq4xs.in, op_iq4xs.out),
        "dequant_iq4xs_f32");

    overall_pass &= record_dispatch(
        vkquant_quantize_q8_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                  op_q8rt.in, op_q8rt.qbytes),
        "quantize_q8_0_f32");
    overall_pass &= record_dispatch(
        vkquant_quantize_q4_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                  op_q4rt.in, op_q4rt.qbytes),
        "quantize_q4_0_f32");

    /* Quantized bytes -> round-trip dequant. */
    record_compute_to_compute_barrier(h.cmd);

    overall_pass &= record_dispatch(
        vkquant_dequant_q8_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                 op_q8rt.qbytes, op_q8rt.out),
        "dequant(q8_0 rt)");
    overall_pass &= record_dispatch(
        vkquant_dequant_q4_0_f32(h.quant_ctx, h.cmd, TEST_QUANT_BLOCKS,
                                 op_q4rt.qbytes, op_q4rt.out),
        "dequant(q4_0 rt)");

    /* Make the shader writes visible to the transfer readback copies.      */
    record_compute_to_transfer_barrier(h.cmd);

    record_copy_readback(h.cmd, op_q8.out, h.staging,
                         op_q8.off_readback, total * sizeof(float));
    record_copy_readback(h.cmd, op_q4.out, h.staging,
                         op_q4.off_readback, total * sizeof(float));
    record_copy_readback(h.cmd, op_q4k.out, h.staging,
                         op_q4k.off_readback, TEST_K_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_q6k.out, h.staging,
                         op_q6k.off_readback, TEST_K_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_iq4xs.out, h.staging,
                         op_iq4xs.off_readback, TEST_K_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_q8rt.out, h.staging,
                         op_q8rt.off_readback, TEST_QUANT_ELEMS * sizeof(float));
    record_copy_readback(h.cmd, op_q4rt.out, h.staging,
                         op_q4rt.off_readback, TEST_QUANT_ELEMS * sizeof(float));

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
    overall_pass &= check_output("dequant_q4k_f32", h.mapped, op_q4k.off_readback,
                                 exp_q4k, TEST_K_ELEMS, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_q6k_f32", h.mapped, op_q6k.off_readback,
                                 exp_q6k, TEST_K_ELEMS, TEST_F32_TOLERANCE);
    overall_pass &= check_output("dequant_iq4xs_f32", h.mapped, op_iq4xs.off_readback,
                                 exp_iq4, TEST_K_ELEMS, TEST_F32_TOLERANCE);
    overall_pass &= check_roundtrip("quant+q8_0_rt", h.mapped, op_q8rt.off_readback,
                                    TEST_QUANT_ELEMS, TEST_QUANT_TOLERANCE);
    overall_pass &= check_roundtrip("quant+q4_0_rt", h.mapped, op_q4rt.off_readback,
                                    TEST_QUANT_ELEMS, TEST_QUANT_TOLERANCE);

cleanup:
    if (h.quant_ctx) vkquant_destroy_context(h.quant_ctx);
    if (op_q8.in)   vkDestroyBuffer(h.device, op_q8.in, NULL);
    if (op_q8.out)  vkDestroyBuffer(h.device, op_q8.out, NULL);
    if (op_q4.in)   vkDestroyBuffer(h.device, op_q4.in, NULL);
    if (op_q4.out)  vkDestroyBuffer(h.device, op_q4.out, NULL);
    if (op_q4k.in)  vkDestroyBuffer(h.device, op_q4k.in, NULL);
    if (op_q4k.out) vkDestroyBuffer(h.device, op_q4k.out, NULL);
    if (op_q6k.in)  vkDestroyBuffer(h.device, op_q6k.in, NULL);
    if (op_q6k.out) vkDestroyBuffer(h.device, op_q6k.out, NULL);
    if (op_iq4xs.in)  vkDestroyBuffer(h.device, op_iq4xs.in, NULL);
    if (op_iq4xs.out) vkDestroyBuffer(h.device, op_iq4xs.out, NULL);
    if (op_q8rt.in)    vkDestroyBuffer(h.device, op_q8rt.in, NULL);
    if (op_q8rt.qbytes) vkDestroyBuffer(h.device, op_q8rt.qbytes, NULL);
    if (op_q8rt.out)   vkDestroyBuffer(h.device, op_q8rt.out, NULL);
    if (op_q4rt.in)    vkDestroyBuffer(h.device, op_q4rt.in, NULL);
    if (op_q4rt.qbytes) vkDestroyBuffer(h.device, op_q4rt.qbytes, NULL);
    if (op_q4rt.out)   vkDestroyBuffer(h.device, op_q4rt.out, NULL);
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
