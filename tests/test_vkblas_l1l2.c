/**
 * \file test_vkblas_l1l2.c
 * \brief Public-API test harness for the vkblas_l1l2 companion library.
 *
 * Bootstraps a minimal Vulkan instance + physical/logical device (shaderInt64
 * enabled, queue family 0, one command buffer, 1 MiB host-coherent staging),
 * creates a VkBLASContext via vkblas_create_context(), then records and
 * validates BLAS Level-1 (axpy, scal, dot, nrm2, asum, amax) and Level-2
 * (gemv) dispatches, including f16 variants of axpy/scal/dot/gemv, against
 * CPU references.
 *
 * Build (Windows, gcc):
 *   gcc -std=c99 -IC:/VulkanSDK/1.4.357.0/Include -IF:/VAiT/include \
 *       F:/VAiT/tests/test_vkblas_l1l2.c <libs>/vkblas_l1l2.o <libs>/vkblas.o \
 *       C:/VulkanSDK/1.4.357.0/Lib/vulkan-1.lib -o test_vkblas_l1l2.exe
 *
 * Exit status: 0 when all checks pass, 1 on any failure.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkblas_l1l2/vkblas_l1l2.h"

/* ===========================================================================
 * Test parameters
 * ========================================================================== */

#define TEST_N            64        /**< Vector length for L1 ops.            */
#define TEST_INCX_STRIDED 2         /**< Strided axpy increment.             */
#define TEST_M            8         /**< gemv rows.                          */
#define TEST_N_GEMV       8         /**< gemv columns.                       */
#define TEST_STAGING_SIZE ((VkDeviceSize)(1u << 20)) /**< 1 MiB host buffer. */
#define TEST_F32_TOL      1e-4f     /**< f32 comparison tolerance.           */
#define TEST_F16_TOL      1e-2f     /**< f16 comparison tolerance.           */

#define TEST_AMAX_INDEX   17        /**< Known amax index (lowest of a tie). */
#define TEST_AMAX_TIE     33        /**< Second index with the same value.   */
#define TEST_AMAX_VALUE   100.0f    /**< Max |x| value.                      */

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
    VkBLASContext *blas_ctx;
} harness_t;

/* ===========================================================================
 * Bootstrap helpers (mirrors test_vkmath.c)
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
 * \brief Allocate a sub-buffer over a byte region of the staging memory.
 *        Returns the buffer handle and the region offset.
 */
static VkResult make_region(harness_t *h, VkDeviceSize bytes,
                            VkBuffer *out_buf, VkDeviceSize *out_off)
{
    *out_off = take_region(&h->cursor, h->align, bytes);
    return create_sub_buffer(h->device, h->mem, *out_off, bytes, out_buf);
}

/**
 * \brief Reserve a plain byte region (readback/expected) without creating a
 *        VkBuffer — readbacks copy straight into the staging buffer.
 */
static VkDeviceSize take_readback(harness_t *h, VkDeviceSize bytes)
{
    return take_region(&h->cursor, h->align, bytes);
}

/* ===========================================================================
 * Host fp16 helpers (must match the GLSL bit-pack/unpack)
 * ========================================================================== */

static float fp16_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u);
    uint32_t expo = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t man  = (uint32_t)(h & 0x3FFu);
    float mag;
    if (expo == 0u) {
        mag = (float)man * (1.0f / 16777216.0f);
    } else if (expo == 31u) {
        mag = 1.0e30f;
    } else {
        mag = (1.0f + (float)man / 1024.0f) * exp2f((float)((int)expo - 15));
    }
    return sign ? -mag : mag;
}

static uint16_t float_to_fp16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t e32  = (u >> 23) & 0xFFu;
    uint32_t m32  = u & 0x7FFFFFu;
    uint32_t h;

    if (e32 == 255u) {
        h = 0x7C00u;
    } else if (e32 == 0u) {
        h = 0u;
    } else {
        int e16 = (int)e32 - 127 + 15;
        if (e16 >= 31) {
            h = 0x7C00u;
        } else if (e16 <= 0) {
            if (e16 < -10) {
                h = 0u;
            } else {
                m32 |= 0x800000u;
                uint32_t shift = 1u - (uint32_t)e16;
                h = (m32 >> shift) + 0x1000u;
                h = (h >> 13) & 0x3FFu;
            }
        } else {
            h = ((uint32_t)e16 << 10) | (m32 >> 13);
            uint32_t rem = m32 & 0x1FFFu;
            if (rem > 0x1000u) h += 1u;
            else if (rem == 0x1000u && (h & 1u) == 1u) h += 1u;
        }
    }
    return (uint16_t)(sign | h);
}

/* ===========================================================================
 * Recording / comparison helpers
 * ========================================================================== */

static int record_dispatch(VkResult result, const char *name)
{
    if (result == VK_SUCCESS) {
        printf("  %-20s : recorded\n", name);
        return 1;
    }
    printf("  %-20s : FAIL (record, VkResult=%d)\n", name, (int)result);
    return 0;
}

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

static int check_float(const char *name, const void *mapped,
                       VkDeviceSize off, const float *expected,
                       uint32_t count, float tolerance)
{
    const float *got = (const float *)((const char *)mapped + off);
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
    printf("  %-20s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/* f16 buffers hold one half per uint32 (low 16 bits). */
static int check_f16(const char *name, const void *mapped, VkDeviceSize off,
                     const uint16_t *expected, uint32_t count, float tolerance)
{
    const uint32_t *got = (const uint32_t *)((const char *)mapped + off);
    int pass = 1;
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < count; i++) {
        float g = fp16_to_float((uint16_t)(got[i] & 0xFFFFu));
        float e = fp16_to_float(expected[i]);
        float diff = fabsf(g - e);
        if (diff > tolerance) {
            if (mismatches < 8) {
                printf("    mismatch[%u]: got %.6f expected %.6f (diff %.3e)\n",
                       i, g, e, diff);
            }
            mismatches++;
            pass = 0;
        }
    }
    printf("  %-20s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

/* ===========================================================================
 * main
 * ========================================================================== */

int main(void)
{
    harness_t h;
    memset(&h, 0, sizeof(h));

    /* CPU-side data and expected arrays */
    float x_f32[TEST_N], y_f32[TEST_N];
    float xstr_packed[TEST_N * TEST_INCX_STRIDED];   /* strided x (incx=2)   */
    float ystr[TEST_N];                              /* contiguous y         */
    float amax_data[TEST_N];
    float A[TEST_M * TEST_N_GEMV], x_gemv[TEST_N_GEMV], y_gemv[TEST_M];
    float y_gemvT[TEST_M];
    uint16_t xh[TEST_N], yh[TEST_N], xh_scal[TEST_N];
    uint16_t xh_gemv[TEST_N_GEMV], yh_gemv[TEST_M];
    uint16_t Ah[TEST_M * TEST_N_GEMV];

    float exp_axpy[TEST_N], exp_axpy_str[TEST_N], exp_scal[TEST_N];
    float exp_dot, exp_nrm2, exp_asum, exp_amax;
    float exp_gemv[TEST_M], exp_gemvT[TEST_M];
    uint16_t exp_axpy_f16[TEST_N], exp_scal_f16[TEST_N];
    float exp_dot_f16;
    uint16_t exp_gemv_f16[TEST_M];

    /* GPU buffer/region handles */
    VkBuffer xb, yb, xstrb, ystrb, amaxb, Ab, xg, yg, ygT;
    VkBuffer r_dot, r_nrm2, r_asum, r_amax, r_dot16;
    VkBuffer x16, y16, xs16, A16, xg16, yg16;
    VkDeviceSize off_x, off_y, off_xstr, off_ystr, off_amax, off_A, off_xg, off_yg, off_ygT;
    VkDeviceSize off_r_dot, off_r_nrm2, off_r_asum, off_r_amax, off_r_dot16;
    VkDeviceSize off_x16, off_y16, off_xs16, off_A16, off_xg16, off_yg16;
    VkDeviceSize off_rb_axpy, off_rb_axpy_str, off_rb_scal, off_rb_dot, off_rb_nrm2,
                 off_rb_asum, off_rb_amax, off_rb_gemv, off_rb_gemvT;
    VkDeviceSize off_rb_axpy16, off_rb_scal16, off_rb_dot16, off_rb_gemv16;

    int overall_pass = 1;
    VkResult r;

    const float alpha = 2.0f;
    const float beta_gemv = 2.0f;
    const float alpha_gemv = 0.5f;
    const float alpha_f16 = 2.0f;
    const float alpha_scal_f16 = 3.0f;

    /* ── 1. Instance ────────────────────────────────────────────────────── */
    r = create_instance("test_vkblas_l1l2", &h.instance);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkCreateInstance failed (%d)\n", (int)r);
        return 1;
    }

    /* ── 2. Physical device ─────────────────────────────────────────────── */
    r = find_physical_device(h.instance, &h.physical_device);
    if (r != VK_SUCCESS) {
        printf("test_vkblas_l1l2: SKIP (no physical device found)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 3. shaderInt64 gate (context creation uses shaderInt64-free shaders,
           but the bootstrap mirrors test_vkmath) ────────────────────────── */
    if (query_shader_int64(h.physical_device) == VK_FALSE) {
        printf("test_vkblas_l1l2: SKIP (shaderInt64 not supported)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 4. Queue family gate ───────────────────────────────────────────── */
    if (queue_family_supports_compute(h.physical_device, 0) == VK_FALSE) {
        printf("test_vkblas_l1l2: SKIP (queue family 0 lacks compute)\n");
        vkDestroyInstance(h.instance, NULL);
        return 0;
    }

    /* ── 5. Logical device ──────────────────────────────────────────────── */
    r = create_device(h.physical_device, &h.device);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkCreateDevice failed (%d)\n", (int)r);
        goto cleanup;
    }
    vkGetDeviceQueue(h.device, 0, 0, &h.queue);

    /* ── 6. Command pool + one command buffer ───────────────────────────── */
    r = create_command_pool_and_buffer(h.device, &h.cmd_pool, &h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: command pool/buffer failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 7. 1 MiB host-visible/host-coherent staging memory ─────────────── */
    r = allocate_staging_memory(h.physical_device, h.device, TEST_STAGING_SIZE,
                                &h.mem, &h.staging, &h.align);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: staging allocation failed (%d)\n", (int)r);
        goto cleanup;
    }
    r = vkMapMemory(h.device, h.mem, 0, VK_WHOLE_SIZE, 0, &h.mapped);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkMapMemory failed (%d)\n", (int)r);
        goto cleanup;
    }

    /* ── 8. VkBLASContext (shared, reused by vkblas_l1l2) ───────────────── */
    r = vkblas_create_context(h.instance, h.physical_device, h.device, &h.blas_ctx);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkblas_create_context failed (%d)\n", (int)r);
        goto cleanup;
    }
    printf("test_vkblas_l1l2: device ready (arch=%s, tier=%u, staging=%u)\n",
           vkblas_get_arch_name(h.blas_ctx), vkblas_get_arch_index(h.blas_ctx),
           (unsigned)TEST_STAGING_SIZE);

    /* ── 9. Allocate buffers and readback regions ───────────────────────── */
    {
        VkDeviceSize bytes64 = (VkDeviceSize)TEST_N * sizeof(float);
        VkDeviceSize bytes128 = (VkDeviceSize)(TEST_N * TEST_INCX_STRIDED) * sizeof(float);
        VkDeviceSize bytes_gemv = (VkDeviceSize)(TEST_M * TEST_N_GEMV) * sizeof(float);
        VkDeviceSize bytes_m = (VkDeviceSize)TEST_M * sizeof(float);

        /* f32 vector ops */
        if (make_region(&h, bytes64, &xb, &off_x) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &yb, &off_y) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes128, &xstrb, &off_xstr) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &ystrb, &off_ystr) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &amaxb, &off_amax) != VK_SUCCESS) goto cleanup;

        /* gemv */
        if (make_region(&h, bytes_gemv, &Ab, &off_A) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes_m, &xg, &off_xg) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes_m, &yg, &off_yg) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes_m, &ygT, &off_ygT) != VK_SUCCESS) goto cleanup;

        /* reduction result buffers (scratch room for partials) */
        if (make_region(&h, bytes64, &r_dot, &off_r_dot) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &r_nrm2, &off_r_nrm2) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &r_asum, &off_r_asum) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes128, &r_amax, &off_r_amax) != VK_SUCCESS) goto cleanup; /* pairs */

        /* f16 buffers (one half per uint32) */
        if (make_region(&h, bytes64, &x16, &off_x16) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &y16, &off_y16) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &xs16, &off_xs16) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes_gemv, &A16, &off_A16) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes_m, &xg16, &off_xg16) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes_m, &yg16, &off_yg16) != VK_SUCCESS) goto cleanup;
        if (make_region(&h, bytes64, &r_dot16, &off_r_dot16) != VK_SUCCESS) goto cleanup;

        /* readback regions (one per checked output) */
        off_rb_axpy    = take_readback(&h, bytes64);
        off_rb_axpy_str = take_readback(&h, bytes64);
        off_rb_scal    = take_readback(&h, bytes64);
        off_rb_dot     = take_readback(&h, bytes64);
        off_rb_nrm2    = take_readback(&h, bytes64);
        off_rb_asum    = take_readback(&h, bytes64);
        off_rb_amax    = take_readback(&h, bytes64);
        off_rb_gemv    = take_readback(&h, bytes_m);
        off_rb_gemvT   = take_readback(&h, bytes_m);
        off_rb_axpy16  = take_readback(&h, bytes64);
        off_rb_scal16  = take_readback(&h, bytes64);
        off_rb_dot16   = take_readback(&h, bytes64);
        off_rb_gemv16  = take_readback(&h, bytes_m);
    }

    /* ── 10. Fill inputs with known values ──────────────────────────────── */
    for (int i = 0; i < TEST_N; i++) {
        x_f32[i] = (float)((int)(i % 5) - 2);            /* -2,-1,0,1,2 ...  */
        y_f32[i] = 1.0f + 0.25f * (float)i;              /* 1.0 .. 16.75     */
        ystr[i] = y_f32[i];
        xstr_packed[i] = 0.0f;                           /* even slots below  */
        amax_data[i] = (float)(int)(i % 5) - 2.0f;       /* -2..2            */
        xh[i] = float_to_fp16(0.5f * (float)(i % 4));    /* 0,0.5,1,1.5 ...  */
        yh[i] = float_to_fp16(1.0f + 0.25f * (float)i);  /* 1.0 .. 16.75     */
        xh_scal[i] = float_to_fp16(0.5f + 0.5f * (float)(i % 4)); /* 0.5..2 */
    }
    for (int i = 0; i < TEST_N; i++) {
        xstr_packed[i * TEST_INCX_STRIDED] = 0.5f * (float)(i % 3); /* incx=2 */
    }
    amax_data[TEST_AMAX_INDEX] = TEST_AMAX_VALUE;
    amax_data[TEST_AMAX_TIE] = TEST_AMAX_VALUE;          /* tie -> lowest idx */

    for (int col = 0; col < TEST_N_GEMV; col++) {
        for (int row = 0; row < TEST_M; row++) {
            A[row + col * TEST_M] = 1.0f + (float)row + (float)col;
        }
    }
    for (int i = 0; i < TEST_N_GEMV; i++) {
        x_gemv[i] = (float)((i % 3) + 1);                /* 1,2,3 ...        */
    }
    for (int i = 0; i < TEST_M; i++) {
        y_gemv[i] = 100.0f + (float)i;                   /* beta path uses y */
        y_gemvT[i] = 10.0f * (float)(i + 1);
    }
    for (int i = 0; i < TEST_N_GEMV; i++) {
        xh_gemv[i] = float_to_fp16((float)((i % 3) + 1));
    }
    for (int i = 0; i < TEST_M; i++) {
        yh_gemv[i] = float_to_fp16(100.0f + (float)i);
    }
    for (int col = 0; col < TEST_N_GEMV; col++) {
        for (int row = 0; row < TEST_M; row++) {
            Ah[row + col * TEST_M] = float_to_fp16(1.0f + (float)row + (float)col);
        }
    }

    memcpy((char *)h.mapped + off_x, x_f32, sizeof(x_f32));
    memcpy((char *)h.mapped + off_y, y_f32, sizeof(y_f32));
    memcpy((char *)h.mapped + off_xstr, xstr_packed, sizeof(xstr_packed));
    memcpy((char *)h.mapped + off_ystr, ystr, sizeof(ystr));
    memcpy((char *)h.mapped + off_amax, amax_data, sizeof(amax_data));
    memcpy((char *)h.mapped + off_A, A, sizeof(A));
    memcpy((char *)h.mapped + off_xg, x_gemv, sizeof(x_gemv));
    memcpy((char *)h.mapped + off_yg, y_gemv, sizeof(y_gemv));
    memcpy((char *)h.mapped + off_ygT, y_gemvT, sizeof(y_gemvT));

    /* f16 buffers hold ONE HALF PER UINT32 (low 16 bits). A plain memcpy of
       the uint16_t arrays would pack two halves per slot, so write each half
       into its own 32-bit slot explicitly. */
    {
        uint32_t *p;
        p = (uint32_t *)((char *)h.mapped + off_x16);
        for (int i = 0; i < TEST_N; i++) p[i] = (uint32_t)xh[i];
        p = (uint32_t *)((char *)h.mapped + off_y16);
        for (int i = 0; i < TEST_N; i++) p[i] = (uint32_t)yh[i];
        p = (uint32_t *)((char *)h.mapped + off_xs16);
        for (int i = 0; i < TEST_N; i++) p[i] = (uint32_t)xh_scal[i];
        p = (uint32_t *)((char *)h.mapped + off_A16);
        for (int i = 0; i < TEST_M * TEST_N_GEMV; i++) p[i] = (uint32_t)Ah[i];
        p = (uint32_t *)((char *)h.mapped + off_xg16);
        for (int i = 0; i < TEST_N_GEMV; i++) p[i] = (uint32_t)xh_gemv[i];
        p = (uint32_t *)((char *)h.mapped + off_yg16);
        for (int i = 0; i < TEST_M; i++) p[i] = (uint32_t)yh_gemv[i];
    }

    /* ── 11. CPU reference values ───────────────────────────────────────── */
    for (int i = 0; i < TEST_N; i++) {
        exp_axpy[i] = alpha * x_f32[i] + y_f32[i];
        exp_axpy_str[i] = alpha * xstr_packed[i * TEST_INCX_STRIDED] + ystr[i];
        exp_scal[i] = alpha * x_f32[i];
    }

    exp_dot = 0.0f;
    for (int i = 0; i < TEST_N; i++) exp_dot += x_f32[i] * y_f32[i];

    exp_nrm2 = 0.0f;
    for (int i = 0; i < TEST_N; i++) exp_nrm2 += x_f32[i] * x_f32[i];
    exp_nrm2 = sqrtf(exp_nrm2);

    exp_asum = 0.0f;
    for (int i = 0; i < TEST_N; i++) exp_asum += fabsf(x_f32[i]);

    exp_amax = (float)TEST_AMAX_INDEX;                   /* 0-based, lowest idx */

    for (int row = 0; row < TEST_M; row++) {
        float s = 0.0f;
        for (int col = 0; col < TEST_N_GEMV; col++) {
            s += A[row + col * TEST_M] * x_gemv[col];
        }
        exp_gemv[row] = alpha_gemv * s + beta_gemv * y_gemv[row];

        float t = 0.0f;
        for (int col = 0; col < TEST_N_GEMV; col++) {
            t += A[col + row * TEST_M] * x_gemv[col];
        }
        exp_gemvT[row] = t;                              /* alpha=1,beta=0    */
    }

    for (int i = 0; i < TEST_N; i++) {
        exp_axpy_f16[i] = float_to_fp16(
            alpha_f16 * fp16_to_float(xh[i]) + fp16_to_float(yh[i]));
        exp_scal_f16[i] = float_to_fp16(
            alpha_scal_f16 * fp16_to_float(xh_scal[i]));
    }

    exp_dot_f16 = 0.0f;
    for (int i = 0; i < TEST_N; i++) {
        exp_dot_f16 += fp16_to_float(xh[i]) * fp16_to_float(yh[i]);
    }

    for (int row = 0; row < TEST_M; row++) {
        float s = 0.0f;
        for (int col = 0; col < TEST_N_GEMV; col++) {
            s += fp16_to_float(Ah[row + col * TEST_M]) * fp16_to_float(xh_gemv[col]);
        }
        exp_gemv_f16[row] = float_to_fp16(s);
    }

    /* ── 12. Record all dispatches into one command buffer ──────────────── */
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    r = vkBeginCommandBuffer(h.cmd, &begin_info);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkBeginCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    overall_pass &= record_dispatch(
        vkblas_l1_dot(h.blas_ctx, h.cmd, TEST_N, xb, 1, yb, 1, r_dot),
        "dot_f32");

    overall_pass &= record_dispatch(
        vkblas_l1_nrm2(h.blas_ctx, h.cmd, TEST_N, xb, 1, r_nrm2),
        "nrm2_f32");

    overall_pass &= record_dispatch(
        vkblas_l1_asum(h.blas_ctx, h.cmd, TEST_N, xb, 1, r_asum),
        "asum_f32");

    overall_pass &= record_dispatch(
        vkblas_l1_amax(h.blas_ctx, h.cmd, TEST_N, amaxb, 1, r_amax),
        "amax_f32");

    overall_pass &= record_dispatch(
        vkblas_l2_gemv(h.blas_ctx, h.cmd, VKBLAS_OP_N, TEST_M, TEST_N_GEMV,
                       &alpha_gemv, Ab, TEST_M, xg, 1, &beta_gemv, yg, 1),
        "gemv_f32_N");

    overall_pass &= record_dispatch(
        vkblas_l2_gemv(h.blas_ctx, h.cmd, VKBLAS_OP_T, TEST_M, TEST_N_GEMV,
                       &(const float){1.0f}, Ab, TEST_M, xg, 1, &(const float){0.0f}, ygT, 1),
        "gemv_f32_T");

    /* In-place L1 modifiers run last so reductions read pristine inputs.    */
    overall_pass &= record_dispatch(
        vkblas_l1_axpy(h.blas_ctx, h.cmd, TEST_N, &alpha, xb, 1, yb, 1),
        "axpy_f32");

    overall_pass &= record_dispatch(
        vkblas_l1_axpy(h.blas_ctx, h.cmd, TEST_N, &alpha, xstrb, TEST_INCX_STRIDED, ystrb, 1),
        "axpy_f32_strided");

    overall_pass &= record_dispatch(
        vkblas_l1_scal(h.blas_ctx, h.cmd, TEST_N, &alpha, xb, 1),
        "scal_f32");

    overall_pass &= record_dispatch(
        vkblas_l1_dot_f16(h.blas_ctx, h.cmd, TEST_N, x16, 1, y16, 1, r_dot16),
        "dot_f16");

    overall_pass &= record_dispatch(
        vkblas_l1_axpy_f16(h.blas_ctx, h.cmd, TEST_N, &alpha_f16, x16, 1, y16, 1),
        "axpy_f16");

    overall_pass &= record_dispatch(
        vkblas_l1_scal_f16(h.blas_ctx, h.cmd, TEST_N, &alpha_scal_f16, xs16, 1),
        "scal_f16");

    overall_pass &= record_dispatch(
        vkblas_l2_gemv_f16(h.blas_ctx, h.cmd, VKBLAS_OP_N, TEST_M, TEST_N_GEMV,
                           &(const float){1.0f}, A16, TEST_M, xg16, 1,
                           &(const float){0.0f}, yg16, 1),
        "gemv_f16");

    /* Make the shader writes visible to the transfer readback copies.      */
    record_compute_to_transfer_barrier(h.cmd);

    record_copy_readback(h.cmd, yb, h.staging, off_rb_axpy, sizeof(y_f32));
    record_copy_readback(h.cmd, ystrb, h.staging, off_rb_axpy_str,
                         (VkDeviceSize)TEST_N * sizeof(float));
    record_copy_readback(h.cmd, xb, h.staging, off_rb_scal, sizeof(x_f32));
    record_copy_readback(h.cmd, r_dot, h.staging, off_rb_dot, sizeof(float));
    record_copy_readback(h.cmd, r_nrm2, h.staging, off_rb_nrm2, sizeof(float));
    record_copy_readback(h.cmd, r_asum, h.staging, off_rb_asum, sizeof(float));
    record_copy_readback(h.cmd, r_amax, h.staging, off_rb_amax, sizeof(float));
    record_copy_readback(h.cmd, yg, h.staging, off_rb_gemv, sizeof(y_gemv));
    record_copy_readback(h.cmd, ygT, h.staging, off_rb_gemvT, sizeof(y_gemvT));
    record_copy_readback(h.cmd, y16, h.staging, off_rb_axpy16,
                         (VkDeviceSize)TEST_N * sizeof(uint32_t));
    record_copy_readback(h.cmd, xs16, h.staging, off_rb_scal16,
                         (VkDeviceSize)TEST_N * sizeof(uint32_t));
    record_copy_readback(h.cmd, r_dot16, h.staging, off_rb_dot16, sizeof(float));
    record_copy_readback(h.cmd, yg16, h.staging, off_rb_gemv16,
                         (VkDeviceSize)TEST_M * sizeof(uint32_t));

    r = vkEndCommandBuffer(h.cmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkEndCommandBuffer failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }

    /* ── 13. One submit, one fence, device idle ─────────────────────────── */
    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(h.device, &fence_info, NULL, &h.fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "test_vkblas_l1l2: vkCreateFence failed (%d)\n", (int)r);
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
        fprintf(stderr, "test_vkblas_l1l2: vkQueueSubmit failed (%d)\n", (int)r);
        overall_pass = 0;
        goto cleanup;
    }
    vkWaitForFences(h.device, 1, &h.fence, VK_TRUE, UINT64_MAX);
    vkQueueWaitIdle(h.queue);

    /* ── 14. Compare GPU results against the CPU references ─────────────── */
    printf("\n-- verification (tolerances f32=%.0e f16=%.0e) --\n",
           TEST_F32_TOL, TEST_F16_TOL);

    overall_pass &= check_float("axpy_f32", h.mapped, off_rb_axpy,
                                exp_axpy, TEST_N, TEST_F32_TOL);
    overall_pass &= check_float("axpy_f32_strided", h.mapped, off_rb_axpy_str,
                                exp_axpy_str, TEST_N, TEST_F32_TOL);
    overall_pass &= check_float("scal_f32", h.mapped, off_rb_scal,
                                exp_scal, TEST_N, TEST_F32_TOL);
    overall_pass &= check_float("dot_f32", h.mapped, off_rb_dot,
                                &exp_dot, 1, TEST_F32_TOL);
    overall_pass &= check_float("nrm2_f32", h.mapped, off_rb_nrm2,
                                &exp_nrm2, 1, TEST_F32_TOL);
    overall_pass &= check_float("asum_f32", h.mapped, off_rb_asum,
                                &exp_asum, 1, TEST_F32_TOL);
    overall_pass &= check_float("amax_f32", h.mapped, off_rb_amax,
                                &exp_amax, 1, TEST_F32_TOL);
    overall_pass &= check_float("gemv_f32_N", h.mapped, off_rb_gemv,
                                exp_gemv, TEST_M, TEST_F32_TOL);
    overall_pass &= check_float("gemv_f32_T", h.mapped, off_rb_gemvT,
                                exp_gemvT, TEST_M, TEST_F32_TOL);
    overall_pass &= check_f16("axpy_f16", h.mapped, off_rb_axpy16,
                              exp_axpy_f16, TEST_N, TEST_F16_TOL);
    overall_pass &= check_f16("scal_f16", h.mapped, off_rb_scal16,
                              exp_scal_f16, TEST_N, TEST_F16_TOL);
    overall_pass &= check_float("dot_f16", h.mapped, off_rb_dot16,
                                &exp_dot_f16, 1, TEST_F16_TOL);
    overall_pass &= check_f16("gemv_f16", h.mapped, off_rb_gemv16,
                              exp_gemv_f16, TEST_M, TEST_F16_TOL);

cleanup:
    if (h.blas_ctx) vkblas_destroy_context(h.blas_ctx);
    vkDestroyBuffer(h.device, xb, NULL);
    vkDestroyBuffer(h.device, yb, NULL);
    vkDestroyBuffer(h.device, xstrb, NULL);
    vkDestroyBuffer(h.device, ystrb, NULL);
    vkDestroyBuffer(h.device, amaxb, NULL);
    vkDestroyBuffer(h.device, Ab, NULL);
    vkDestroyBuffer(h.device, xg, NULL);
    vkDestroyBuffer(h.device, yg, NULL);
    vkDestroyBuffer(h.device, ygT, NULL);
    vkDestroyBuffer(h.device, r_dot, NULL);
    vkDestroyBuffer(h.device, r_nrm2, NULL);
    vkDestroyBuffer(h.device, r_asum, NULL);
    vkDestroyBuffer(h.device, r_amax, NULL);
    vkDestroyBuffer(h.device, r_dot16, NULL);
    vkDestroyBuffer(h.device, x16, NULL);
    vkDestroyBuffer(h.device, y16, NULL);
    vkDestroyBuffer(h.device, xs16, NULL);
    vkDestroyBuffer(h.device, A16, NULL);
    vkDestroyBuffer(h.device, xg16, NULL);
    vkDestroyBuffer(h.device, yg16, NULL);
    if (h.staging) vkDestroyBuffer(h.device, h.staging, NULL);
    if (h.mapped) vkUnmapMemory(h.device, h.mem);
    if (h.mem != VK_NULL_HANDLE) vkFreeMemory(h.device, h.mem, NULL);
    if (h.fence != VK_NULL_HANDLE) vkDestroyFence(h.device, h.fence, NULL);
    if (h.cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(h.device, h.cmd_pool, 1, &h.cmd);
    if (h.cmd_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(h.device, h.cmd_pool, NULL);
    if (h.device != VK_NULL_HANDLE) vkDestroyDevice(h.device, NULL);
    if (h.instance != VK_NULL_HANDLE) vkDestroyInstance(h.instance, NULL);

    printf("test_vkblas_l1l2: %s\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
