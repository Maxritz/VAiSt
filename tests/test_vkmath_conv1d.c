/**
 * \file test_vkmath_conv1d.c
 * \brief Conv1D test harness for VKMath.
 * Validates conv1d_f32 (1D convolution via conv2d wrapper with kh=1).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <vkmath/vkmath.h>

static VkResult create_instance(VkInstance *out) {
    VkApplicationInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "test_conv1d";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    return vkCreateInstance(&ci, NULL, out);
}

static VkBool32 query_shader_int64(VkPhysicalDevice pd) {
    VkPhysicalDeviceFeatures2 f2;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    return f2.features.shaderInt64;
}

static VkResult create_device(VkPhysicalDevice pd, VkDevice *out_dev, VkQueue *out_q) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi;
    memset(&qi, 0, sizeof(qi));
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = 0;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan11Features f11;
    memset(&f11, 0, sizeof(f11));
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    VkPhysicalDeviceVulkan12Features f12;
    memset(&f12, 0, sizeof(f12));
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 f2;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f12;
    f12.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(pd, &f2);

    VkPhysicalDeviceFeatures2 en;
    memset(&en, 0, sizeof(en));
    en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    en.features.shaderInt64 = VK_TRUE;
    en.pNext = &f12;
    f12.pNext = &f11;

    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &en;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;

    VkResult r = vkCreateDevice(pd, &dci, NULL, out_dev);
    if (r == VK_SUCCESS) {
        vkGetDeviceQueue(*out_dev, 0, 0, out_q);
    }
    return r;
}

static VkResult alloc_mem(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                          VkDeviceMemory *mem, VkBuffer *buf) {
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(dev, &bi, NULL, buf);
    if (r != VK_SUCCESS) return r;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, *buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    uint32_t mi = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            mi = i; break;
        }
    }
    if (mi == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mi;

    r = vkAllocateMemory(dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) { vkDestroyBuffer(dev, *buf, NULL); return r; }
    vkBindBufferMemory(dev, *buf, *mem, 0);
    return VK_SUCCESS;
}

#define IN_W    8
#define KW      3
#define STRIDE  1
#define PAD     0
#define IN_C    1
#define OUT_C   1

static float kernel[KW] = {1.0f, 2.0f, 3.0f};
static float input[IN_W] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting conv1d test...\n");

    uint32_t out_w = (IN_W + 2*PAD - (KW-1)) / STRIDE + 1;
    printf("out_w = %u\n", out_w);

    float* expected = (float*)malloc(out_w * sizeof(float));
    for (uint32_t ow = 0; ow < out_w; ow++) {
        float val = 0.0f;
        for (uint32_t j = 0; j < KW; j++) {
            int src = (int)(ow * STRIDE - PAD + j);
            if (src >= 0 && src < IN_W) val += kernel[j] * input[src];
        }
        expected[ow] = val;
        printf("  expected[%u] = %.4f\n", ow, val);
    }

    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkMathContext *ctx = NULL;
    VkBuffer in_buf = VK_NULL_HANDLE, wb_buf = VK_NULL_HANDLE, out_buf = VK_NULL_HANDLE;
    VkDeviceMemory in_mem = VK_NULL_HANDLE, wb_mem = VK_NULL_HANDLE, out_mem = VK_NULL_HANDLE;
    int pass = 1;

    if (create_instance(&inst) != VK_SUCCESS) { printf("conv1d: FAIL (vkCreateInstance)\n"); return 1; }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) { printf("conv1d: SKIP (no GPU)\n"); vkDestroyInstance(inst, NULL); return 0; }
    vkEnumeratePhysicalDevices(inst, &n, &pd);

    if (!query_shader_int64(pd)) { printf("conv1d: SKIP (no shaderInt64)\n"); vkDestroyInstance(inst, NULL); return 0; }

    if (create_device(pd, &dev, &q) != VK_SUCCESS) { printf("conv1d: FAIL (vkCreateDevice)\n"); pass = 0; goto cleanup; }
    if (vkmath_create_context(pd, dev, &ctx) != VK_SUCCESS) { printf("conv1d: FAIL (vkmath_create_context)\n"); pass = 0; goto cleanup; }

    uint32_t w_elems = OUT_C * IN_C * 1 * KW;
    uint32_t b_elems = OUT_C;
    VkDeviceSize in_bytes = IN_C * 1 * IN_W * sizeof(float);
    VkDeviceSize wb_bytes = (w_elems + b_elems) * sizeof(float);
    VkDeviceSize out_bytes = OUT_C * 1 * out_w * sizeof(float);

    if (alloc_mem(pd, dev, in_bytes, &in_mem, &in_buf) != VK_SUCCESS) { printf("conv1d: FAIL (alloc in)\n"); pass = 0; goto cleanup; }
    if (alloc_mem(pd, dev, wb_bytes, &wb_mem, &wb_buf) != VK_SUCCESS) { printf("conv1d: FAIL (alloc wb)\n"); pass = 0; goto cleanup; }
    if (alloc_mem(pd, dev, out_bytes, &out_mem, &out_buf) != VK_SUCCESS) { printf("conv1d: FAIL (alloc out)\n"); pass = 0; goto cleanup; }

    /* Fill input */
    void *m;
    vkMapMemory(dev, in_mem, 0, in_bytes, 0, &m);
    memcpy(m, input, in_bytes);
    vkUnmapMemory(dev, in_mem);

    /* Fill weights */
    float *wb = (float*)malloc(wb_bytes);
    memcpy(wb, kernel, w_elems * sizeof(float));
    wb[w_elems] = 0.0f; /* bias = 0 */
    vkMapMemory(dev, wb_mem, 0, wb_bytes, 0, &m);
    memcpy(m, wb, wb_bytes);
    vkUnmapMemory(dev, wb_mem);
    free(wb);

    /* Cmd buffer */
    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    vkCreateCommandPool(dev, &cpi, NULL, &pool);
    VkCommandBufferAllocateInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    vkAllocateCommandBuffers(dev, &cbi, &cmd);

    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(cmd, &cbbi);

    VkResult vr = vkmath_conv1d_f32(ctx, cmd, IN_W, KW, STRIDE, PAD, IN_C, OUT_C,
        0, 0, in_buf, wb_buf, out_buf);
    if (vr != VK_SUCCESS) {
        printf("conv1d: FAIL (vkmath_conv1d_f32 VkResult=%d)\n", (int)vr);
        pass = 0;
    } else {
        printf("  conv1d_f32        : recorded\n");
    }
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(q);

    /* Read back */
    vkMapMemory(dev, out_mem, 0, out_bytes, 0, &m);
    float *out = (float*)m;
    printf("\nConv1D result:\n");
    for (uint32_t i = 0; i < out_w; i++) {
        float diff = fabsf(out[i] - expected[i]);
        if (diff > 0.01f) {
            printf("  MISMATCH out[%u]: expected=%.4f, got=%.4f\n", i, expected[i], out[i]);
            pass = 0;
        } else {
            printf("  out[%u] = %.4f\n", i, out[i]);
        }
    }
    printf("  conv1d_f32        : %s\n", pass ? "PASS" : "FAIL");

cleanup:
    if (out_mem) vkFreeMemory(dev, out_mem, NULL);
    if (out_buf) vkDestroyBuffer(dev, out_buf, NULL);
    if (wb_mem) vkFreeMemory(dev, wb_mem, NULL);
    if (wb_buf) vkDestroyBuffer(dev, wb_buf, NULL);
    if (in_mem) vkFreeMemory(dev, in_mem, NULL);
    if (in_buf) vkDestroyBuffer(dev, in_buf, NULL);
    if (pool) vkDestroyCommandPool(dev, pool, NULL);
    if (ctx) vkmath_destroy_context(ctx);
    if (dev) vkDestroyDevice(dev, NULL);
    if (inst) vkDestroyInstance(inst, NULL);
    free(expected);

    printf("\ntest_vkmath_conv1d: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
