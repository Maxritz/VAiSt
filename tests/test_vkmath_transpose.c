/**
 * \file test_vkmath_transpose.c
 * \brief Minimal transpose test harness for VKMath.
 * Tests 4x3 → 3x4 transpose.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkmath/vkmath.h"

static VkResult create_instance(VkInstance *out) {
    VkApplicationInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "test_trans";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    return vkCreateInstance(&ci, NULL, out);
}

static VkBool32 query_shader_int64(VkPhysicalDevice pd) {
    VkPhysicalDeviceFeatures2 f2 = {0};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    return f2.features.shaderInt64;
}

static VkResult create_device(VkPhysicalDevice pd, VkDevice *out_dev, VkQueue *out_q) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {0};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = 0;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures2 f2 = {0};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    VkResult r = vkCreateDevice(pd, &dci, NULL, out_dev);
    if (r == VK_SUCCESS) vkGetDeviceQueue(*out_dev, 0, 0, out_q);
    return r;
}

static VkResult create_cmd(VkDevice dev, VkCommandPool *pool, VkCommandBuffer *cmd) {
    VkCommandPoolCreateInfo pci = {0};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = 0;
    VkResult r = vkCreateCommandPool(dev, &pci, NULL, pool);
    if (r != VK_SUCCESS) return r;
    VkCommandBufferAllocateInfo aci = {0};
    aci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    aci.commandPool = *pool;
    aci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    aci.commandBufferCount = 1;
    return vkAllocateCommandBuffers(dev, &aci, cmd);
}

static VkResult alloc_mem(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                          VkDeviceMemory *mem, VkBuffer *buf, void **mapped) {
    VkBufferCreateInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mi;
    r = vkAllocateMemory(dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) { vkDestroyBuffer(dev, *buf, NULL); return r; }
    vkBindBufferMemory(dev, *buf, *mem, 0);
    return vkMapMemory(dev, *mem, 0, mr.size, 0, mapped);
}

int main(void) {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkMathContext *ctx = NULL;
    VkBuffer in_buf = VK_NULL_HANDLE, out_buf = VK_NULL_HANDLE;
    VkDeviceMemory in_mem = VK_NULL_HANDLE, out_mem = VK_NULL_HANDLE;
    void *in_mapped = NULL, *out_mapped = NULL;
    int pass = 1;

    if (create_instance(&inst) != VK_SUCCESS) { printf("trans: FAIL (instance)\n"); return 1; }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) { printf("trans: SKIP\n"); vkDestroyInstance(inst, NULL); return 0; }
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    if (!query_shader_int64(pd)) { printf("trans: SKIP\n"); vkDestroyInstance(inst, NULL); return 0; }
    if (create_device(pd, &dev, &q) != VK_SUCCESS) { printf("trans: FAIL (device)\n"); pass = 0; goto cleanup; }
    if (vkmath_create_context(pd, dev, &ctx) != VK_SUCCESS) { printf("trans: FAIL (ctx)\n"); pass = 0; goto cleanup; }
    if (create_cmd(dev, &pool, &cmd) != VK_SUCCESS) { printf("trans: FAIL (cmd)\n"); pass = 0; goto cleanup; }

    uint32_t rows = 4, cols = 3;
    uint32_t total = rows * cols;
    VkDeviceSize bytes = total * sizeof(float);

    if (alloc_mem(pd, dev, bytes, &in_mem, &in_buf, &in_mapped) != VK_SUCCESS) {
        printf("trans: FAIL (alloc in)\n"); pass = 0; goto cleanup;
    }
    if (alloc_mem(pd, dev, bytes, &out_mem, &out_buf, &out_mapped) != VK_SUCCESS) {
        printf("trans: FAIL (alloc out)\n"); pass = 0; goto cleanup;
    }

    float in[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    memcpy(in_mapped, in, sizeof(in));

    float expected[] = {
        1.0f, 4.0f, 7.0f, 10.0f,
        2.0f, 5.0f, 8.0f, 11.0f,
        3.0f, 6.0f, 9.0f, 12.0f
    };

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferMemoryBarrier bmbs[2] = {0};
    bmbs[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmbs[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmbs[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmbs[0].buffer = in_buf;
    bmbs[0].size = VK_WHOLE_SIZE;

    bmbs[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmbs[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmbs[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmbs[1].buffer = out_buf;
    bmbs[1].size = VK_WHOLE_SIZE;

    vkBeginCommandBuffer(cmd, &bi);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 2, bmbs, 0, NULL);

    VkResult vr = vkmath_transpose_f32(ctx, cmd, rows, cols, in_buf, out_buf);
    if (vr != VK_SUCCESS) {
        printf("trans: FAIL (VkResult=%d)\n", (int)vr); pass = 0; goto submit;
    }
    printf("  transpose_f32      : recorded\n");

    bmbs[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmbs[1].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &bmbs[1], 0, NULL);

submit:
    vkEndCommandBuffer(cmd);
    vkCreateFence(dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL, &fence);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(q, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDeviceWaitIdle(dev);

    float *got = (float *)malloc(bytes);
    memcpy(got, out_mapped, bytes);

    int check_pass = 1;
    float tol = 1e-4f;
    for (uint32_t i = 0; i < total; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > tol) {
            printf("transpose MISMATCH [%u]: got=%.1f exp=%.1f\n", i, got[i], expected[i]);
            check_pass = 0; break;
        }
    }
    printf("  transpose_f32      : %s\n", check_pass ? "PASS" : "FAIL");
    pass &= check_pass;

cleanup:
    if (cmd) vkFreeCommandBuffers(dev, pool, 1, &cmd);
    if (pool) vkDestroyCommandPool(dev, pool, NULL);
    if (in_buf) vkDestroyBuffer(dev, in_buf, NULL);
    if (out_buf) vkDestroyBuffer(dev, out_buf, NULL);
    if (in_mem) { if (in_mapped) vkUnmapMemory(dev, in_mem); vkFreeMemory(dev, in_mem, NULL); }
    if (out_mem) { if (out_mapped) vkUnmapMemory(dev, out_mem); vkFreeMemory(dev, out_mem, NULL); }
    if (fence) vkDestroyFence(dev, fence, NULL);
    if (ctx) vkmath_destroy_context(ctx);
    if (dev) vkDestroyDevice(dev, NULL);
    if (inst) vkDestroyInstance(inst, NULL);
    free(got);

    printf("test_vkmath_transpose: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
