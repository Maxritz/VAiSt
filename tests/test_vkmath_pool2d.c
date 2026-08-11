/**
 * \file test_vkmath_pool2d.c
 * \brief Minimal pool2d test harness for VKMath.
 * Tests 3x3 max and avg pooling on a 4x4 input with 1 channel.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/vkmath/vkmath.h"

static VkResult create_instance(VkInstance *out) {
    VkApplicationInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "test_pool2d";
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
    VkPhysicalDeviceFeatures2 f2;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    VkResult r = vkCreateDevice(pd, &dci, NULL, out_dev);
    if (r == VK_SUCCESS) vkGetDeviceQueue(*out_dev, 0, 0, out_q);
    return r;
}

static VkResult create_cmd(VkDevice dev, VkCommandPool *pool, VkCommandBuffer *cmd) {
    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = 0;
    VkResult r = vkCreateCommandPool(dev, &pci, NULL, pool);
    if (r != VK_SUCCESS) return r;
    VkCommandBufferAllocateInfo aci;
    memset(&aci, 0, sizeof(aci));
    aci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    aci.commandPool = *pool;
    aci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    aci.commandBufferCount = 1;
    return vkAllocateCommandBuffers(dev, &aci, cmd);
}

static VkResult alloc_mem(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                          VkDeviceMemory *mem, VkBuffer *buf, void **mapped) {
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
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
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
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
    VkBuffer in_buf = VK_NULL_HANDLE, out_buf = VK_NULL_HANDLE, out_buf2 = VK_NULL_HANDLE;
    VkDeviceMemory in_mem = VK_NULL_HANDLE, out_mem = VK_NULL_HANDLE, out_mem2 = VK_NULL_HANDLE;
    void *in_mapped = NULL, *out_mapped = NULL, *out_mapped2 = NULL;

    int pass = 1;

    if (create_instance(&inst) != VK_SUCCESS) {
        printf("pool2d: FAIL (create instance)\n"); return 1;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) { printf("pool2d: SKIP\n"); vkDestroyInstance(inst, NULL); return 0; }
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    if (!query_shader_int64(pd)) {
        printf("pool2d: SKIP (no shaderInt64)\n"); vkDestroyInstance(inst, NULL); return 0;
    }
    if (create_device(pd, &dev, &q) != VK_SUCCESS) {
        printf("pool2d: FAIL (create device)\n"); pass = 0; goto cleanup;
    }
    if (vkmath_create_context(pd, dev, &ctx) != VK_SUCCESS) {
        printf("pool2d: FAIL (create context)\n"); pass = 0; goto cleanup;
    }
    if (create_cmd(dev, &pool, &cmd) != VK_SUCCESS) {
        printf("pool2d: FAIL (create cmd)\n"); pass = 0; goto cleanup;
    }

    /* 4x4 input, 1 channel, 3x3 pool, stride 1, pad 0 → 3x3 output */
    uint32_t in_w = 4, in_h = 4, in_c = 1;
    uint32_t kw = 3, kh = 3, sw = 1, sh = 1, pw = 0, ph = 0;
    uint32_t out_c = in_c; /* same for pool */
    uint32_t out_w = (in_w + 2*pw - (kw-1)) / sw + 1; /* 3 */
    uint32_t out_h = (in_h + 2*ph - (kh-1)) / sh + 1; /* 3 */

    uint32_t in_elems = in_c * in_h * in_w;
    uint32_t out_elems = out_c * out_h * out_w;

    VkDeviceSize in_bytes = in_elems * sizeof(float);
    VkDeviceSize out_bytes = out_elems * sizeof(float);

    if (alloc_mem(pd, dev, in_bytes, &in_mem, &in_buf, &in_mapped) != VK_SUCCESS) {
        printf("pool2d: FAIL (alloc in)\n"); pass = 0; goto cleanup;
    }
    if (alloc_mem(pd, dev, out_bytes, &out_mem, &out_buf, &out_mapped) != VK_SUCCESS) {
        printf("pool2d: FAIL (alloc out)\n"); pass = 0; goto cleanup;
    }
    if (alloc_mem(pd, dev, out_bytes, &out_mem2, &out_buf2, &out_mapped2) != VK_SUCCESS) {
        printf("pool2d: FAIL (alloc out2)\n"); pass = 0; goto cleanup;
    }

    float in[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    memcpy(in_mapped, in, sizeof(in));

    /* CPU references for max pool on 3x3 windows (4 output positions) */
    /* out(0,0): rows 0-2,cols 0-2 → 1,2,3,5,6,7,9,10,11 max=11
       out(0,1): rows 0-2,cols 1-3 → 2,3,4,6,7,8,10,11,12 max=12
       out(0,2): rows 0-2,cols 2-4 → only cols 2-3 (col 4 OOB) → 3,4,7,8,11,12 max=12
       out(1,0): rows 1-3,cols 0-2 → 5,6,7,9,10,11,13,14,15 max=15
       out(1,1): rows 1-3,cols 1-3 → 6,7,8,10,11,12,14,15,16 max=16
       out(1,2): rows 1-3,cols 2-4 → 7,8,11,12,15,16 max=16
       out(2,0): rows 2-4,cols 0-2 → only rows 2-3 → 9,10,11,13,14,15 max=15
       out(2,1): rows 2-4,cols 1-3 → 10,11,12,14,15,16 max=16
       out(2,2): rows 2-4,cols 2-4 → 11,12,15,16 max=16
    */
    float max_expected[] = { 11.0f, 12.0f, 12.0f, 15.0f, 16.0f, 16.0f, 15.0f, 16.0f, 16.0f };

    /* avg pool: same windows, sum/count */
    /* out(0,0): 1+2+3+5+6+7+9+10+11=54, n=9, avg=6.0
       out(0,1): 2+3+4+6+7+8+10+11+12=63, avg=7.0
       out(0,2): 3+4+7+8+11+12=45, n=6, avg=7.5
       out(1,0): 5+6+7+9+10+11+13+14+15=90, avg=10.0
       out(1,1): 6+7+8+10+11+12+14+15+16=109, avg=12.111...
       out(1,2): 7+8+11+12+15+16=69, n=6, avg=11.5
       out(2,0): 9+10+11+13+14+15=72, n=6, avg=12.0
       out(2,1): 10+11+12+14+15+16=78, n=6, avg=13.0
       out(2,2): 11+12+15+16=54, n=4, avg=13.5
    */
    float avg_expected[] = {
        6.0f, 7.0f, 7.5f,
        10.0f, 99.0f/9.0f, 11.5f,
        12.0f, 13.0f, 13.5f
    };

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    /* barriers for host→compute */
    VkBufferMemoryBarrier bmb_in, bmb_out;
    memset(&bmb_in, 0, sizeof(bmb_in));
    bmb_in.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_in.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmb_in.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb_in.buffer = in_buf;
    bmb_in.size = VK_WHOLE_SIZE;

    memset(&bmb_out, 0, sizeof(bmb_out));
    bmb_out.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_out.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmb_out.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb_out.buffer = out_buf;
    bmb_out.size = VK_WHOLE_SIZE;

    VkBufferMemoryBarrier barriers_in[] = { bmb_in, bmb_out };

    vkBeginCommandBuffer(cmd, &bi);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 2, barriers_in, 0, NULL);

    /* max pool */
    VkResult vr = vkmath_pool2d_f32(ctx, cmd, in_w, in_h, kw, kh,
                                    sw, sh, pw, ph, in_c, 0u, in_buf, out_buf);
    if (vr != VK_SUCCESS) {
        printf("pool2d: FAIL (vkmath_pool2d_f32 max VkResult=%d)\n", (int)vr);
        pass = 0; goto submit;
    }
    printf("  pool2d_f32_max     : recorded\n");

    /* barrier: compute write → host read */
    VkBufferMemoryBarrier bmb_out2;
    memset(&bmb_out2, 0, sizeof(bmb_out2));
    bmb_out2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_out2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb_out2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb_out2.buffer = out_buf;
    bmb_out2.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL,
                         1, &bmb_out2, 0, NULL);

    /* readback max pool result */
    float max_got[4];
    /* We need to wait, read, then re-record for avg pool. Simplest: end cmd, submit, wait, check. */

submit:
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fi;
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(dev, &fi, NULL, &fence);

    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(q, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDeviceWaitIdle(dev);

    memcpy(max_got, out_mapped, out_bytes);
    int max_pass = 1;
    float tol = 1e-4f;
    for (uint32_t i = 0; i < out_elems; i++) {
        float diff = fabsf(max_got[i] - max_expected[i]);
        if (diff > tol) {
            printf("pool2d max MISMATCH [%u]: got=%.6f exp=%.6f\n", i, max_got[i], max_expected[i]);
            max_pass = 0;
        }
    }
    printf("  pool2d_f32_max     : %s\n", max_pass ? "PASS" : "FAIL");
    pass &= max_pass;

    /* ── Now test avg pool ── */
    vkResetCommandBuffer(cmd, 0);
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferMemoryBarrier barriers2[] = { bmb_in, bmb_out2 };
    bmb_out2.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb_out2.buffer = out_buf2;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 2, barriers2, 0, NULL);

    vr = vkmath_pool2d_f32(ctx, cmd, in_w, in_h, kw, kh,
                           sw, sh, pw, ph, in_c, 1u, in_buf, out_buf2);
    if (vr != VK_SUCCESS) {
        printf("pool2d: FAIL (vkmath_pool2d_f32 avg VkResult=%d)\n", (int)vr);
        pass = 0; goto cleanup2;
    }
    printf("  pool2d_f32_avg     : recorded\n");

    bmb_out2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb_out2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb_out2.buffer = out_buf2;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL,
                         1, &bmb_out2, 0, NULL);

cleanup2:
    vkEndCommandBuffer(cmd);
    vkResetFences(dev, 1, &fence);
    vkQueueSubmit(q, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDeviceWaitIdle(dev);

    float avg_got[9];
    memcpy(avg_got, out_mapped2, out_bytes);
    int avg_pass = 1;
    for (uint32_t i = 0; i < out_elems; i++) {
        float diff = fabsf(avg_got[i] - avg_expected[i]);
        if (diff > tol) {
            printf("pool2d avg MISMATCH [%u]: got=%.6f exp=%.6f\n", i, avg_got[i], avg_expected[i]);
            avg_pass = 0;
        }
    }
    printf("  pool2d_f32_avg     : %s\n", avg_pass ? "PASS" : "FAIL");
    pass &= avg_pass;

cleanup:
    if (cmd) vkFreeCommandBuffers(dev, pool, 1, &cmd);
    if (pool) vkDestroyCommandPool(dev, pool, NULL);
    if (out_buf) vkDestroyBuffer(dev, out_buf, NULL);
    if (out_buf2) vkDestroyBuffer(dev, out_buf2, NULL);
    if (in_buf) vkDestroyBuffer(dev, in_buf, NULL);
    if (in_mem) { if (in_mapped) vkUnmapMemory(dev, in_mem); vkFreeMemory(dev, in_mem, NULL); }
    if (out_mem) { if (out_mapped) vkUnmapMemory(dev, out_mem); vkFreeMemory(dev, out_mem, NULL); }
    if (out_mem2) { if (out_mapped2) vkUnmapMemory(dev, out_mem2); vkFreeMemory(dev, out_mem2, NULL); }
    if (fence) vkDestroyFence(dev, fence, NULL);
    if (ctx) vkmath_destroy_context(ctx);
    if (dev) vkDestroyDevice(dev, NULL);
    if (inst) vkDestroyInstance(inst, NULL);

    printf("test_vkmath_pool2d: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
