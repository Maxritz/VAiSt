/**
 * \file test_vkmath_conv2d.c
 * \brief Minimal conv2d test harness for VKMath.
 *
 * Bootstraps a Vulkan instance/device, creates a VkMathContext, dispatches
 * vkmath_conv2d_f32 with a 3x3 kernel on a 5x5 input, validates against a CPU
 * reference, and reports PASS/FAIL.
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
    ai.pApplicationName = "test_conv2d";
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
    en.features.shaderInt16 = VK_FALSE;
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
                          VkDeviceMemory *mem, VkBuffer *buf) {
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
            mi = i;
            break;
        }
    }
    if (mi == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mi;

    r = vkAllocateMemory(dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(dev, *buf, NULL);
        return r;
    }
    vkBindBufferMemory(dev, *buf, *mem, 0);
    return VK_SUCCESS;
}

static VkResult alloc_staging(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                              VkDeviceMemory *mem, VkBuffer *buf, void **mapped) {
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
            mi = i;
            break;
        }
    }
    if (mi == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mi;

    r = vkAllocateMemory(dev, &ai, NULL, mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(dev, *buf, NULL);
        return r;
    }
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
    VkBuffer in_buf = VK_NULL_HANDLE, wb_buf = VK_NULL_HANDLE, out_buf = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory in_mem = VK_NULL_HANDLE, wb_mem = VK_NULL_HANDLE, out_mem = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void *mapped = NULL;

    int pass = 1;

    /* ── 1. Instance ── */
    if (create_instance(&inst) != VK_SUCCESS) {
        printf("conv2d: FAIL (vkCreateInstance)\n");
        return 1;
    }

    /* ── 2. Physical device ── */
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) {
        printf("conv2d: SKIP (no GPU)\n");
        vkDestroyInstance(inst, NULL);
        return 0;
    }
    vkEnumeratePhysicalDevices(inst, &n, &pd);

    /* ── 3. shaderInt64 gate ── */
    if (!query_shader_int64(pd)) {
        printf("conv2d: SKIP (no shaderInt64)\n");
        vkDestroyInstance(inst, NULL);
        return 0;
    }

    /* ── 4. Device ── */
    if (create_device(pd, &dev, &q) != VK_SUCCESS) {
        printf("conv2d: FAIL (vkCreateDevice)\n");
        pass = 0; goto cleanup;
    }

    /* ── 5. VKMath context ── */
    if (vkmath_create_context(pd, dev, &ctx) != VK_SUCCESS) {
        printf("conv2d: FAIL (vkmath_create_context)\n");
        pass = 0; goto cleanup;
    }

    /* ── 6. Buffers: 5x5 input, 3x3 kernel, 1 out channel ── */
    uint32_t in_c = 1, out_c = 1;
    uint32_t in_w = 5, in_h = 5;
    uint32_t kw = 3, kh = 3;
    uint32_t stride_w = 1, stride_h = 1;
    uint32_t pad_w = 1, pad_h = 1;
    uint32_t out_w = (in_w + 2*pad_w - (kw-1)) / stride_w + 1; /* 5 */
    uint32_t out_h = (in_h + 2*pad_h - (kh-1)) / stride_h + 1; /* 5 */

    uint32_t in_elems = in_c * in_h * in_w;
    uint32_t w_elems = out_c * in_c * kh * kw;
    uint32_t b_elems = out_c;
    uint32_t out_elems = out_c * out_h * out_w;

    uint32_t w_offset = 0;
    uint32_t b_offset = w_elems;

    VkDeviceSize in_bytes = in_elems * sizeof(float);
    VkDeviceSize wb_bytes = (w_elems + b_elems) * sizeof(float);
    VkDeviceSize out_bytes = out_elems * sizeof(float);

    if (alloc_mem(pd, dev, in_bytes, &in_mem, &in_buf) != VK_SUCCESS) {
        printf("conv2d: FAIL (alloc in)\n"); pass = 0; goto cleanup;
    }
    if (alloc_mem(pd, dev, wb_bytes, &wb_mem, &wb_buf) != VK_SUCCESS) {
        printf("conv2d: FAIL (alloc wb)\n"); pass = 0; goto cleanup;
    }
    if (alloc_mem(pd, dev, out_bytes, &out_mem, &out_buf) != VK_SUCCESS) {
        printf("conv2d: FAIL (alloc out)\n"); pass = 0; goto cleanup;
    }

    /* ── 7. Staging buffer ── */
    if (alloc_staging(pd, dev, out_bytes, &staging_mem, &staging, &mapped) != VK_SUCCESS) {
        printf("conv2d: FAIL (alloc staging)\n"); pass = 0; goto cleanup;
    }

    /* ── 8. Fill input + weight data on CPU ── */
    float *in = (float *)malloc(in_bytes);
    float *wb = (float *)malloc(wb_bytes);
    float *expected = (float *)malloc(out_bytes);

    for (uint32_t i = 0; i < in_elems; i++) in[i] = (float)(i % 7) - 3.0f;
    for (uint32_t i = 0; i < w_elems; i++) wb[i] = (float)((i % 3) - 1) * 0.5f + 0.1f;
    wb[b_offset] = 0.5f; /* bias for output channel 0 */

    /* Actually, let's just map directly */
    void *in_map;
    vkMapMemory(dev, in_mem, 0, in_bytes, 0, &in_map);
    memcpy(in_map, in, in_bytes);
    vkUnmapMemory(dev, in_mem);

    void *wb_map;
    vkMapMemory(dev, wb_mem, 0, wb_bytes, 0, &wb_map);
    memcpy(wb_map, wb, wb_bytes);
    vkUnmapMemory(dev, wb_mem);

    /* ── 9. CPU reference ── */
    for (uint32_t oc = 0; oc < out_c; oc++) {
        for (uint32_t oh = 0; oh < out_h; oh++) {
            for (uint32_t ow = 0; ow < out_w; ow++) {
                float acc = 0.0f;
                for (uint32_t fy = 0; fy < kh; fy++) {
                    int iy = (int)(oh * stride_h) - (int)pad_h + (int)fy;
                    if (iy < 0 || (uint32_t)iy >= in_h) continue;
                    for (uint32_t fx = 0; fx < kw; fx++) {
                        int ix = (int)(ow * stride_w) - (int)pad_w + (int)fx;
                        if (ix < 0 || (uint32_t)ix >= in_w) continue;
                        float in_val = in[(uint32_t)iy * in_w + (uint32_t)ix];
                        float w_val = wb[oc * in_c * kh * kw + fy * kw * in_c + fx * in_c];
                        acc += in_val * w_val;
                    }
                }
                expected[oc * out_h * out_w + oh * out_w + ow] = acc + wb[b_offset + oc];
            }
        }
    }

    /* ── 10. Command buffer ── */
    if (create_cmd(dev, &pool, &cmd) != VK_SUCCESS) {
        printf("conv2d: FAIL (create cmd)\n"); pass = 0; goto cleanup;
    }

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &bi);

    /* barrier: make host writes visible to compute */
    VkBufferMemoryBarrier bmb_in, bmb_wb;
    memset(&bmb_in, 0, sizeof(bmb_in));
    bmb_in.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_in.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmb_in.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb_in.srcQueueFamilyIndex = 0;
    bmb_in.dstQueueFamilyIndex = 0;
    bmb_in.buffer = in_buf;
    bmb_in.size = VK_WHOLE_SIZE;

    memset(&bmb_wb, 0, sizeof(bmb_wb));
    bmb_wb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_wb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmb_wb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb_wb.buffer = wb_buf;
    bmb_wb.size = VK_WHOLE_SIZE;

    VkBufferMemoryBarrier bmb_out;
    memset(&bmb_out, 0, sizeof(bmb_out));
    bmb_out.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_out.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmb_out.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb_out.buffer = out_buf;
    bmb_out.size = VK_WHOLE_SIZE;

    VkBufferMemoryBarrier barriers[] = { bmb_in, bmb_wb, bmb_out };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 3, barriers, 0, NULL);

    VkResult vr = vkmath_conv2d_f32(ctx, cmd, in_w, in_h, kw, kh,
                                    stride_w, stride_h, pad_w, pad_h,
                                    in_c, out_c, w_offset, b_offset,
                                    in_buf, wb_buf, out_buf);
    if (vr != VK_SUCCESS) {
        printf("conv2d: FAIL (vkmath_conv2d_f32 VkResult=%d)\n", (int)vr);
        pass = 0; goto submit;
    }
    printf("  conv2d_f32        : recorded (VkResult=%d)\n", (int)vr);

    /* barrier: make compute write visible to transfer */
    VkBufferMemoryBarrier bmb_out2;
    memset(&bmb_out2, 0, sizeof(bmb_out2));
    bmb_out2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_out2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb_out2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bmb_out2.buffer = out_buf;
    bmb_out2.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 1, &bmb_out2, 0, NULL);

submit:
    /* copy output to staging for readback */
    VkBufferCopy bc;
    memset(&bc, 0, sizeof(bc));
    bc.srcOffset = 0;
    bc.dstOffset = 0;
    bc.size = out_bytes;
    vkCmdCopyBuffer(cmd, out_buf, staging, 1, &bc);

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

    /* ── 11. Verify ── */
    float *got = (float *)malloc(out_bytes);
    memcpy(got, mapped, out_bytes);

    int check_pass = 1;
    float tol = 1e-4f;
    for (uint32_t i = 0; i < out_elems; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > tol) {
            printf("conv2d: MISMATCH at [%u]: got=%.6f expected=%.6f diff=%.6f\n",
                   i, got[i], expected[i], diff);
            check_pass = 0;
            break;
        }
    }
    if (check_pass) {
        printf("  conv2d_f32        : PASS\n");
    } else {
        printf("  conv2d_f32        : FAIL\n");
        pass = 0;
    }

cleanup:
    if (cmd) vkFreeCommandBuffers(dev, pool, 1, &cmd);
    if (pool) vkDestroyCommandPool(dev, pool, NULL);
    if (in_buf) vkDestroyBuffer(dev, in_buf, NULL);
    if (wb_buf) vkDestroyBuffer(dev, wb_buf, NULL);
    if (out_buf) vkDestroyBuffer(dev, out_buf, NULL);
    if (staging) vkDestroyBuffer(dev, staging, NULL);
    if (in_mem) vkFreeMemory(dev, in_mem, NULL);
    if (wb_mem) vkFreeMemory(dev, wb_mem, NULL);
    if (out_mem) vkFreeMemory(dev, out_mem, NULL);
    if (staging_mem) {
        if (mapped) vkUnmapMemory(dev, staging_mem);
        vkFreeMemory(dev, staging_mem, NULL);
    }
    if (fence) vkDestroyFence(dev, fence, NULL);
    if (ctx) vkmath_destroy_context(ctx);
    if (dev) vkDestroyDevice(dev, NULL);
    if (inst) vkDestroyInstance(inst, NULL);

    free(in); free(wb); free(expected); free(got);

    printf("test_vkmath_conv2d: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
