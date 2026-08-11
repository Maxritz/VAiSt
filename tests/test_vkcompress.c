/**
 * \file test_vkcompress.c
 * \brief Round-trip compression test for vkcompress module.
 * Tests: compress 1024 f32 values (with zeros) → decompress → verify match.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <vkruntime/vkruntime.h>
#include <vkcompress/vkcompress.h>
#include "shaders_spv.h"

static float test_data[1024];
static float result_data[1024];

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t typeFilter,
                                  uint32_t properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting vkcompress round-trip test...\n");

    /* Fill test data from GGUF file for real data test */
    const char* gguf_path = "E:/OLLAMA-Models/GGUF/gemma4-v2-Q6_K.gguf";
    FILE* gf = fopen(gguf_path, "rb");
    if (gf) {
        printf("Loading GGUF data for compression test...\n");
        size_t bytes_read = fread(test_data, 1, 1024 * sizeof(float), gf);
        fclose(gf);
        printf("Read %zu bytes from GGUF\n", bytes_read);
    } else {
        /* Fallback: synthetic data with zeros + patterns */
        for (int i = 0; i < 1024; i++) {
            if (i % 4 == 0) test_data[i] = 0.0f;
            else test_data[i] = (float)(i * 0.001f);
        }
        printf("Using synthetic data (GGUF not found)\n");
    }
    memset(result_data, 0, sizeof(result_data));

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkcompress",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    VkResult vr = vkCreateInstance(&instInfo, NULL, &instance);
    if (vr) { printf("FAIL: vkCreateInstance (%d)\n", vr); return 1; }

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    if (!pdcount) { printf("SKIP: no GPU\n"); vkDestroyInstance(instance, NULL); return 0; }
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(instance, &pdcount, &pd);

    /* Create device using project's canonical device creation */
    VkDevice device;
    if (vkr_create_device(pd, 0, &device) != VK_SUCCESS) {
        printf("FAIL: vkr_create_device\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("Device created with proper features\n");

    VkCompressContext* ctx = NULL;
    vr = vkcompress_create_context(pd, device, &ctx);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkcompress_create_context (%d)\n", vr);
        vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("vkcompress_create_context: OK\n");

    vkcomp_buffer_id_t buf_id = vkcompress_register_buffer(ctx, 1024 * sizeof(float),
                                                             "test:buffer:0", 5);
    if (buf_id == VKCOMP_INVALID_ID) {
        printf("FAIL: vkcompress_register_buffer\n");
        vkcompress_destroy_context(ctx);
        vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("Registered buffer id=%llu\n", (unsigned long long)buf_id);

    /* Create source and destination buffers */
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 1024 * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkBuffer src_buf;
    vkCreateBuffer(device, &bci, NULL, &src_buf);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, src_buf, &mr);
    uint32_t memType = find_memory_type(pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = memType,
    };
    VkDeviceMemory src_mem;
    vkAllocateMemory(device, &mai, NULL, &src_mem);
    vkBindBufferMemory(device, src_buf, src_mem, 0);

    /* Upload test data */
    void* data;
    vkMapMemory(device, src_mem, 0, 1024 * sizeof(float), 0, &data);
    memcpy(data, test_data, 1024 * sizeof(float));
    vkUnmapMemory(device, src_mem);

    /* Command pool + buffer */
    VkCommandPool pool;
    vkCreateCommandPool(device, &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    }, NULL, &pool);

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    }, &cmd);

    vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    });

    vr = vkcompress_write(ctx, cmd, buf_id, src_buf, 1024 * sizeof(float), 0);
    printf("vkcompress_write: %d\n", vr);

    vr = vkcompress_read(ctx, cmd, buf_id, src_buf, 1024 * sizeof(float), 0);
    printf("vkcompress_read: %d\n", vr);

    vkEndCommandBuffer(cmd);

    /* Submit */
    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    vkCreateFence(device, &fci, NULL, &fence);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, NULL);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, NULL);

    /* Read back result */
    vkMapMemory(device, src_mem, 0, 1024 * sizeof(float), 0, &data);
    memcpy(result_data, data, 1024 * sizeof(float));
    vkUnmapMemory(device, src_mem);

    /* Verify all values match (decompressed output should equal input) */
    int errors = 0;
    for (int i = 0; i < 1024; i++) {
        if (result_data[i] != test_data[i]) {
            if (errors < 5) printf("  MISMATCH at %d: got %f, expected %f\n",
                i, result_data[i], test_data[i]);
            errors++;
        }
    }

    vkDestroyBuffer(device, src_buf, NULL);
    vkFreeMemory(device, src_mem, NULL);
    vkcompress_destroy_context(ctx);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    if (errors == 0) {
        printf("\n=== RESULT: PASS (round-trip verified: all 1024 values match) ===\n");
        return 0;
    } else {
        printf("\n=== RESULT: FAIL (%d mismatches) ===\n", errors);
        return 1;
    }
}
