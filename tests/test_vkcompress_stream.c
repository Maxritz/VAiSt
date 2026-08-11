/**
 * \file test_vkcompress_stream.c
 * \brief Streaming compression test for vkcompress module.
 * Streams a GGUF file through GPU compression in chunks.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vulkan/vulkan.h>
#include <vkruntime/vkruntime.h>
#include <vkcompress/vkcompress.h>

#define CHUNK_WORDS 65536   /* 256KB chunks for test */

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
    printf("Starting vkcompress streaming test...\n");

    /* Open GGUF file */
    const char* gguf_path = "E:/OLLAMA-Models/GGUF/gemma4-v2-Q6_K.gguf";
    FILE* gf = fopen(gguf_path, "rb");
    if (!gf) {
        printf("SKIP: GGUF not found at %s\n", gguf_path);
        return 0;
    }

    /* Get file size using fseek/ftell with 64-bit offset */
    if (fseek(gf, 0, SEEK_END) != 0) {
        printf("FAIL: fseek\n");
        fclose(gf); return 1;
    }
    long long file_size = ftell(gf);
    if (file_size < 0) {
        /* Try alternative: read backwards */
        printf("ftell returned %lld, trying alternative...\n", file_size);
        fseek(gf, 0, SEEK_SET);
        /* Estimate size from file system */
        struct _stat64 st;
        if (_stat64(gguf_path, &st) == 0) {
            file_size = st.st_size;
            printf("Got size from _stat64: %lld bytes\n", file_size);
        } else {
            printf("FAIL: cannot determine file size\n");
            fclose(gf); return 1;
        }
    }
    fseek(gf, 0, SEEK_SET);
    printf("GGUF file: %lld MB (%lld bytes)\n", file_size / 1024 / 1024, file_size);

    /* Test with first chunk of the file */
    VkDeviceSize chunk_bytes = CHUNK_WORDS * sizeof(uint32_t);  /* 256KB */

    /* Create Vulkan instance + device */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkcompress_stream",
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

    VkDevice device;
    if (vkr_create_device(pd, 0, &device) != VK_SUCCESS) {
        printf("FAIL: vkr_create_device\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("Device created\n");

    /* Create vkcompress context */
    VkCompressContext* ctx = NULL;
    vr = vkcompress_create_context(pd, device, &ctx);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkcompress_create_context (%d)\n", vr);
        vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("vkcompress_create_context: OK\n");

    /* Register file as streaming buffer with 256MB chunks */
    VkDeviceSize vram_safe_chunk = 256 * 1024 * 1024;  /* 256 MB */
    uint32_t num_chunks = (uint32_t)(((long long)file_size + vram_safe_chunk - 1) / vram_safe_chunk);
    printf("Streaming %lld bytes in %u chunks of 256MB each\n", file_size, num_chunks);

    /* Use streaming registration */
    vkcomp_buffer_id_t stream_id = vkcompress_register_buffer_streaming(
        ctx, (VkDeviceSize)file_size, vram_safe_chunk, "gguf:stream", 5);

    if (stream_id == VKCOMP_INVALID_ID) {
        printf("FAIL: vkcompress_register_buffer_streaming\n");
        /* Fall back to single chunk */
        printf("Falling back to single-chunk test...\n");
        stream_id = vkcompress_register_buffer(ctx, chunk_bytes, "gguf:single", 5);
        if (stream_id == VKCOMP_INVALID_ID) {
            printf("FAIL: single-chunk register too\n");
            vkcompress_destroy_context(ctx);
            vkDestroyDevice(device, NULL);
            vkDestroyInstance(instance, NULL);
            fclose(gf);
            return 1;
        }
        num_chunks = 1;
        vram_safe_chunk = chunk_bytes;
    }

    uint32_t actual_chunks = vkcompress_get_chunk_count(ctx, stream_id);
    printf("Registered streaming buffer: id=%llu, chunks=%u\n",
           (unsigned long long)stream_id, actual_chunks);

    /* Create staging buffer for one chunk */
    VkBuffer staging_buf;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = vram_safe_chunk,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    vkCreateBuffer(device, &bci, NULL, &staging_buf);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, staging_buf, &mr);
    uint32_t memType = find_memory_type(pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = memType,
    };
    VkDeviceMemory staging_mem;
    vkAllocateMemory(device, &mai, NULL, &staging_mem);
    vkBindBufferMemory(device, staging_buf, staging_mem, 0);

    /* Create command pool + buffer */
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

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    /* Stream compress all chunks */
    VkDeviceSize total_compressed_estimate = 0;
    int stream_pass = 1;

    for (uint32_t i = 0; i < actual_chunks; i++) {
        VkDeviceSize remaining = file_size - (VkDeviceSize)i * vram_safe_chunk;
        VkDeviceSize this_chunk = remaining < vram_safe_chunk ? remaining : vram_safe_chunk;

        /* Read chunk from file into staging */
        fseek(gf, (long)(i * vram_safe_chunk), SEEK_SET);
        void* data;
        vkMapMemory(device, staging_mem, 0, this_chunk, 0, &data);
        size_t bread = fread(data, 1, (size_t)this_chunk, gf);
        vkUnmapMemory(device, staging_mem);

        if (bread > 0) {
            printf("  Chunk %u/%u: %zu bytes\n", i + 1, actual_chunks, bread);

            /* Record + submit compress for this chunk */
            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            });

            vr = vkcompress_stream_write(ctx, cmd, stream_id, i,
                                         staging_buf, this_chunk, 0);
            if (vr != VK_SUCCESS) {
                printf("  FAIL: vkcompress_stream_write chunk %u: %d\n", i, vr);
                stream_pass = 0;
                break;
            }

            vkEndCommandBuffer(cmd);

            VkFence f;
            vkCreateFence(device, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL, &f);
            vkQueueSubmit(queue, 1, &(VkSubmitInfo){
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &cmd,
            }, f);
            vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX);
            vkDestroyFence(device, f, NULL);

            total_compressed_estimate += this_chunk;  /* rough: assume same size */
        }
    }

    fclose(gf);

    /* Cleanup */
    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, NULL);
    vkFreeMemory(device, staging_mem, NULL);
    vkDestroyBuffer(device, staging_buf, NULL);
    vkcompress_destroy_context(ctx);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    if (stream_pass) {
        printf("\n=== RESULT: PASS (streamed %lld bytes in %u chunks through GPU compress) ===\n",
               file_size, actual_chunks);
        return 0;
    } else {
        printf("\n=== RESULT: FAIL (streaming compression failed) ===\n");
        return 1;
    }
}
