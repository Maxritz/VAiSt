#include "vkstream.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

/* VK_EXT_memory_budget */
typedef void (VKAPI_PTR* PFN_vkGetPhysicalDeviceMemoryBudgetPropertiesEXT)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryBudgetPropertiesEXT* pProperties);

/* VK_EXT_memory_priority */
typedef void (VKAPI_PTR* PFN_vkSetDeviceMemoryPriorityEXT)(VkDevice device, VkDeviceMemory memory, float priority);

struct vkstream_request {
    vkstream_result_t result;
    vkstream_path_t actual_path;
    vkstream_complete_cb callback;
    void* user_data;
    VkBuffer dest_buffer;
    VkDeviceMemory external_memory;
};

static PFN_vkGetMemoryHostPointerPropertiesEXT g_fpGetMemoryHostPointerProperties = NULL;
static PFN_vkGetPhysicalDeviceMemoryBudgetPropertiesEXT g_fpGetMemoryBudget = NULL;
static PFN_vkSetDeviceMemoryPriorityEXT g_fpSetMemoryPriority = NULL;

static void* vkstream_aligned_alloc(VkDeviceSize size, VkDeviceSize alignment) {
    void* ptr = NULL;
#if defined(_MSC_VER)
    ptr = _aligned_malloc((size_t)size, (size_t)alignment);
#elif defined(__unix__)
    if (posix_memalign(&ptr, (size_t)alignment, (size_t)size) != 0) ptr = NULL;
#else
    ptr = malloc((size_t)size);
#endif
    return ptr;
}

static void vkstream_aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__unix__)
    free(ptr);
#else
    free(ptr);
#endif
}

static VkBool32 vkstream_check_external_memory_host(VkPhysicalDevice pdevice, VkDevice device) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, NULL);
    VkExtensionProperties* exts = (VkExtensionProperties*)calloc(extCount, sizeof(VkExtensionProperties));
    if (!exts) return VK_FALSE;
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, exts);

    VkBool32 found = VK_FALSE;
    for (uint32_t i = 0; i < extCount; i++) {
        if (strcmp(exts[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
            found = VK_TRUE;
            break;
        }
    }
    free(exts);

    if (found) {
        g_fpGetMemoryHostPointerProperties = (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetDeviceProcAddr(device, "vkGetMemoryHostPointerPropertiesEXT");
    }

    if (g_fpGetMemoryHostPointerProperties) {
        g_fpGetMemoryBudget = (PFN_vkGetPhysicalDeviceMemoryBudgetPropertiesEXT)
            vkGetInstanceProcAddr(NULL, "vkGetPhysicalDeviceMemoryBudgetPropertiesEXT");
        g_fpSetMemoryPriority = (PFN_vkSetDeviceMemoryPriorityEXT)
            vkGetDeviceProcAddr(device, "vkSetDeviceMemoryPriorityEXT");
    }

    return g_fpGetMemoryHostPointerProperties != NULL;
}

static VkBool32 vkstream_check_bar(vkstream_context_t* ctx) {
    ctx->bar_supported = VK_FALSE;
    ctx->bar_aperture_size = 0;

    if (!g_fpGetMemoryHostPointerProperties) return VK_FALSE;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
    };
    if (g_fpGetMemoryBudget) {
        g_fpGetMemoryBudget(ctx->pdevice, &budgetProps);
    }

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx->pdevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        const VkMemoryType* mt = &memProps.memoryTypes[i];
        if ((mt->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (mt->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            VkMemoryHeap* heap = &memProps.memoryHeaps[mt->heapIndex];
            if (heap->flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                ctx->bar_supported = VK_TRUE;
                ctx->bar_aperture_size = (VkDeviceSize)heap->size;
                break;
            }
        }
    }

    if (!ctx->bar_supported && g_fpGetMemoryBudget) {
        /* Use VK_EXT_memory_budget as fallback for BAR detection */
        for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
            if (budgetProps.heapBudget[i] > ctx->bar_aperture_size) {
                ctx->bar_aperture_size = budgetProps.heapBudget[i];
                ctx->bar_supported = VK_TRUE;
            }
        }
    }

    if (!ctx->bar_supported) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx->pdevice, &props);
        ctx->bar_aperture_size = props.limits.maxStorageBufferRange;
    }

    return ctx->bar_supported;
}

static VkResult vkstream_create_external_buffer(
    vkstream_context_t* ctx,
    VkBufferCreateInfo* bufInfo,
    VkBuffer* buffer,
    VkDeviceMemory* memory,
    void** host_ptr,
    VkDeviceSize size
) {
    VkResult r;

    /* Allocate aligned host memory — importable via VK_EXT_external_memory_host */
    VkDeviceSize alignment = 64; /* cache line */
    *host_ptr = vkstream_aligned_alloc(size, alignment);
    if (!*host_ptr) return VK_ERROR_OUT_OF_HOST_MEMORY;

    r = vkCreateBuffer(ctx->device, bufInfo, NULL, buffer);
    if (r != VK_SUCCESS) goto fail_alloc;

    /* Create imported memory from host pointer via VK_EXT_external_memory_host */
    VkImportMemoryHostPointerInfoEXT importInfo = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pHostPointer = *host_ptr,
         .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
    };

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &importInfo,
        .allocationSize = size,
    };

    /* Find memory type matching buffer requirements */
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx->pdevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (memReqs.memoryTypeBits & (1u << i)) {
            if ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        r = VK_ERROR_FEATURE_NOT_PRESENT;
        goto fail_buffer;
    }

    allocInfo.memoryTypeIndex = memTypeIndex;

    /* Apply priority if available */
    VkMemoryPriorityAllocateInfoEXT priorityInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
        .pNext = &importInfo,
        .priority = 1.0f  /* High priority — keep resident */
    };
    if (g_fpSetMemoryPriority) {
        allocInfo.pNext = &priorityInfo;
    }

    r = vkAllocateMemory(ctx->device, &allocInfo, NULL, memory);
    if (r != VK_SUCCESS) goto fail_buffer;

    r = vkBindBufferMemory(ctx->device, *buffer, *memory, 0);
    if (r != VK_SUCCESS) {
        vkFreeMemory(ctx->device, *memory, NULL);
        goto fail_buffer;
    }

    return VK_SUCCESS;

fail_buffer:
    vkDestroyBuffer(ctx->device, *buffer, NULL);
fail_alloc:
    vkstream_aligned_free(*host_ptr);
    return r;
}

static VkResult vkstream_create_staging_buffer(vkstream_context_t* ctx) {
    VkBufferCreateInfo bufInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = ctx->staging_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    return vkstream_create_external_buffer(ctx, &bufInfo,
        &ctx->staging_buffer, &ctx->staging_memory,
        NULL, ctx->staging_size);
}

vkstream_context_t* vkstream_create(
    VkDevice device,
    VkPhysicalDevice pdevice,
    uint32_t transfer_qfamily,
    uint32_t compute_qfamily
) {
    vkstream_context_t* ctx = (vkstream_context_t*)calloc(1, sizeof(vkstream_context_t));
    if (!ctx) return NULL;

    ctx->device = device;
    ctx->pdevice = pdevice;
    ctx->transfer_queue_family = transfer_qfamily;
    ctx->compute_queue_family = compute_qfamily;

    vkGetDeviceQueue(device, transfer_qfamily, 0, &ctx->transfer_queue);
    vkGetDeviceQueue(device, compute_qfamily, 0, &ctx->compute_queue);

    ctx->external_memory_host = vkstream_check_external_memory_host(pdevice, device);
    vkstream_check_bar(ctx);

    ctx->staging_size = 64 * 1024 * 1024;
    if (ctx->external_memory_host) {
        vkstream_create_staging_buffer(ctx);
    } else {
        /* Fallback: traditional device-local staging */
        VkBufferCreateInfo bufInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = ctx->staging_size,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        vkCreateBuffer(ctx->device, &bufInfo, NULL, &ctx->staging_buffer);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(ctx->device, ctx->staging_buffer, &memReqs);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(ctx->pdevice, &memProps);

        uint32_t memTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                memTypeIndex = i;
                break;
            }
        }

        if (memTypeIndex != UINT32_MAX) {
            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs.size,
                .memoryTypeIndex = memTypeIndex
            };
            if (vkAllocateMemory(ctx->device, &allocInfo, NULL, &ctx->staging_memory) == VK_SUCCESS) {
                vkBindBufferMemory(ctx->device, ctx->staging_buffer, ctx->staging_memory, 0);
            } else {
                vkDestroyBuffer(ctx->device, ctx->staging_buffer, NULL);
                ctx->staging_buffer = VK_NULL_HANDLE;
            }
        }
    }

    ctx->fence_count = 32;
    ctx->fences = (VkFence*)calloc(ctx->fence_count, sizeof(VkFence));
    for (uint32_t i = 0; i < ctx->fence_count; i++) {
        VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        vkCreateFence(device, &fenceInfo, NULL, &ctx->fences[i]);
    }

    ctx->request_count = 256;
    ctx->requests = (vkstream_request_t*)calloc(ctx->request_count, sizeof(vkstream_request_t));

    return ctx;
}

void vkstream_destroy(vkstream_context_t* ctx) {
    if (!ctx) return;

    if (ctx->staging_buffer) {
        if (ctx->external_memory_host && ctx->staging_memory) {
            vkFreeMemory(ctx->device, ctx->staging_memory, NULL);
        }
        vkDestroyBuffer(ctx->device, ctx->staging_buffer, NULL);
    }

    if (ctx->fences) {
        for (uint32_t i = 0; i < ctx->fence_count; i++) {
            if (ctx->fences[i]) vkDestroyFence(ctx->device, ctx->fences[i], NULL);
        }
        free(ctx->fences);
    }

    if (ctx->requests) free(ctx->requests);
    free(ctx);
}

vkstream_request_t* vkstream_load_tensor(
    vkstream_context_t* ctx,
    const vkstream_load_desc_t* desc,
    vkstream_complete_cb callback,
    void* user_data
) {
    vkstream_request_t* req = NULL;
    for (uint32_t i = 0; i < ctx->request_count; i++) {
        if (ctx->requests[i].result.result == VK_NOT_READY) {
            req = &ctx->requests[i];
            break;
        }
    }
    if (!req) req = &ctx->requests[0]; /* overflow */

    memset(req, 0, sizeof(vkstream_request_t));
    req->result.result = VK_NOT_READY;
    req->callback = callback;
    req->user_data = user_data;
    req->dest_buffer = desc->dest_buffer;

    /* Determine actual path */
    if (ctx->external_memory_host && desc->size <= ctx->staging_size) {
        req->actual_path = VKSTREAM_PATH_BAR_ZERO_COPY;
    } else {
        req->actual_path = VKSTREAM_PATH_STAGING;
    }

    FILE* fp = fopen(desc->filepath, "rb");
    if (!fp) {
        req->result.result = VK_ERROR_INITIALIZATION_FAILED;
        if (callback) callback(req, user_data);
        return req;
    }

    if (req->actual_path == VKSTREAM_PATH_BAR_ZERO_COPY) {
        /* Zero-copy path via VK_EXT_external_memory_host */
        void* host_ptr = NULL;
        VkBufferCreateInfo bufInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = desc->size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };

        VkResult r = vkstream_create_external_buffer(ctx, &bufInfo,
            &req->dest_buffer, &req->external_memory,
            &host_ptr, desc->size);

        if (r != VK_SUCCESS) {
            /* Fallback to staging */
            req->actual_path = VKSTREAM_PATH_STAGING;
            goto staging_path;
        }

        /* Read file directly into host-imported buffer (zero-copy) */
        size_t read = fread(host_ptr, 1, (size_t)desc->size, fp);
        fclose(fp);

        req->result.bytes_transferred = (VkDeviceSize)read;
        req->result.result = VK_SUCCESS;

        /* Apply memory priority if available */
        if (g_fpSetMemoryPriority && req->external_memory) {
            g_fpSetMemoryPriority(ctx->device, req->external_memory, 1.0f);
        }

    } else {
staging_path:
        /* Staging path */
        if (!ctx->staging_buffer) {
            fclose(fp);
            req->result.result = VK_ERROR_FEATURE_NOT_PRESENT;
            if (callback) callback(req, user_data);
            return req;
        }

        void* mapped = NULL;
        if (ctx->staging_memory && ctx->external_memory_host == VK_FALSE) {
            vkMapMemory(ctx->device, ctx->staging_memory, 0, desc->size, 0, &mapped);
        }

        /* For external memory path, use staging memory */
        if (!mapped && ctx->external_memory_host) {
            VkMemoryRequirements memReqs;
            vkGetBufferMemoryRequirements(ctx->device, ctx->staging_buffer, &memReqs);
            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs.size,
            };
            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(ctx->pdevice, &memProps);
            for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
                if ((memReqs.memoryTypeBits & (1u << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                    allocInfo.memoryTypeIndex = i;
                    break;
                }
            }
            VkDeviceMemory stage_mem;
            if (vkAllocateMemory(ctx->device, &allocInfo, NULL, &stage_mem) == VK_SUCCESS) {
                vkBindBufferMemory(ctx->device, ctx->staging_buffer, stage_mem, 0);
                vkMapMemory(ctx->device, stage_mem, 0, desc->size, 0, &mapped);
                ctx->staging_memory = stage_mem;
            }
        }

        if (!mapped) {
            fclose(fp);
            req->result.result = VK_ERROR_FEATURE_NOT_PRESENT;
            if (callback) callback(req, user_data);
            return req;
        }

        size_t read = fread(mapped, 1, (size_t)desc->size, fp);
        fclose(fp);

        if (ctx->external_memory_host == VK_FALSE) {
            vkUnmapMemory(ctx->device, ctx->staging_memory);
        }

        /* Copy to dest buffer */
        VkCommandPool pool;
        VkCommandPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = ctx->transfer_queue_family,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        };
        vkCreateCommandPool(ctx->device, &poolInfo, NULL, &pool);

        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        vkAllocateCommandBuffers(ctx->device, &allocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion = {
            .srcOffset = 0,
            .dstOffset = desc->dest_offset,
            .size = (VkDeviceSize)read
        };
        vkCmdCopyBuffer(cmd, ctx->staging_buffer, desc->dest_buffer, 1, &copyRegion);
        vkEndCommandBuffer(cmd);

        VkFence fence;
        VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        vkCreateFence(ctx->device, &fenceInfo, NULL, &fence);

        VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd
        };
        vkQueueSubmit(ctx->transfer_queue, 1, &submitInfo, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

        req->result.bytes_transferred = (VkDeviceSize)read;
        req->result.result = VK_SUCCESS;

        vkDestroyFence(ctx->device, fence, NULL);
        vkDestroyCommandPool(ctx->device, pool, NULL);
    }

    if (callback) callback(req, user_data);
    return req;
}

void vkstream_poll(vkstream_context_t* ctx) {
    (void)ctx;
}

void vkstream_flush(vkstream_context_t* ctx) {
    vkDeviceWaitIdle(ctx->device);
}

vkstream_result_t vkstream_get_result(vkstream_request_t* req) {
    return req->result;
}

VkBool32 vkstream_is_bar_available(vkstream_context_t* ctx) {
    return ctx->bar_supported && ctx->external_memory_host;
}

VkDeviceSize vkstream_get_bar_aperture(vkstream_context_t* ctx) {
    return ctx->bar_aperture_size;
}

VkBool32 vkstream_is_external_memory_host_supported(vkstream_context_t* ctx) {
    return ctx->external_memory_host;
}
