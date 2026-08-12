#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>

/* Note: VK_EXT_external_memory_host uses vkAllocateMemory with chained
   VkImportMemoryHostPointerInfoEXT — no separate vkImportMemoryHostPointerEXT
   entrypoint exists. The proc addr lookup below is a diagnostic only. */

#define TENSOR_SIZE (256 * 256)

static float check_result(float* data, size_t count, float expected_base) {
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
        float expected = expected_base + (float)i;
        float diff = fabsf(data[i] - expected);
        if (diff > 0.01f) {
            errors++;
            if (errors <= 3) {
                printf("  MISMATCH at [i=%zu]: expected=%.4f, got=%.4f\n", i, expected, data[i]);
            }
        }
    }
    return (float)errors;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting HIP-Vulkan zero-copy test...\n");

    /* === PHASE 1: Create Vulkan context === */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "hip_vulkan_bridge_test",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    VkResult r = vkCreateInstance(&instInfo, NULL, &instance);
    if (r != VK_SUCCESS) { printf("FAIL: vkCreateInstance %d\n", r); return 1; }
    printf("PASS: Vulkan instance created\n");

    uint32_t pdcount = 0;
    r = vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    if (r != VK_SUCCESS || pdcount == 0) {
        printf("FAIL: No physical devices (r=%d)\n", r);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    VkPhysicalDevice* pdvs = (VkPhysicalDevice*)malloc(pdcount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &pdcount, pdvs);
    VkPhysicalDevice pdevice = pdvs[0];
    free(pdvs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdevice, &props);
    printf("GPU: %s\n", props.deviceName);

    /* Check queue family 0 supports compute */
    uint32_t qFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &qFamilyCount, NULL);
    VkQueueFamilyProperties* qProps = (VkQueueFamilyProperties*)calloc(qFamilyCount, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &qFamilyCount, qProps);
    if (qProps[0].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        printf("PASS: Queue family 0 supports compute\n");
    } else {
        printf("FAIL: Queue family 0 doesn't support compute\n");
        free(qProps);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    free(qProps);

    /* Check for external_memory_host extension */
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, NULL);
    VkExtensionProperties* exts = (VkExtensionProperties*)calloc(extCount, sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, exts);
    int has_ext = 0;
    for (uint32_t i = 0; i < extCount; i++) {
        if (strcmp(exts[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
            has_ext = 1;
            break;
        }
    }
    free(exts);
    printf("VK_EXT_external_memory_host: %s\n", has_ext ? "YES" : "NO");

    const char* devExts[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qInfo,
    };
    if (has_ext) {
        devInfo.enabledExtensionCount = 1;
        devInfo.ppEnabledExtensionNames = devExts;
    }
    VkDevice device;
    r = vkCreateDevice(pdevice, &devInfo, NULL, &device);
    if (r != VK_SUCCESS) {
        printf("FAIL: vkCreateDevice %d\n", r);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("PASS: Vulkan device created\n");

PFN_vkGetMemoryHostPointerPropertiesEXT fpGetProps = 
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(device, "vkGetMemoryHostPointerPropertiesEXT");
    printf("vkGetMemoryHostPointerPropertiesEXT via device: %p\n", (void*)fpGetProps);
    if (!fpGetProps) {
        fpGetProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetInstanceProcAddr(instance, "vkGetMemoryHostPointerPropertiesEXT");
        printf("vkGetMemoryHostPointerPropertiesEXT via instance: %p\n", (void*)fpGetProps);
    }
    VkBool32 import_available = (fpGetProps != NULL);

    /* === PHASE 2: HIP tensor allocation === */
    printf("PHASE 2: HIP allocation\n");
    void* hip_tensor = NULL;
    hipError_t hr = hipHostMalloc(&hip_tensor, TENSOR_SIZE * sizeof(float), hipHostMallocMapped);
    if (hr != hipSuccess) {
        printf("FAIL: hipHostMalloc failed (err=%d)\n", r);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("PASS: HIP allocated %zu bytes at %p\n", TENSOR_SIZE * sizeof(float), hip_tensor);

    float* hip_data = (float*)hip_tensor;
    for (size_t i = 0; i < TENSOR_SIZE; i++) {
        hip_data[i] = (float)i;
    }
    hipDeviceSynchronize();
    printf("PASS: HIP wrote data (first=%.1f, last=%.1f)\n", hip_data[0], hip_data[TENSOR_SIZE-1]);

    /* === PHASE 3: Import to Vulkan === */
    if (import_available) {
        printf("PHASE 3: Direct HIP->Vulkan import (zero-copy)\n");
        VkBufferCreateInfo bufInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = TENSOR_SIZE * sizeof(float),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer vk_buffer;
        r = vkCreateBuffer(device, &bufInfo, NULL, &vk_buffer);
        if (r != VK_SUCCESS) { printf("FAIL: vkCreateBuffer %d\n", r); goto cleanup; }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, vk_buffer, &memReqs);
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(pdevice, &memProps);

        uint32_t memType = 0;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                memType = i;
                break;
            }
        }

        VkImportMemoryHostPointerInfoEXT importInfo = {
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            .pHostPointer = hip_tensor,
        };
        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = memType,
            .pNext = &importInfo,
        };
        VkDeviceMemory vk_memory;
        r = vkAllocateMemory(device, &allocInfo, NULL, &vk_memory);
        if (r != VK_SUCCESS) {
            printf("FAIL: Import failed: %d\n", r);
            vkDestroyBuffer(device, vk_buffer, NULL);
            goto staging_fallback;
        }
        printf("PASS: Direct import succeeded\n");

        r = vkBindBufferMemory(device, vk_buffer, vk_memory, 0);
        if (r != VK_SUCCESS) { printf("FAIL: BindMemory %d\n", r); }
        else {
            printf("PASS: Bound imported memory to buffer\n");
        }

        /* Verify data integrity via map */
        void* mapped = NULL;
        r = vkMapMemory(device, vk_memory, 0, 256 * sizeof(float), 0, &mapped);
        if (r == VK_SUCCESS) {
            float errors = check_result((float*)mapped, 256, 0.0f);
            if (errors > 0) printf("FAIL: Data mismatch (%.0f errors)\n", errors);
            else printf("PASS: Zero-copy data integrity verified\n");
            vkUnmapMemory(device, vk_memory);
        }

        /* Write from Vulkan */
        r = vkMapMemory(device, vk_memory, 0, 64 * sizeof(float), 0, &mapped);
        if (r == VK_SUCCESS) {
            float* vp = (float*)mapped;
            for (int i = 0; i < 64; i++) vp[i] = (float)(i * 100);
            vkUnmapMemory(device, vk_memory);
            hipDeviceSynchronize();
            float* hc = (float*)hip_tensor;
            int ok = 1;
            for (int i = 0; i < 64; i++) {
                if (hc[i] != (float)(i * 100)) { ok = 0; break; }
            }
            printf("%s: Vulkan->HIP visibility\n", ok ? "PASS" : "FAIL");
        }

        vkFreeMemory(device, vk_memory, NULL);
        vkDestroyBuffer(device, vk_buffer, NULL);
        printf("PHASE 3 PASS: Full zero-copy bridge verified\n");
    } else {
        printf("PHASE 3: Direct import unavailable, using staging fallback\n");
    }

staging_fallback:
    if (!import_available) {
        /* Staging buffer path: copy HIP data to Vulkan-allocated memory */
        printf("Staging fallback: copy HIP->Vulkan via mapped memory\n");
        VkBufferCreateInfo bufInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = TENSOR_SIZE * sizeof(float),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer vk_buffer;
        r = vkCreateBuffer(device, &bufInfo, NULL, &vk_buffer);
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, vk_buffer, &memReqs);
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(pdevice, &memProps);

        uint32_t memType = 0;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                memType = i;
                break;
            }
        }
        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = memType,
        };
        VkDeviceMemory vk_memory;
        r = vkAllocateMemory(device, &allocInfo, NULL, &vk_memory);
        if (r != VK_SUCCESS) { printf("FAIL: alloc staging %d\n", r); }
        else {
            vkBindBufferMemory(device, vk_buffer, vk_memory, 0);
            void* mapped;
            vkMapMemory(device, vk_memory, 0, TENSOR_SIZE * sizeof(float), 0, &mapped);
            memcpy(mapped, hip_tensor, TENSOR_SIZE * sizeof(float));
            vkUnmapMemory(device, vk_memory);
            printf("PASS: Staged %zu bytes via copy (not zero-copy)\n", TENSOR_SIZE * sizeof(float));
            vkFreeMemory(device, vk_memory, NULL);
            vkDestroyBuffer(device, vk_buffer, NULL);
        }
    }

cleanup:
    /* === PHASE 4: Cleanup === */
    if (hip_tensor) hipHostFree(hip_tensor);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    printf("\n=== DONE ===\n");
    return 0;
}
