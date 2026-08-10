#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>

#define VK_SHARING_MODE_PRIVATE 0
#define VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_POINTER_EXT 0x00000001u
#define VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT 1000176000
#define VK_STRUCTURE_TYPE_MEMORY_DEDUPLICATE_INFO_EXT 1000358000

typedef VkResult (VKAPI_PTR *PFN_vkImportMemoryHostPointerEXT_t)(
    VkDevice device,
    const VkMemoryAllocateInfo* pAllocInfo,
    const VkImportMemoryHostPointerInfoEXT* pImportInfo,
    VkDeviceMemory* pMemory);

typedef VkResult (VKAPI_PTR *PFN_vkGetMemoryOpaqueCaptureAddressKHR_t)(
    VkDevice device,
    const VkBufferDeviceAddressInfo* pInfo,
    uint64_t* pAddress);

typedef void (VKAPI_PTR *PFN_vkCmdDebugMarkerInsertEXT_t)(
    VkCommandBuffer commandBuffer,
    const void* pLabel);

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Extension Proc Address Audit ===\n\n");

    /* Create Vulkan instance */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ext_audit",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    VkResult r = vkCreateInstance(&instInfo, NULL, &instance);
    if (r != VK_SUCCESS) { printf("vkCreateInstance failed: %d\n", r); return 1; }

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    VkPhysicalDevice* pdvs = malloc(pdcount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &pdcount, pdvs);
    VkPhysicalDevice pdevice = pdvs[0];
    free(pdvs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdevice, &props);
    printf("GPU: %s\n", props.deviceName);

    /* Enumerate all device extensions */
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, NULL);
    VkExtensionProperties* exts = calloc(extCount, sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, exts);

    /* Check key extensions from gap_analysis */
    const char* extensions_to_check[] = {
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,           /* Zero-copy HIP import */
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,            /* Linux FD import */
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,         /* Semaphore import */
        VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,             /* Fence import */
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,                    /* Debug markers */
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME,                    /* Debug report */
        VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,               /* Host image copy */
        VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME,              /* Host query reset */
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,                   /* Memory budget */
        VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,               /* Memory priority */
        VK_EXT_SEPARATE_STENCIL_USAGE_EXTENSION_NAME,        /* Stencil usage */
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,             /* Descriptor indexing */
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,             /* Shader atomics */
        VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME,       /* Vertex divisor */
        VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,      /* Conservative raster */
        VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME,                /* Sample locations */
        VK_EXT_BLEND_OPERATION_ADVANCED_EXTENSION_NAME,        /* Blend operations */
    };
    int num_exts = sizeof(extensions_to_check) / sizeof(extensions_to_check[0]);

    printf("\n--- Extension Availability ---\n");
    VkBool32 has_host_ext = VK_FALSE;
    for (int i = 0; i < num_exts; i++) {
        VkBool32 found = VK_FALSE;
        for (uint32_t j = 0; j < extCount; j++) {
            if (strcmp(exts[j].extensionName, extensions_to_check[i]) == 0) {
                found = VK_TRUE;
                if (i == 0) has_host_ext = VK_TRUE;
                break;
            }
        }
        printf("  %-45s %s\n", extensions_to_check[i], found ? "LISTED" : "no");
    }
    free(exts);

    /* Create device with host_memory_host extension */
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
        .enabledExtensionCount = has_host_ext ? 1 : 0,
        .ppEnabledExtensionNames = devExts,
    };
    VkDevice device;
    r = vkCreateDevice(pdevice, &devInfo, NULL, &device);
    if (r != VK_SUCCESS) {
        printf("\nvkCreateDevice failed: %d\n", r);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("\n--- Device Extensions Enabled: %u ---\n", has_host_ext ? 1 : 0);

    /* Check proc addresses for key entrypoints */
    const char* entrypoints[] = {
        "vkImportMemoryHostPointerEXT",
        "vkGetMemoryOpaqueCaptureAddressKHR",
        "vkGetBufferOpaqueCaptureAddressKHR",
        "vkCmdDebugMarkerInsertEXT",
        "vkCmdDebugMarkerBeginEXT",
        "vkCmdDebugMarkerEndEXT",
        "vkGetPhysicalDeviceMemoryProperties2KHR",
        "vkGetPhysicalDeviceExternalMemoryHostPropertiesEXT",
    };
    int num_eps = sizeof(entrypoints) / sizeof(entrypoints[0]);

    printf("\n--- Proc Address Resolution ---\n");
    for (int i = 0; i < num_eps; i++) {
        void* dev_pa = vkGetDeviceProcAddr(device, entrypoints[i]);
        void* inst_pa = vkGetInstanceProcAddr(instance, entrypoints[i]);
        printf("  %-45s device=%-18p instance=%-18p\n",
               entrypoints[i],
               dev_pa, inst_pa);
    }

    /* Test actual import capability */
    if (has_host_ext) {
        printf("\n--- Import Test ---\n");
        void* hip_ptr = NULL;
        hipError_t herr = hipHostMalloc(&hip_ptr, 4096, hipHostMallocMapped);
        if (herr != hipSuccess) {
            printf("  hipHostMalloc failed: %d\n", herr);
        } else {
            printf("  HIP host memory: %p\n", hip_ptr);

            PFN_vkImportMemoryHostPointerEXT_t fpImport =
                (PFN_vkImportMemoryHostPointerEXT_t)vkGetDeviceProcAddr(device, "vkImportMemoryHostPointerEXT");

            if (!fpImport) {
                printf("  vkImportMemoryHostPointerEXT: NULL (AMD driver bug)\n");
                printf("  STAGING FALLBACK REQUIRED\n");
            } else {
                printf("  vkImportMemoryHostPointerEXT: AVAILABLE\n");

                /* Query host properties */
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProps = {
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
                };
                /* Need to use vkGetPhysicalDeviceProperties2 to get pNext chain */
                VkPhysicalDeviceProperties2 props2 = {
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                    .pNext = &hostProps,
                };
                vkGetPhysicalDeviceProperties2(pdevice, &props2);
                printf("  minImportedHostPointerAlignment: 0x%llx\n",
                       (unsigned long long)hostProps.minImportedHostPointerAlignment);

                /* Try import */
                VkImportMemoryHostPointerInfoEXT importInfo = {
                    .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_POINTER_EXT,
                    .pHostPointer = hip_ptr,
                };
                VkMemoryAllocateInfo allocInfo = {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = 4096,
                    .memoryTypeIndex = 0,
                    .pNext = &importInfo,
                };
                VkDeviceMemory mem;
                VkResult ir = fpImport(device, &allocInfo, &importInfo, &mem);
                printf("  Import result: %d (%s)\n", ir, ir == VK_SUCCESS ? "SUCCESS" : "FAIL");
            }
            hipHostFree(hip_ptr);
        }
    }

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    printf("\n=== Audit Complete ===\n");
    return 0;
}
