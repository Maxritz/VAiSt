#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>

#define VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT 1000176000
#define VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_POINTER_EXT 0x00000001u
#define VK_SHARING_MODE_PRIVATE 0

typedef VkResult (VKAPI_PTR *PFN_vkImportMemoryHostPointerEXT_t)(
    VkDevice device,
    const VkMemoryAllocateInfo* pAllocInfo,
    const VkImportMemoryHostPointerInfoEXT* pImportInfo,
    VkDeviceMemory* pMemory);

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "host_props_test",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    if (vkCreateInstance(&instInfo, NULL, &instance) != VK_SUCCESS) {
        printf("vkCreateInstance failed\n");
        return 1;
    }

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    VkPhysicalDevice* pdvs = malloc(pdcount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &pdcount, pdvs);
    VkPhysicalDevice pdevice = pdvs[0];
    free(pdvs);

    /* Query the host properties via pNext chain of VkPhysicalDeviceProperties2 */
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProps = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &hostProps,
    };
    vkGetPhysicalDeviceProperties2(pdevice, &props2);

    printf("VkPhysicalDeviceExternalMemoryHostPropertiesEXT:\n");
    printf("  minImportedHostPointerAlignment: 0x%llx (%llu bytes)\n",
           (unsigned long long)hostProps.minImportedHostPointerAlignment,
           (unsigned long long)hostProps.minImportedHostPointerAlignment);

    /* Query memory properties */
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pdevice, &memProps);
    printf("  memoryTypeCount: %u\n", memProps.memoryTypeCount);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
        uint32_t heap = memProps.memoryTypes[i].heapIndex;
        char buf[256];
        snprintf(buf, sizeof(buf), "  [%u]: flags=0x%x heap=%u",
                 i, (unsigned)flags, heap);
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) strcat(buf, " HOST_VISIBLE");
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) strcat(buf, " HOST_COHERENT");
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) strcat(buf, " DEVICE_LOCAL");
        if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) strcat(buf, " HOST_CACHED");
        printf("%s\n", buf);
    }

    /* Check actual HIP pointer alignment */
    void* hip_ptr = NULL;
    hipHostMalloc(&hip_ptr, 4096, hipHostMallocMapped);
    printf("\nHIP host pointer: %p (alignment: %llu)\n",
           hip_ptr, (unsigned long long)((uintptr_t)hip_ptr % 
           (uintptr_t)hostProps.minImportedHostPointerAlignment));

    /* Check if hipHostAlloc can give us aligned memory */
    void* hip_ptr2 = NULL;
    hipHostMalloc(&hip_ptr2, 4096, hipHostMallocMapped | hipHostMallocPortable);
    printf("HIP host ptr2: %p\n", hip_ptr2);

    hipHostFree(hip_ptr);
    hipHostFree(hip_ptr2);
    vkDestroyInstance(instance, NULL);
    return 0;
}
