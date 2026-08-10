#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <windows.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting minimal Vulkan test...\n");

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test",
        .apiVersion = VK_API_VERSION_1_0,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    VkResult r = vkCreateInstance(&instInfo, NULL, &instance);
    if (r != VK_SUCCESS) {
        printf("FAIL: vkCreateInstance %d\n", r);
        return 1;
    }
    printf("PASS: instance created\n");

    uint32_t pdcount = 0;
    r = vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    printf("enum PD result: %d, count: %u\n", r, pdcount);
    if (pdcount == 0) {
        printf("FAIL: No physical devices\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    VkPhysicalDevice* pdvs = calloc(pdcount, sizeof(VkPhysicalDevice));
    r = vkEnumeratePhysicalDevices(instance, &pdcount, pdvs);
    printf("enum PD2 result: %d\n", r);
    VkPhysicalDevice pdevice = pdvs[0];
    free(pdvs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdevice, &props);
    printf("GPU: %s\n", props.deviceName);

    /* Check extensions */
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, NULL);
    printf("Device extensions: %u\n", extCount);

    VkExtensionProperties* exts = calloc(extCount, sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(pdevice, NULL, &extCount, exts);

    VkBool32 has_ext = VK_FALSE;
    for (uint32_t i = 0; i < extCount; i++) {
        if (strstr(exts[i].extensionName, "host") || strstr(exts[i].extensionName, "Host")) {
            printf("  Found host-related ext: %s\n", exts[i].extensionName);
        }
        if (strcmp(exts[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
            has_ext = VK_TRUE;
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
    printf("PASS: device created\n");

    /* Try getting proc addr */
    void* fptr = vkGetDeviceProcAddr(device, "vkImportMemoryHostPointerEXT");
    printf("vkGetDeviceProcAddr: %p\n", fptr);

    fptr = vkGetInstanceProcAddr(instance, "vkImportMemoryHostPointerEXT");
    printf("vkGetInstanceProcAddr: %p\n", fptr);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pdevice, &memProps);
    printf("Memory types: %u\n", memProps.memoryTypeCount);

    /* Check queue families */
    uint32_t qFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &qFamilyCount, NULL);
    printf("Queue families: %u\n", qFamilyCount);
    VkQueueFamilyProperties* qProps = calloc(qFamilyCount, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &qFamilyCount, qProps);
    for (uint32_t i = 0; i < qFamilyCount; i++) {
        printf("  Q[%u]: flags=0x%x count=%u\n", i, qProps[i].queueFlags, qProps[i].queueCount);
    }
    free(qProps);

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    printf("DONE\n");
    return 0;
}
