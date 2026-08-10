#include "vkstream/vkstream.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pdevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue transfer_q = VK_NULL_HANDLE;
    VkQueue compute_q = VK_NULL_HANDLE;

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vkstream_test",
        .apiVersion = VK_API_VERSION_1_4,
    };

    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };

    if (vkCreateInstance(&instInfo, NULL, &instance) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed\n");
        return -1;
    }

    uint32_t pdevCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdevCount, NULL);
    assert(pdevCount > 0);

    VkPhysicalDevice* pdevices = (VkPhysicalDevice*)calloc(pdevCount, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &pdevCount, pdevices);
    pdevice = pdevices[0];
    free(pdevices);

    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    };

    const char* devExts[] = { VK_KHR_MAINTENANCE3_EXTENSION_NAME };
    devInfo.enabledExtensionCount = 1;
    devInfo.ppEnabledExtensionNames = devExts;

    vkCreateDevice(pdevice, &devInfo, NULL, &device);
    vkGetDeviceQueue(device, 0, 0, &transfer_q);
    vkGetDeviceQueue(device, 0, 0, &compute_q);

    vkstream_context_t* ctx = vkstream_create(device, pdevice, 0, 0);
    assert(ctx != NULL);

    printf("vkstream: BAR available = %s, aperture = %llu bytes\n",
           vkstream_is_bar_available(ctx) ? "yes" : "no",
           (unsigned long long)vkstream_get_bar_aperture(ctx));

    vkstream_destroy(ctx);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    printf("test_vkstream: PASS\n");
    return 0;
}
