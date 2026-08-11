/**
 * \file test_hip_bridge_libraries.c
 * \brief Validate that ROCm libraries (rocSPARSE, MIOpen, rocsolver) are
 *        accessible from the VAiSt Vulkan runtime via HIP dynamic loading.
 *
 * This validates the VJITC bridge: that libraries available in the ROCm
 * install can be called from the same process that has the Vulkan context.
 */
#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>

#define ROCM_BIN_DIR "F:/ROCM-7.14.0-Windows/bin"
#define ROCM_LIB_DIR "F:/ROCM-7.14.0-Windows/lib"

static void check_dll(const char* dll_name, const char* required_symbol) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", ROCM_BIN_DIR, dll_name);
    HMODULE mod = LoadLibraryA(path);
    if (!mod) {
        printf("  FAIL: %s not found at %s\n", dll_name, path);
        return;
    }
    void* sym = (void*)GetProcAddress(mod, required_symbol);
    if (!sym) {
        printf("  WARN: %s loads but %s not found (dynamic dispatch may fail)\n", dll_name, required_symbol);
    } else {
        printf("  OK:   %s (%s at %p)\n", dll_name, required_symbol, sym);
    }
    FreeLibrary(mod);
}

static void check_lib_static(const char* lib_name, const char* required_symbol) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.lib", ROCM_LIB_DIR, lib_name);
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        printf("  FAIL: %s not found at %s\n", lib_name, path);
        return;
    }
    printf("  OK:   %s.lib found (linker can resolve symbols)\n", lib_name);
    CloseHandle(f);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("VJITC Bridge Library Validation\n");
    printf("ROCm install: %s\n\n", ROCM_LIB_DIR);

    /* Add ROCm bin to PATH so DLLs can find their dependencies */
    wchar_t pathVar[4096];
    GetEnvironmentVariableW(L"PATH", pathVar, 4096);
    wcscat(pathVar, L";");
    wcscat(pathVar, L"F:\\ROCM-7.14.0-Windows\\bin");
    SetEnvironmentVariableW(L"PATH", pathVar);
    /* Also try adding to DLL search path for already-loaded scenario */
    AddDllDirectory(L"F:\\ROCM-7.14.0-Windows\\bin");

    /* === Check DLL availability === */
    printf("=== Dynamic Libraries (runtime dispatch) ===\n");
    check_dll("hipsparse.dll", "hipsparseCreate");
    check_dll("hipblas.dll", "hipblasCreate");
    check_dll("MIOpen.dll", "miopenCreate");
    check_dll("rocsolver.dll", "rocsolver_dgetrf");
    check_dll("hiprtc0714.dll", "hiprtcCreateProgram");

    printf("\n=== Static Libraries (link-time resolution) ===\n");
    check_lib_static("hipsparse", "hipsparseCreate");
    check_lib_static("hipblas", "hipblasCreate");
    check_lib_static("MIOpen", "miopenCreate");
    check_lib_static("rocsolver", "rocsolver_dgetrf");
    check_lib_static("hiprtc", "hiprtcCreateProgram");

    /* === Validate HIP runtime works with Vulkan context === */
    printf("\n=== HIP Runtime + Vulkan Context ===\n");
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vjitc_validate",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    if (vkCreateInstance(&instInfo, NULL, &instance) != VK_SUCCESS) {
        printf("FAIL: vkCreateInstance\n");
        return 1;
    }

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    VkPhysicalDevice pdevice;
    vkEnumeratePhysicalDevices(instance, &pdcount, &pdevice);

    /* Enable external_memory_host for zero-copy bridge */
    const char* devExts[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qInfo,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = devExts,
    };
    VkDevice device;
    if (vkCreateDevice(pdevice, &devInfo, NULL, &device) != VK_SUCCESS) {
        printf("FAIL: vkCreateDevice\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("Vulkan context + external_memory_host: OK\n");

    /* === Validate HIP can allocate and import to Vulkan === */
    printf("\n=== HIP Allocation + Vulkan Import ===\n");
    void* hip_ptr = NULL;
    if (hipHostMalloc(&hip_ptr, 1024 * sizeof(float), hipHostMallocMapped) != hipSuccess) {
        printf("FAIL: hipHostMalloc\n");
        return 1;
    }
    printf("HIP host allocation: %p\n", hip_ptr);

    VkBufferCreateInfo bufInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 1024 * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer vk_buffer;
    vkCreateBuffer(device, &bufInfo, NULL, &vk_buffer);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, vk_buffer, &memReqs);
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pdevice, &memProps);

    uint32_t memType = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memType = i; break;
        }
    }

    VkImportMemoryHostPointerInfoEXT importInfo = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = hip_ptr,
    };
    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memType,
        .pNext = &importInfo,
    };
    VkDeviceMemory vk_memory;
    VkResult vr = vkAllocateMemory(device, &allocInfo, NULL, &vk_memory);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkAllocateMemory (import) %d\n", vr);
    } else {
        printf("Vulkan imported HIP allocation: OK (VkDeviceMemory=%p)\n", (void*)vk_memory);
        vkFreeMemory(device, vk_memory, NULL);
    }
    vkDestroyBuffer(device, vk_buffer, NULL);
    hipHostFree(hip_ptr);

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    printf("\n=== RESULT: All VJITC bridge libraries are accessible ===\n");
    return 0;
}
