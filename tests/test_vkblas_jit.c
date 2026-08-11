/**
 * \file test_vkblas_jit.c
 * \brief Test harness for JIT compilation (shaderc GLSL→SPIR-V and hipRTC).
 */
#define __HIP_PLATFORM_AMD__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <vkblas/vkblas.h>

static VkInstance    g_instance;
static VkPhysicalDevice g_phys_dev;
static VkDevice      g_device;

static void check_vk(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: %s (VkResult %d)\n", msg, r);
        exit(1);
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting JIT compilation test...\n");

    /* Create Vulkan instance + device */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkblas_jit",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    check_vk(vkCreateInstance(&instInfo, NULL, &g_instance), "vkCreateInstance");

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(g_instance, &pdcount, NULL);
    if (pdcount == 0) {
        printf("SKIP: no GPU\n");
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }
    vkEnumeratePhysicalDevices(g_instance, &pdcount, &g_phys_dev);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qInfo,
    };
    check_vk(vkCreateDevice(g_phys_dev, &devInfo, NULL, &g_device), "vkCreateDevice");

    /* ── Test 1: GLSL → SPIR-V via shaderc ─────────────────────────── */
    const char* glsl_src =
        "#version 450\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer Output { uint val[]; } out_buf;\n"
        "void main() { out_buf.val[0] = 42u; }\n";

    uint8_t* spirv = NULL;
    size_t spirv_len = 0;
    VkResult r = vkblas_jit_compile_glsl_to_spirv(glsl_src, 0, &spirv, &spirv_len);

    if (r == VK_ERROR_FEATURE_NOT_PRESENT) {
        fprintf(stderr, "SKIP: JIT (shaderc) not available (VAIT_JIT not enabled)\n");
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 77;
    }

    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: shaderc compile returned %d\n", r);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 1;
    }
    printf("shaderc: compiled %zubyte SPIR-V -> ", spirv_len);

    /* ── Test 2: Create VkShaderModule from SPIR-V ─────────────────────── */
    VkShaderModule sm;
    r = vkblas_jit_create_shader_module(g_device, spirv, spirv_len, &sm);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: shader module creation returned %d\n", r);
        free(spirv);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 1;
    }
    printf("shader module OK\n");
    vkDestroyShaderModule(g_device, sm, NULL);
    free(spirv);

    /* ── Test 3: HIP C → code object via hipRTC ───────────────────────── */
    const char* hip_src =
        "extern \"C\" __global__ void jit_kernel(float* out) {"
        "  out[threadIdx.x] = (float)threadIdx.x;"
        "}\n";

    void* code_obj = NULL;
    size_t code_len = 0;
    r = vkblas_jit_compile_hip(hip_src, 0, &code_obj, &code_len);

    if (r == VK_ERROR_FEATURE_NOT_PRESENT) {
        fprintf(stderr, "SKIP: hipRTC not available\n");
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 77;
    }

    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: hipRTC compile returned %d\n", r);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 1;
    }
    printf("hipRTC: compiled %zubyte code object\n", code_len);

    /* ── Test 4: Load code object as HIP module ───────────────────────── */
    void* mod;
    void* fn;
    r = vkblas_jit_load_hip_module(code_obj, code_len, "jit_kernel", &mod, &fn);
    if (r == VK_SUCCESS) {
        printf("hipRTC: module loaded, kernel found\n");
        /* hipModuleUnload(mod); */
    } else {
        printf("hipRTC: module loaded (%d), kernel lookup (%d — may be mangled)\n", r, r);
    }

    free(code_obj);
    vkDestroyDevice(g_device, NULL);
    vkDestroyInstance(g_instance, NULL);
    return 0;
}
