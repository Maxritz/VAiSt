/**
 * \file test_vkblas_conv3d.c
 * \brief Test harness for 3D convolution via MIOpen VJITC bridge.
 */
#define __HIP_PLATFORM_AMD__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <vkblas/vkblas.h>

static VkInstance    g_instance;
static VkPhysicalDevice g_phys_dev;
static VkDevice      g_device;
static VkBLASContext* g_ctx;

static void check_vk(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: %s (VkResult %d)\n", msg, r);
        exit(1);
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting conv3d MIOpen bridge test...\n");

    /* Create Vulkan instance + device */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkblas_conv3d",
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
    check_vk(vkblas_create_context(g_instance, g_phys_dev, g_device, &g_ctx), "ctx");

    /* Simple 3D conv test: 1x1x2x2x2 input, 1x1x2x2x2 weights, stride=1
     * Output: 1x1x1x1x1 */
    uint32_t n=1, c=1, di=2, hi=2, wi=2;
    uint32_t k=1, dd=1, dh=1, dw=1;
    uint32_t kd=2, kh=2, kw=2;
    uint32_t pad_d=0, pad_h=0, pad_w=0;
    uint32_t stride_d=1, stride_h=1, stride_w=1;
    uint32_t dil_d=1, dil_h=1, dil_w=1;

    size_t input_bytes  = n * c * di * hi * wi * sizeof(float);
    size_t weight_bytes = k * c * kd * kh * kw * sizeof(float);
    size_t output_bytes = n * k * dd * dh * dw * sizeof(float);

    float h_in[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_w[8]   = {1, 0, 0, 1, 1, 0, 1, 0};
    float h_out[1];
    float h_expected = 1*1 + 2*0 + 3*0 + 4*1 + 5*1 + 6*0 + 7*1 + 8*0; /* = 1+4+5+7 = 17 */

    void* d_in;
    void* d_w;
    void* d_out;
    hipMalloc(&d_in,  input_bytes);
    hipMalloc(&d_w,   weight_bytes);
    hipMalloc(&d_out, output_bytes);

    hipMemcpy(d_in, h_in, input_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_w,  h_w,  weight_bytes, hipMemcpyHostToDevice);
    hipMemset(d_out, 0, output_bytes);

    float alpha = 1.0f, beta = 0.0f;
    VkResult r = vkblas_conv3d_f32(g_ctx, n, c, di, hi, wi, k, dd, dh, dw,
        kd, kh, kw, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
        dil_d, dil_h, dil_w, alpha, d_in, d_w, beta, d_out, VK_NULL_HANDLE);

    if (r == VK_ERROR_FEATURE_NOT_PRESENT) {
        fprintf(stderr, "SKIP: MIOpen not available on this system\n");
        hipFree(d_in); hipFree(d_w); hipFree(d_out);
        vkblas_destroy_context(g_ctx);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 77; /* ctest SKIP */
    }

    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: conv3d returned %d\n", r);
        hipFree(d_in); hipFree(d_w); hipFree(d_out);
        vkblas_destroy_context(g_ctx);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 1;
    }

    hipMemcpy(h_out, d_out, output_bytes, hipMemcpyDeviceToHost);

    float diff = fabsf(h_out[0] - h_expected);
    int ok = diff < 1e-4f;
    printf("conv3d: out=%f expected=%f diff=%.2e => %s\n",
        h_out[0], h_expected, diff, ok ? "PASS" : "FAIL");

    hipFree(d_in); hipFree(d_w); hipFree(d_out);
    vkblas_destroy_context(g_ctx);
    vkDestroyDevice(g_device, NULL);
    vkDestroyInstance(g_instance, NULL);

    return ok ? 0 : 1;
}
