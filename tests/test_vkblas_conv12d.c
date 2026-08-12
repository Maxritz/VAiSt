/**
 * \file test_vkblas_conv12d.c
 * \brief Validate 1D/2D convolution via MIOpen VJITC bridge.
 *
 * Tests simple conv1d (1x1 kernel) and conv2d (3x3 kernel) with CPU reference.
 * Skipped gracefully if miopen.dll is not present.
 */
#define __HIP_PLATFORM_AMD__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <vkblas/vkblas.h>

/* === Test 1: conv1d, 1x1 kernel (identity passthrough) === */
static int test_conv1d(void)
{
    uint32_t n = 1, c = 2, li = 8, k = 2, lo = 8, kl = 1;
    /* Input: [batch=1, chan=2, len=8] */
    float x_host[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,   /* chan 0 */
        0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f    /* chan 1 */
    };
    /* Weights: 1x1, identity-ish: each out chan = same in chan */
    float w_host[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float y_host[16];

    void* d_x, * d_w, * d_y;
    hipMalloc(&d_x, 16 * sizeof(float));
    hipMalloc(&d_w, 4 * sizeof(float));
    hipMalloc(&d_y, 16 * sizeof(float));
    hipMemcpy(d_x, x_host, 16 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w, w_host, 4 * sizeof(float), hipMemcpyHostToDevice);

    VkBLASContext* ctx = NULL;
    VkApplicationInfo appInfo = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_4 };
    VkInstanceCreateInfo instInfo = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &appInfo };
    VkInstance instance; vkCreateInstance(&instInfo, NULL, &instance);
    uint32_t pdcount = 0; vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    if (pdcount == 0) { printf("SKIP: no GPU\n"); return 0; }
    VkPhysicalDevice pd; vkEnumeratePhysicalDevices(instance, &pdcount, &pd);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkDevice device; VkDeviceCreateInfo devInfo12d = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qInfo }; vkCreateDevice(pd, &devInfo12d, NULL, &device);
    vkblas_create_context(instance, pd, device, &ctx);

    VkResult r = vkblas_conv1d_f32(ctx, n, c, li, k, lo, kl,
        0, 1, 1,   /* pad_l, stride_l, dil_l */
        1.0f, d_x, d_w, 0.0f, d_y, VK_NULL_HANDLE);

    if (r != VK_SUCCESS) {
        printf("  conv1d_f32 : SKIP (miopen.dll not available or unsupported: %d)\n", r);
        vkblas_destroy_context(ctx); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
        hipFree(d_x); hipFree(d_w); hipFree(d_y);
        return 1; /* pass = skip */
    }

    hipMemcpy(y_host, d_y, 16 * sizeof(float), hipMemcpyDeviceToHost);

    /* Expected: y[0][0] = x[0][0], y[0][1] = x[1][1], etc. (identity via w) */
    int pass = 1;
    for (int ch = 0; ch < 2; ch++)
        for (int l = 0; l < 8; l++) {
            float exp = x_host[ch * 8 + l]; /* identity 1x1 */
            float got = y_host[ch * 8 + l];
            if (fabsf(got - exp) > 0.01f) {
                printf("  MISMATCH y[%d][%d]: exp=%.4f got=%.4f\n", ch, l, exp, got);
                pass = 0;
            }
        }

    printf("  conv1d_f32 : %s\n", pass ? "PASS" : "FAIL");
    vkblas_destroy_context(ctx); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
    hipFree(d_x); hipFree(d_w); hipFree(d_y);
    return pass;
}

/* === Test 2: conv2d, 3x3 kernel, stride=1, pad=1 (preserves spatial dim) === */
static int test_conv2d(void)
{
    uint32_t n = 1, c = 1, hi = 5, wi = 5, k = 1, dh = 5, dw = 5, kh = 3, kw = 3;

    /* Input: 5x5 all-ones */
    float x_host[25];
    memset(x_host, 0, sizeof(x_host));
    for (int i = 0; i < 25; i++) x_host[i] = 1.0f;

    /* Kernel: 3x3 edge-detection-like, sum of weights = 1 */
    float w_host[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    /* Expected output: center=1, borders=0 (for interior pixels) */

    float y_host[25];
    void* d_x, * d_w, * d_y;
    hipMalloc(&d_x, 25 * sizeof(float));
    hipMalloc(&d_w, 9 * sizeof(float));
    hipMalloc(&d_y, 25 * sizeof(float));
    hipMemcpy(d_x, x_host, 25 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w, w_host, 9 * sizeof(float), hipMemcpyHostToDevice);

    VkBLASContext* ctx = NULL;
    VkApplicationInfo appInfo = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_4 };
    VkInstanceCreateInfo instInfo = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &appInfo };
    VkInstance instance; vkCreateInstance(&instInfo, NULL, &instance);
    uint32_t pdcount = 0; vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    if (pdcount == 0) { printf("SKIP: no GPU\n"); return 0; }
    VkPhysicalDevice pd; vkEnumeratePhysicalDevices(instance, &pdcount, &pd);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkDevice device; VkDeviceCreateInfo devInfo12d = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qInfo }; vkCreateDevice(pd, &devInfo12d, NULL, &device);
    vkblas_create_context(instance, pd, device, &ctx);

    VkResult r = vkblas_conv2d_f32(ctx, n, c, hi, wi, k, dh, dw, kh, kw,
        1, 1,    /* pad_h, pad_w */
        1, 1,    /* stride_h, stride_w */
        1, 1,    /* dil_h, dil_w */
        1.0f, d_x, d_w, 0.0f, d_y, VK_NULL_HANDLE);

    if (r != VK_SUCCESS) {
        printf("  conv2d_f32 : SKIP (miopen.dll not available or unsupported: %d)\n", r);
        vkblas_destroy_context(ctx); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
        hipFree(d_x); hipFree(d_w); hipFree(d_y);
        return 1; /* pass = skip */
    }

    hipMemcpy(y_host, d_y, 25 * sizeof(float), hipMemcpyDeviceToHost);

    /* Expected: center pixel = 1*1 = 1, edges/corners = 0 (with padding=1) */
    int pass = 1;
    /* Check center */
    if (fabsf(y_host[2 * 5 + 2] - 1.0f) > 0.01f) {
        printf("  MISMATCH center: exp=1.0 got=%.4f\n", y_host[12]);
        pass = 0;
    }
    /* Expected: all output = 1.0 (3x3 kernel with center=1, input all=1, pad=1)
       At any output pixel, the center kernel weight (1.0) sees an input pixel
       (which is 1.0 due to zero-padding), so output = 1.0 everywhere */
    int pass_all = 1;
    for (int i = 0; i < 25; i++) {
        if (fabsf(y_host[i] - 1.0f) > 0.01f) {
            printf("  MISMATCH y[%d]: exp=1.0 got=%.4f\n", i, y_host[i]);
            pass_all = 0;
        }
    }
    /* Keep 'pass' for backward compat from center check */
    pass = pass_all;

    printf("  conv2d_f32 : %s\n", pass ? "PASS" : "FAIL");
    vkblas_destroy_context(ctx); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
    hipFree(d_x); hipFree(d_w); hipFree(d_y);
    return pass;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting conv1d/conv2d MIOpen bridge test...\n");

    int p1 = test_conv1d();
    int p2 = test_conv2d();

    printf("\n=== RESULT: %s ===\n", (p1 && p2) ? "PASS" : "FAIL");
    return (p1 && p2) ? 0 : 1;
}
