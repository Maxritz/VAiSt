/**
 * \file test_vkblas_sparse.c
 * \brief Validate sparse GEMM bridge via VJITC path to rocSPARSE.
 *
 * Tests: C = A_sparse (4x4, 6 nonzeros) @ B (4x4 dense) + 0 * C
 * Expected result is computed on CPU and compared.
 */
#define __HIP_PLATFORM_AMD__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <vkblas/vkblas.h>

/* 4x4 sparse matrix A (6 nonzeros in CSR format)
        [ 1.0  0.0  2.0  0.0 ]
        [ 0.0  0.0  0.0  3.0 ]
        [ 4.0  0.0  5.0  0.0 ]
        [ 0.0  6.0  0.0  7.0 ]
   CSR row_ptr: [0, 2, 3, 5, 7]
   CSR col_ind: [0, 2, 3, 0, 2, 1, 3]
   CSR val:     [1, 2, 3, 4, 5, 6, 7]
   nnz = 7 */

static float A_sparse[7] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
static uint32_t A_row_ptr[5] = {0, 2, 3, 5, 7};
static uint32_t A_col_ind[7] = {0, 2, 3, 0, 2, 1, 3};

/* Dense B (4x4) */
static float B_host[16] = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f
};

static float compute_expected(int row, int col) {
    /* C = A @ B (col-major storage for both B and C)
       A is row-major sparse, B is col-major dense:
       C[row][col] = sum_k A[row][k] * B[k][col] */
    float sum = 0.0f;
    for (int k = 0; k < 4; k++) {
        /* Look up A[row][k] in CSR */
        float a_val = 0.0f;
        int found = 0;
        for (uint32_t idx = A_row_ptr[row]; idx < A_row_ptr[row+1]; idx++) {
            if (A_col_ind[idx] == (uint32_t)k) {
                a_val = A_sparse[idx];
                found = 1;
                break;
            }
        }
        if (found) {
            sum += a_val * B_host[col * 4 + k]; /* B is col-major */
        }
    }
    return sum;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting sparse GEMM bridge test...\n");

    /* === Create Vulkan context (minimal) === */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_vkblas_sparse",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    VkResult vr = vkCreateInstance(&instInfo, NULL, &instance);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkCreateInstance (%d)\n", vr);
        return 1;
    }

    uint32_t pdcount = 0;
    vkEnumeratePhysicalDevices(instance, &pdcount, NULL);
    if (pdcount == 0) {
        printf("SKIP: no GPU\n");
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    VkPhysicalDevice pdevice;
    vkEnumeratePhysicalDevices(instance, &pdcount, &pdevice);

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
    vr = vkCreateDevice(pdevice, &devInfo, NULL, &device);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkCreateDevice (%d)\n", vr);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("Vulkan device created with external_memory_host\n");

    /* === Create VkBLASContext via standard API === */
    VkBLASContext* ctx = NULL;
    vr = vkblas_create_context(instance, pdevice, device, &ctx);
    if (vr != VK_SUCCESS || !ctx) {
        printf("FAIL: vkblas_create_context (%d)\n", vr);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("vkblas_create_context: OK\n");

    /* === Allocate HIP device pointers === */
    void* d_B = NULL;
    void* d_C = NULL;

    hipError_t he = hipMalloc(&d_B, 16 * sizeof(float));
    if (he != hipSuccess) {
        printf("FAIL: hipMalloc B (%d)\n", he);
        return 1;
    }
    hipMemcpy(d_B, B_host, 16 * sizeof(float), hipMemcpyHostToDevice);

    he = hipMalloc(&d_C, 16 * sizeof(float));
    if (he != hipSuccess) {
        printf("FAIL: hipMalloc C (%d)\n", he);
        hipFree(d_B);
        return 1;
    }
    /* Initialize C to zero */
    float zeros[16] = {0};
    hipMemcpy(d_C, zeros, 16 * sizeof(float), hipMemcpyHostToDevice);

    printf("HIP device pointers allocated: B=%p, C=%p\n", d_B, d_C);

    /* Allocate CSR arrays on device */
    void* d_row_ptr = NULL;
    void* d_col_ind = NULL;
    void* d_val = NULL;
    hipMalloc(&d_row_ptr, 5 * sizeof(uint32_t));
    hipMalloc(&d_col_ind, 7 * sizeof(uint32_t));
    hipMalloc(&d_val, 7 * sizeof(float));
    hipMemcpy(d_row_ptr, A_row_ptr, 5 * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_col_ind, A_col_ind, 7 * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_val, A_sparse, 7 * sizeof(float), hipMemcpyHostToDevice);

    /* === Call sparse GEMM via VJITC bridge === */
    float alpha = 1.0f;
    float beta = 0.0f;

    VkCommandBuffer cmd = VK_NULL_HANDLE; /* unused for bridge calls */

    VkResult result = vkblas_sparse_gemm_f32(
        ctx,
        VKBLAS_OP_N, VKBLAS_OP_N,
        4, 4, 4,  /* m=4, n=4, k=4 */
        7,         /* nnz */
        &alpha,
        (uint32_t*)d_row_ptr,
        (uint32_t*)d_col_ind,
        (float*)d_val,
        (void*)d_B,
        &beta,
        (void*)d_C,
        cmd);

    if (result != VK_SUCCESS) {
        printf("FAIL: vkblas_sparse_gemm_f32 returned %d\n", result);
        hipFree(d_val);
        hipFree(d_col_ind);
        hipFree(d_row_ptr);
        hipFree(d_C);
        hipFree(d_B);
        vkblas_destroy_context(ctx);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    printf("vkblas_sparse_gemm_f32: OK\n");

    /* === Read back result and verify === */
    float C_host[16];
    hipMemcpy(C_host, d_C, 16 * sizeof(float), hipMemcpyDeviceToHost);

    printf("\nResult matrix C (col-major, expect non-zero only where A has data):\n");
    int pass = 1;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            float expected = compute_expected(row, col);
            float got = C_host[col * 4 + row]; /* col-major access */
            float diff = fabsf(got - expected);
            if (diff > 0.01f) {
                printf("  MISMATCH C[%d,%d]: expected=%.4f, got=%.4f (diff=%.4f)\n",
                       row, col, expected, got, diff);
                pass = 0;
            } else {
                printf("  C[%d,%d] = %.4f (expected %.4f)\n", row, col, got, expected);
            }
        }
    }

    /* === Sparse triangular solve: A_L * y = x (lower unit triangular) ===
       A_L =
         [ 1  0  0  0 ]
         [ 2  1  0  0 ]
         [ 3  0  1  0 ]
         [ 4  0  5  1 ]
       CSR row_ptr: [0,1,3,5,8]
       CSR col_ind: [0, 0,1, 0,2, 0,2,3]
       CSR val:     [1, 2,1, 3,1, 4,5,1]
       x = [1, 5, 6, 10]^T  -> forward substitution gives
         y0 = 1
         y1 = 5 - 2*1        = 3
         y2 = 6 - 3*1        = 3
         y3 = 10 - 4*1 - 5*3 = -9
    */
    {
        uint32_t Lt_row_ptr[5] = {0, 1, 3, 5, 8};
        uint32_t Lt_col_ind[8] = {0, 0, 1, 0, 2, 0, 2, 3};
        float    Lt_val[8]     = {1.f, 2.f, 1.f, 3.f, 1.f, 4.f, 5.f, 1.f};
        float    x_host[4]     = {1.f, 5.f, 6.f, 10.f};
        float    y_expect[4]   = {1.f, 3.f, 3.f, -9.f};
        float    y_host[4]     = {0.f, 0.f, 0.f, 0.f};

        void* dL_row = NULL; void* dL_col = NULL; void* dL_val = NULL;
        void* d_x = NULL;     void* d_y = NULL;
        hipMalloc(&dL_row, 5 * sizeof(uint32_t));
        hipMalloc(&dL_col, 8 * sizeof(uint32_t));
        hipMalloc(&dL_val, 8 * sizeof(float));
        hipMalloc(&d_x, 4 * sizeof(float));
        hipMalloc(&d_y, 4 * sizeof(float));
        hipMemcpy(dL_row, Lt_row_ptr, 5 * sizeof(uint32_t), hipMemcpyHostToDevice);
        hipMemcpy(dL_col, Lt_col_ind, 8 * sizeof(uint32_t), hipMemcpyHostToDevice);
        hipMemcpy(dL_val, Lt_val, 8 * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_x, x_host, 4 * sizeof(float), hipMemcpyHostToDevice);

        float spsv_alpha = 1.0f;
        VkResult sres = vkblas_sparse_spsv_f32(ctx, VKBLAS_OP_N, 4, 8,
            &spsv_alpha, (uint32_t*)dL_row, (uint32_t*)dL_col, (float*)dL_val,
            (void*)d_x, (void*)d_y, cmd);
        if (sres != VK_SUCCESS) {
            printf("FAIL: vkblas_sparse_spsv_f32 returned %d\n", sres);
            pass = 0;
        } else {
            hipMemcpy(y_host, d_y, 4 * sizeof(float), hipMemcpyDeviceToHost);
            for (int i = 0; i < 4; i++) {
                if (fabsf(y_host[i] - y_expect[i]) > 0.01f) {
                    printf("  MISMATCH spsv y[%d]: expected=%.4f, got=%.4f\n",
                           i, y_expect[i], y_host[i]);
                    pass = 0;
                } else {
                    printf("  spsv y[%d] = %.4f (expected %.4f)\n",
                           i, y_host[i], y_expect[i]);
                }
            }
        }
        hipFree(dL_row); hipFree(dL_col); hipFree(dL_val);
        hipFree(d_x);    hipFree(d_y);
    }

    /* === Cleanup === */
    hipFree(d_val);
    hipFree(d_col_ind);
    hipFree(d_row_ptr);
    hipFree(d_C);
    hipFree(d_B);
    vkblas_destroy_context(ctx);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    printf("\n=== RESULT: %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
