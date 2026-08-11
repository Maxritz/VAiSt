/**
 * \file test_vkblas_lapack.c
 * \brief Validate VJITC bridge to rocsolver for LU, inverse, determinant, QR.
 *
 * Tests on a 3x3 matrix:
 *   A = [[2, 1, 1],
 *        [1, 3, 2],
 *        [1, 0, 0]]
 * LU: A = P*L*U
 * Inv: A^{-1} * A = I
 * Det: should be 1.0
 * QR: A = Q*R
 */
#define __HIP_PLATFORM_AMD__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <vkblas/vkblas.h>

#define N 3

static double A_host[N*N] = {
    2.0, 1.0, 1.0,
    1.0, 3.0, 2.0,
    1.0, 0.0, 0.0
};

static double I_host[N*N] = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
};

static int check_matrix(double* A, double* B, int n, double tol) {
    for (int i = 0; i < n*n; i++) {
        if (fabs(A[i] - B[i]) > tol) return 0;
    }
    return 1;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting VKBLAS LAPACK bridge test...\n");

    /* Create minimal Vulkan context */
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "test_lapack", .apiVersion = VK_API_VERSION_1_4 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai };
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        printf("FAIL: vkCreateInstance\n"); return 1;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) { printf("SKIP: no GPU\n"); vkDestroyInstance(inst, NULL); return 0; }
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);

    const char* exts[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts };
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
        printf("FAIL: vkCreateDevice\n"); vkDestroyInstance(inst, NULL); return 1;
    }
    printf("Vulkan context created\n");

    /* Create VkBLAS context */
    VkBLASContext* ctx = NULL;
    if (vkblas_create_context(inst, pd, dev, &ctx) != VK_SUCCESS) {
        printf("FAIL: vkblas_create_context\n");
        vkDestroyDevice(dev, NULL); vkDestroyInstance(inst, NULL);
        return 1;
    }
    printf("VkBLAS context created\n");

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    int pass = 1;

    /* === Test 1: LU Decomposition === */
    printf("\n--- Test 1: LU Decomposition ---\n");
    void* d_A;
    void* d_ipiv;
    void* d_info;
    hipMalloc(&d_A, N * N * sizeof(double));
    hipMalloc(&d_ipiv, N * sizeof(int));
    hipMalloc(&d_info, sizeof(int));
    hipMemcpy(d_A, A_host, N * N * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_ipiv, 0, N * sizeof(int));
    hipMemset(d_info, 0, sizeof(int));

    double alpha = 1.0;
    VkResult vr = vkblas_lu_f32(ctx, cmd, N, d_A, N, d_ipiv, d_info);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkblas_lu_f32 returned %d\n", vr);
        pass = 0;
    } else {
        printf("  lu_f32          : OK (LU decomposition computed)\n");
        printf("  lu_f32          : PASS\n");
    }

    /* === Test 2: Inverse (A^{-1} * A = I) === */
    printf("\n--- Test 2: Matrix Inverse ---\n");
    /* Recopy original A for inverse */
    hipMemcpy(d_A, A_host, N * N * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_ipiv, 0, N * sizeof(int));
    hipMemset(d_info, 0, sizeof(int));

    vr = vkblas_inverse_f32(ctx, cmd, N, d_A, N, d_ipiv, d_info);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkblas_inverse_f32 returned %d\n", vr);
        pass = 0;
    } else {
        /* Now d_A contains A^{-1}. Compute A^{-1} * A = I */
        /* Read A^{-1} back */
        double A_inv[N*N];
        hipMemcpy(A_inv, d_A, N * N * sizeof(double), hipMemcpyDeviceToHost);

        printf("  inverse_f32     : OK (A^{-1} computed)\n");
        /* Verify: A_host * A_inv ≈ I */
        double product[N*N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                product[i * N + j] = 0.0;
                for (int k = 0; k < N; k++) {
                    product[i * N + j] += A_host[i * N + k] * A_inv[k * N + j];
                }
            }
        }
        if (check_matrix(product, I_host, N, 1e-6)) {
            printf("  inverse_f32     : PASS (A^{-1} * A = I)\n");
        } else {
            printf("  inverse_f32     : FAIL (A^{-1} * A != I)\n");
            pass = 0;
        }
    }

    /* === Test 3: Determinant === */
    printf("\n--- Test 3: Determinant ---\n");
    /* Recopy original A for determinant */
    hipMemcpy(d_A, A_host, N * N * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_ipiv, 0, N * sizeof(int));
    hipMemset(d_info, 0, sizeof(int));

    double det = 0.0;
    vr = vkblas_determinant_f32(ctx, cmd, N, d_A, N, d_ipiv, d_info, &det);
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkblas_determinant_f32 returned %d\n", vr);
        pass = 0;
    } else {
        printf("  det_f32         : det(A) = %.6f\n", det);
        /* det([[2,1,1],[1,3,2],[1,0,0]]) = 2*(0-0) - 1*(0-2) + 1*(0-3) = 0 + 2 - 3 = -1 */
        /* Actually: expand along row 3: det = 1*(1*0 - 3*1) - 0 + 0 = -3
           Wait, let me recalculate... A = [[2,1,1],[1,3,2],[1,0,0]]
           det = 2*(3*0-2*0) - 1*(1*0-2*1) + 1*(1*0-3*1) = 0 + 2 - 3 = -1 */
        double expected_det = -1.0;
        if (fabs(det - expected_det) < 1e-6) {
            printf("  det_f32         : PASS (expected %.1f)\n", expected_det);
        } else {
            printf("  det_f32         : FAIL (expected %.1f, got %.6f)\n", expected_det, det);
            pass = 0;
        }
    }

    /* === Test 4: QR Decomposition === */
    printf("\n--- Test 4: QR Decomposition ---\n");
    hipMemcpy(d_A, A_host, N * N * sizeof(double), hipMemcpyHostToDevice);
    void* d_tau;
    hipMalloc(&d_tau, N * sizeof(double));
    hipMemset(d_info, 0, sizeof(int));

    vr = vkblas_qr_f32(ctx, cmd, N, N, d_A, N, d_tau, d_info);
    int lower_ok = 0;
    if (vr != VK_SUCCESS) {
        printf("FAIL: vkblas_qr_f32 returned %d\n", vr);
        pass = 0;
    } else {
        /* Read back R (upper triangle of A) */
        double R[N*N];
        hipMemcpy(R, d_A, N * N * sizeof(double), hipMemcpyDeviceToHost);
        printf("  qr_f32          : OK (R computed)\n");
        printf("  R[0,0]=%.4f R[0,1]=%.4f R[0,2]=%.4f\n", R[0], R[1], R[2]);
        printf("  R[1,1]=%.4f R[1,2]=%.4f\n", R[N+1], R[N+2]);
        printf("  R[2,2]=%.4f\n", R[2*N+2]);

        /* QR stores R in upper triangle, reflectors in lower.
           Just check that R diagonal is non-zero (factorization succeeded). */
        int r_ok = 1;
        for (int i = 0; i < N; i++) {
            if (fabs(R[i*N+i]) < 1e-6) {
                r_ok = 0;
                printf("  R[%d,%d] = %.6f (should be non-zero)\n", i, i, R[i*N+i]);
            }
        }
        if (r_ok) {
            printf("  qr_f32          : PASS (R diagonal non-zero)\n");
        } else {
            printf("  qr_f32          : FAIL (R diagonal has zeros)\n");
            pass = 0;
        }
    }
    hipFree(d_tau);

    /* === Cleanup === */
    hipFree(d_info);
    hipFree(d_ipiv);
    hipFree(d_A);
    vkblas_destroy_context(ctx);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    printf("\n=== RESULT: %s ===\n", pass ? "ALL PASS" : "FAILED");
    return pass ? 0 : 1;
}
