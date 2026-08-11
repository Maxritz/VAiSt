/**
 * \file vkblas_lapack.c
 * \brief LAPACK bridge via VJITC to rocsolver.
 *
 * Implements LU decomposition, matrix inverse, determinant, and QR
 * decomposition by bridging Vulkan device memory to HIP device pointers
 * via the zero-copy import path (VK_EXT_external_memory_host), then
 * calling rocsolver functions.
 */
#define __HIP_PLATFORM_AMD__ 1

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <string.h>

static rocblas_handle get_rocsolver_handle(void) {
    static rocblas_handle handle = NULL;
    if (!handle) {
        rocblas_create_handle(&handle);
    }
    return handle;
}

/* === LU Decomposition: A = P * L * U === */
VkResult vkblas_lu_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                       uint32_t n, void* A, uint32_t lda,
                       void* ipiv, void* info) {
    if (!ctx || !A) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    rocblas_handle h = get_rocsolver_handle();
    if (!h) return VK_ERROR_INITIALIZATION_FAILED;

    rocblas_status st = rocsolver_dgetrf(h, (int)n, (int)n,
                                         (double*)A, (int)lda,
                                         (rocblas_int*)ipiv,
                                         (rocblas_int*)info);
    hipDeviceSynchronize();

    return (st == rocblas_status_success) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

/* === Matrix Inverse: A^{-1} (via LU) === */
VkResult vkblas_inverse_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                            uint32_t n, void* A, uint32_t lda,
                            void* ipiv, void* info) {
    if (!ctx || !A) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    rocblas_handle h = get_rocsolver_handle();
    if (!h) return VK_ERROR_INITIALIZATION_FAILED;

    /* First do LU, then invert */
    rocblas_status st = rocsolver_dgetrf(h, (int)n, (int)n,
                                         (double*)A, (int)lda,
                                         (rocblas_int*)ipiv,
                                         (rocblas_int*)info);
    if (st != rocblas_status_success) {
        hipDeviceSynchronize();
        return VK_ERROR_UNKNOWN;
    }

    st = rocsolver_dgetri(h, (int)n,
                          (double*)A, (int)lda,
                          (rocblas_int*)ipiv,
                          (rocblas_int*)info);
    hipDeviceSynchronize();

    return (st == rocblas_status_success) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

/* === Determinant: from LU decomposition === */
VkResult vkblas_determinant_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                                uint32_t n, void* A, uint32_t lda,
                                void* ipiv, void* info, double* det) {
    if (!ctx || !A || !det) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    rocblas_handle h = get_rocsolver_handle();
    if (!h) return VK_ERROR_INITIALIZATION_FAILED;

    rocblas_status st = rocsolver_dgetrf(h, (int)n, (int)n,
                                         (double*)A, (int)lda,
                                         (rocblas_int*)ipiv,
                                         (rocblas_int*)info);
    if (st != rocblas_status_success) {
        hipDeviceSynchronize();
        return VK_ERROR_UNKNOWN;
    }

    /* Copy LU result + ipiv back to compute determinant on CPU */
    double* host_A = (double*)malloc(n * lda * sizeof(double));
    int* host_ipiv = (int*)malloc(n * sizeof(int));
    int host_info = 0;

    hipMemcpy(host_A, A, n * lda * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(host_ipiv, ipiv, n * sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    /* Determinant = product of diagonal of U * sign of permutation */
    double result = 1.0;
    int sign = 1;
    for (uint32_t i = 0; i < n; i++) {
        result *= host_A[i * lda + i];  /* U[i,i] */
        if (host_ipiv[i] != (int)(i + 1)) {
            sign = -sign;  /* row swap changes sign */
        }
    }
    *det = result * sign;

    free(host_A);
    free(host_ipiv);
    (void)host_info;
    return VK_SUCCESS;
}

/* === QR Decomposition: A = Q * R === */
VkResult vkblas_qr_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                       uint32_t m, uint32_t n,
                       void* A, uint32_t lda,
                       void* tau, void* info) {
    if (!ctx || !A) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    rocblas_handle h = get_rocsolver_handle();
    if (!h) return VK_ERROR_INITIALIZATION_FAILED;

    rocblas_status st = rocsolver_dgeqrf(h, (int)m, (int)n,
                                         (double*)A, (int64_t)lda,
                                         (double*)tau);
    hipDeviceSynchronize();

    return (st == rocblas_status_success) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

/* === Cholesky Decomposition: A = L * L^T (SPD matrix) === */
VkResult vkblas_cholesky_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                             uint32_t n, void* A, uint32_t lda,
                             void* info) {
    if (!ctx || !A) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    rocblas_handle h = get_rocsolver_handle();
    if (!h) return VK_ERROR_INITIALIZATION_FAILED;

    rocblas_status st = rocsolver_spotrf(h, rocblas_fill_lower, (int)n,
                                         (float*)A, (int)lda,
                                         (rocblas_int*)info);
    hipDeviceSynchronize();

    return (st == rocblas_status_success) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

/* === Eigenvalue Decomposition: A = V * diag(W) * V^T (symmetric) === */
VkResult vkblas_eigendecomp_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                                uint32_t n, void* A, uint32_t lda,
                                void* W, void* info) {
    if (!ctx || !A || !W) return VK_ERROR_INITIALIZATION_FAILED;
    (void)cmd;

    rocblas_handle h = get_rocsolver_handle();
    if (!h) return VK_ERROR_INITIALIZATION_FAILED;

    size_t n2 = (size_t)n * (size_t)lda;
    void* E = NULL;
    hipMalloc(&E, n * sizeof(double));

    rocblas_status st = rocsolver_dsyev(h, rocblas_evect_original, rocblas_fill_lower,
                                        (int)n, (double*)A, (int)lda,
                                        (double*)W, (double*)E,
                                        (rocblas_int*)info);
    hipDeviceSynchronize();
    hipFree(E);

    return (st == rocblas_status_success) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}
