/**
 * \file vkblas_sparse.c
 * \brief Sparse GEMM via VJITC bridge to rocSPARSE.
 *
 * Bridges Vulkan device memory to HIP device pointers via the zero-copy
 * import path (VK_EXT_external_memory_host), then calls hipSPARSE spmm.
 *
 * All buffer arguments are HIP device pointers that must also be bound to
 * VkBuffers via VK_EXT_external_memory_host (see test_hip_bridge_libraries).
 * The caller is responsible for the memory bridge — this function only
 * performs the ROCm library call.
 */
#define __HIP_PLATFORM_AMD__ 1

#include "vkblas/vkblas.h"
#include "vkblas_internal.h"
#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
#include <string.h>

VkResult vkblas_sparse_gemm_f32(
    VkBLASContext* ctx,
    VkBLASOperation_t op_A,
    VkBLASOperation_t op_B,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t nnz,
    const float* alpha,
    const uint32_t* csr_row_ptr,   /* (rows_A + 1) */
    const uint32_t* csr_col_ind,   /* nnz */
    const float* csr_val,          /* nnz */
    const void* B,                 /* device pointer */
    const float* beta,
    void* C,                       /* device pointer */
    VkCommandBuffer cmd)
{
    if (!ctx || !csr_val || !csr_row_ptr || !csr_col_ind || !B || !C || !alpha || !beta) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    (void)cmd;  /* Bridge calls happen on the HIP stream, not via VkCommandBuffer */

    hipsparseHandle_t sp_handle;
    hipsparseStatus_t st = hipsparseCreate(&sp_handle);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    /* Map VkBLASOperation to hipsparseOperation_t */
    hipsparseOperation_t sp_op_A = (op_A == VKBLAS_OP_T) ? HIPSPARSE_OPERATION_TRANSPOSE :
                                  (op_A == VKBLAS_OP_C) ? HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t sp_op_B = (op_B == VKBLAS_OP_T) ? HIPSPARSE_OPERATION_TRANSPOSE :
                                  (op_B == VKBLAS_OP_C) ? HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;

    /* Rows/cols for A (CSR): if op_A transposed, dimensions swap */
    int64_t A_rows = (op_A == VKBLAS_OP_N) ? m : k;
    int64_t A_cols = (op_A == VKBLAS_OP_N) ? k : m;

    /* Dense matrix B dimensions: B is op_B(K, N) where K=inner dim */
    int64_t B_rows = (op_B != VKBLAS_OP_N) ? n : k;
    int64_t B_cols = (op_B != VKBLAS_OP_N) ? k : n;

    /* Dense matrix C dimensions */
    int64_t C_rows = m;
    int64_t C_cols = n;

    /* === Create sparse CSR descriptor === */
    hipsparseSpMatDescr_t A_descr;
    st = hipsparseCreateCsr(&A_descr,
        A_rows, A_cols, nnz,
        (void*)csr_row_ptr,
        (void*)csr_col_ind,
        (void*)csr_val,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO,
        HIP_R_32F);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Create dense descriptors for B and C === */
    hipsparseDnMatDescr_t B_descr;
    st = hipsparseCreateDnMat(&B_descr,
        B_rows, B_cols, B_rows,
        (void*)B,
        HIP_R_32F,
        HIPSPARSE_ORDER_COL);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    hipsparseDnMatDescr_t C_descr;
    st = hipsparseCreateDnMat(&C_descr,
        C_rows, C_cols, C_rows,
        C,
        HIP_R_32F,
        HIPSPARSE_ORDER_COL);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroyDnMat(B_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Query buffer size === */
    size_t buffer_size = 0;
    st = hipsparseSpMM_bufferSize(sp_handle,
        sp_op_A, sp_op_B,
        alpha, A_descr, B_descr, beta, C_descr,
        HIP_R_32F,
        HIPSPARSE_SPMM_ALG_DEFAULT,
        &buffer_size);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroyDnMat(C_descr);
        hipsparseDestroyDnMat(B_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Allocate buffer and preprocess === */
    void* buffer = NULL;
    if (buffer_size > 0) {
        hipMalloc(&buffer, buffer_size);
    }

    st = hipsparseSpMM_preprocess(sp_handle,
        sp_op_A, sp_op_B,
        alpha, A_descr, B_descr, beta, C_descr,
        HIP_R_32F,
        HIPSPARSE_SPMM_ALG_DEFAULT,
        buffer);

    if (st != HIPSPARSE_STATUS_SUCCESS) {
        if (buffer) hipFree(buffer);
        hipsparseDestroyDnMat(C_descr);
        hipsparseDestroyDnMat(B_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Execute sparse matmul === */
    st = hipsparseSpMM(sp_handle,
        sp_op_A, sp_op_B,
        alpha, A_descr, B_descr, beta, C_descr,
        HIP_R_32F,
        HIPSPARSE_SPMM_ALG_DEFAULT,
        buffer);

    /* === Cleanup === */
    if (buffer) hipFree(buffer);
    hipsparseDestroyDnMat(C_descr);
    hipsparseDestroyDnMat(B_descr);
    hipsparseDestroySpMat(A_descr);
    hipsparseDestroy(sp_handle);

    if (st != HIPSPARSE_STATUS_SUCCESS) {
        return VK_ERROR_UNKNOWN;
    }

    hipDeviceSynchronize();
    return VK_SUCCESS;
}

VkResult vkblas_sparse_spsv_f32(
    VkBLASContext* ctx,
    VkBLASOperation_t op_A,
    uint32_t m,
    uint32_t nnz,
    const float* alpha,
    const uint32_t* csr_row_ptr,
    const uint32_t* csr_col_ind,
    const float* csr_val,
    const void* x,
    void* y,
    VkCommandBuffer cmd)
{
    if (!ctx || !csr_val || !csr_row_ptr || !csr_col_ind || !x || !y || !alpha || m == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    (void)cmd;  /* Bridge calls happen on the HIP stream, not via VkCommandBuffer */

    hipsparseHandle_t sp_handle;
    hipsparseStatus_t st = hipsparseCreate(&sp_handle);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    hipsparseOperation_t sp_op_A = (op_A == VKBLAS_OP_T) ? HIPSPARSE_OPERATION_TRANSPOSE
                                                         : HIPSPARSE_OPERATION_NON_TRANSPOSE;

    /* === Create sparse CSR descriptor === */
    hipsparseSpMatDescr_t A_descr;
    st = hipsparseCreateCsr(&A_descr,
        (int64_t)m, (int64_t)m, (int64_t)nnz,
        (void*)csr_row_ptr,
        (void*)csr_col_ind,
        (void*)csr_val,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO,
        HIP_R_32F);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Create dense vector descriptors for x and y === */
    hipsparseDnVecDescr_t x_descr;
    st = hipsparseCreateDnVec(&x_descr, (int64_t)m, (void*)x, HIP_R_32F);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    hipsparseDnVecDescr_t y_descr;
    st = hipsparseCreateDnVec(&y_descr, (int64_t)m, y, HIP_R_32F);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroyDnVec(x_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Create SpSV descriptor === */
    hipsparseSpSVDescr_t spsv_descr;
    st = hipsparseSpSV_createDescr(&spsv_descr);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseDestroyDnVec(y_descr);
        hipsparseDestroyDnVec(x_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Query buffer size === */
    size_t buffer_size = 0;
    st = hipsparseSpSV_bufferSize(sp_handle, sp_op_A, alpha,
        A_descr, x_descr, y_descr, HIP_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsv_descr, &buffer_size);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        hipsparseSpSV_destroyDescr(spsv_descr);
        hipsparseDestroyDnVec(y_descr);
        hipsparseDestroyDnVec(x_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    void* buffer = NULL;
    if (buffer_size > 0) {
        hipMalloc(&buffer, buffer_size);
    }

    /* === Analysis === */
    st = hipsparseSpSV_analysis(sp_handle, sp_op_A, alpha,
        A_descr, x_descr, y_descr, HIP_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsv_descr, buffer);
    if (st != HIPSPARSE_STATUS_SUCCESS) {
        if (buffer) hipFree(buffer);
        hipsparseSpSV_destroyDescr(spsv_descr);
        hipsparseDestroyDnVec(y_descr);
        hipsparseDestroyDnVec(x_descr);
        hipsparseDestroySpMat(A_descr);
        hipsparseDestroy(sp_handle);
        return VK_ERROR_UNKNOWN;
    }

    /* === Solve op(A) * y = alpha * x === */
    st = hipsparseSpSV_solve(sp_handle, sp_op_A, alpha,
        A_descr, x_descr, y_descr, HIP_R_32F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsv_descr);

    /* === Cleanup === */
    if (buffer) hipFree(buffer);
    hipsparseSpSV_destroyDescr(spsv_descr);
    hipsparseDestroyDnVec(y_descr);
    hipsparseDestroyDnVec(x_descr);
    hipsparseDestroySpMat(A_descr);
    hipsparseDestroy(sp_handle);

    if (st != HIPSPARSE_STATUS_SUCCESS) {
        return VK_ERROR_UNKNOWN;
    }

    hipDeviceSynchronize();
    return VK_SUCCESS;
}
