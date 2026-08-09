/**
 * \file vkblas.h
 * \brief Vulkan-native BLAS interface mirroring the rocBLAS / hipBLAS API.
 *
 * VKBLAS is a Vulkan compute BLAS implementation targeting AMD RDNA2 and
 * RDNA4 architectures. All memory and command objects are standard Vulkan
 * handles (VkBuffer, VkCommandBuffer, etc.). The caller owns memory and
 * synchronization; VKBLAS only records compute work.
 */
#ifndef VKBLAS_H
#define VKBLAS_H

#include <vulkan/vulkan.h>
#include <stddef.h>  /* for size_t used by gemm_ex */

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief Operation type for op(X) in GEMM: C = alpha * op(A) * op(B) + beta * C.
 *
 * Mirrors hipblasOperation_t and rocblas_operation.
 */
typedef enum {
    VKBLAS_OP_N = 0,  /**< No transpose: op(X) = X.               */
    VKBLAS_OP_T = 1,  /**< Transpose:     op(X) = X^T.            */
    VKBLAS_OP_C = 2,  /**< Conjugate transpose: op(X) = X^H.      */
} VkBLASOperation_t;

/**
 * \brief Pointer mode for alpha/beta scalars (host or device memory).
 *
 * In Vulkan there is no host/device pointer ambiguity, but we keep the
 * concept to mirror rocBLAS semantics. Use VKBLAS_POINTER_MODE_HOST when
 * the pointer points to host memory (default), and VKBLAS_POINTER_MODE_DEVICE
 * when it points to a VkBuffer or device address.
 */
typedef enum {
    VKBLAS_POINTER_MODE_HOST   = 0,
    VKBLAS_POINTER_MODE_DEVICE = 1,
} VkBLASPointerMode_t;

/**
 * \brief Computation type for mixed-precision GEMM-ex operations.
 *
 * Mirrors hipblasComputeType_t. The computation type specifies the precision
 * used for intermediate accumulation.
 */
typedef enum {
    VKBLAS_COMPUTE_32F         = 0,  /**< f32 compute */
    VKBLAS_COMPUTE_16F         = 1,  /**< f16 compute */
    VKBLAS_COMPUTE_16B         = 2,  /**< bf16 compute */
    VKBLAS_COMPUT_32F_FAST_TF32 = 3, /**< tf32 compute (truncated f32) */
} VkBLASComputeType_t;

/**
 * \brief Algorithm selection flags for GEMM.
 *
 * Mirrors hipblasGemmFlags_t. Controls tiling, atomics usage, and
 * precision-switching behavior.
 */
typedef enum {
    VKBLAS_GEMM_FLAGS_NONE                             = 0x0,
    VKBLAS_GEMM_FLAGS_USE_CU_EFFICIENCY                = 0x2,
    VKBLAS_GEMM_FLAGS_FP16_ALT_IMPL                    = 0x4,
    VKBLAS_GEMM_FLAGS_CHECK_SOLUTION_INDEX             = 0x8,
    VKBLAS_GEMM_FLAGS_FP16_ALT_IMPL_RNZ                = 0x10,
} VkBLASGemmFlags_t;

/**
 * \brief Flags controlling GEMM behavior at dispatch time.
 *
 * These are passed as a bitmask to control optional features.
 */
typedef enum {
    VKBLAS_GEMM_FLAG_NONE       = 0x0,
    VKBLAS_GEMM_FLAG_TRANS_A    = 0x1,  /**< Transpose A before use.    */
    VKBLAS_GEMM_FLAG_TRANS_B    = 0x2,  /**< Transpose B before use.    */
    VKBLAS_GEMM_FLAG_CONJ_TRANS_A = 0x4, /**< Conjugate-transpose A.     */
    VKBLAS_GEMM_FLAG_CONJ_TRANS_B = 0x8, /**< Conjugate-transpose B.     */
    VKBLAS_GEMM_FLAG_BETA_ZERO  = 0x10, /**< Beta is known to be zero; skip C read. */
} VkBLASGemmDispatchFlags_t;

/**
 * \brief Vulkan-native BLAS context.
 *
 * Wraps the VkDevice, a pipeline cache (lazily populated), a descriptor
 * set pool, and the detected GPU architecture. The caller creates this
 * once per VkDevice and reuses it for all GEMM calls. It is thread-unsafe;
 * callers must serialize concurrent access.
 *
 * Internal layout is hidden; treat as opaque.
 */
typedef struct VkBLASContext VkBLASContext;

/**
 * \brief Describes a buffer location for batched GEMM.
 */
typedef struct {
    VkBuffer buffer;    /**< Buffer containing the base element.    */
    VkDeviceSize offset; /**< Byte offset into buffer.               */
} VkBLASBufferLoc;

/* ===========================================================================
 * Context lifecycle
 * ========================================================================== */

/**
 * \brief Create a VkBLASContext bound to a VkDevice.
 *
 * The context lazily creates Vulkan compute pipelines on first GEMM call
 * and caches them for subsequent dispatches. No device memory is allocated
 * by this function; the caller retains full memory ownership.
 *
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the context will bind to.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Device queries or shader load failed.
 */
VkResult vkblas_create_context(VkPhysicalDevice physicalDevice,
                              VkDevice device,
                              VkBLASContext** pContext);

/**
 * \brief Destroy a VkBLASContext and release all cached pipelines / descriptors.
 *
 * Pipelines and descriptor sets owned by the context are destroyed.
 * The caller is responsible for freeing any VkBuffer memory that was
 * used through this context.
 *
 * \param context Pointer to the context to destroy. May be NULL.
 */
void vkblas_destroy_context(VkBLASContext* context);

/**
 * \brief Query the GPU architecture index the context is bound to.
 *
 * 0 = baseline (vendor-agnostic), 1 = RDNA2 (GFX10), 2 = RDNA4 (GFX12).
 * Used for shader variant selection and debugging.
 *
 * \param context Valid context.
 * \return Architecture index.
 */
uint32_t vkblas_get_arch_index(VkBLASContext* context);

/**
 * \brief Get the human-readable GPU architecture string.
 *
 * \param context Valid context.
 * \return e.g. "rdna2", "rdna4", or "baseline".
 */
const char* vkblas_get_arch_name(VkBLASContext* context);

/**
 * \brief Query GPU capabilities and select the optimal shader tier.
 *
 * Called automatically by vkblas_create_context(). Can be called again
 * after context creation to re-detect on a different physical device,
 * e.g. after vkDeviceWaitIdle.
 *
 * \param context Valid context.
 * \param physicalDevice Physical device to query.
 * \retval VK_SUCCESS
 */
VkResult vkblas_init_capabilities(VkBLASContext* context,
                                  VkPhysicalDevice physicalDevice);

/* ===========================================================================
 * GEMM — single-precision (f32)
 * ========================================================================== */

/**
 * \brief f32 GEMM:  D = alpha * op(A) * op(B) + beta * C
 *
 * Mirrors hipblasSgemm parameter order exactly.
 * Alpha and beta are host pointers (pass by value via VKBLAS_POINTER_MODE_HOST).
 *
 * \param context  Valid VkBLASContext.
 * \param cmd      Command buffer to record into (must be in recording state).
 * \param transA   VKBLAS_OP_N / VKBLAS_OP_T / VKBLAS_OP_C.
 * \param transB   VKBLAS_OP_N / VKBLAS_OP_T / VKBLAS_OP_C.
 * \param m        Rows of op(A) and D.
 * \param n        Cols of op(B) and D.
 * \param k        Cols of op(A), rows of op(B).
 * \param alpha    Host pointer to scalar alpha.
 * \param A        VkBuffer holding matrix A (column-major).
 * \param lda      Leading dimension of A (>= max(1, transA==N ? m : k)).
 * \param B        VkBuffer holding matrix B (column-major).
 * \param ldb      Leading dimension of B (>= max(1, transB==N ? k : n)).
 * \param beta     Host pointer to scalar beta.
 * \param C        VkBuffer holding matrix C (read). May be VK_NULL_HANDLE
 *                 when beta == 0.
 * \param ldc      Leading dimension of C.
 * \param D        VkBuffer to store result D (write-only output).
 * \param ldd      Leading dimension of D.
 * \retval VK_SUCCESS
 */
VkResult vkblas_sgemm(VkBLASContext*  context,
                      VkCommandBuffer cmd,
                      VkBLASOperation_t transA,
                      VkBLASOperation_t transB,
                      int32_t           m,
                      int32_t           n,
                      int32_t           k,
                      const float*      alpha,
                      VkBuffer          A,
                      int32_t           lda,
                      VkBuffer          B,
                      int32_t           ldb,
                      const float*      beta,
                      VkBuffer          C,
                      int32_t           ldc,
                      VkBuffer          D,
                      int32_t           ldd);

/**
 * \brief Strided-batched f32 GEMM.
 *
 * Performs `batchCount` independent GEMM operations with fixed row/stride
 * between consecutive matrices. All matrices live in the same VkBuffer.
 *
 * Mirrors hipblasSgemmStridedBatched: adds strideA, strideB, strideC, strideD.
 */
VkResult vkblas_sgemm_strided_batched(VkBLASContext*    context,
                                      VkCommandBuffer   cmd,
                                      VkBLASOperation_t   transA,
                                      VkBLASOperation_t   transB,
                                      int32_t             m,
                                      int32_t             n,
                                      int32_t             k,
                                      const float*        alpha,
                                      VkBuffer            A,
                                      int32_t             lda,
                                      int64_t             strideA,
                                      VkBuffer            B,
                                      int32_t             ldb,
                                      int64_t             strideB,
                                      const float*        beta,
                                      VkBuffer            C,
                                      int32_t             ldc,
                                      int64_t             strideC,
                                      VkBuffer            D,
                                      int32_t             ldd,
                                      int64_t             strideD,
                                      int32_t             batchCount);

/**
 * \brief Batched f32 GEMM with per-matrix buffer arrays.
 *
 * Each matrix is stored in a separate VkBuffer. The caller provides arrays
 * of buffer handles and offsets. This is more flexible than strided-batched
 * but requires 4× the descriptor bindings (or multiple sets).
 *
 * Mirrors hipblasSgemmBatched interface conceptually.
 */
VkResult vkblas_sgemm_batched(VkBLASContext*    context,
                              VkCommandBuffer   cmd,
                              VkBLASOperation_t transA,
                              VkBLASOperation_t transB,
                              int32_t           m,
                              int32_t           n,
                              int32_t           k,
                              const float*      alpha,
                              const VkBLASBufferLoc* A,
                              int32_t           lda,
                              const VkBLASBufferLoc* B,
                              int32_t           ldb,
                              const float*      beta,
                              const VkBLASBufferLoc* C,
                              int32_t           ldc,
                              VkBLASBufferLoc*  D,
                              int32_t           ldd,
                              int32_t           batchCount);

/* ===========================================================================
 * GEMM — double-precision (f64)
 * ========================================================================== */

VkResult vkblas_dgemm(VkBLASContext*    context,
                      VkCommandBuffer   cmd,
                      VkBLASOperation_t transA,
                      VkBLASOperation_t transB,
                      int32_t           m,
                      int32_t           n,
                      int32_t           k,
                      const double*     alpha,
                      VkBuffer          A,
                      int32_t           lda,
                      VkBuffer          B,
                      int32_t           ldb,
                      const double*     beta,
                      VkBuffer          C,
                      int32_t           ldc,
                      VkBuffer          D,
                      int32_t           ldd);

VkResult vkblas_dgemm_strided_batched(VkBLASContext*    context,
                                      VkCommandBuffer   cmd,
                                      VkBLASOperation_t transA,
                                      VkBLASOperation_t transB,
                                      int32_t           m,
                                      int32_t           n,
                                      int32_t           k,
                                      const double*     alpha,
                                      VkBuffer          A,
                                      int32_t           lda,
                                      int64_t           strideA,
                                      VkBuffer          B,
                                      int32_t           ldb,
                                      int64_t           strideB,
                                      const double*     beta,
                                      VkBuffer          C,
                                      int32_t           ldc,
                                      int64_t           strideC,
                                      VkBuffer          D,
                                      int32_t           ldd,
                                      int64_t           strideD,
                                      int32_t           batchCount);

/* ===========================================================================
 * GEMM — half-precision (f16)
 * ========================================================================== */

VkResult vkblas_hgemm(VkBLASContext*    context,
                      VkCommandBuffer   cmd,
                      VkBLASOperation_t transA,
                      VkBLASOperation_t transB,
                      int32_t           m,
                      int32_t           n,
                      int32_t           k,
                      const uint16_t*   alpha,
                      VkBuffer          A,
                      int32_t           lda,
                      VkBuffer          B,
                      int32_t           ldb,
                      const uint16_t*   beta,
                      VkBuffer          C,
                      int32_t           ldc,
                      VkBuffer          D,
                      int32_t           ldd);

VkResult vkblas_hgemm_strided_batched(VkBLASContext*    context,
                                      VkCommandBuffer   cmd,
                                      VkBLASOperation_t transA,
                                      VkBLASOperation_t transB,
                                      int32_t           m,
                                      int32_t           n,
                                      int32_t           k,
                                      const uint16_t*   alpha,
                                      VkBuffer          A,
                                      int32_t           lda,
                                      int64_t           strideA,
                                      VkBuffer          B,
                                      int32_t           ldb,
                                      int64_t           strideB,
                                      const uint16_t*   beta,
                                      VkBuffer          C,
                                      int32_t           ldc,
                                      int64_t           strideC,
                                      VkBuffer          D,
                                      int32_t           ldd,
                                      int64_t           strideD,
                                      int32_t           batchCount);

/* ===========================================================================
 * GEMM — bfloat16 (bf16)
 * ========================================================================== */

VkResult vkblas_bgemm(VkBLASContext*    context,
                      VkCommandBuffer   cmd,
                      VkBLASOperation_t transA,
                      VkBLASOperation_t transB,
                      int32_t           m,
                      int32_t           n,
                      int32_t           k,
                      const uint16_t*   alpha,
                      VkBuffer          A,
                      int32_t           lda,
                      VkBuffer          B,
                      int32_t           ldb,
                      const uint16_t*   beta,
                      VkBuffer          C,
                      int32_t           ldc,
                      VkBuffer          D,
                      int32_t           ldd);

VkResult vkblas_bgemm_strided_batched(VkBLASContext*    context,
                                      VkCommandBuffer   cmd,
                                      VkBLASOperation_t transA,
                                      VkBLASOperation_t transB,
                                      int32_t           m,
                                      int32_t           n,
                                      int32_t           k,
                                      const uint16_t*   alpha,
                                      VkBuffer          A,
                                      int32_t           lda,
                                      int64_t           strideA,
                                      VkBuffer          B,
                                      int32_t           ldb,
                                      int64_t           strideB,
                                      const uint16_t*   beta,
                                      VkBuffer          C,
                                      int32_t           ldc,
                                      int64_t           strideC,
                                      VkBuffer          D,
                                      int32_t           ldd,
                                      int64_t           strideD,
                                      int32_t           batchCount);

/* ===========================================================================
 * GEMM-ex — mixed-precision with explicit compute type
 * ========================================================================== */

/**
 * \brief Mixed-precision GEMM with explicit compute type.
 *
 * Allows e.g. f16 input with f32 accumulation, or f16 input with f16 compute.
 * The data-type suffix reflects the *storage* type of A, B, C, D; the
 * computeType argument controls the accumulation precision.
 *
 * Mirrors hipblasGemmEx.
 */
VkResult vkblas_gemm_ex(VkBLASContext*       context,
                        VkCommandBuffer      cmd,
                        VkBLASOperation_t    transA,
                        VkBLASOperation_t    transB,
                        int32_t              m,
                        int32_t              n,
                        int32_t              k,
                        const void*          alpha,
                        VkBuffer             A,
                        int32_t              lda,
                        size_t               strideA_element,  /* bytes between elements if packed */
                        VkBuffer             B,
                        int32_t              ldb,
                        size_t               strideB_element,
                        const void*          beta,
                        VkBuffer             C,
                        int32_t              ldc,
                        size_t               strideC_element,
                        VkBuffer             D,
                        int32_t              ldd,
                        size_t               strideD_element,
                        VkBLASComputeType_t  computeType,
                        VkBLASGemmFlags_t    flags);

/* ===========================================================================
 * GEMM-ex strided-batched
 * ========================================================================== */

VkResult vkblas_gemm_ex_strided_batched(VkBLASContext*       context,
                                        VkCommandBuffer      cmd,
                                        VkBLASOperation_t    transA,
                                        VkBLASOperation_t    transB,
                                        int32_t              m,
                                        int32_t              n,
                                        int32_t              k,
                                        const void*          alpha,
                                        VkBuffer             A,
                                        int32_t              lda,
                                        int64_t              strideA,
                                        size_t               strideA_element,
                                        VkBuffer             B,
                                        int32_t              ldb,
                                        int64_t              strideB,
                                        size_t               strideB_element,
                                        const void*          beta,
                                        VkBuffer             C,
                                        int32_t              ldc,
                                        int64_t              strideC,
                                        size_t               strideC_element,
                                        VkBuffer             D,
                                        int32_t              ldd,
                                        int64_t              strideD,
                                        size_t               strideD_element,
                                        int32_t              batchCount,
                                        VkBLASComputeType_t  computeType,
                                        VkBLASGemmFlags_t    flags);

/* ===========================================================================
 * GEMM — fused quantized (dequant-in-matmul): Q8_0 / Q4_K / Q4_0 / Q5_K /
 * Q6_K / Q3_K / IQ4_XS weights
 * ========================================================================== */

/**
 * \brief Fused Q8_0 quantized GEMM: y = alpha * (dequant(Wq) * x) + beta * y.
 *
 * The weight matrix is dequantized *inside* the matmul kernel: quantized
 * blocks are read from Wq, decoded to f32 in shared memory, and accumulated
 * against the f32 activation x. No separate dequant pass is required.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 32 elements.
 * A Q8_0 block is 36 bytes: an f32 scale `d` (bytes 0..3) followed by 32
 * int8 values `qs` (bytes 4..35);  dequant(i) = d * qs[i].
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/32) blocks. ldw is the byte stride between rows
 * (>= ceil(k/32) * 36, i.e. a multiple of 4).
 *
 * \par Activation / output layout
 * x is (k x m) f32 column-major (x[col*ldx + row]); y is (n x m) f32
 * column-major (y[col*ldy + row]) and is read for the beta term and written
 * in place.
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the Q8_0-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_q8_0_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                               int32_t m, int32_t n, int32_t k,
                               const float* alpha, VkBuffer Wq, int32_t ldw,
                               VkBuffer x, int32_t ldx,
                               const float* beta, VkBuffer y, int32_t ldy);

/**
 * \brief Fused Q4_K quantized GEMM: y = alpha * (dequant(Wq) * x) + beta * y.
 *
 * Weight matrix dequantized *inside* the matmul kernel using the canonical
 * ggml Q4_K block format (see below). Same dispatch semantics as
 * vkblas_qgemm_q8_0_f32.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 256 elements.
 * A Q4_K block is 144 bytes in ggml block_q4_K order:
 *   bytes   0..1   f16 d
 *   bytes   2..3   f16 dmin
 *   bytes   4..15  uint8 scales[12]
 *   bytes  16..143 uint8 qs[128]  (packed 4-bit nibbles)
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/256) blocks. ldw is the byte stride between rows
 * (>= ceil(k/256) * 144, a multiple of 4).
 *
 * Per-32-element-group dequant (canonical ggml get_scale_min_k4), where
 * `is` is the group index (0..7) and `nib` the element nibble (0..15):
 *   is<4:  sc = scales[is]&63,        mn = scales[is+4]&63
 *   else:  sc = (scales[is+4]&0xF)|((scales[is-4]>>6)<<4)
 *          mn = (scales[is+4]>>4)   |((scales[is]>>6)<<4)
 *   out = d*sc*nib - dmin*mn
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the Q4_K-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_q4k_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y, int32_t ldy);

/**
 * \brief Fused Q4_0 quantized GEMM: y = alpha * (dequant(Wq) * x) + beta * y.
 *
 * Weight matrix dequantized *inside* the matmul kernel using the VAIT Q4_0
 * block format. Same dispatch semantics as vkblas_qgemm_q8_0_f32.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 32 elements.
 * A Q4_0 block is 20 bytes: an f32 scale `d` (bytes 0..3) followed by 16
 * uint8 packed nibbles (bytes 4..19). For element i in [0,32):
 *   nib = qs[i>>1] >> (4*(i&1)) & 0xF;  dequant(i) = d * (nib - 8).
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/32) blocks. ldw is the byte stride between rows
 * (>= ceil(k/32) * 20, a multiple of 4).
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the Q4_0-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_q4_0_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                               int32_t m, int32_t n, int32_t k,
                               const float* alpha, VkBuffer Wq, int32_t ldw,
                               VkBuffer x, int32_t ldx,
                               const float* beta, VkBuffer y, int32_t ldy);

/**
 * \brief Fused Q5_K quantized GEMM: y = alpha * (dequant(Wq) * x) + beta * y.
 *
 * Weight matrix dequantized *inside* the matmul kernel using the canonical
 * ggml Q5_K block format. Same dispatch semantics as vkblas_qgemm_q8_0_f32.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 256 elements.
 * A Q5_K block is 176 bytes in ggml block_q5_K order:
 *   bytes   0..1   f16 d
 *   bytes   2..3   f16 dmin
 *   bytes   4..15  uint8 scales[12]
 *   bytes  16..47  uint8 qh[32]   (5th bits)
 *   bytes  48..175 uint8 qs[128]  (packed 4-bit nibbles)
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/256) blocks. ldw is the byte stride between rows
 * (>= ceil(k/256) * 176, a multiple of 4).
 *
 * Per-32-element-group dequant (canonical ggml dequantize_row_q5_K):
 *   super = i>>6, hi = (i>>5)&1, l = i&31, is = super*2 + hi
 *   sc/mn = get_scale_min_k4(is)  (6-bit scale/min pair, same packing as Q4_K)
 *   nib = low/high nibble of qs[32*super + l]
 *   level = nib + ((qh[l] >> (2*super + hi)) & 1 ? 16 : 0)
 *   out = d*sc*level - dmin*mn
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the Q5_K-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_q5k_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y, int32_t ldy);

/**
 * \brief Fused Q6_K quantized GEMM: y = alpha * (dequant(Wq) * x) + beta * y.
 *
 * Weight matrix dequantized *inside* the matmul kernel using the canonical
 * ggml Q6_K block format. Same dispatch semantics as vkblas_qgemm_q8_0_f32.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 256 elements.
 * A Q6_K block is 210 bytes in ggml block_q6_K order:
 *   bytes   0..127   uint8 ql[128]   (low 4 bits)
 *   bytes 128..191   uint8 qh[64]    (high 2 bits)
 *   bytes 192..207   int8 scales[16]
 *   bytes 208..209   f16 d
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/256) blocks. ldw is the byte stride between rows
 * (>= ceil(k/256) * 210, a multiple of 4).
 *
 * Dequant (canonical ggml dequantize_row_q6_K):
 *   chunk = i>>7, sub = (i>>5)&3, l = i&31, is = l>>4
 *   ql4 = (sub<2 ? ql&0xF : ql>>4); qh2 = (qh >> (sub*2)) & 3
 *   level = (ql4 | (qh2 << 4)) - 32
 *   out = d * scales[chunk*8 + is + sub*2] * level
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the Q6_K-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_q6k_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y, int32_t ldy);

/**
 * \brief Fused Q3_K quantized GEMM: y = alpha * (dequant(Wq) * x) + beta * y.
 *
 * Weight matrix dequantized *inside* the matmul kernel using the canonical
 * ggml Q3_K block format. Same dispatch semantics as vkblas_qgemm_q8_0_f32.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 256 elements.
 * A Q3_K block is 110 bytes in ggml block_q3_K order:
 *   bytes   0..31    uint8 hmask[32]  (high/sign bits)
 *   bytes  32..95    uint8 qs[64]     (2-bit levels, 4 per byte)
 *   bytes  96..107   uint8 scales[12] (16 x 6-bit packed int8 scales)
 *   bytes 108..109   f16 d
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/256) blocks. ldw is the byte stride between rows
 * (>= ceil(k/256) * 110, a multiple of 4).
 *
 * Dequant (canonical ggml dequantize_row_q3_K):
 *   half = i>>7, j = (i&127)>>5, hi = (i>>4)&1, ll = i&15
 *   q2 = (qs[half*32 + ll + hi*16] >> 2*j) & 3
 *   level = q2 - ((hmask[ll + hi*16] >> (j + half*4)) & 1 ? 0 : 4)
 *   the 16 x int8 scales are recovered from the 12 packed bytes; see
 *   shaders/vkblas/baseline/qgemm_q3k.comp for the bit math.
 *   out = d * (sc - 32) * level
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the Q3_K-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_q3k_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y, int32_t ldy);

/**
 * \brief Fused IQ4_XS quantized GEMM: y = alpha * (dequant(Wq) * x) + beta*y.
 *
 * Weight matrix dequantized *inside* the matmul kernel using the canonical
 * ggml IQ4_XS block format (non-linear 4-bit lookup). Same dispatch
 * semantics as vkblas_qgemm_q8_0_f32.
 *
 * \par Weight layout (Wq)
 * W is n rows x k columns, stored row-major in blocks of 256 elements.
 * A IQ4_XS block is 136 bytes in ggml block_iq4_xs order:
 *   bytes   0..1    f16 d
 *   bytes   2..3    uint16 scales_h
 *   bytes   4..7    uint8 scales_l[4]
 *   bytes   8..135  uint8 qs[128]  (packed nibbles)
 * Row r occupies byte offset r * ldw; its blocks are contiguous, so a row
 * needs ceil(k/256) blocks. ldw is the byte stride between rows
 * (>= ceil(k/256) * 136, a multiple of 4).
 *
 * Dequant (canonical ggml dequantize_row_iq4_xs):
 *   ib = i>>5, j = i&15, hi = (i>>4)&1
 *   ls = (scales_l[ib>>1] >> 4*(ib&1) & 0xF) | ((scales_h >> 2*ib) & 3) << 4
 *   out = d * (ls - 32) * iq4nl[low/high nibble of qs[ib*16 + j]]
 *   iq4nl = kvalues_iq4nl (ggml-common.h).
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y (activation/batch dimension).
 * \param n     Rows of W and y (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha.
 * \param Wq    VkBuffer holding the IQ4_XS-quantized weight matrix.
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta.
 * \param y     VkBuffer holding output y (n x m f32); read + written.
 * \param ldy   Leading dimension of y (>= n).
 * \retval VK_SUCCESS On success.
 */
VkResult vkblas_qgemm_iq4xs_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                                int32_t m, int32_t n, int32_t k,
                                const float* alpha, VkBuffer Wq, int32_t ldw,
                                VkBuffer x, int32_t ldx,
                                const float* beta, VkBuffer y, int32_t ldy);

/* ===========================================================================
 * Pointer mode control
 * ========================================================================== */

/**
 * \brief Set whether alpha/beta are interpreted as host or device pointers.
 *
 * When VKBLAS_POINTER_MODE_DEVICE is set, the alpha/beta pointers are treated
 * as VkDeviceAddress values (the caller must have created buffers with
 * VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT).
 */
VkResult vkblas_set_pointer_mode(VkBLASContext*          context,
                                 VkBLASPointerMode_t     mode);

VkBLASPointerMode_t vkblas_get_pointer_mode(VkBLASContext* context);

/* ===========================================================================
 * Utility
 * ========================================================================== */

/**
 * \brief Flush cached pipelines associated with the context.
 *
 * Call before vkDeviceWaitIdle if the device will be reset.
 */
void vkblas_flush_pipelines(VkBLASContext* context);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKBLAS_H */
