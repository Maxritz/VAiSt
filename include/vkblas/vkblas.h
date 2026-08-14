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
 * \brief Triangle selector for triangular/symmetric storage formats.
 *
 * Used by trsv, trsm, symv, hemv, symm, hemm, syrk, herk.
 */
typedef enum {
    VKBLAS_FILL_LOWER = 0,  /**< Only the lower triangle is authoritative. */
    VKBLAS_FILL_UPPER = 1,  /**< Only the upper triangle is authoritative. */
} VkBLASFillMode_t;

/**
 * \brief Diagonal storage mode for triangular-matrix formats.
 *
 * Only used by trsv and trsm.
 */
typedef enum {
    VKBLAS_DIAG_NON_UNIT = 0,  /**< Diagonal entries are read from A. */
    VKBLAS_DIAG_UNIT     = 1,  /**< Diagonal entries implied 1 (not read). */
} VkBLASDiagType_t;

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
    VKBLAS_COMPUTE_32F_FAST_TF32 = 3, /**< tf32 compute (truncated f32) */
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
 * \param instance        Vulkan instance (used for instance-level extension
 *                        queries such as cooperative-matrix enumeration).
 * \param physicalDevice Physical device handle (for capability queries).
 * \param device          Logical device the context will bind to.
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host allocation failed.
 * \retval VK_ERROR_INITIALIZATION_FAILED Device queries or shader load failed.
 */
VkResult vkblas_create_context(VkInstance instance,
                               VkPhysicalDevice physicalDevice,
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

/**
 * \brief Fused quantized-GEMM weight formats (argument to
 *        vkblas_qgemm_get_tier).
 *
 * The numeric values match the internal VKBLAS dtype codes so a format value
 * can be passed through to the dispatcher unchanged.
 */
typedef enum VkBLASQGemmFormat_t {
    VKBLAS_QGEMM_Q8_0  = 5,  /**< Q8_0 weights (36 B/block of 32) */
    VKBLAS_QGEMM_Q4K   = 6,  /**< Q4_K weights (ggml, 144 B/block of 256) */
    VKBLAS_QGEMM_Q4_0  = 7,  /**< Q4_0 weights (20 B/block of 32) */
    VKBLAS_QGEMM_Q5K   = 8,  /**< Q5_K weights (ggml, 176 B/block of 256) */
    VKBLAS_QGEMM_Q6K   = 9,  /**< Q6_K weights (ggml, 210 B/block of 256) */
    VKBLAS_QGEMM_Q3K   = 10, /**< Q3_K weights (ggml, 110 B/block of 256) */
    VKBLAS_QGEMM_IQ4XS = 11, /**< IQ4_XS weights (ggml, 136 B/block of 256) */
} VkBLASQGemmFormat_t;

/**
 * \brief Report the execution tier a fused qgemm kernel resolves to.
 *
 * Execution tiers: 0 = baseline (portable shared-memory), 1 = subgroup
 * (one subgroup per output tile, subgroup-shuffle broadcasts), 2 = coopmatrix
 * (dormant by default, see context capabilities).
 *
 * The reported value reflects the pipeline actually in use for the format:
 * on a subgroup-capable device, qgemm Q8_0 resolves to the subgroup tier,
 * while the remaining formats (Q4_K/Q4_0/Q5_K/Q6_K/Q3_K/IQ4_XS) still resolve
 * to the baseline tier. If a higher-tier pipeline failed to create on a given
 * driver, the reported tier is the fallback actually dispatched, not the
 * theoretical active tier.
 *
 * \param ctx      Valid VkBLASContext.
 * \param format   Quantized weight format to query.
 * \param out_tier Receives the resolved tier index (0/1/2).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT If format is not a valid qgemm format.
 */
VkResult vkblas_qgemm_get_tier(VkBLASContext* ctx, VkBLASQGemmFormat_t format,
                               uint32_t* out_tier);

/* ═════════════════════════════════════════════════════════════════════════
 * Fused quantized GEMM — FP16 output storage
 * ═════════════════════════════════════════════════════════════════════════
 *
 * The vkblas_qgemm_*_f16 functions compute the same fused quantized GEMM as
 * their *_f32 counterparts but store the output y as IEEE-754 half (fp16)
 * instead of f32. Everything else is identical:
 *   - Wq layout and ldw are byte-for-byte the format contract on the *_f32
 *     functions (Q8_0 36 B/block of 32, Q4_0 20 B/block of 32, Q4_K/Q5_K/
 *     Q6_K/Q3_K/IQ4_XS ggml 144/176/210/110/136 B/block of 256).
 *   - x is f32 (k x m), column-major; ldx >= k.
 *   - y16 is the output (n x m), column-major, element type uint16_t holding
 *     fp16 bits; ldy >= n. It is read for the beta term and written in place,
 *     exactly like y in the *_f32 variants.
 *   - alpha/beta are f32 host scalars; accumulation happens in f32 and the
 *     result is rounded to fp16 only at the final store (no fp16 arithmetic
 *     inside the kernel).
 *
 * \par Device requirements
 * The fp16 output path is a subgroup-tier feature: the kernels dispatch from
 * the subgroup shaders (one 32-lane subgroup per 32x8 output tile) and need
 * storageBuffer16BitAccess + scalarBlockLayout on the device/queue (the
 * shaderFloat16 feature is NOT required — fp16 is only stored, never
 * arithmetically computed). On devices that resolve qgemm to the baseline
 * tier, these functions return VK_ERROR_FEATURE_NOT_PRESENT. See
 * vkblas_qgemm_get_tier to query tier support.
 *
 * \param ctx   Valid VkBLASContext.
 * \param cmd   Command buffer to record into (recording state).
 * \param m     Columns of x and y16 (activation/batch dimension).
 * \param n     Rows of W and y16 (output dimension).
 * \param k     Columns of W, rows of x (contraction dimension).
 * \param alpha Host pointer to scalar alpha (f32).
 * \param Wq    VkBuffer holding the quantized weight matrix (format layout
 *              identical to the corresponding *_f32 function).
 * \param ldw   Byte stride between weight rows.
 * \param x     VkBuffer holding activation x (k x m f32).
 * \param ldx   Leading dimension of x (>= k).
 * \param beta  Host pointer to scalar beta (f32).
 * \param y16   VkBuffer holding output y16 (n x m fp16 bits); read + written.
 * \param ldy   Leading dimension of y16 (>= n).
 * \retval VK_SUCCESS On success.
 * \retval VK_ERROR_FEATURE_NOT_PRESENT If the device resolves qgemm to the
 *         baseline tier (no subgroup fp16-output kernel available).
 */
VkResult vkblas_qgemm_q8_0_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                               int32_t m, int32_t n, int32_t k,
                               const float* alpha, VkBuffer Wq, int32_t ldw,
                               VkBuffer x, int32_t ldx,
                               const float* beta, VkBuffer y16, int32_t ldy);
VkResult vkblas_qgemm_q4k_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y16, int32_t ldy);
VkResult vkblas_qgemm_q4_0_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                               int32_t m, int32_t n, int32_t k,
                               const float* alpha, VkBuffer Wq, int32_t ldw,
                               VkBuffer x, int32_t ldx,
                               const float* beta, VkBuffer y16, int32_t ldy);
VkResult vkblas_qgemm_q5k_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y16, int32_t ldy);
VkResult vkblas_qgemm_q6k_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y16, int32_t ldy);
VkResult vkblas_qgemm_q3k_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                              int32_t m, int32_t n, int32_t k,
                              const float* alpha, VkBuffer Wq, int32_t ldw,
                              VkBuffer x, int32_t ldx,
                              const float* beta, VkBuffer y16, int32_t ldy);
VkResult vkblas_qgemm_iq4xs_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                                int32_t m, int32_t n, int32_t k,
                                const float* alpha, VkBuffer Wq, int32_t ldw,
                                VkBuffer x, int32_t ldx,
                                const float* beta, VkBuffer y16, int32_t ldy);

/* ===========================================================================
 * Extended BLAS Level-2 / Level-3 ops
 *
 * These mirror rocBLAS / cuBLAS conventions: A, B, C, D, x, y are VkBuffer
 * handles, column-major layout, and alpha/beta are host-pointer scalars.
 * The triangular solves (trsv, trsm) OVERWRITE the RHS buffer with the
 * solution.  All f16 variants use f32 accumulation internally and narrow
 * to fp16 only at the final store. The f16 ext BLAS APIs (trsv/trsm/symv/
 * hemv/symm/hemm/syrk/herk) interpret alpha/beta pointers as uint16_t f16
 * bit patterns and promote them to f32 internally via `vkblas_f16_to_f32`.
 * ========================================================================== */

/**
 * \brief Triangular solve for vectors (BLAS-2).
 *
 * Solves  op(A) * x = alpha * b  in place: the input vector b is scaled by
 * alpha and then overwritten with the solution x. b and x use the same
 * buffer handle.
 *
 * \param ctx     VKBLAS context.
 * \param cmd     Vulkan command buffer (compute commands are recorded here).
 * \param uplo    Which triangle of A is authoritative (lower or upper).
 * \param transA  Whether op(A) = A or op(A) = A^T.
 * \param diag    Whether the diagonal of A is all-ones (not read from A).
 * \param n       Order of A (square: A is n x n). No-op if n <= 0.
 * \param alpha   Scale for the RHS vector b (host scalar).
 * \param A       VkBuffer containing the n x n triangular matrix, column-major.
 * \param lda     Leading dimension of A (>= n).
 * \param b       VkBuffer containing b; overwritten with x.
 * \param ldb     Leading dimension/stride for b (must equal n for a contiguous vector).
 */
VkResult vkblas_trsv_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t transA,
                         VkBLASDiagType_t diag,
                         int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer b, int32_t ldb);
VkResult vkblas_trsv_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t transA,
                         VkBLASDiagType_t diag,
                         int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer b, int32_t ldb);

/**
 * \brief Triangular solve for matrices (BLAS-3).
 *
 * Solves  op(A) * X = alpha * B  column-by-column: each column of B is
 * solved as an independent trsv. B is overwritten with X.
 *
 * \param m       Order of A (square n = m, since A is m x m; B is m x nrhs).
 * \param nrhs    Number of columns in B.
 */
VkResult vkblas_trsm_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t transA,
                         VkBLASDiagType_t diag,
                         int32_t m, int32_t nrhs, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer B, int32_t ldb);
VkResult vkblas_trsm_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t transA,
                         VkBLASDiagType_t diag,
                         int32_t m, int32_t nrhs, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer B, int32_t ldb);

/**
 * \brief Symmetric (real) matrix-vector product (BLAS-2): y = alpha*A*x + beta*y.
 */
VkResult vkblas_symv_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer x, int32_t incx,
                         const float* beta,
                         VkBuffer y, int32_t incy);
VkResult vkblas_symv_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer x, int32_t incx,
                         const float* beta,
                         VkBuffer y, int32_t incy);

/**
 * \brief Hermitian matrix-vector product (BLAS-2): y = alpha*A*x + beta*y.
 *
 * Identical to symv when A is real-valued (f32/f16).
 */
VkResult vkblas_hemv_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer x, int32_t incx,
                         const float* beta,
                         VkBuffer y, int32_t incy);
VkResult vkblas_hemv_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer x, int32_t incx,
                         const float* beta,
                         VkBuffer y, int32_t incy);

/**
 * \brief Symmetric matrix-matrix multiply (BLAS-3): C = alpha*A*B + beta*C.
 *
 * A is symmetric on the left with leading dimension lda; B is m x k.
 * A is square (m x m).  Note the dimensionality convention: pc.m carries the
 * order of A and the row count of both B and C, while pc.k carries the
 * column count of B and C.
 */
VkResult vkblas_symm_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t m, int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer B, int32_t ldb,
                         const float* beta,
                         VkBuffer C, int32_t ldc);
VkResult vkblas_symm_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t m, int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer B, int32_t ldb,
                         const float* beta,
                         VkBuffer C, int32_t ldc);

/**
 * \brief Hermitian matrix-matrix multiply (BLAS-3): C = alpha*A*B + beta*C.
 *
 * Identical to symm when A is real-valued (f32/f16).
 */
VkResult vkblas_hemm_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t m, int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer B, int32_t ldb,
                         const float* beta,
                         VkBuffer C, int32_t ldc);
VkResult vkblas_hemm_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo,
                         int32_t m, int32_t n, const float* alpha,
                         VkBuffer A, int32_t lda,
                         VkBuffer B, int32_t ldb,
                         const float* beta,
                         VkBuffer C, int32_t ldc);

/**
 * \brief Symmetric rank-k update (BLAS-3): C = alpha * op(A) * op(A)^T + beta * C.
 *
 * C is n x n, symmetric. Only the selected triangle is read/written; the
 * shader mirrors the off-diagonal half so the output C is fully symmetric.
 */
VkResult vkblas_syrk_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t trans,
                         int32_t n, int32_t k,
                         const float* alpha,
                         VkBuffer A, int32_t lda,
                         const float* beta,
                         VkBuffer C, int32_t ldc);
VkResult vkblas_syrk_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t trans,
                         int32_t n, int32_t k,
                         const float* alpha,
                         VkBuffer A, int32_t lda,
                         const float* beta,
                         VkBuffer C, int32_t ldc);

/**
 * \brief Hermitian rank-k update (BLAS-3): C = alpha * op(A) * op(A)^T + beta * C.
 *
 * Identical to syrk when A is real-valued (f32/f16).
 */
VkResult vkblas_herk_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t trans,
                         int32_t n, int32_t k,
                         const float* alpha,
                         VkBuffer A, int32_t lda,
                         const float* beta,
                         VkBuffer C, int32_t ldc);
VkResult vkblas_herk_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                         VkBLASFillMode_t uplo, VkBLASOperation_t trans,
                         int32_t n, int32_t k,
                         const float* alpha,
                         VkBuffer A, int32_t lda,
                         const float* beta,
                         VkBuffer C, int32_t ldc);

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

/**
 * \brief VJITC-bridge sparse-dense GEMM: C = alpha * op(A)_sparse @ op(B) + beta * C
 *
 * A is a CSR-format sparse matrix; B and C are dense. The matrix A and vectors
 * B, C are HIP device pointers that must be allocated via hipMalloc and exported
 * to Vulkan via VK_EXT_external_memory_host (see vkstream.h for the import path).
 *
 * This function bridges into rocSPARSE's hipsparseSpMM — no Vulkan shader
 * dispatch occurs. The caller's VkCommandBuffer is unused (the HIP call
 * executes on the default stream).
 *
 * \param ctx     Valid VkBLASContext (for capability detection).
 * \param op_A    Operation for A (VKBLAS_OP_N, VKBLAS_OP_T, VKBLAS_OP_C).
 * \param op_B    Operation for B.
 * \param m       Number of rows in C (= rows in op(A)).
 * \param n       Number of columns in C (= cols in op(B)).
 * \param k       Inner dimension (cols in op(A) = rows in op(B)).
 * \param nnz     Number of non-zero elements in A.
 * \param alpha   Host scalar multiplier for A*B.
 * \param csr_row_ptr  CSR row pointers (length = rows_A + 1).
 * \param csr_col_ind  CSR column indices (length = nnz).
 * \param csr_val      CSR values (length = nnz).
 * \param B       Dense matrix B (device pointer, op_B(K, N) column-major).
 * \param beta    Host scalar multiplier for C.
 * \param C       Dense matrix C (device pointer, (M, N) column-major, in/out).
 * \param cmd     VkCommandBuffer (unused for bridge calls).
 * \retval VK_SUCCESS on success.
 */
VkResult vkblas_sparse_gemm_f32(
    VkBLASContext* ctx,
    VkBLASOperation_t op_A,
    VkBLASOperation_t op_B,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t nnz,
    const float* alpha,
    const uint32_t* csr_row_ptr,
    const uint32_t* csr_col_ind,
    const float* csr_val,
    const void* B,
    const float* beta,
    void* C,
    VkCommandBuffer cmd);

/**
 * \brief VJITC-bridge sparse triangular solve: op(A) * y = alpha * x (f32).
 *
 * A is a square CSR-format sparse matrix; x is the dense right-hand side and
 * y the dense solution (may alias x). This is the building block for direct
 * sparse solvers (e.g. a lower then upper SpSV after a sparse LU/ILU
 * factorization), useful for scientific workloads as well as ML.
 *
 * The matrix A and vectors x, y are HIP device pointers that must be
 * allocated via hipMalloc and exported to Vulkan via VK_EXT_external_memory_host
 * (see vkstream.h for the import path).
 *
 * This function bridges into rocSPARSE's hipsparseSpSV — no Vulkan shader
 * dispatch occurs. The caller's VkCommandBuffer is unused (the HIP call
 * executes on the default stream).
 *
 * \param ctx     Valid VkBLASContext (for capability detection).
 * \param op_A    Operation for A (VKBLAS_OP_N or VKBLAS_OP_T; A is square).
 * \param m       Matrix dimension (A is m x m; x/y length m).
 * \param nnz     Number of non-zero elements in A.
 * \param alpha   Host scalar multiplier (alpha * x).
 * \param csr_row_ptr  CSR row pointers (length = m + 1).
 * \param csr_col_ind  CSR column indices (length = nnz).
 * \param csr_val      CSR values (length = nnz).
 * \param x       Dense right-hand side (device pointer, length m).
 * \param y       Dense solution (device pointer, length m, in/out; may == x).
 * \param cmd     VkCommandBuffer (unused for bridge calls).
 * \retval VK_SUCCESS on success.
 */
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
    VkCommandBuffer cmd);

/**
 * \brief VJITC-bridge LU decomposition: A = P * L * U
 *
 * A is overwritten with L (lower, unit diagonal) and U (upper).
 * piv is the pivot vector (1-based, LAPACK style).
 * info = 0 on success, >0 if U is singular.
 *
 * Device pointers must be allocated via hipMalloc and exported to Vulkan
 * via VK_EXT_external_memory_host.
 */
VkResult vkblas_lu_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                       uint32_t n, void* A, uint32_t lda,
                       void* ipiv, void* info);

/**
 * \brief VJITC-bridge matrix inverse via LU.
 * A must be n x n, column-major. On return, A contains A^{-1}.
 * Requires lu (ipiv) from a prior vkblas_lu_f32 call on A.
 */
VkResult vkblas_inverse_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                            uint32_t n, void* A, uint32_t lda,
                            void* ipiv, void* info);

/**
 * \brief VJITC-bridge determinant via LU.
 * Computes det(A) by performing LU on a copy of A, then multiplying
 * the diagonal of U with the permutation sign.
 */
VkResult vkblas_determinant_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                                uint32_t n, void* A, uint32_t lda,
                                void* ipiv, void* info, double* det);

/**
 * \brief VJITC-bridge QR decomposition: A = Q * R
 * A is m x n (m >= n), column-major. On return, A contains R (upper)
 * and tau contains the elementary reflectors for Q.
 */
VkResult vkblas_qr_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                       uint32_t m, uint32_t n,
                       void* A, uint32_t lda,
                       void* tau, void* info);

/**
 * \brief VJITC-bridge Cholesky decomposition: A = L * L^T
 * A is n x n (column-major), symmetric positive-definite. On return, A
 * contains L in its lower triangle.
 */
VkResult vkblas_cholesky_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                             uint32_t n, void* A, uint32_t lda,
                             void* info);

/**
 * \brief VJITC-bridge Eigenvalue Decomposition: A = V * diag(W) * V^T
 * A is n x n (column-major), symmetric. On return, A contains eigenvectors
 * (columns of V), and W contains eigenvalues.
 */
VkResult vkblas_eigendecomp_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
                                uint32_t n, void* A, uint32_t lda,
                                void* W, void* info);

/* ===========================================================================
 * VJITC-bridge: 3D convolution (MIOpen) and JIT compilation (hipRTC/shaderc)
 * ========================================================================== */

/**
 * \brief Native Vulkan 3D convolution: y = alpha * conv3d(x, w) + beta * y
 *
 * Dispatches the register-blocked direct kernel (RB=2, one workgroup per
 * (n, k), 512-thread workgroups). x, w, y are VkBuffer handles (f32, NCDHW /
 * KCDHW / NKDHW layouts). Records a compute dispatch into cmd; the caller
 * owns all synchronization.
 *
 * \param ctx    Valid VkBLASContext.
 * \param n      Batch size.
 * \param c      Input channels.
 * \param di     Input depth.
 * \param hi     Input height.
 * \param wi     Input width.
 * \param k      Output channels.
 * \param dd     Output depth.
 * \param dh     Output height.
 * \param dw     Output width.
 * \param kd     Kernel depth.
 * \param kh     Kernel height.
 * \param kw     Kernel width.
 * \param pad_d  Padding depth.
 * \param pad_h  Padding height.
 * \param pad_w  Padding width.
 * \param stride_d  Stride depth.
 * \param stride_h  Stride height.
 * \param stride_w  Stride width.
 * \param dil_d  Dilation depth.
 * \param dil_h  Dilation height.
 * \param dil_w  Dilation width.
 * \param alpha  Input scale (host scalar).
 * \param x      Input tensor (NCDHW, f32, VkBuffer).
 * \param w      Weight tensor (KCDHW, f32, VkBuffer).
 * \param beta   Output scale (host scalar; accumulation into y not yet fused).
 * \param y      Output tensor (NKDHW, f32, VkBuffer).
 * \param cmd    VkCommandBuffer to record the dispatch into.
 * \retval VK_SUCCESS on success.
 */
VkResult vkblas_conv3d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t di, uint32_t hi, uint32_t wi,
    uint32_t k, uint32_t dd, uint32_t dh, uint32_t dw,
    uint32_t kd, uint32_t kh, uint32_t kw,
    uint32_t pad_d, uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_d, uint32_t stride_h, uint32_t stride_w,
    uint32_t dil_d, uint32_t dil_h, uint32_t dil_w,
    float alpha, VkBuffer x, VkBuffer w, float beta, VkBuffer y,
    VkCommandBuffer cmd);

/**
 * \brief Native Vulkan 1D convolution: y = alpha * conv1d(x, w) + beta * y
 *
 * Spatial dim = 1, tensor format NCL.
 * x: (n, c, li)  w: (k, c, kl)  y: (n, k, lo)
 */
VkResult vkblas_conv1d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t li,
    uint32_t k, uint32_t lo,
    uint32_t kl,
    uint32_t pad_l, uint32_t stride_l, uint32_t dil_l,
    float alpha, VkBuffer x, VkBuffer w, float beta, VkBuffer y,
    VkCommandBuffer cmd);

/**
 * \brief Native Vulkan 2D convolution: y = alpha * conv2d(x, w) + beta * y
 *
 * Spatial dim = 2, tensor format NCHW.
 * x: (n, c, hi, wi)  w: (k, c, kh, kw)  y: (n, k, dh, dw)
 */
VkResult vkblas_conv2d_f32(
    VkBLASContext* ctx,
    uint32_t n, uint32_t c, uint32_t hi, uint32_t wi,
    uint32_t k, uint32_t dh, uint32_t dw,
    uint32_t kh, uint32_t kw,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t dil_h, uint32_t dil_w,
    float alpha, VkBuffer x, VkBuffer w, float beta, VkBuffer y,
    VkCommandBuffer cmd);

/**
 * \brief JIT compile GLSL source to SPIR-V at runtime via shaderc.
 *
 * Only available when the library is compiled with -DVAIT_JIT=ON.
 * Returns VK_ERROR_FEATURE_NOT_PRESENT when JIT is disabled.
 *
 * \param source    GLSL source string (NUL-terminated).
 * \param source_len Length in bytes (0 = use strlen).
 * \param out_spirv Receives malloc'd SPIR-V buffer (caller frees).
 * \param out_len   Receives SPIR-V byte length.
 */
VkResult vkblas_jit_compile_glsl_to_spirv(
    const char* source, size_t source_len,
    uint8_t** out_spirv, size_t* out_len);

/**
 * \brief Create a VkShaderModule from runtime-compiled SPIR-V.
 *        Requires VAIT_JIT enabled and a shaderc-built SPIR-V blob.
 */
VkResult vkblas_jit_create_shader_module(VkDevice device,
    const uint8_t* spirv, size_t spirv_len, VkShaderModule* module);

/**
 * \brief JIT compile HIP C source to a code object via hipRTC.
 *
 * Only available when the library is compiled with -DVAIT_JIT=ON.
 * The code object can be loaded via vkblas_jit_load_hip_module.
 *
 * \param source      HIP C source string.
 * \param source_len  Length in bytes (0 = use strlen).
 * \param out_code    Receives malloc'd code object (caller frees).
 * \param out_len     Receives code object size.
 */
VkResult vkblas_jit_compile_hip(const char* source, size_t source_len,
    void** out_code, size_t* out_len);

/**
 * \brief Load a HIP code object and get a kernel function handle.
 * \param code      Code object (from vkblas_jit_compile_hip).
 * \param code_len  Size of code object.
 * \param func_name Kernel function name within the code object.
 * \param module    Receives the HIP module handle (call hipModuleUnload when done).
 * \param kernel    Receives the HIP function handle.
 */
VkResult vkblas_jit_load_hip_module(const void* code, size_t code_len,
    const char* func_name,
    void* module, void* kernel);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKBLAS_H */
