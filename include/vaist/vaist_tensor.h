/**
 * \file vaist_tensor.h
 * \brief Tensor type and dtype enumeration.
 */
#ifndef VAIST_TENSOR_H
#define VAIST_TENSOR_H
#include "vaist_core.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Data type enumeration.
 * Values 0-4 match the original ABI. VAIST_F16 reuses VAIST_U8 slot
 * (fp16 is stored as uint16 in memory; the dtype distinguishes semantics
 * at the API boundary). Quantized types are VaistQuantType values.
 */
typedef enum VaistDType {
    VAIST_F32  = 0,  /**< 32-bit IEEE 754 float */
    VAIST_F64  = 1,  /**< 64-bit IEEE 754 float (treated as I32 in SafeTensors mapping) */
    VAIST_I32  = 2,  /**< 32-bit signed integer */
    VAIST_I8   = 3,  /**< 8-bit signed integer */
    VAIST_U8   = 4,  /**< 8-bit unsigned integer (also used for fp16/uint16 storage) */
    VAIST_F16 = 5,   /**< 16-bit float (stored as uint16) */
    VAIST_BF16 = 7,  /**< Brain float16 (stored as uint16) — non-ABI-compatible extension */
    VAIST_F8_E4M3 = 8,  /**< FP8 E4M3 (1 byte per element) */
    VAIST_F8_E5M2 = 9,  /**< FP8 E5M2 (1 byte per element) */
    VAIST_DTYPE_Q8_0 = 6,  /**< GGML Q8_0 quantized type for storage */
} VaistDType;

typedef struct VaistTensor VaistTensor;

VAIST_API size_t vaist_dtype_size(VaistDType t);
VAIST_API VaistStatus vaist_tensor_create(VaistDType t, uint32_t rank, const uint64_t *shape, VaistTensor **out);
VAIST_API VaistStatus vaist_tensor_view(VaistTensor *src, uint32_t rank, const uint64_t *shape, const int64_t *stride, VaistTensor **out);
VAIST_API void vaist_tensor_destroy(VaistTensor *t);
VAIST_API VaistStatus vaist_tensor_reshape(VaistTensor *t, uint32_t rank, const uint64_t *shape);
VAIST_API VaistStatus vaist_tensor_transpose2(VaistTensor *t);
VAIST_API VaistStatus vaist_tensor_copy(const VaistTensor *src, VaistTensor *dst);
VAIST_API void *vaist_tensor_data(VaistTensor *t);
VAIST_API const void *vaist_tensor_const_data(const VaistTensor *t);
VAIST_API uint32_t vaist_tensor_rank(const VaistTensor *t);
VAIST_API uint64_t vaist_tensor_numel(const VaistTensor *t);
VAIST_API VaistDType vaist_tensor_dtype(const VaistTensor *t);
VAIST_API const uint64_t *vaist_tensor_shape(const VaistTensor *t);
VAIST_API const int64_t *vaist_tensor_stride(const VaistTensor *t);

#ifdef __cplusplus
}
#endif
#endif
