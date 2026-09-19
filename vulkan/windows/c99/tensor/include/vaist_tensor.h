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

typedef enum VaistDType {
    VAIST_F32  = 0,
    VAIST_F64  = 1,
    VAIST_I32  = 2,
    VAIST_I8   = 3,
    VAIST_U8   = 4,
    VAIST_F16 = 5,
    VAIST_DTYPE_Q8_0 = 6,
    VAIST_BF16 = 7,
    VAIST_F8_E4M3 = 8,
    VAIST_F8_E5M2 = 9,
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
