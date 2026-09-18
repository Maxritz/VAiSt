#ifndef VAIST_XPU_H
#define VAIST_XPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t count;
} vaist_xpu_vector_params_t;

typedef struct {
    size_t rows;
    size_t cols;
} vaist_xpu_matrix_params_t;

typedef struct {
    size_t count;
    size_t stride;
} vaist_xpu_reduce_params_t;

void vaist_xpu_vector_add(const float* A, const float* B, float* C, size_t count);
void vaist_xpu_matrix_transpose(const float* A, float* B, size_t rows, size_t cols);
void vaist_xpu_reduce_sum(const float* A, float* B, size_t count, size_t stride);

#ifdef __cplusplus
}
#endif

#endif // VAIST_XPU_H