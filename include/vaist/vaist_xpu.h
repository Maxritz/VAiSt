#ifndef VAIST_XPU_H
#define VAIST_XPU_H
#include "vaist_core.h"
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

typedef enum {
    VAIST_OPTIMIZATION_NONE,
    VAIST_OPTIMIZATION_BASIC,
    VAIST_OPTIMIZATION_ADVANCED,
    VAIST_OPTIMIZATION_ULTIMATE
} vaist_xpu_optimization_level;

/* Performance hint, mirrors OpenVINO performance_mode (LATENCY/THROUGHPUT):
 * selects blocking shape; DEFAULT = THROUGHPUT. Orthogonal to the
 * capability ladder in vaist_xpu_optimization_level. */
typedef enum {
    VAIST_XPU_HINT_DEFAULT = 0,
    VAIST_XPU_HINT_LATENCY = 1,
    VAIST_XPU_HINT_THROUGHPUT = 2
} vaist_xpu_perf_hint_t;

VAIST_API VaistStatus vaist_xpu_set_perf_hint(vaist_xpu_perf_hint_t hint);
VAIST_API vaist_xpu_perf_hint_t vaist_xpu_get_perf_hint(void);

VAIST_API VaistStatus vaist_xpu_vector_add(const float* A, const float* B, float* C,
                                           vaist_xpu_vector_params_t* params);
VAIST_API VaistStatus vaist_xpu_matrix_transpose(const float* A, float* B,
                                                 vaist_xpu_matrix_params_t* params);
VAIST_API VaistStatus vaist_xpu_reduce_sum(const float* A, float* B,
                                           vaist_xpu_reduce_params_t* params);
VAIST_API int vaist_xpu_is_available(void);
VAIST_API vaist_xpu_optimization_level vaist_xpu_get_optimization_level(void);

#ifdef __cplusplus
}
#endif
#endif