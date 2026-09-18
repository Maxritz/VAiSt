#ifndef VAIST_NPU_H
#define VAIST_NPU_H
#include "vaist_core.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define VAIST_NPU_BACKEND_INTEL_AI_BOOST 1
#define VAIST_NPU_BACKEND_GOOGLE_EDGE_TPU 2

/* GEMM flags (vaist_npu_gemm_params_t.flags). */
#define VAIST_NPU_FLAG_SPARSE 1u

/* Performance hint, mirrors OpenVINO performance_mode + NPU efficiency override:
 * LATENCY = smallest tiles/fastest first-result, THROUGHPUT = largest tiles,
 * EFFICIENCY = latency-shaped tiles tuned for power. DEFAULT = THROUGHPUT. */
typedef enum {
    VAIST_NPU_HINT_DEFAULT = 0,
    VAIST_NPU_HINT_LATENCY = 1,
    VAIST_NPU_HINT_THROUGHPUT = 2,
    VAIST_NPU_HINT_EFFICIENCY = 3
} vaist_npu_perf_hint_t;

VAIST_API VaistStatus vaist_npu_set_perf_hint(vaist_npu_perf_hint_t hint);
VAIST_API vaist_npu_perf_hint_t vaist_npu_get_perf_hint(void);

typedef struct {
    int32_t M, N, K;
    uint32_t a_off, b_off, c_off;
    uint32_t flags;
    float alpha, beta;
} vaist_npu_gemm_params_t;

typedef struct {
    int32_t i0, n;
    uint32_t w_off, s_off, x_off, y_off;
} vaist_npu_mv_params_t;

typedef struct {
    int in_h, in_w, in_c;
    int kernel_h, kernel_w, out_c;
    int stride;
    int pad;
    int dilation;
} vaist_npu_conv2d_params_t;

typedef enum {
    VAIST_NPU_OPT_GEMM = 0,
    VAIST_NPU_OPT_CONV2D = 1,
    VAIST_NPU_OPT_ACTIVATION = 2,
    VAIST_NPU_OPT_POOLING = 3
} vaist_npu_operation_t;

VAIST_API VaistStatus vaist_npu_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                                          size_t M, size_t N, size_t K,
                                          const vaist_npu_gemm_params_t* params);
VAIST_API VaistStatus vaist_npu_conv2d_int8(const int8_t* input, const int8_t* kernel, int32_t* output,
                                            const vaist_npu_conv2d_params_t* params);
VAIST_API VaistStatus vaist_npu_relu_int32(const int32_t* input, int32_t* output, size_t count);
VAIST_API int vaist_npu_is_available(void);
VAIST_API uint32_t vaist_npu_device_id(void);
VAIST_API VaistStatus vaist_npu_device_probe(void);

#ifdef __cplusplus
}
#endif
#endif