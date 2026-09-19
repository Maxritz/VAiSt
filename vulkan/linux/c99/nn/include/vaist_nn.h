#ifndef VAIST_NN_H
#define VAIST_NN_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
VAIST_API VaistStatus vaist_nn_relu_f32(float*x,size_t n);
VAIST_API VaistStatus vaist_nn_gelu_f32(float*x,size_t n);
VAIST_API VaistStatus vaist_nn_silu_f32(float*x,size_t n);
VAIST_API VaistStatus vaist_nn_rmsnorm_f32(float*x,size_t n,float eps);
VAIST_API VaistStatus vaist_nn_softmax_f32(float*x,size_t n);
VAIST_API VaistStatus vaist_nn_rope_f32(float*x,size_t pairs,float theta,size_t position);

/**
 * \brief Fused dual RMSNorm for DeepSeek V2/V4 QK-norm pattern.
 *
 * Computes two independent RMSNorms (query branch and KV branch) in a single
 * pass, mirroring AITER fused_qknorm_idxrqknorm / ATOM _fuse_rmsnorm_quant.
 *
 * Replaces two separate vaist_nn_rmsnorm_f32 calls with one kernel launch,
 * halving the thread-divergence overhead and avoiding a redundant second
 * mean-square reduction over the same input rows.
 *
 * Replaces: atom/model_ops/layernorm.py:rmsnorm2d_fwd x2 (q_norm + kv_norm)
 *
 * \param x         Input tensor                     (num_rows * num_cols)
 * \param w_q       Q-norm weight                    (num_cols)
 * \param w_kv      KV-norm weight                   (num_cols)
 * \param y_q       Q-norm output (f32 ref)          (num_rows * num_cols)
 * \param y_kv      KV-norm output (f32 ref)         (num_rows * num_cols)
 * \param num_rows  Number of rows to process
 * \param num_cols  Columns per row (hidden dim)
 * \param eps       RMSNorm epsilon (typically 1e-6)
 * \return VAIST_OK on success
 */
VAIST_API VaistStatus vaist_nn_fused_kv_rmsnorm_f32(
    const float* x, const float* w_q, const float* w_kv,
    float* y_q, float* y_kv,
    size_t num_rows, size_t num_cols, float eps
);
#ifdef __cplusplus
}
#endif
#endif