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
#ifdef __cplusplus
}
#endif
#endif