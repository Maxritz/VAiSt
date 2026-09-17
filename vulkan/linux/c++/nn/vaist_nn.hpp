#include "vaist_nn.h"
namespace vaist { inline VaistStatus relu(float*x,size_t n){return vaist_nn_relu_f32(x,n);} inline VaistStatus softmax(float*x,size_t n){return vaist_nn_softmax_f32(x,n);} }