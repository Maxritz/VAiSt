#include "vaist_compute.h"
namespace vaist { inline VaistStatus matmul(const float*a,const float*b,float*o,size_t m,size_t k,size_t n){return vaist_matmul_f32(a,b,o,m,k,n);} }