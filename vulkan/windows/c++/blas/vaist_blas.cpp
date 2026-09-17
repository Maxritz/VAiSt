#include "vaist_blas.h"
namespace vaist {
inline VaistStatus gemm(const VaistRuntime*rt,const float*A,const float*B,float*C,size_t m,size_t k,size_t n,const void*w=0,VaistWeightFormat wf=VAIST_W_FP16){
    return vaist_blas_gemm(rt,A,B,C,m,k,n,w,wf);
}
inline VaistStatus matvec_ternary(const VaistRuntime*rt,const int8_t*sign,const float*scale,size_t k,size_t n,const float*x,float*y){
    return vaist_blas_matvec_ternary(rt,sign,scale,k,n,x,y);
}
inline VaistStatus matvec_binary(const VaistRuntime*rt,const uint8_t*bit,const float*scale,size_t k,size_t n,const float*x,float*y){
    return vaist_blas_matvec_binary(rt,bit,scale,k,n,x,y);
}
inline VaistComputePath best_path(const VaistRuntime*rt,size_t m,size_t k,size_t n,VaistWeightFormat wf){
    return vaist_blas_best_path(rt,m,k,n,wf);
}
}
