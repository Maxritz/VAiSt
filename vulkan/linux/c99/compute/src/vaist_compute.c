#include "vaist_compute.h"
#include "vaist_core.h"
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

static int bytes_ok(size_t count, size_t item) { return item != 0u && count <= SIZE_MAX / item; }
static int bad_args(const void*a,const void*b,void*c,size_t n){
    return !a||!b||!c||n==0u||!bytes_ok(n,sizeof(float));
}

/* ---- existing elementwise / reduce (unchanged behavior) ---- */
VAIST_API VaistStatus vaist_add_f32(const float*a,const float*b,float*o,size_t n){
    size_t i; if(bad_args(a,b,o,n))return VAIST_INVALID_ARGUMENT;
    for(i=0;i<n;i++)o[i]=a[i]+b[i]; return VAIST_OK;
}
VAIST_API VaistStatus vaist_mul_f32(const float*a,const float*b,float*o,size_t n){
    size_t i; if(bad_args(a,b,o,n))return VAIST_INVALID_ARGUMENT;
    for(i=0;i<n;i++)o[i]=a[i]*b[i]; return VAIST_OK;
}
VAIST_API VaistStatus vaist_reduce_sum_f32(const float*a,float*o,size_t n){
    size_t i; if(!a||!o||!n||!bytes_ok(n,sizeof(float)))return VAIST_INVALID_ARGUMENT;
    float s=0.0f; for(i=0;i<n;i++)s+=a[i]; *o=s; return VAIST_OK;
}
VAIST_API VaistStatus vaist_dot_i8(const int8_t*a,const int8_t*b,int32_t*o,size_t n){
    size_t i; int32_t s=0; if(!a||!b||!o||!n)return VAIST_INVALID_ARGUMENT;
    for(i=0;i<n;i++){int32_t x=(int32_t)a[i]*(int32_t)b[i];
        if((x>0&&s>INT32_MAX-x)||(x<0&&s<INT32_MIN-x))return VAIST_INVALID_ARGUMENT; s+=x;}
    *o=s; return VAIST_OK;
}

/* ---- classic matmul (kept as scalar fallback; new matmul-free kernels
 *     below are the recommended path on weak GPUs) ---- */
VAIST_API VaistStatus vaist_matmul_f32(const float*a,const float*b,float*o,size_t m,size_t k,size_t n){
    size_t i,j,p;
    if(!a||!b||!o||!m||!k||!n||m>SIZE_MAX/k||m*k>SIZE_MAX/n||k>SIZE_MAX/n||n>SIZE_MAX/k)
        return VAIST_INVALID_ARGUMENT;
    for(i=0;i<m;i++)for(j=0;j<n;j++){float s=0.0f;for(p=0;p<k;p++)s+=a[i*k+p]*b[p*n+j];o[i*n+j]=s;}
    return VAIST_OK;
}

/* ---- MatMul-free kernels (2406.02528 ternary, 2608.03142/01528 1-bit xnor) ---- */
VAIST_API VaistStatus vaist_matvec_ternary_f32(const int8_t*wsign,const float*scale,
        size_t k,size_t n,const float*x,float*o){
    /* wsign: k*n packed as {-1,0,+1}; scale: one float per row (n). */
    size_t i,j;
    if(!wsign||!scale||!x||!o||!k||!n)return VAIST_INVALID_ARGUMENT;
    for(j=0;j<n;j++){float acc=0.0f; const int8_t*w=wsign+j*k;
        for(i=0;i<k;i++){int8_t s=w[i]; if(s>0)acc+=x[i]; else if(s<0)acc-=x[i];}
        o[j]=acc*scale[j];}
    return VAIST_OK;
}
VAIST_API VaistStatus vaist_matvec_binary_f32(const uint8_t*wbit,const float*scale,
        size_t k,size_t n,const float*x,float*o){
    /* wbit: packed 1 bit/weight, MSB first per byte; scale: one float per row (n). */
    size_t i,j;
    if(!wbit||!scale||!x||!o||!k||!n)return VAIST_INVALID_ARGUMENT;
    for(j=0;j<n;j++){int32_t acc=0; const uint8_t*w=wbit+(j*k)/8;
        for(i=0;i<k;i++)acc += (int)((w[i>>3] >> (7u-(i&7u))) & 1u);
        float dot=0.0f; for(i=0;i<k;i++){int bit=(int)((w[i>>3]>>(7u-(i&7u)))&1u); dot += (bit?x[i]:-x[i]);}
        o[j]=dot*scale[j];}
    return VAIST_OK;
}

/* ---- capability ladder: real selection over device caps ---- */
VAIST_API VaistComputePath vaist_compute_best_path(const VaistRuntime*rt,
        size_t m,size_t k,size_t n,int prefer_matmul_free){
    (void)m; (void)k; (void)n;
    if(prefer_matmul_free){ return VAIST_PATH_MATMUL_FREE; }
    if(rt){
        VaistRuntimeInfo info;
        if(vaist_runtime_info(rt,&info)==VAIST_OK){
            if(info.backend==VAIST_BACKEND_CPU) return VAIST_PATH_SIMD;
            if(info.vulkan_available && info.vulkan_api_version){
                VaistDeviceCaps dc;
                if(vaist_runtime_device_caps(rt,&dc)==VAIST_OK){
                    if(dc.integer_dot_product_8bit_accelerated) return VAIST_PATH_VULKAN_DOT;
                    return VAIST_PATH_VULKAN_TILE;  /* tiled GPU matmul, no dot */
                }
                return VAIST_PATH_VULKAN_TILE;
            }
        }
    }
    return VAIST_PATH_MATMUL_FREE; /* guaranteed zero-mul fallback for weak GPUs */
}
