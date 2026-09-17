#include "vaist_quant.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef struct Q4B{float d;int8_t q[32];}Q4B;
typedef struct Q8B{float d;int8_t q[32];}Q8B;
static size_t blocks(size_t n){return (n+31u)/32u;}
VAIST_API size_t vaist_quant_block_bytes(VaistQuantType q){return q==VAIST_Q8_0?sizeof(Q8B):sizeof(Q4B);}
VAIST_API VaistStatus vaist_quantize_f32(VaistQuantType q,const float*s,size_t n,void*d,size_t cap,size_t*u){size_t nb,i,j;float a,sc;unsigned char*o=(unsigned char*)d;if(!s||!d||!n||!u||(q!=VAIST_Q8_0&&q!=VAIST_Q4_0&&q!=VAIST_Q4_1))return VAIST_INVALID_ARGUMENT;nb=blocks(n);if(cap<nb*vaist_quant_block_bytes(q))return VAIST_OUT_OF_MEMORY;for(i=0;i<nb;i++){size_t len=(n-i*32u<32u?n-i*32u:32u);a=0;for(j=0;j<len;j++){float x=fabsf(s[i*32u+j]);if(x>a)a=x;}sc=(q==VAIST_Q8_0)?(a/127.0f):(a/7.0f);if(sc==0)sc=1.0f;if(q==VAIST_Q8_0){Q8B*b=(Q8B*)(o+i*sizeof(Q8B));b->d=sc;for(j=0;j<32;j++)b->q[j]=(j<len)?(int8_t)lrintf(s[i*32u+j]/sc):0;}else{Q4B*b=(Q4B*)(o+i*sizeof(Q4B));b->d=sc;for(j=0;j<32;j++)b->q[j]=(j<len)?(int8_t)lrintf(s[i*32u+j]/sc):0;}}*u=nb*vaist_quant_block_bytes(q);return VAIST_OK;}
VAIST_API VaistStatus vaist_dequantize_f32(VaistQuantType q,const void*d,size_t bytes,float*o,size_t n){size_t nb,i,j,bs;if(!d||!o||!n)return VAIST_INVALID_ARGUMENT;bs=vaist_quant_block_bytes(q);nb=blocks(n);if(bytes<nb*bs)return VAIST_INVALID_ARGUMENT;for(i=0;i<nb;i++){size_t len=(n-i*32u<32u?n-i*32u:32u);float sc;if(q==VAIST_Q8_0){const Q8B*b=(const Q8B*)((const unsigned char*)d+i*bs);sc=b->d;for(j=0;j<len;j++)o[i*32u+j]=sc*(float)b->q[j];}else{const Q4B*b=(const Q4B*)((const unsigned char*)d+i*bs);sc=b->d;for(j=0;j<len;j++)o[i*32u+j]=sc*(float)b->q[j];}}return VAIST_OK;}
