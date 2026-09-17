#include "vaist_distributed.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
VAIST_API VaistStatus vaist_comm_init(VaistCommunicator*c,uint32_t r,uint32_t s){if(!c||s==0u||r>=s)return VAIST_INVALID_ARGUMENT;c->rank=r;c->size=s;return VAIST_OK;}
VAIST_API VaistStatus vaist_allreduce_sum_f32(VaistCommunicator*c,float*d,size_t n){if(!c||!d||!n)return VAIST_INVALID_ARGUMENT;if(c->size==1u)return VAIST_OK;return VAIST_UNSUPPORTED;}
VAIST_API VaistStatus vaist_allgather_f32(VaistCommunicator*c,const float*i,size_t n,float*o,size_t cap){size_t total;if(!c||!i||!o||!n)return VAIST_INVALID_ARGUMENT;if((size_t)c->size>SIZE_MAX/n)return VAIST_INVALID_ARGUMENT;total=n*(size_t)c->size;if(total>cap)return VAIST_OUT_OF_MEMORY;if(c->size!=1u)return VAIST_UNSUPPORTED;memcpy(o,i,n*sizeof(float));return VAIST_OK;}
VAIST_API VaistStatus vaist_reduce_scatter_f32(VaistCommunicator*c,const float*i,size_t n,float*o,size_t on){if(!c||!i||!o||!on||n<on)return VAIST_INVALID_ARGUMENT;if(c->size!=1u)return VAIST_UNSUPPORTED;memcpy(o,i,on*sizeof(float));return VAIST_OK;}
VAIST_API VaistStatus vaist_broadcast_f32(VaistCommunicator*c,float*d,size_t n,uint32_t root){if(!c||!d||!n||root>=c->size)return VAIST_INVALID_ARGUMENT;return c->size==1u?VAIST_OK:VAIST_UNSUPPORTED;}
