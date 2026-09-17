#ifndef VAIST_DISTRIBUTED_H
#define VAIST_DISTRIBUTED_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VaistCommunicator{uint32_t rank,size;} VaistCommunicator;
VAIST_API VaistStatus vaist_comm_init(VaistCommunicator*c,uint32_t rank,uint32_t size);
VAIST_API VaistStatus vaist_allreduce_sum_f32(VaistCommunicator*c,float*data,size_t n);
VAIST_API VaistStatus vaist_allgather_f32(VaistCommunicator*c,const float*in,size_t n,float*out,size_t cap);
VAIST_API VaistStatus vaist_reduce_scatter_f32(VaistCommunicator*c,const float*in,size_t n,float*out,size_t out_n);
VAIST_API VaistStatus vaist_broadcast_f32(VaistCommunicator*c,float*data,size_t n,uint32_t root);
#ifdef __cplusplus
}
#endif
#endif