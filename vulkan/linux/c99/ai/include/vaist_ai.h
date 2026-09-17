#ifndef VAIST_AI_H
#define VAIST_AI_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
VAIST_API VaistStatus vaist_embedding_normalize(float*x,size_t n);
VAIST_API VaistStatus vaist_vector_dot(const float*a,const float*b,size_t n,float*out);
VAIST_API VaistStatus vaist_vector_cosine(const float*a,const float*b,size_t n,float*out);
VAIST_API VaistStatus vaist_autograd_sgd(float*x,const float*g,size_t n,float lr);
VAIST_API VaistStatus vaist_agent_validate_tool_name(const char*name);
#ifdef __cplusplus
}
#endif
#endif