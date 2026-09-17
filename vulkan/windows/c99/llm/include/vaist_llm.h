#ifndef VAIST_LLM_H
#define VAIST_LLM_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VaistKVCache VaistKVCache;
typedef struct VaistSampler VaistSampler;
typedef struct VaistTokenizer VaistTokenizer;
VAIST_API VaistStatus vaist_kv_create(size_t capacity,size_t width,VaistKVCache**out);
VAIST_API void vaist_kv_destroy(VaistKVCache*k);
VAIST_API VaistStatus vaist_kv_write(VaistKVCache*k,size_t pos,const float*data);
VAIST_API VaistStatus vaist_kv_read(const VaistKVCache*k,size_t pos,float*out);
VAIST_API size_t vaist_kv_length(const VaistKVCache*k);
VAIST_API VaistStatus vaist_sampler_create(float temperature,uint32_t top_k,float top_p,VaistSampler**out);
VAIST_API void vaist_sampler_destroy(VaistSampler*s);
VAIST_API VaistStatus vaist_sample_greedy(const float*logits,size_t n,uint32_t*out);
VAIST_API VaistStatus vaist_sample(const float*logits,size_t n,VaistSampler*s,uint64_t*state,uint32_t*out);
VAIST_API VaistStatus vaist_tokenizer_create(VaistTokenizer**out);
VAIST_API void vaist_tokenizer_destroy(VaistTokenizer*t);
VAIST_API VaistStatus vaist_tokenize_bytes(const char*text,uint32_t*out,size_t cap,size_t*count);
#ifdef __cplusplus
}
#endif
#endif