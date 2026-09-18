#include "vaist_llm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
struct VaistKVCache{size_t cap,width,len;float*data;};
struct VaistSampler{float temperature,top_p;uint32_t top_k;};
struct VaistTokenizer{uint32_t reserved;};
VAIST_API VaistStatus vaist_kv_create(size_t c,size_t w,VaistKVCache**o){VaistKVCache*k;if(!o||!c||!w){DBG_TRACE("path=guard-fail -> INVALID_ARGUMENT");return VAIST_INVALID_ARGUMENT;}k=(VaistKVCache*)calloc(1,sizeof(*k));if(!k)return VAIST_OUT_OF_MEMORY;k->data=(float*)calloc(c*w,sizeof(float));if(!k->data){free(k);return VAIST_OUT_OF_MEMORY;}k->cap=c;k->width=w;*o=k;DBG_TRACE("kv_create cap=%lu width=%lu -> OK",(unsigned long)c,(unsigned long)w);return VAIST_OK;}
VAIST_API void vaist_kv_destroy(VaistKVCache*k){if(k){free(k->data);free(k);}}
VAIST_API VaistStatus vaist_kv_write(VaistKVCache*k,size_t p,const float*d){if(!k||!d||p>=k->cap){DBG_TRACE("path=guard-fail p=%lu cap=%lu -> INVALID_ARGUMENT",(unsigned long)p,(unsigned long)(k?k->cap:0ul));return VAIST_INVALID_ARGUMENT;}memcpy(k->data+p*k->width,d,k->width*sizeof(float));if(p+1>k->len)k->len=p+1;DBG_TRACE("kv_write p=%lu len=%lu d0=%f d1=%f",(unsigned long)p,(unsigned long)k->len,d[0],k->width>1?d[1]:0.0f);return VAIST_OK;}
VAIST_API VaistStatus vaist_kv_read(const VaistKVCache*k,size_t p,float*o){if(!k||!o||p>=k->len){DBG_TRACE("path=guard-fail p=%lu len=%lu -> INVALID_ARGUMENT",(unsigned long)p,(unsigned long)(k?k->len:0ul));return VAIST_INVALID_ARGUMENT;}memcpy(o,k->data+p*k->width,k->width*sizeof(float));DBG_TRACE("kv_read p=%lu len=%lu width=%lu o0=%f o1=%f",(unsigned long)p,(unsigned long)k->len,(unsigned long)k->width,o[0],k->width>1?o[1]:0.0f);return VAIST_OK;}
VAIST_API size_t vaist_kv_length(const VaistKVCache*k){return k?k->len:0;}
VAIST_API VaistStatus vaist_sampler_create(float t,uint32_t k,float p,VaistSampler**o){VaistSampler*s;if(!o||t<0||p<=0||p>1)return VAIST_INVALID_ARGUMENT;s=(VaistSampler*)calloc(1,sizeof(*s));if(!s)return VAIST_OUT_OF_MEMORY;s->temperature=t;s->top_k=k;s->top_p=p;*o=s;return VAIST_OK;}
VAIST_API void vaist_sampler_destroy(VaistSampler*s){free(s);}
VAIST_API VaistStatus vaist_sample_greedy(const float*l,size_t n,uint32_t*o){size_t i,b=0;if(!l||!o||!n)return VAIST_INVALID_ARGUMENT;for(i=1;i<n;i++)if(l[i]>l[b])b=i;*o=(uint32_t)b;return VAIST_OK;}
static uint64_t rng(uint64_t*x){*x^=*x<<13;*x^=*x>>7;*x^=*x<<17;return*x;}
VAIST_API VaistStatus vaist_sample(const float*l,size_t n,VaistSampler*s,uint64_t*st,uint32_t*o){size_t i;float sum=0,r;uint64_t q;if(!l||!s||!st||!o||!n)return VAIST_INVALID_ARGUMENT;if(s->temperature==0)return vaist_sample_greedy(l,n,o);float*m=(float*)malloc(n*sizeof(float));if(!m)return VAIST_OUT_OF_MEMORY;for(i=0;i<n;i++){m[i]=expf(l[i]/s->temperature);sum+=m[i];}q=rng(st);r=((float)(q&0xffffffu)/16777216.0f)*sum;for(i=0;i<n;i++){r-=m[i];if(r<=0)break;}if(i>=n)i=n-1;*o=(uint32_t)i;free(m);return VAIST_OK;}
VAIST_API VaistStatus vaist_tokenizer_create(VaistTokenizer**o){if(!o)return VAIST_INVALID_ARGUMENT;*o=(VaistTokenizer*)calloc(1,sizeof(VaistTokenizer));return *o?VAIST_OK:VAIST_OUT_OF_MEMORY;}
VAIST_API void vaist_tokenizer_destroy(VaistTokenizer*t){free(t);}
VAIST_API VaistStatus vaist_tokenize_bytes(const char*x,uint32_t*o,size_t c,size_t*n){size_t i,L;if(!x||!o||!n)return VAIST_INVALID_ARGUMENT;L=strlen(x);if(c<L)return VAIST_OUT_OF_MEMORY;for(i=0;i<L;i++)o[i]=(uint32_t)(unsigned char)x[i];*n=L;return VAIST_OK;}
