#include "vaist_llm.h"
#include <cstdio>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { std::fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)
#endif
namespace vaist { class KV { VaistKVCache*p_; public: KV():p_(nullptr){DBG_TRACE("KV ctor");} ~KV(){DBG_TRACE("KV dtor p=%p",(void*)p_); vaist_kv_destroy(p_);} VaistStatus create(size_t c,size_t w){VaistStatus s=vaist_kv_create(c,w,&p_); DBG_TRACE("KV create c=%lu w=%lu s=%d p=%p",(unsigned long)c,(unsigned long)w,(int)s,(void*)p_); return s;} }; }
