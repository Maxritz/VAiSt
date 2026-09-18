#include "vaist_ai.h"
#include <cstdio>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { std::fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)
#endif
namespace vaist { inline VaistStatus cosine(const float*a,const float*b,size_t n,float*o){DBG_TRACE("cosine enter n=%lu",(unsigned long)n); VaistStatus s=vaist_vector_cosine(a,b,n,o); DBG_TRACE("cosine s=%d o=%f",(int)s,(s==VAIST_OK&&o)?*o:0.0f); return s;} }
