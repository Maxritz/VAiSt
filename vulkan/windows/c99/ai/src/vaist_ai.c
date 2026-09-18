#include "vaist_ai.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
VAIST_API VaistStatus vaist_embedding_normalize(float*x,size_t n){size_t i;float s=0;if(!x||!n)return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)s+=x[i]*x[i];s=sqrtf(s);if(s==0)return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)x[i]/=s;return VAIST_OK;}
VAIST_API VaistStatus vaist_vector_dot(const float*a,const float*b,size_t n,float*o){size_t i;float s=0;if(!a||!b||!o||!n)return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)s+=a[i]*b[i];*o=s;return VAIST_OK;}
VAIST_API VaistStatus vaist_vector_cosine(const float*a,const float*b,size_t n,float*o){float d,na,nb;size_t i;if(!a||!b||!o||!n){DBG_TRACE("path=guard-fail -> INVALID_ARGUMENT");return VAIST_INVALID_ARGUMENT;}d=na=nb=0;for(i=0;i<n;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}DBG_TRACE("cosine d=%f na=%f nb=%f",d,na,nb);if(na==0||nb==0){DBG_TRACE("path=zero-norm -> INVALID_ARGUMENT");return VAIST_INVALID_ARGUMENT;}*o=d/(sqrtf(na)*sqrtf(nb));DBG_TRACE("cosine result=%f",*o);return VAIST_OK;}
VAIST_API VaistStatus vaist_autograd_sgd(float*x,const float*g,size_t n,float lr){size_t i;if(!x||!g||!n||lr<0)return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)x[i]-=lr*g[i];return VAIST_OK;}
VAIST_API VaistStatus vaist_agent_validate_tool_name(const char*n){size_t i,L;if(!n||!*n)return VAIST_INVALID_ARGUMENT;L=strlen(n);if(L>64)return VAIST_INVALID_ARGUMENT;for(i=0;i<L;i++)if(!((n[i]>='a'&&n[i]<='z')||(n[i]>='A'&&n[i]<='Z')||(n[i]>='0'&&n[i]<='9')||n[i]=='_'||n[i]=='-'))return VAIST_INVALID_ARGUMENT;return VAIST_OK;}
