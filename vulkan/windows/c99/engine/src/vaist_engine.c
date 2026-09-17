#include "vaist_engine.h"
#include "vaist_llm.h"
#include <stdlib.h>
#include <string.h>
struct VaistEngine{uint32_t live;};
struct VaistSession{VaistEngine*e;uint32_t cancelled;};
VAIST_API VaistStatus vaist_engine_create(VaistEngine**o){VaistEngine*e;if(!o)return VAIST_INVALID_ARGUMENT;e=(VaistEngine*)calloc(1,sizeof(*e));if(!e)return VAIST_OUT_OF_MEMORY;e->live=1;*o=e;return VAIST_OK;}
VAIST_API void vaist_engine_destroy(VaistEngine*e){free(e);}
VAIST_API VaistStatus vaist_session_create(VaistEngine*e,VaistSession**o){VaistSession*s;if(!e||!e->live||!o)return VAIST_INVALID_ARGUMENT;s=(VaistSession*)calloc(1,sizeof(*s));if(!s)return VAIST_OUT_OF_MEMORY;s->e=e;*o=s;return VAIST_OK;}
VAIST_API void vaist_session_destroy(VaistSession*s){free(s);}
VAIST_API VaistStatus vaist_engine_cancel(VaistSession*s){if(!s)return VAIST_INVALID_ARGUMENT;s->cancelled=1;return VAIST_OK;}
VAIST_API VaistStatus vaist_generate(VaistSession*s,const char*p,size_t n,char*o,size_t c){size_t L,i;if(!s||!p||!o||c==0)return VAIST_INVALID_ARGUMENT;if(s->cancelled)return VAIST_CANCELLED;L=strlen(p);if(L>=c)return VAIST_OUT_OF_MEMORY;if(n>c-L-1)n=c-L-1;memcpy(o,p,L);for(i=0;i<n;i++){if(s->cancelled){o[L+i]=0;return VAIST_CANCELLED;}o[L+i]=(char)(' ');if(i%4==0&&L+i+1<c)o[L+i]='.';}o[L+n]=0;return VAIST_OK;}
