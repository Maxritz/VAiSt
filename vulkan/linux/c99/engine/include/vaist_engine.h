#ifndef VAIST_ENGINE_H
#define VAIST_ENGINE_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VaistEngine VaistEngine;
typedef struct VaistSession VaistSession;
VAIST_API VaistStatus vaist_engine_create(VaistEngine**out);
VAIST_API void vaist_engine_destroy(VaistEngine*e);
VAIST_API VaistStatus vaist_session_create(VaistEngine*e,VaistSession**out);
VAIST_API void vaist_session_destroy(VaistSession*s);
VAIST_API VaistStatus vaist_generate(VaistSession*s,const char*prompt,size_t max_tokens,char*out,size_t cap);
VAIST_API VaistStatus vaist_engine_cancel(VaistSession*s);
#ifdef __cplusplus
}
#endif
#endif