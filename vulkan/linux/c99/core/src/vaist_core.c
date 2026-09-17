#include "vaist_core.h"
#include <stdio.h>
#include <string.h>
#if defined(_WIN32)
#define VAIST_THREAD_LOCAL __declspec(thread)
#else
#define VAIST_THREAD_LOCAL _Thread_local
#endif
static VAIST_THREAD_LOCAL char g_error[512];
VAIST_API const char *vaist_get_version(void){return "1.0.0";}
VAIST_API VaistVersion vaist_get_abi_version(void){VaistVersion v={VAIST_ABI_MAJOR,VAIST_ABI_MINOR,VAIST_ABI_PATCH};return v;}
VAIST_API const char *vaist_get_build_info(void){return "VAiSt Vulkan 1.4.357; C99 ABI; CPU fallback; cooperative matrix excluded";}
VAIST_API const char *vaist_status_string(VaistStatus s){
 switch(s){case VAIST_OK:return "ok";case VAIST_INVALID_ARGUMENT:return "invalid_argument";case VAIST_INVALID_STATE:return "invalid_state";case VAIST_UNSUPPORTED:return "unsupported";case VAIST_OUT_OF_MEMORY:return "out_of_memory";case VAIST_DEVICE_ERROR:return "device_error";case VAIST_DEVICE_LOST:return "device_lost";case VAIST_TIMEOUT:return "timeout";case VAIST_CANCELLED:return "cancelled";case VAIST_IO_ERROR:return "io_error";case VAIST_MODEL_ERROR:return "model_error";case VAIST_COMPILE_ERROR:return "compile_error";case VAIST_RUNTIME_ERROR:return "runtime_error";default:return "internal_error";}
}
VAIST_API VaistStatus vaist_set_last_error(const char *message){size_t n;if(!message)return VAIST_INVALID_ARGUMENT;n=strlen(message);if(n>=sizeof(g_error))n=sizeof(g_error)-1;memcpy(g_error,message,n);g_error[n]=0;return VAIST_OK;}
VAIST_API VaistStatus vaist_last_error(char *dst,size_t cap){size_t n;if(!dst||cap==0)return VAIST_INVALID_ARGUMENT;n=strlen(g_error);if(n>=cap)n=cap-1;memcpy(dst,g_error,n);dst[n]=0;return VAIST_OK;}
