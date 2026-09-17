#ifndef VAIST_CORE_H
#define VAIST_CORE_H
#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
#define VAIST_API __declspec(dllexport)
#else
#define VAIST_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define VAIST_ABI_MAJOR 1u
#define VAIST_ABI_MINOR 0u
#define VAIST_ABI_PATCH 0u
typedef enum VaistStatus {
 VAIST_OK=0, VAIST_INVALID_ARGUMENT=1, VAIST_INVALID_STATE=2, VAIST_UNSUPPORTED=3,
 VAIST_OUT_OF_MEMORY=4, VAIST_DEVICE_ERROR=5, VAIST_DEVICE_LOST=6, VAIST_TIMEOUT=7,
 VAIST_CANCELLED=8, VAIST_IO_ERROR=9, VAIST_MODEL_ERROR=10, VAIST_COMPILE_ERROR=11,
 VAIST_RUNTIME_ERROR=12, VAIST_INTERNAL_ERROR=13
} VaistStatus;
typedef struct VaistVersion { uint32_t major,minor,patch; } VaistVersion;
VAIST_API const char *vaist_get_version(void);
VAIST_API VaistVersion vaist_get_abi_version(void);
VAIST_API const char *vaist_get_build_info(void);
VAIST_API const char *vaist_status_string(VaistStatus s);
VAIST_API VaistStatus vaist_set_last_error(const char *message);
VAIST_API VaistStatus vaist_last_error(char *dst,size_t cap);
#ifdef __cplusplus
}
#endif
#endif
