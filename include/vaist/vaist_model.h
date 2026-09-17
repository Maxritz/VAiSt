#ifndef VAIST_MODEL_H
#define VAIST_MODEL_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum VaistModelFormat { VAIST_MODEL_UNKNOWN=0,VAIST_MODEL_GGUF=1,VAIST_MODEL_SAFETENSORS=2,VAIST_MODEL_ONNX=3,VAIST_MODEL_OPENVINO=4 } VaistModelFormat;
typedef struct VaistModelInfo{VaistModelFormat format;uint64_t file_size;uint32_t version;uint32_t tensor_count;}VaistModelInfo;
VAIST_API VaistStatus vaist_model_inspect(const char*path,VaistModelInfo*out);
VAIST_API const char *vaist_model_format_string(VaistModelFormat f);
#ifdef __cplusplus
}
#endif
#endif