#include "vaist_model.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint64_t file_size(FILE *f) {
    long end;
    if (fseek(f, 0, SEEK_END) != 0) return 0u;
    end = ftell(f);
    if (end < 0) return 0u;
    if (fseek(f, 0, SEEK_SET) != 0) return 0u;
    return (uint64_t)(unsigned long)end;
}

VAIST_API const char *vaist_model_format_string(VaistModelFormat f) {
    switch (f) { case VAIST_MODEL_GGUF:return "GGUF"; case VAIST_MODEL_SAFETENSORS:return "SafeTensors"; case VAIST_MODEL_ONNX:return "ONNX"; case VAIST_MODEL_OPENVINO:return "OpenVINO"; default:return "unknown"; }
}

VAIST_API VaistStatus vaist_model_inspect(const char *path, VaistModelInfo *out) {
    FILE *f; unsigned char h[16]; size_t got; uint64_t sz;
    if (!path || !out) return VAIST_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    f = fopen(path, "rb"); if (!f) return VAIST_IO_ERROR;
    sz = file_size(f); if (sz == 0u) { fclose(f); return VAIST_MODEL_ERROR; }
    got = fread(h, 1u, sizeof(h), f); fclose(f);
    if (got < 4u) return VAIST_MODEL_ERROR;
    out->file_size = sz; out->tensor_count = 0u;
    if (memcmp(h, "GGUF", 4u) == 0) {
        out->format = VAIST_MODEL_GGUF;
        if (got < 8u) return VAIST_MODEL_ERROR;
        out->version = (uint32_t)h[4] | ((uint32_t)h[5]<<8) | ((uint32_t)h[6]<<16) | ((uint32_t)h[7]<<24);
        if (out->version == 0u || out->version > 3u) return VAIST_MODEL_ERROR;
        return VAIST_OK;
    }
    if (got >= 8u && memcmp(h + 4u, "ORTM", 4u) == 0) { out->format = VAIST_MODEL_ONNX; return VAIST_OK; }
    if (got >= 8u && h[0] == 0x7f && h[1] == 'E' && h[2] == 'L' && h[3] == 'F') { out->format = VAIST_MODEL_OPENVINO; return VAIST_OK; }
    return VAIST_UNSUPPORTED;
}
