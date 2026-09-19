/**
 * \file vaist_model.c
 * \brief Model weight loading: SafeTensors and GGUF format parsers,
 *        tensor reading with dequantization.
 */
#include "vaist_model.h"
#include "vaist_quant.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

static uint64_t file_size(FILE *f) {
    long end;
    if (fseek(f, 0, SEEK_END) != 0) return 0u;
    end = ftell(f);
    if (end < 0) return 0u;
    if (fseek(f, 0, SEEK_SET) != 0) return 0u;
    return (uint64_t)(unsigned long)end;
}

VAIST_API const char *vaist_model_format_string(VaistModelFormat f) {
    switch (f) {
        case VAIST_MODEL_GGUF:   return "GGUF";
        case VAIST_MODEL_SAFETENSORS: return "SafeTensors";
        case VAIST_MODEL_ONNX:   return "ONNX";
        case VAIST_MODEL_OPENVINO: return "OpenVINO";
        default: return "unknown";
    }
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
        out->version = (uint32_t)h[4] | ((uint32_t)h[5] << 8) | ((uint32_t)h[6] << 16) | ((uint32_t)h[7] << 24);
        if (out->version == 0u || out->version > 3u) return VAIST_MODEL_ERROR;
        return VAIST_OK;
    }
    if (got >= 8u) {
        uint64_t hdr_len = (uint64_t)h[0] | ((uint64_t)h[1] << 8) | ((uint64_t)h[2] << 16) |
                           ((uint64_t)h[3] << 24) | ((uint64_t)h[4] << 32) | ((uint64_t)h[5] << 40) |
                           ((uint64_t)h[6] << 48) | ((uint64_t)h[7] << 56);
        if (hdr_len > 0 && hdr_len < sz && hdr_len + 8 <= sz) {
            out->format = VAIST_MODEL_SAFETENSORS;
            out->version = (uint32_t)hdr_len;
            return VAIST_OK;
        }
    }
    return VAIST_UNSUPPORTED;
}

/* ---- Minimal JSON parser for SafeTensors headers ---- */

static int json_find_value(const char *json, const char *key,
                           const char **val_out, size_t *val_len) {
    char pattern[256];
    size_t klen;
    const char *p, *start;
    *val_out = NULL;
    *val_len = 0;
    klen = strlen(key);
    if (klen > 250) return 0;
    pattern[0] = '"';
    memcpy(pattern + 1, key, klen);
    pattern[1 + klen] = '"';
    pattern[2 + klen] = ':';
    pattern[3 + klen] = '\0';
    p = strstr(json, pattern);
    if (!p) return 0;
    p += 3 + klen;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    start = p;
    if (*p == '"') {
        p++;
        while (*p && *p != '"') p++;
        if (*p == '"') { *val_out = start + 1; *val_len = (size_t)(p - start - 1); return 1; }
        return 0;
    } else if (*p == '[') {
        int depth = 1;
        p++;
        start = p;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            if (depth > 0) p++;
        }
        if (depth == 0) { *val_out = start; *val_len = (size_t)(p - start); return 1; }
        return 0;
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
        *val_out = start;
        *val_len = (size_t)(p - start);
        return 1;
    }
}

static int parse_json_uint64(const char *s, size_t len, uint64_t *out) {
    char buf[64];
    if (len > 63) return 0;
    memcpy(buf, s, len);
    buf[len] = '\0';
    char *endptr;
    unsigned long long v = strtoull(buf, &endptr, 10);
    if (endptr == buf) return 0;
    *out = (uint64_t)v;
    return 1;
}

static VaistDType map_safetensors_dtype(const char *dt, size_t len) {
    if (!strncmp(dt, "F32", len > 3 ? 3 : len) && len >= 3) return VAIST_F32;
    if (!strncmp(dt, "F64", len > 3 ? 3 : len) && len >= 3) return VAIST_I32;
    if (!strncmp(dt, "F16", len > 3 ? 3 : len) && len >= 3) return VAIST_F16;
    if (!strncmp(dt, "BF16", len > 4 ? 4 : len) && len >= 4) return VAIST_U8;
    if (!strncmp(dt, "I64", len > 3 ? 3 : len) && len >= 3) return VAIST_I32;
    if (!strncmp(dt, "I32", len > 3 ? 3 : len) && len >= 3) return VAIST_I32;
    if (!strncmp(dt, "I16", len > 3 ? 3 : len) && len >= 3) return VAIST_U8;
    if (!strncmp(dt, "I8", len > 2 ? 2 : len) && len >= 2) return VAIST_I32;
    if (!strncmp(dt, "U8", len > 2 ? 2 : len) && len >= 2) return VAIST_U8;
    if (!strncmp(dt, "BOOL", len > 4 ? 4 : len) && len >= 4) return VAIST_U8;
    return VAIST_F32;
}

VAIST_API VaistStatus vaist_model_load_safetensors(
    const char *path,
    VaistTensorDesc **tensors_out,
    size_t *count,
    size_t *capacity) {

    FILE *f;
    uint64_t header_len = 0, total_size;
    char *header_json = NULL;
    size_t i, num_tensors;
    VaistTensorDesc *out_arr = NULL;
    const char *names_val, *dtypes_val, *shapes_val, *p, *end;
    size_t names_len, dtypes_len, shapes_len;

    if (!path || !tensors_out || !count || !capacity) return VAIST_INVALID_ARGUMENT;
    *tensors_out = NULL; *count = 0; *capacity = 0;

    f = fopen(path, "rb");
    if (!f) return VAIST_IO_ERROR;
    total_size = file_size(f);
    if (total_size < 8) { fclose(f); return VAIST_MODEL_ERROR; }

    {
        unsigned char hdr[8];
        if (fread(hdr, 1, 8, f) < 8) { fclose(f); return VAIST_IO_ERROR; }
        header_len = (uint64_t)hdr[0] | ((uint64_t)hdr[1] << 8) |
                     ((uint64_t)hdr[2] << 16) | ((uint64_t)hdr[3] << 24) |
                     ((uint64_t)hdr[4] << 32) | ((uint64_t)hdr[5] << 40) |
                     ((uint64_t)hdr[6] << 48) | ((uint64_t)hdr[7] << 56);
    }
    if (header_len == 0 || header_len > total_size - 8) { fclose(f); return VAIST_MODEL_ERROR; }

    header_json = (char *)malloc(header_len + 1);
    if (!header_json) { fclose(f); return VAIST_OUT_OF_MEMORY; }
    if (fread(header_json, 1, header_len, f) < header_len) { free(header_json); fclose(f); return VAIST_IO_ERROR; }
    header_json[header_len] = '\0';
    fclose(f);

    /* Parse names array */
    if (!json_find_value(header_json, "names", &names_val, &names_len)) {
        free(header_json);
        return VAIST_MODEL_ERROR;
    }

    /* Count strings in names array */
    num_tensors = 0;
    p = names_val; end = names_val + names_len;
    while (p < end) {
        p = (const char *)memchr(p, '"', end - p);
        if (!p) break;
        p++;
        p = (const char *)memchr(p, '"', end - p);
        if (!p) break;
        p++;
        num_tensors++;
    }
    if (num_tensors == 0) { free(header_json); return VAIST_MODEL_ERROR; }

    out_arr = (VaistTensorDesc *)calloc(num_tensors, sizeof(VaistTensorDesc));
    if (!out_arr) { free(header_json); return VAIST_OUT_OF_MEMORY; }

    /* Parse names */
    p = names_val; end = names_val + names_len;
    for (i = 0; i < num_tensors && p < end; i++) {
        p = (const char *)memchr(p, '"', end - p);
        if (!p || p >= end) break;
        p++;
        const char *name_start = p;
        p = (const char *)memchr(p, '"', end - p);
        if (!p) break;
        size_t name_len = (size_t)(p - name_start);
        if (name_len > 255) name_len = 255;
        memcpy(out_arr[i].name, name_start, name_len);
        out_arr[i].name[name_len] = '\0';
        p++;
        p = (const char *)memchr(p, ',', end - p);
        if (!p) break;
        p++;
    }

    /* Parse dtypes */
    if (json_find_value(header_json, "dtypes", &dtypes_val, &dtypes_len)) {
        p = dtypes_val; end = dtypes_val + dtypes_len;
        for (i = 0; i < num_tensors && p < end; i++) {
            p = (const char *)memchr(p, '"', end - p);
            if (!p) break;
            p++;
            const char *dt_start = p;
            p = (const char *)memchr(p, '"', end - p);
            if (!p) break;
            size_t dt_len = (size_t)(p - dt_start);
            p++;
            out_arr[i].dtype = map_safetensors_dtype(dt_start, dt_len);
        }
    }

    /* Parse shapes */
    if (json_find_value(header_json, "shapes", &shapes_val, &shapes_len)) {
        p = shapes_val; end = shapes_val + shapes_len;
        for (i = 0; i < num_tensors; i++) {
            p = (const char *)memchr(p, '[', end - p);
            if (!p) break;
            p++;
            out_arr[i].rank = 0;
            while (p < end) {
                while (p < end && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
                if (p >= end || *p == ']') break;
                const char *num_start = p;
                while (p < end && *p >= '0' && *p <= '9') p++;
                if (out_arr[i].rank < 8) {
                    if (parse_json_uint64(num_start, (size_t)(p - num_start), &out_arr[i].shape[out_arr[i].rank]))
                        out_arr[i].rank++;
                }
            }
            p = (const char *)memchr(p, ']', end - p);
            if (!p) break;
            p++;
        }
    }

    /* Parse data_offsets */
    {
        char offset_pattern[256];
        size_t olen = strlen("data_offsets");
        offset_pattern[0] = '"';
        memcpy(offset_pattern + 1, "data_offsets", olen);
        offset_pattern[1 + olen] = '"';
        offset_pattern[2 + olen] = ':';
        offset_pattern[3 + olen] = '\0';
        const char *off_start = strstr(header_json, offset_pattern);
        if (off_start) {
            p = off_start + 3 + olen;
            end = header_json + header_len;
            for (i = 0; i < num_tensors && p < end; i++) {
                p = (const char *)memchr(p, '[', end - p);
                if (!p) break;
                p++;
                while (p < end && (*p == ' ' || *p == ',')) p++;
                if (!parse_json_uint64(p, end - p, &out_arr[i].offset)) {
                    const char *ns = p;
                    while (p < end && *p >= '0' && *p <= '9') p++;
                    parse_json_uint64(ns, (size_t)(p - ns), &out_arr[i].offset);
                }
                p = (const char *)memchr(p, ',', end - p);
                if (!p) break;
                p++;
                while (p < end && (*p == ' ')) p++;
                const char *end_start = p;
                while (p < end && *p >= '0' && *p <= '9') p++;
                uint64_t end_val = 0;
                parse_json_uint64(end_start, (size_t)(p - end_start), &end_val);
                out_arr[i].byte_size = end_val - out_arr[i].offset;
                p = (const char *)memchr(p, ']', end - p);
                if (!p) break;
                p++;
            }
        }
    }

    free(header_json);
    *tensors_out = out_arr;
    *count = num_tensors;
    *capacity = num_tensors;
    return VAIST_OK;
}

/* ---- GGUF loader ---- */

typedef enum {
    GGUF_TYPE_F32       = 0,
    GGUF_TYPE_F16       = 1,
    GGUF_TYPE_Q4_0      = 2,
    GGUF_TYPE_Q4_1      = 3,
    GGUF_TYPE_Q5_0      = 6,
    GGUF_TYPE_Q5_1      = 7,
    GGUF_TYPE_Q8_0      = 8,
    GGUF_TYPE_Q2_K      = 10,
    GGUF_TYPE_Q3_K      = 14,
    GGUF_TYPE_Q5_K      = 15,
    GGUF_TYPE_Q6_K      = 16,
    GGUF_TYPE_Q8_1      = 12,
} GgufType;

/**
 * \brief Map GGUF ggml_type to a dtype value for storage in VaistTensorDesc.
 * Quantized types are stored as their raw ggml_type value (offset by
 * GGUF_DTYPE_OFFSET to avoid overlap with VaistDType values 0-6).
 */
#define GGUF_DTYPE_OFFSET 100

static VaistDType map_gguf_type(uint32_t ggml_type, size_t *element_size) {
    switch (ggml_type) {
        case GGUF_TYPE_F32:  *element_size = sizeof(float);     return VAIST_F32;
        case GGUF_TYPE_F16:  *element_size = sizeof(uint16_t);   return VAIST_F16;
        case GGUF_TYPE_Q4_0: *element_size = sizeof(vaist_block_q4_0); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q4_1: *element_size = sizeof(vaist_block_q4_1); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q5_0: *element_size = sizeof(vaist_block_q5_0); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q5_1: *element_size = sizeof(vaist_block_q5_1); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q8_0: *element_size = sizeof(vaist_block_q8_0); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q8_1: *element_size = sizeof(vaist_block_q8_1); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q2_K: *element_size = sizeof(vaist_block_q2_K); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q3_K: *element_size = sizeof(vaist_block_q3_K); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q5_K: *element_size = sizeof(vaist_block_q5_K); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        case GGUF_TYPE_Q6_K: *element_size = sizeof(vaist_block_q6_K); return (VaistDType)(GGUF_DTYPE_OFFSET + ggml_type);
        default: *element_size = sizeof(float); return VAIST_F32;
    }
}

static int gguf_read_u64(FILE *f, uint64_t *out) {
    unsigned char tmp[8];
    size_t i;
    if (fread(tmp, 1, 8, f) < 8) return 0;
    *out = 0;
    for (i = 0; i < 8; i++) *out |= ((uint64_t)tmp[i]) << (8 * i);
    return 1;
}

static int gguf_read_u32(FILE *f, uint32_t *out) {
    unsigned char tmp[4];
    if (fread(tmp, 1, 4, f) < 4) return 0;
    *out = (uint32_t)tmp[0] | ((uint32_t)tmp[1] << 8) |
           ((uint32_t)tmp[2] << 16) | ((uint32_t)tmp[3] << 24);
    return 1;
}

VAIST_API VaistStatus vaist_model_load_gguf(
    const char *path,
    VaistTensorDesc **tensors_out,
    size_t *count,
    size_t *capacity) {

    FILE *f;
    unsigned char magic[4];
    uint32_t version;
    uint64_t tc, metadata_kv_count;
    size_t i;
    VaistTensorDesc *out_arr;

    if (!path || !tensors_out || !count || !capacity) return VAIST_INVALID_ARGUMENT;
    *tensors_out = NULL; *count = 0; *capacity = 0;

    f = fopen(path, "rb");
    if (!f) return VAIST_IO_ERROR;

    if (fread(magic, 1, 4, f) < 4) { fclose(f); return VAIST_MODEL_ERROR; }
    if (memcmp(magic, "GGUF", 4) != 0) { fclose(f); return VAIST_MODEL_ERROR; }
    if (!gguf_read_u32(f, &version)) { fclose(f); return VAIST_MODEL_ERROR; }
    if (version == 0 || version > 3) { fclose(f); return VAIST_MODEL_ERROR; }
    if (!gguf_read_u64(f, &tc)) { fclose(f); return VAIST_MODEL_ERROR; }
    if (!gguf_read_u64(f, &metadata_kv_count)) { fclose(f); return VAIST_MODEL_ERROR; }

    /* Skip metadata */
    for (i = 0; i < metadata_kv_count; i++) {
        char key[1024];
        uint64_t key_len;
        uint32_t val_type;
        if (!gguf_read_u64(f, &key_len)) { fclose(f); return VAIST_IO_ERROR; }
        if (key_len > 1023) { fclose(f); return VAIST_MODEL_ERROR; }
        if (fread(key, 1, key_len, f) < key_len) { fclose(f); return VAIST_IO_ERROR; }
        if (!gguf_read_u32(f, &val_type)) { fclose(f); return VAIST_IO_ERROR; }
        switch (val_type) {
            case 0u: { uint64_t slen; if (!gguf_read_u64(f,&slen)) { fclose(f);return VAIST_IO_ERROR; }
                       if (slen > 0) { if (fseek(f, (long)slen, SEEK_CUR) != 0) { fclose(f);return VAIST_IO_ERROR; } } break; }
            case 1u: case 2u: case 3u: case 4u: if (fseek(f, 8, SEEK_CUR) != 0) { fclose(f);return VAIST_IO_ERROR; } break;
            case 5u: { uint64_t dcount; if (!gguf_read_u64(f,&dcount)) { fclose(f);return VAIST_IO_ERROR; }
                       for (size_t j = 0; j < dcount; j++) {
                           uint64_t knlen; uint32_t vtype;
                           if (!gguf_read_u64(f,&knlen)) { fclose(f);return VAIST_IO_ERROR; }
                           if (fseek(f,(long)knlen,SEEK_CUR)!=0){fclose(f);return VAIST_IO_ERROR;}
                           if (!gguf_read_u32(f,&vtype)) { fclose(f);return VAIST_IO_ERROR; }
                           switch (vtype) {
                               case 0u: { uint64_t slen2; if (!gguf_read_u64(f,&slen2)){fclose(f);return VAIST_IO_ERROR;}
                                        if (fseek(f,(long)slen2,SEEK_CUR)!=0){fclose(f);return VAIST_IO_ERROR;} break; }
                               case 1u: case 2u: case 3u: case 4u: if (fseek(f,8,SEEK_CUR)!=0){fclose(f);return VAIST_IO_ERROR;} break;
                               default: fclose(f); return VAIST_MODEL_ERROR;
                           }
                       } break; }
            case 6u: { uint32_t atype; uint64_t acount;
                       if (!gguf_read_u32(f,&atype)) { fclose(f);return VAIST_IO_ERROR; }
                       if (!gguf_read_u64(f,&acount)) { fclose(f);return VAIST_IO_ERROR; }
                       switch (atype) {
                           case 0u: for (size_t j=0;j<acount;j++){uint64_t slen2;if(!gguf_read_u64(f,&slen2)){fclose(f);return VAIST_IO_ERROR;}
                                      if (fseek(f,(long)slen2,SEEK_CUR)!=0){fclose(f);return VAIST_IO_ERROR;}} break;
                           case 1u: case 2u: case 3u: case 4u: { size_t elem_sz = (atype == 4u) ? 1 : 8;
                                      if (fseek(f,(long)(acount*elem_sz),SEEK_CUR)!=0){fclose(f);return VAIST_IO_ERROR;} break; }
                           default: fclose(f); return VAIST_MODEL_ERROR;
                       } break; }
            default: fclose(f); return VAIST_MODEL_ERROR;
        }
    }

    (void)ftell(f);
    if (tc == 0) { fclose(f); *tensors_out = NULL; *count = 0; *capacity = 0; return VAIST_OK; }

    out_arr = (VaistTensorDesc *)calloc(tc, sizeof(VaistTensorDesc));
    if (!out_arr) { fclose(f); return VAIST_OUT_OF_MEMORY; }

    for (i = 0; i < tc; i++) {
        VaistTensorDesc *td = &out_arr[i];
        uint64_t name_len;
        uint32_t n_dimensions, ggml_type;
        size_t elem_sz;
        uint32_t d;

        if (!gguf_read_u64(f, &name_len)) { free(out_arr); fclose(f); return VAIST_IO_ERROR; }
        if (name_len > 255) { free(out_arr); fclose(f); return VAIST_MODEL_ERROR; }
        if (fread(td->name, 1, name_len, f) < name_len) { free(out_arr); fclose(f); return VAIST_IO_ERROR; }
        td->name[name_len] = '\0';

        if (!gguf_read_u32(f, &n_dimensions)) { free(out_arr); fclose(f); return VAIST_IO_ERROR; }
        td->rank = n_dimensions;
        for (d = 0; d < n_dimensions && d < 8; d++) {
            uint64_t dim_val;
            if (!gguf_read_u64(f, &dim_val)) { free(out_arr); fclose(f); return VAIST_IO_ERROR; }
            td->shape[d] = dim_val;
        }

        if (!gguf_read_u32(f, &ggml_type)) { free(out_arr); fclose(f); return VAIST_IO_ERROR; }
        td->dtype = map_gguf_type(ggml_type, &elem_sz);

        {
            uint64_t numel = 1;
            for (d = 0; d < n_dimensions && d < 8; d++) numel *= td->shape[d];
            if (ggml_type == GGUF_TYPE_F32 || ggml_type == GGUF_TYPE_F16) {
                td->byte_size = numel * elem_sz;
            } else {
                VaistQuantType qt;
                switch (ggml_type) {
                    case GGUF_TYPE_Q4_0:  qt = VAIST_Q4_0;  break;
                    case GGUF_TYPE_Q4_1:  qt = VAIST_Q4_1;  break;
                    case GGUF_TYPE_Q5_0:  qt = VAIST_Q5_0;  break;
                    case GGUF_TYPE_Q5_1:  qt = VAIST_Q5_1;  break;
                    case GGUF_TYPE_Q8_0:  qt = VAIST_Q8_0;  break;
                    case GGUF_TYPE_Q8_1:  qt = VAIST_Q8_1;  break;
                    case GGUF_TYPE_Q2_K:  qt = VAIST_Q2_K;  break;
                    case GGUF_TYPE_Q3_K:  qt = VAIST_Q3_K;  break;
                    case GGUF_TYPE_Q5_K:  qt = VAIST_Q5_K;  break;
                    case GGUF_TYPE_Q6_K:  qt = VAIST_Q6_K;  break;
                    default: qt = VAIST_Q8_0; break;
                }
                size_t block_sz = vaist_quant_block_size(qt);
                size_t block_bytes = vaist_quant_block_bytes(qt);
                uint64_t nblocks = (numel + block_sz - 1) / block_sz;
                td->byte_size = nblocks * block_bytes;
            }
        }

        uint64_t offset;
        if (!gguf_read_u64(f, &offset)) { free(out_arr); fclose(f); return VAIST_IO_ERROR; }
        td->offset = offset;
    }

    fclose(f);
    *tensors_out = out_arr;
    *count = (size_t)tc;
    *capacity = (size_t)tc;
    return VAIST_OK;
}

/* ---- Tensor reading ---- */

static uint64_t tensor_numel(const VaistTensorDesc *desc) {
    uint64_t n = 1;
    uint32_t i;
    for (i = 0; i < desc->rank && i < 8; i++) n *= desc->shape[i];
    return n;
}

VAIST_API VaistStatus vaist_model_tensor_read(
    const char *path,
    const VaistTensorDesc *desc,
    float *dst,
    size_t dst_cap) {

    FILE *f;
    uint64_t numel;
    VaistStatus st;

    if (!path || !desc || !dst || dst_cap == 0) return VAIST_INVALID_ARGUMENT;
    numel = tensor_numel(desc);
    if (numel == 0) return VAIST_INVALID_ARGUMENT;
    if (dst_cap < numel) return VAIST_INVALID_ARGUMENT;

    f = fopen(path, "rb");
    if (!f) return VAIST_IO_ERROR;
    if (fseek(f, (long)desc->offset, SEEK_SET) != 0) { fclose(f); return VAIST_IO_ERROR; }

    switch (desc->dtype) {
        case VAIST_F32:
            if (fread(dst, sizeof(float), numel, f) < numel) { fclose(f); return VAIST_IO_ERROR; }
            fclose(f); return VAIST_OK;
        case VAIST_F64: {
            double *tmp = (double *)malloc(numel * sizeof(double));
            uint64_t i;
            if (!tmp) { fclose(f); return VAIST_OUT_OF_MEMORY; }
            if (fread(tmp, sizeof(double), numel, f) < numel) { free(tmp); fclose(f); return VAIST_IO_ERROR; }
            for (i = 0; i < numel; i++) dst[i] = (float)tmp[i];
            free(tmp); fclose(f); return VAIST_OK;
        }
        case VAIST_I32: {
            int32_t *tmp = (int32_t *)malloc(numel * sizeof(int32_t));
            uint64_t i;
            if (!tmp) { fclose(f); return VAIST_OUT_OF_MEMORY; }
            if (fread(tmp, sizeof(int32_t), numel, f) < numel) { free(tmp); fclose(f); return VAIST_IO_ERROR; }
            for (i = 0; i < numel; i++) dst[i] = (float)tmp[i];
            free(tmp); fclose(f); return VAIST_OK;
        }
        case VAIST_I8: {
            int8_t *tmp = (int8_t *)malloc(numel * sizeof(int8_t));
            uint64_t i;
            if (!tmp) { fclose(f); return VAIST_OUT_OF_MEMORY; }
            if (fread(tmp, sizeof(int8_t), numel, f) < numel) { free(tmp); fclose(f); return VAIST_IO_ERROR; }
            for (i = 0; i < numel; i++) dst[i] = (float)tmp[i];
            free(tmp); fclose(f); return VAIST_OK;
        }
        case VAIST_F16: {
            uint16_t *half_data = (uint16_t *)malloc(numel * sizeof(uint16_t));
            uint64_t i;
            if (!half_data) { fclose(f); return VAIST_OUT_OF_MEMORY; }
            if (fread(half_data, sizeof(uint16_t), numel, f) < numel) { free(half_data); fclose(f); return VAIST_IO_ERROR; }
            for (i = 0; i < numel; i++) {
                uint16_t h = half_data[i];
                uint32_t sign = ((uint32_t)h >> 15) & 1u;
                uint32_t exp = (h >> 10) & 0x1fu;
                uint32_t mant = h & 0x3ffu;
                uint32_t fv;
                if (exp == 0u) {
                    if (mant == 0u) { fv = (sign << 31); }
                    else { uint32_t m = mant; int e = 1; while ((m & 0x400u) == 0u) { m <<= 1; e--; } m &= ~0x400u; fv = (sign << 31) | (((uint32_t)(127 - 15 - e)) << 23) | m; }
                } else if (exp == 0x1fu) { fv = (sign << 31) | 0x7f800000u | (mant ? 0x400000u : 0u); }
                else { fv = (sign << 31) | (((uint32_t)exp + (127u - 15u)) << 23) | (mant << 13); }
                memcpy(&dst[i], &fv, 4);
            }
            free(half_data); fclose(f); return VAIST_OK;
        }
        case VAIST_U8: {
            uint8_t *tmp = (uint8_t *)malloc(numel * sizeof(uint8_t));
            uint64_t i;
            if (!tmp) { fclose(f); return VAIST_OUT_OF_MEMORY; }
            if (fread(tmp, sizeof(uint8_t), numel, f) < numel) { free(tmp); fclose(f); return VAIST_IO_ERROR; }
            for (i = 0; i < numel; i++) dst[i] = (float)tmp[i];
            free(tmp); fclose(f); return VAIST_OK;
        }
        default: {
            uint32_t raw;
            VaistQuantType qt;
            size_t block_bytes, block_size;
            uint64_t nblocks;
            uint8_t *qdata;
            if ((uint32_t)desc->dtype < GGUF_DTYPE_OFFSET) return VAIST_UNSUPPORTED;
            raw = (uint32_t)desc->dtype - GGUF_DTYPE_OFFSET;
            qt = (VaistQuantType)raw;
            block_bytes = vaist_quant_block_bytes(qt);
            block_size = vaist_quant_block_size(qt);
            if (block_bytes == 0 || block_size == 0) return VAIST_UNSUPPORTED;
            nblocks = (numel + block_size - 1) / block_size;
            qdata = (uint8_t *)malloc(nblocks * block_bytes);
            if (!qdata) { fclose(f); return VAIST_OUT_OF_MEMORY; }
            if (fread(qdata, 1, desc->byte_size, f) < desc->byte_size) { free(qdata); fclose(f); return VAIST_IO_ERROR; }
            st = vaist_dequantize_f32(qt, qdata, nblocks * block_bytes, dst, numel);
            free(qdata); fclose(f); return st;
        }
    }
}

VAIST_API VaistStatus vaist_model_tensor_list(
    VaistTensorDesc **tensors,
    size_t count) {

    size_t i;
    if (!tensors || count == 0) return VAIST_INVALID_ARGUMENT;
    for (i = 0; i < count; i++) {
        VaistTensorDesc *td = tensors[i];
        printf("  [%lu] %s: dtype=%u rank=%u bytes=%llu offset=%llu shape=[",
               (unsigned long)i, td->name, (unsigned)td->dtype, td->rank,
               (unsigned long long)td->byte_size, (unsigned long long)td->offset);
        {
            uint32_t j;
            for (j = 0; j < td->rank && j < 8; j++) printf("%s%llu", j ? "," : "", (unsigned long long)td->shape[j]);
        }
        printf("]\n");
    }
    return VAIST_OK;
}
