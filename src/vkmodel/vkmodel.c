/**
 * \file vkmodel.c
 * \brief VKModel implementation: GGUF host parser + per-tensor GPU upload.
 *
 * Design:
 *  - Host parsing is pure C99 + FILE* I/O (no Vulkan). A little-endian byte
 *    reader walks the header, the metadata key-value pairs (all 13 GGUF value
 *    types incl. nested arrays, stored as typed heap copies), and the tensor
 *    infos. Tensor data is never loaded whole-file: each tensor is read in
 *    bounded VKMODEL_STREAM_CHUNK slices via fseek/fread and uploaded with
 *    vkr_upload() into a device buffer obtained from vkr_malloc().
 *  - One command pool + one command buffer are created at load and reused
 *    across every tensor upload (vkr_upload() resets it after each submit).
 *    The queue is resolved as family 0 / index 0, the stack-wide convention
 *    used by every harness when creating the runtime.
 *  - Tensor byte sizes follow the stack's canonical (classic ggml) block
 *    layout — the same layout VKQuant dequantizes (Q8_0 = 36 B f32 scale,
 *    Q4_0 = 20 B f32 scale, K-format super-blocks unchanged). The gguf
 *    per-tensor offset field is used verbatim to locate data.
 */
#include "vkmodel_internal.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define VKMODEL_FSEEK _fseeki64
#else
#define VKMODEL_FSEEK fseeko
#endif

/* ===========================================================================
 * ggml_type tables (elements per block, bytes per block)
 *
 * Enum values are the GGUF-spec ggml_type numbers (Q5_0=6, Q5_1=7, Q8_0=8,
 * Q8_1=9, Q2_K=10 ... Q6_K=14). Byte sizes match the stack's VKQuant block
 * layouts for the formats it dequantizes and the classic ggml layouts for the
 * rest. Removed/unknown enum slots default to 0 (treated as an error).
 * ========================================================================== */

static const uint32_t vkmodel_blck_elems_tbl[] = {
    [0]  = 1,    /* F32     */
    [1]  = 1,    /* F16     */
    [2]  = 32,   /* Q4_0    */
    [3]  = 32,   /* Q4_1    */
    [6]  = 32,   /* Q5_0    */
    [7]  = 32,   /* Q5_1    */
    [8]  = 32,   /* Q8_0    */
    [9]  = 32,   /* Q8_1    */
    [10] = 256,  /* Q2_K    */
    [11] = 256,  /* Q3_K    */
    [12] = 256,  /* Q4_K    */
    [13] = 256,  /* Q5_K    */
    [14] = 256,  /* Q6_K    */
    [15] = 256,  /* Q8_K    */
    [16] = 256,  /* IQ2_XXS */
    [17] = 256,  /* IQ2_XS  */
    [18] = 256,  /* IQ3_XXS */
    [19] = 256,  /* IQ1_S   */
    [20] = 32,   /* IQ4_NL  */
    [21] = 256,  /* IQ3_S   */
    [22] = 256,  /* IQ2_S   */
    [23] = 256,  /* IQ4_XS  */
    [24] = 1,    /* I8      */
    [25] = 1,    /* I16     */
    [26] = 1,    /* I32     */
    [27] = 1,    /* I64     */
    [28] = 1,    /* F64     */
    [29] = 256,  /* IQ1_M   */
    [30] = 1,    /* BF16    */
    [34] = 256,  /* TQ1_0   */
    [35] = 256,  /* TQ2_0   */
    [39] = 32,   /* MXFP4   */
    [40] = 64,   /* NVFP4   */
    [41] = 128,  /* Q1_0    */
    [42] = 64,   /* Q2_0    */
};

static const uint32_t vkmodel_blck_bytes_tbl[] = {
    [0]  = 4,     /* F32     */
    [1]  = 2,     /* F16     */
    [2]  = 20,    /* Q4_0    (f32 scale + 16 nibbles)                 */
    [3]  = 24,    /* Q4_1    (f32 d + f32 m + 16 nibbles)             */
    [6]  = 24,    /* Q5_0    (f32 d + qh + 16 nibbles)                */
    [7]  = 28,    /* Q5_1    (f32 d + f32 m + qh + 16 nibbles)        */
    [8]  = 36,    /* Q8_0    (f32 scale + 32 x int8)                  */
    [9]  = 40,    /* Q8_1    (f32 d + f32 s + 32 x int8)              */
    [10] = 84,    /* Q2_K    */
    [11] = 110,   /* Q3_K    */
    [12] = 144,   /* Q4_K    (f16 d/dmin + 12 scales + 128 nibbles)   */
    [13] = 176,   /* Q5_K    */
    [14] = 210,   /* Q6_K    (ql + qh + 16 int8 scales + f16 d)       */
    [15] = 258,   /* Q8_K    */
    [16] = 42,    /* IQ2_XXS */
    [17] = 42,    /* IQ2_XS  */
    [18] = 82,    /* IQ3_XXS */
    [19] = 44,    /* IQ1_S   */
    [20] = 18,    /* IQ4_NL  */
    [21] = 110,   /* IQ3_S   */
    [22] = 50,    /* IQ2_S   */
    [23] = 136,   /* IQ4_XS  (f16 d + u16 scales_h + scales_l + qs)   */
    [24] = 1,     /* I8      */
    [25] = 2,     /* I16     */
    [26] = 4,     /* I32     */
    [27] = 8,     /* I64     */
    [28] = 8,     /* F64     */
    [29] = 56,    /* IQ1_M   */
    [30] = 2,     /* BF16    */
    [34] = 54,    /* TQ1_0   */
    [35] = 66,    /* TQ2_0   */
    [39] = 17,    /* MXFP4   */
    [40] = 36,    /* NVFP4   */
    [41] = 18,    /* Q1_0    */
    [42] = 18,    /* Q2_0    */
};

static uint32_t vkmodel_tbl_entry(const uint32_t* tbl, uint32_t count,
                                  uint32_t type)
{
    return type < count ? tbl[type] : 0;
}

static uint32_t vkmodel_blck_elems_of(uint32_t type)
{
    return vkmodel_tbl_entry(vkmodel_blck_elems_tbl,
                             (uint32_t)(sizeof(vkmodel_blck_elems_tbl) /
                                        sizeof(vkmodel_blck_elems_tbl[0])),
                             type);
}

static uint32_t vkmodel_blck_bytes_of(uint32_t type)
{
    return vkmodel_tbl_entry(vkmodel_blck_bytes_tbl,
                             (uint32_t)(sizeof(vkmodel_blck_bytes_tbl) /
                                        sizeof(vkmodel_blck_bytes_tbl[0])),
                             type);
}

static VkDeviceSize vkmodel_tensor_bytes(uint32_t type, uint64_t nelems)
{
    uint32_t blck = vkmodel_blck_elems_of(type);
    uint32_t bsz  = vkmodel_blck_bytes_of(type);
    if (blck == 0 || bsz == 0) return 0;
    uint64_t blocks = (nelems + blck - 1) / blck;
    return (VkDeviceSize)blocks * (VkDeviceSize)bsz;
}

static uint64_t vkmodel_align_up(uint64_t value, uint64_t align)
{
    if (align == 0) return value;
    uint64_t rem = value % align;
    return rem == 0 ? value : value + (align - rem);
}

/* ===========================================================================
 * Little-endian file reader
 * ========================================================================== */

typedef struct {
    FILE    *fp;
    uint64_t pos;   /**< Sequential read position (parse phase only). */
} vkmodel_reader_t;

static VkResult vkmodel_rd_fill(vkmodel_reader_t* r, void* dst, uint64_t n)
{
    if (n > (uint64_t)(size_t)-1) return VK_ERROR_INITIALIZATION_FAILED;
    if (fread(dst, 1, (size_t)n, r->fp) != n) return VK_ERROR_INITIALIZATION_FAILED;
    r->pos += n;
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_u16(vkmodel_reader_t* r, uint16_t* out)
{
    uint8_t b[2];
    VkResult res = vkmodel_rd_fill(r, b, 2);
    if (res != VK_SUCCESS) return res;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_u32(vkmodel_reader_t* r, uint32_t* out)
{
    uint8_t b[4];
    VkResult res = vkmodel_rd_fill(r, b, 4);
    if (res != VK_SUCCESS) return res;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_u64(vkmodel_reader_t* r, uint64_t* out)
{
    uint8_t b[8];
    VkResult res = vkmodel_rd_fill(r, b, 8);
    if (res != VK_SUCCESS) return res;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    *out = v;
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_f32(vkmodel_reader_t* r, float* out)
{
    uint32_t u;
    VkResult res = vkmodel_rd_u32(r, &u);
    if (res != VK_SUCCESS) return res;
    memcpy(out, &u, sizeof(u));
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_f64(vkmodel_reader_t* r, double* out)
{
    uint64_t u;
    VkResult res = vkmodel_rd_u64(r, &u);
    if (res != VK_SUCCESS) return res;
    memcpy(out, &u, sizeof(u));
    return VK_SUCCESS;
}

static VkResult vkmodel_rd_str(vkmodel_reader_t* r, char** out)
{
    uint64_t len = 0;
    VkResult res = vkmodel_rd_u64(r, &len);
    if (res != VK_SUCCESS) return res;
    if (len > VKMODEL_MAX_STRING) return VK_ERROR_INITIALIZATION_FAILED;

    char* s = (char*)malloc((size_t)len + 1);
    if (!s) return VK_ERROR_OUT_OF_HOST_MEMORY;
    res = vkmodel_rd_fill(r, s, len);
    if (res != VK_SUCCESS) {
        free(s);
        return res;
    }
    s[len] = '\0';
    *out = s;
    return VK_SUCCESS;
}

/* ===========================================================================
 * Metadata value parsing
 * ========================================================================== */

void vkmodel_value_free(VkModelValue* value)
{
    if (!value) return;
    switch (value->type) {
    case VKMODEL_VAL_STRING:
        free(value->v.str);
        break;
    case VKMODEL_VAL_ARRAY:
        for (uint64_t i = 0; i < value->v.array.count; i++) {
            vkmodel_value_free(&value->v.array.elems[i]);
        }
        free(value->v.array.elems);
        break;
    default:
        break;
    }
    memset(value, 0, sizeof(*value));
}

static VkResult vkmodel_parse_value(vkmodel_reader_t* r, uint32_t type,
                                    VkModelValue* out)
{
    memset(out, 0, sizeof(*out));
    out->type = type;

    switch (type) {
    case VKMODEL_VAL_UINT8: {
        uint8_t v;
        VkResult res = vkmodel_rd_fill(r, &v, 1);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT8: {
        uint8_t b;
        VkResult res = vkmodel_rd_fill(r, &b, 1);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = (uint64_t)(int64_t)(int8_t)b;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_UINT16: {
        uint16_t v;
        VkResult res = vkmodel_rd_u16(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT16: {
        uint16_t u;
        VkResult res = vkmodel_rd_u16(r, &u);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = (uint64_t)(int64_t)(int16_t)u;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_UINT32: {
        uint32_t v;
        VkResult res = vkmodel_rd_u32(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT32: {
        uint32_t u;
        VkResult res = vkmodel_rd_u32(r, &u);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = (uint64_t)(int64_t)(int32_t)u;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_UINT64: {
        uint64_t v;
        VkResult res = vkmodel_rd_u64(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_INT64: {
        uint64_t u;
        VkResult res = vkmodel_rd_u64(r, &u);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = u; /* two's complement preserved */
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_FLOAT32: {
        float v;
        VkResult res = vkmodel_rd_f32(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.f64 = (double)v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_FLOAT64: {
        double v;
        VkResult res = vkmodel_rd_f64(r, &v);
        if (res != VK_SUCCESS) return res;
        out->v.f64 = v;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_BOOL: {
        uint8_t b;
        VkResult res = vkmodel_rd_fill(r, &b, 1);
        if (res != VK_SUCCESS) return res;
        out->v.u64 = b ? 1 : 0;
        return VK_SUCCESS;
    }
    case VKMODEL_VAL_STRING:
        return vkmodel_rd_str(r, &out->v.str);
    case VKMODEL_VAL_ARRAY: {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        VkResult res = vkmodel_rd_u32(r, &elem_type);
        if (res != VK_SUCCESS) return res;
        res = vkmodel_rd_u64(r, &count);
        if (res != VK_SUCCESS) return res;
        if (count > (1u << 24)) return VK_ERROR_INITIALIZATION_FAILED;

        out->v.array.elem_type = elem_type;
        out->v.array.count = count;
        if (count == 0) {
            out->v.array.elems = NULL;
            return VK_SUCCESS;
        }
        out->v.array.elems =
            (VkModelValue*)calloc((size_t)count, sizeof(VkModelValue));
        if (!out->v.array.elems) return VK_ERROR_OUT_OF_HOST_MEMORY;
        for (uint64_t i = 0; i < count; i++) {
            res = vkmodel_parse_value(r, elem_type,
                                      &out->v.array.elems[i]);
            if (res != VK_SUCCESS) {
                vkmodel_value_free(out);
                return res;
            }
        }
        return VK_SUCCESS;
    }
    default:
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

/* ===========================================================================
 * Section parsing
 * ========================================================================== */

static VkResult vkmodel_parse_header(vkmodel_reader_t* r, VkModel* m)
{
    uint32_t magic = 0;
    VkResult res = vkmodel_rd_u32(r, &magic);
    if (res != VK_SUCCESS) return res;
    if (magic != VKMODEL_MAGIC_LE) return VK_ERROR_INITIALIZATION_FAILED;

    uint32_t version = 0;
    res = vkmodel_rd_u32(r, &version);
    if (res != VK_SUCCESS) return res;
    if (version < 2 || version > VKMODEL_GGUF_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint64_t tc = 0, kc = 0;
    res = vkmodel_rd_u64(r, &tc);
    if (res != VK_SUCCESS) return res;
    res = vkmodel_rd_u64(r, &kc);
    if (res != VK_SUCCESS) return res;
    if (tc > (1u << 20) || kc > (1u << 20)) return VK_ERROR_INITIALIZATION_FAILED;

    m->tensor_count = (uint32_t)tc;
    m->kv_count = (uint32_t)kc;

    if (m->tensor_count > 0) {
        m->tensors = (VkModelTensor*)calloc(m->tensor_count,
                                            sizeof(VkModelTensor));
        if (!m->tensors) return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (m->kv_count > 0) {
        m->kv = (VkModelKV*)calloc(m->kv_count, sizeof(VkModelKV));
        if (!m->kv) return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return VK_SUCCESS;
}

static VkResult vkmodel_parse_metadata(vkmodel_reader_t* r, VkModel* m)
{
    for (uint64_t i = 0; i < m->kv_count; i++) {
        char* key = NULL;
        VkResult res = vkmodel_rd_str(r, &key);
        if (res != VK_SUCCESS) return res;

        uint32_t vtype = 0;
        res = vkmodel_rd_u32(r, &vtype);
        if (res != VK_SUCCESS) {
            free(key);
            return res;
        }

        VkModelValue val;
        res = vkmodel_parse_value(r, vtype, &val);
        if (res != VK_SUCCESS) {
            free(key);
            return res;
        }

        m->kv[i].key = key;
        m->kv[i].value = val;

        if (vtype == VKMODEL_VAL_UINT32 || vtype == VKMODEL_VAL_UINT64) {
            if (strcmp(key, "general.alignment") == 0) {
                uint64_t a = val.v.u64;
                if (a < 8 || (a % 8) != 0) return VK_ERROR_INITIALIZATION_FAILED;
                m->alignment = a;
            }
        }
    }
    return VK_SUCCESS;
}

static VkResult vkmodel_parse_tensors(vkmodel_reader_t* r, VkModel* m)
{
    for (uint64_t i = 0; i < m->tensor_count; i++) {
        VkModelTensor* t = &m->tensors[i];

        VkResult res = vkmodel_rd_str(r, &t->name);
        if (res != VK_SUCCESS) return res;

        uint32_t n_dims = 0;
        res = vkmodel_rd_u32(r, &n_dims);
        if (res != VK_SUCCESS) return res;
        if (n_dims == 0 || n_dims > VKMODEL_MAX_DIMS)
            return VK_ERROR_INITIALIZATION_FAILED;
        t->n_dims = n_dims;

        t->nelems = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = 0;
            res = vkmodel_rd_u64(r, &dim);
            if (res != VK_SUCCESS) return res;
            if (dim == 0) return VK_ERROR_INITIALIZATION_FAILED;
            t->dims[d] = dim;
            t->nelems *= dim;
        }

        uint32_t dtype = 0;
        res = vkmodel_rd_u32(r, &dtype);
        if (res != VK_SUCCESS) return res;
        t->dtype = dtype;

        uint64_t offset = 0;
        res = vkmodel_rd_u64(r, &offset);
        if (res != VK_SUCCESS) return res;
        t->offset = offset;

        t->size = vkmodel_tensor_bytes(dtype, t->nelems);
        if (t->size == 0) return VK_ERROR_INITIALIZATION_FAILED;
    }

    m->data_offset = vkmodel_align_up(r->pos, m->alignment);
    return VK_SUCCESS;
}

/* ===========================================================================
 * Tensor data upload (streamed)
 * ========================================================================== */

static VkResult vkmodel_upload_tensor(VkModel* m, vkmodel_reader_t* r,
                                      uint32_t idx)
{
    VkModelTensor* t = &m->tensors[idx];
    if (t->size == 0) return VK_SUCCESS;

    VkResult res = vkr_malloc(m->rt, t->size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &t->buffer, &t->memory);
    if (res != VK_SUCCESS) return res;

    uint64_t file_off = m->data_offset + t->offset;
    if (VKMODEL_FSEEK(r->fp, (long long)file_off, SEEK_SET) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    void* chunk = (void*)malloc(VKMODEL_STREAM_CHUNK);
    if (!chunk) return VK_ERROR_OUT_OF_HOST_MEMORY;

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(vkr_get_device(m->rt), 0, 0, &queue);
    if (queue == VK_NULL_HANDLE) {
        free(chunk);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint64_t remaining = t->size;
    VkDeviceSize upload_off = 0;
    while (remaining > 0) {
        uint64_t n = remaining < (uint64_t)VKMODEL_STREAM_CHUNK
                         ? remaining
                         : (uint64_t)VKMODEL_STREAM_CHUNK;
        if (fread(chunk, 1, (size_t)n, r->fp) != n) {
            free(chunk);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        res = vkr_upload(m->rt, m->upload_cmd, queue, chunk, t->buffer,
                         upload_off, (VkDeviceSize)n);
        if (res != VK_SUCCESS) {
            free(chunk);
            return res;
        }
        remaining -= n;
        upload_off += (VkDeviceSize)n;
    }

    free(chunk);
    return VK_SUCCESS;
}

/* ===========================================================================
 * Public API: lifecycle
 * ========================================================================== */

VkResult vkmodel_load(VkRuntime* rt, const char* path, VkModel** pModel)
{
    if (!rt || !path || !pModel) return VK_ERROR_INITIALIZATION_FAILED;
    *pModel = NULL;

    VkModel* m = (VkModel*)calloc(1, sizeof(VkModel));
    if (!m) return VK_ERROR_OUT_OF_HOST_MEMORY;
    m->rt = rt;
    m->alignment = VKMODEL_DEFAULT_ALIGN;
    m->upload_pool = VK_NULL_HANDLE;
    m->upload_cmd = VK_NULL_HANDLE;

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        free(m);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    vkmodel_reader_t reader;
    memset(&reader, 0, sizeof(reader));
    reader.fp = fp;

    VkResult res = vkmodel_parse_header(&reader, m);
    if (res == VK_SUCCESS) res = vkmodel_parse_metadata(&reader, m);
    if (res == VK_SUCCESS) res = vkmodel_parse_tensors(&reader, m);
    if (res != VK_SUCCESS) {
        fclose(fp);
        vkmodel_destroy(m);
        return res;
    }

    res = vkr_create_command_pool(rt, 0, &m->upload_pool);
    if (res != VK_SUCCESS) {
        fclose(fp);
        vkmodel_destroy(m);
        return res;
    }
    {
        VkCommandBufferAllocateInfo ai;
        memset(&ai, 0, sizeof(ai));
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = m->upload_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        res = vkAllocateCommandBuffers(vkr_get_device(rt), &ai, &m->upload_cmd);
        if (res != VK_SUCCESS) {
            fclose(fp);
            vkmodel_destroy(m);
            return res;
        }
    }

    for (uint32_t i = 0; i < m->tensor_count; i++) {
        res = vkmodel_upload_tensor(m, &reader, i);
        if (res != VK_SUCCESS) {
            fclose(fp);
            vkmodel_destroy(m);
            return res;
        }
    }

    fclose(fp);
    *pModel = m;
    return VK_SUCCESS;
}

void vkmodel_destroy(VkModel* m)
{
    if (!m) return;

    if (m->upload_cmd != VK_NULL_HANDLE && m->upload_pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(vkr_get_device(m->rt), m->upload_pool, 1,
                             &m->upload_cmd);
    }
    if (m->upload_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vkr_get_device(m->rt), m->upload_pool, NULL);
    }

    for (uint32_t i = 0; i < m->kv_count; i++) {
        free(m->kv[i].key);
        vkmodel_value_free(&m->kv[i].value);
    }
    free(m->kv);

    for (uint32_t i = 0; i < m->tensor_count; i++) {
        free(m->tensors[i].name);
        if (m->tensors[i].buffer != VK_NULL_HANDLE) {
            vkr_free(m->rt, m->tensors[i].buffer, m->tensors[i].memory);
        }
    }
    free(m->tensors);

    free(m);
}

/* ===========================================================================
 * Public API: metadata access
 * ========================================================================== */

uint32_t vkmodel_get_kv_count(VkModel* m)
{
    return m ? m->kv_count : 0;
}

const char* vkmodel_get_kv_key(VkModel* m, uint32_t i)
{
    if (!m || i >= m->kv_count) return NULL;
    return m->kv[i].key;
}

const char* vkmodel_get_kv_string(VkModel* m, const char* key,
                                  const char* fallback)
{
    if (!m || !key) return fallback;
    for (uint32_t i = 0; i < m->kv_count; i++) {
        if (m->kv[i].key && strcmp(m->kv[i].key, key) == 0) {
            if (m->kv[i].value.type != VKMODEL_VAL_STRING) return NULL;
            return m->kv[i].value.v.str;
        }
    }
    return fallback;
}

/* ===========================================================================
 * Public API: tensor access
 * ========================================================================== */

uint32_t vkmodel_get_tensor_count(VkModel* m)
{
    return m ? m->tensor_count : 0;
}

const char* vkmodel_get_tensor_name(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return NULL;
    return m->tensors[i].name;
}

uint32_t vkmodel_get_tensor_dtype(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return 0;
    return m->tensors[i].dtype;
}

uint32_t vkmodel_get_tensor_nelems(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return 0;
    return (uint32_t)m->tensors[i].nelems;
}

VkBuffer vkmodel_get_tensor_buffer(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return VK_NULL_HANDLE;
    return m->tensors[i].buffer;
}

VkDeviceSize vkmodel_get_tensor_size(VkModel* m, uint32_t i)
{
    if (!m || i >= m->tensor_count) return 0;
    return m->tensors[i].size;
}

/* ===========================================================================
 * Public API: type mapping
 * ========================================================================== */

uint32_t vkmodel_block_elems(uint32_t ggml_type)
{
    return vkmodel_blck_elems_of(ggml_type);
}
