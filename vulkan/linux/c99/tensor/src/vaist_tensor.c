#include "vaist_tensor.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static size_t dtype_size(VaistDType t) {
    switch (t) {
        case VAIST_F32: case VAIST_I32: return 4u;
        case VAIST_F64: return 8u;
        case VAIST_I8: case VAIST_U8: return 1u;
        case VAIST_F16: return 2u;
         case VAIST_DTYPE_Q8_0: return 0u;
        default: return 0u;
    }
}

struct VaistTensor {
    VaistDType dtype;
    uint32_t rank;
    uint64_t *shape;
    int64_t *stride;
    uint64_t numel;
    void *data;
    size_t data_bytes;
    struct VaistTensor *owner;
    uint32_t refcount;
};

static int product_checked(uint32_t rank, const uint64_t *shape, uint64_t *out) {
    uint64_t n = 1u;
    uint32_t i;
    if (!shape || !out || rank == 0u || rank > 8u) return 0;
    for (i = 0u; i < rank; ++i) {
        if (shape[i] == 0u || n > UINT64_MAX / shape[i]) return 0;
        n *= shape[i];
    }
    *out = n;
    return 1;
}

static int bytes_checked(uint64_t elements, size_t item_size, size_t *out) {
    if (!out || item_size == 0u || elements > (uint64_t)SIZE_MAX) return 0;
    if ((uint64_t)SIZE_MAX / (uint64_t)item_size < elements) return 0;
    *out = (size_t)elements * item_size;
    return 1;
}

static int contiguous(const VaistTensor *t) {
    int64_t expected;
    uint32_t i;
    if (!t || t->rank == 0u) return 0;
    expected = 1;
    for (i = t->rank; i-- > 0u;) {
        if (t->stride[i] != expected) return 0;
        if (t->shape[i] > (uint64_t)INT64_MAX) return 0;
        if (i > 0u && expected > INT64_MAX / (int64_t)t->shape[i]) return 0;
        expected *= (int64_t)t->shape[i];
    }
    return 1;
}

static void release_owner(VaistTensor *t) {
    if (!t || !t->owner) return;
    if (t->owner->refcount > 0u) t->owner->refcount--;
    if (t->owner->refcount == 0u) {
        free(t->owner->data);
        free(t->owner->shape);
        free(t->owner->stride);
        free(t->owner);
    }
    t->owner = NULL;
}

VAIST_API size_t vaist_dtype_size(VaistDType t) { return dtype_size(t); }

VAIST_API VaistStatus vaist_tensor_create(VaistDType dtype, uint32_t rank,
                                           const uint64_t *shape, VaistTensor **out) {
    VaistTensor *t;
    uint64_t n;
    size_t bytes;
    uint32_t i;
    if (!out || !shape || rank == 0u || rank > 8u || dtype_size(dtype) == 0u)
        return VAIST_INVALID_ARGUMENT;
    *out = NULL;
    if (!product_checked(rank, shape, &n) || !bytes_checked(n, dtype_size(dtype), &bytes))
        return VAIST_INVALID_ARGUMENT;
    t = (VaistTensor *)calloc(1u, sizeof(*t));
    if (!t) return VAIST_OUT_OF_MEMORY;
    t->shape = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)rank);
    t->stride = (int64_t *)malloc(sizeof(int64_t) * (size_t)rank);
    t->data = calloc(1u, bytes);
    if (!t->shape || !t->stride || !t->data) {
        free(t->shape); free(t->stride); free(t->data); free(t); return VAIST_OUT_OF_MEMORY;
    }
    t->dtype = dtype; t->rank = rank; t->numel = n; t->data_bytes = bytes; t->refcount = 1u;
    for (i = 0u; i < rank; ++i) t->shape[i] = shape[i];
    t->stride[rank - 1u] = 1;
    for (i = rank - 1u; i > 0u; --i) {
        if (t->shape[i] > (uint64_t)INT64_MAX || t->stride[i] > INT64_MAX / (int64_t)t->shape[i]) {
            free(t->shape); free(t->stride); free(t->data); free(t); return VAIST_INVALID_ARGUMENT;
        }
        t->stride[i - 1u] = t->stride[i] * (int64_t)t->shape[i];
    }
    *out = t;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_tensor_view(VaistTensor *src, uint32_t rank,
                                        const uint64_t *shape, const int64_t *stride,
                                        VaistTensor **out) {
    VaistTensor *t;
    uint64_t n;
    uint32_t i;
    if (!src || !shape || !stride || !out || rank == 0u || rank > 8u) return VAIST_INVALID_ARGUMENT;
    *out = NULL;
    if (!product_checked(rank, shape, &n) || n != src->numel) return VAIST_INVALID_ARGUMENT;
    for (i = 0u; i < rank; ++i) if (stride[i] < 0) return VAIST_INVALID_ARGUMENT;
    t = (VaistTensor *)calloc(1u, sizeof(*t));
    if (!t) return VAIST_OUT_OF_MEMORY;
    t->shape = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)rank);
    t->stride = (int64_t *)malloc(sizeof(int64_t) * (size_t)rank);
    if (!t->shape || !t->stride) { free(t->shape); free(t->stride); free(t); return VAIST_OUT_OF_MEMORY; }
    memcpy(t->shape, shape, sizeof(uint64_t) * (size_t)rank);
    memcpy(t->stride, stride, sizeof(int64_t) * (size_t)rank);
    t->dtype = src->dtype; t->rank = rank; t->numel = n; t->data = src->data;
    t->data_bytes = src->data_bytes; t->owner = src; t->refcount = 1u;
    if (src->refcount == UINT32_MAX) { free(t->shape); free(t->stride); free(t); return VAIST_INVALID_STATE; }
    src->refcount++;
    *out = t;
    return VAIST_OK;
}

VAIST_API void vaist_tensor_destroy(VaistTensor *t) {
    if (!t) return;
    if (t->owner) { release_owner(t); free(t->shape); free(t->stride); free(t); return; }
    if (t->refcount > 0u) t->refcount--;
    if (t->refcount == 0u) { free(t->data); free(t->shape); free(t->stride); free(t); }
}

VAIST_API VaistStatus vaist_tensor_reshape(VaistTensor *t, uint32_t rank, const uint64_t *shape) {
    uint64_t n;
    uint64_t *ns;
    int64_t *st;
    uint32_t i;
    if (!t || t->owner || !shape || rank == 0u || rank > 8u) return VAIST_INVALID_ARGUMENT;
    if (!product_checked(rank, shape, &n) || n != t->numel || !contiguous(t)) return VAIST_INVALID_ARGUMENT;
    ns = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)rank);
    st = (int64_t *)malloc(sizeof(int64_t) * (size_t)rank);
    if (!ns || !st) { free(ns); free(st); return VAIST_OUT_OF_MEMORY; }
    for (i = 0u; i < rank; ++i) ns[i] = shape[i];
    st[rank - 1u] = 1;
    for (i = rank - 1u; i > 0u; --i) {
        if (ns[i] > (uint64_t)INT64_MAX || st[i] > INT64_MAX / (int64_t)ns[i]) { free(ns); free(st); return VAIST_INVALID_ARGUMENT; }
        st[i - 1u] = st[i] * (int64_t)ns[i];
    }
    free(t->shape); free(t->stride); t->shape = ns; t->stride = st; t->rank = rank;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_tensor_transpose2(VaistTensor *t) {
    uint64_t q; int64_t z;
    if (!t || t->rank != 2u) return VAIST_INVALID_ARGUMENT;
    q = t->shape[0]; t->shape[0] = t->shape[1]; t->shape[1] = q;
    z = t->stride[0]; t->stride[0] = t->stride[1]; t->stride[1] = z;
    return VAIST_OK;
}

static int copy_recursive(const VaistTensor *s, VaistTensor *d, uint32_t dim,
                          uint64_t so, uint64_t doff, size_t item) {
    uint64_t i;
    if (dim == s->rank) {
        const unsigned char *sp = (const unsigned char *)s->data + (size_t)so * item;
        unsigned char *dp = (unsigned char *)d->data + (size_t)doff * item;
        memcpy(dp, sp, item);
        return 1;
    }
    for (i = 0u; i < s->shape[dim]; ++i) {
        uint64_t sn, dn;
        if (i && s->stride[dim] > (int64_t)(UINT64_MAX / i)) return 0;
        if (i && d->stride[dim] > (int64_t)(UINT64_MAX / i)) return 0;
        sn = so + i * (uint64_t)s->stride[dim];
        dn = doff + i * (uint64_t)d->stride[dim];
        if (!copy_recursive(s, d, dim + 1u, sn, dn, item)) return 0;
    }
    return 1;
}

VAIST_API VaistStatus vaist_tensor_copy(const VaistTensor *src, VaistTensor *dst) {
    size_t bytes;
    if (!src || !dst || src->dtype != dst->dtype || src->numel != dst->numel) return VAIST_INVALID_ARGUMENT;
    if (!bytes_checked(src->numel, dtype_size(src->dtype), &bytes)) return VAIST_INVALID_ARGUMENT;
    if (src->rank == dst->rank && contiguous(src) && contiguous(dst)) { memcpy(dst->data, src->data, bytes); return VAIST_OK; }
    if (src->rank != dst->rank) return VAIST_INVALID_ARGUMENT;
    return copy_recursive(src, dst, 0u, 0u, 0u, dtype_size(src->dtype)) ? VAIST_OK : VAIST_INVALID_ARGUMENT;
}

VAIST_API void *vaist_tensor_data(VaistTensor *t) { return t ? t->data : NULL; }
VAIST_API const void *vaist_tensor_const_data(const VaistTensor *t) { return t ? t->data : NULL; }
VAIST_API uint32_t vaist_tensor_rank(const VaistTensor *t) { return t ? t->rank : 0u; }
VAIST_API uint64_t vaist_tensor_numel(const VaistTensor *t) { return t ? t->numel : 0u; }
VAIST_API VaistDType vaist_tensor_dtype(const VaistTensor *t) { return t ? t->dtype : (VaistDType)-1; }
VAIST_API const uint64_t *vaist_tensor_shape(const VaistTensor *t) { return t ? t->shape : NULL; }
VAIST_API const int64_t *vaist_tensor_stride(const VaistTensor *t) { return t ? t->stride : NULL; }
