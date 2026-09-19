/**
 * \file vaist_model_streaming.c
 * \brief SSD-offload streaming layer: mmap + LRU cache + async prefetch.
 *
 * Ported from Edge0's src/edge0/streaming/mmap.py + cache.py + layer.py:
 *   - SafetensorsMmap -> vaist_gguf_streaming_open (mmap-backed file view)
 *   - SharedExpertCache -> vaist_streaming_cache_create (LRU with shared budget)
 *   - PrefetchBuffer -> vaist_gguf_tensor_prefetch (staging before promote)
 *   - vaist_gguf_tensor_to_gpu -> layer.py's staged decode (host->VRAM DMA)
 */
#include "vaist_model_streaming.h"
#include "vaist_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(_WIN32)
 #include <windows.h>
#else
 #include <sys/mman.h>
 #include <sys/stat.h>
 #include <fcntl.h>
 #include <unistd.h>
#endif

/* ---- Streaming handle ---- */

struct VaistStreamingHandle {
    void *mapped_base;     /* mmap / CreateFileMapping base */
    size_t file_size;      /* total file size */
    size_t data_offset;   /* start of tensor data (after GGUF header) */
    char path[1024];
#if defined(_WIN32)
    HANDLE h_file;
    HANDLE h_mapping;
#else
    int fd;
#endif
};

/* ---- Cache entry LRU list operations ---- */

static void _lru_remove(VaistStreamingCache *c, uint32_t idx) {
    VaistCacheEntry *e = &c->entries[idx];
    if (e->lru_prev != UINT32_MAX)
        c->entries[e->lru_prev].lru_next = e->lru_next;
    else
        c->head = e->lru_next;
    if (e->lru_next != UINT32_MAX)
        c->entries[e->lru_next].lru_prev = e->lru_prev;
    else
        c->tail = e->lru_prev;
    e->lru_prev = UINT32_MAX;
    e->lru_next = UINT32_MAX;
}

static void _lru_push_front(VaistStreamingCache *c, uint32_t idx) {
    /* Insert at head (MRU position) */
    c->entries[idx].lru_prev = UINT32_MAX;
    c->entries[idx].lru_next = c->head;
    if (c->head != UINT32_MAX)
        c->entries[c->head].lru_prev = idx;
    else
        c->tail = idx;
    c->head = idx;
}

static uint32_t _find_entry(VaistStreamingCache *c, const char *name) {
    for (size_t i = 0; i < c->max_entries; i++) {
        if (c->entries[i].gpu_ptr && c->entries[i].name[0] &&
            strcmp(c->entries[i].name, name) == 0)
            return (uint32_t)i;
    }
    return UINT32_MAX;
}

static uint32_t _find_entry_by_name(VaistStreamingCache *c, const char *name, int resident_only) {
    for (size_t i = 0; i < c->max_entries; i++) {
        if (c->entries[i].name[0] && strcmp(c->entries[i].name, name) == 0) {
            if (!resident_only || c->entries[i].gpu_ptr)
                return (uint32_t)i;
        }
    }
    return UINT32_MAX;
}

/* ---- Streaming handle: open/mmap ---- */

VAIST_API VaistStatus vaist_gguf_streaming_open(
    const char *path,
    VaistStreamingHandle **out)
{
    if (!path || !out) return VAIST_INVALID_ARGUMENT;
    *out = NULL;

    VaistStreamingHandle *h = (VaistStreamingHandle *)calloc(1, sizeof(*h));
    if (!h) return VAIST_OUT_OF_MEMORY;
    strncpy(h->path, path, sizeof(h->path) - 1);

#if defined(_WIN32)
    h->h_file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h->h_file == INVALID_HANDLE_VALUE) {
        free(h); return VAIST_IO_ERROR;
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(h->h_file, &file_size)) {
        CloseHandle(h->h_file); free(h); return VAIST_IO_ERROR;
    }
    h->file_size = (size_t)file_size.QuadPart;
    h->h_mapping = CreateFileMappingA(h->h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!h->h_mapping) {
        CloseHandle(h->h_file); free(h); return VAIST_IO_ERROR;
    }
    h->mapped_base = MapViewOfFile(h->h_mapping, FILE_MAP_READ, 0, 0, 0);
    if (!h->mapped_base) {
        CloseHandle(h->h_mapping); CloseHandle(h->h_file); free(h); return VAIST_IO_ERROR;
    }
#else
    h->fd = open(path, O_RDONLY);
    if (h->fd < 0) {
        free(h); return VAIST_IO_ERROR;
    }
    struct stat st;
    if (fstat(h->fd, &st) != 0) {
        close(h->fd); free(h); return VAIST_IO_ERROR;
    }
    h->file_size = (size_t)st.st_size;
    h->mapped_base = mmap(NULL, h->file_size, PROT_READ, MAP_PRIVATE, h->fd, 0);
    if (h->mapped_base == MAP_FAILED) {
        close(h->fd); free(h); return VAIST_IO_ERROR;
    }
    madvise(h->mapped_base, h->file_size, MADV_WILLNEED);
#endif
    *out = h;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_gguf_streaming_close(VaistStreamingHandle *handle) {
    if (!handle) return VAIST_INVALID_ARGUMENT;
#if defined(_WIN32)
    if (handle->mapped_base) UnmapViewOfFile(handle->mapped_base);
    if (handle->h_mapping) CloseHandle(handle->h_mapping);
    if (handle->h_file != INVALID_HANDLE_VALUE) CloseHandle(handle->h_file);
#else
    if (handle->mapped_base && handle->mapped_base != MAP_FAILED)
        munmap(handle->mapped_base, handle->file_size);
    if (handle->fd >= 0) close(handle->fd);
#endif
    free(handle);
    return VAIST_OK;
}

/* ---- Cache creation/destroy ---- */

VAIST_API VaistStatus vaist_streaming_cache_create(
    VaistStreamingHandle *handle,
    size_t max_bytes,
    size_t max_entries,
    VaistStreamingCache **out)
{
    if (!handle || !out) return VAIST_INVALID_ARGUMENT;
    *out = NULL;

    VaistStreamingCache *c = (VaistStreamingCache *)calloc(1, sizeof(*c));
    if (!c) return VAIST_OUT_OF_MEMORY;

    c->handle = handle;
    c->max_bytes = max_bytes;
    c->max_entries = max_entries;
    c->entries = (VaistCacheEntry *)calloc(max_entries, sizeof(VaistCacheEntry));
    if (!c->entries) { free(c); return VAIST_OUT_OF_MEMORY; }

    /* Initialize LRU chain as empty */
    c->head = UINT32_MAX;
    c->tail = UINT32_MAX;
    for (size_t i = 0; i < max_entries; i++) {
        c->entries[i].lru_prev = UINT32_MAX;
        c->entries[i].lru_next = UINT32_MAX;
    }

    *out = c;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_streaming_cache_destroy(VaistStreamingCache *cache) {
    if (!cache) return VAIST_INVALID_ARGUMENT;
    for (size_t i = 0; i < cache->max_entries; i++) {
        if (cache->entries[i].gpu_ptr) {
            free(cache->entries[i].gpu_ptr); /* or vkFreeMemory in real impl */
            cache->entries[i].gpu_ptr = NULL;
        }
        if (cache->entries[i].host_staging) {
            free(cache->entries[i].host_staging);
            cache->entries[i].host_staging = NULL;
        }
    }
    free(cache->entries);
    free(cache);
    return VAIST_OK;
}

/* ---- Async prefetch: SSD -> host staging ---- */

VAIST_API VaistStatus vaist_gguf_tensor_prefetch(
    VaistStreamingCache *cache,
    const char *tensor_name,
    uint64_t byte_offset,
    size_t raw_bytes)
{
    if (!cache || !tensor_name) return VAIST_INVALID_ARGUMENT;

    /* Find or allocate a cache slot */
    uint32_t idx = _find_entry_by_name(cache, tensor_name, 0);
    if (idx == UINT32_MAX) {
        /* Find free slot */
        for (size_t i = 0; i < cache->max_entries; i++) {
            if (cache->entries[i].name[0] == '\0') {
                idx = (uint32_t)i;
                break;
            }
        }
        if (idx == UINT32_MAX) {
            /* No free slot — evict LRU */
            idx = cache->tail;
            if (idx == UINT32_MAX) return VAIST_OUT_OF_MEMORY;
            VaistCacheEntry *e = &cache->entries[idx];
            if (e->gpu_ptr) { free(e->gpu_ptr); e->gpu_ptr = NULL; }
            if (e->host_staging) { free(e->host_staging); e->host_staging = NULL; }
            memset(e->name, 0, sizeof(e->name));
            _lru_remove(cache, idx);
        }
    }

    VaistCacheEntry *e = &cache->entries[idx];
    /* Read from mmap into pinned host staging */
    if (cache->handle && cache->handle->mapped_base) {
        const char *src = (const char *)cache->handle->mapped_base + byte_offset;
        e->host_staging = (float *)malloc(raw_bytes);
        if (!e->host_staging) return VAIST_OUT_OF_MEMORY;
        memcpy(e->host_staging, src, raw_bytes);
    }
    strncpy(e->name, tensor_name, sizeof(e->name) - 1);
    e->offset = byte_offset;
    e->byte_size = raw_bytes;
    return VAIST_OK;
}

/* ---- Sync prefetch ---- */

VAIST_API VaistStatus vaist_streaming_prefetch_wait(VaistStreamingCache *cache) {
    /* In single-threaded CPU fallback, prefetch is synchronous.
     * Real impl would wait on Vulkan fence here. */
    (void)cache;
    return VAIST_OK;
}

/* ---- Host -> GPU transfer with LRU eviction ---- */

VAIST_API VaistStatus vaist_gguf_tensor_to_gpu(
    VaistStreamingCache *cache,
    const char *tensor_name,
    void **gpu_ptr_out)
{
    if (!cache || !tensor_name || !gpu_ptr_out) return VAIST_INVALID_ARGUMENT;

    uint32_t idx = _find_entry_by_name(cache, tensor_name, 1);
    if (idx != UINT32_MAX) {
        /* Already resident — move to MRU in LRU */
        _lru_remove(cache, idx);
        _lru_push_front(cache, idx);
        cache->entries[idx].refcount++;
        *gpu_ptr_out = cache->entries[idx].gpu_ptr;
        return VAIST_OK;
    }

    idx = _find_entry_by_name(cache, tensor_name, 0);
    if (idx == UINT32_MAX) return VAIST_INVALID_ARGUMENT;

    VaistCacheEntry *e = &cache->entries[idx];
    if (!e->host_staging) return VAIST_IO_ERROR;

    /* Evict LRU if over budget */
    while (cache->resident_bytes + e->byte_size > cache->max_bytes && cache->head != UINT32_MAX) {
        uint32_t victim = cache->tail;
        if (victim == UINT32_MAX || victim == idx) break;
        VaistCacheEntry *v = &cache->entries[victim];
        if (v->gpu_ptr) {
            free(v->gpu_ptr); /* or vkDestroyBuffer */
            v->gpu_ptr = NULL;
            cache->resident_bytes -= v->byte_size;
        }
        _lru_remove(cache, victim);
    }

    /* Allocate GPU (or CPU fallback) */
    e->gpu_ptr = malloc(e->byte_size); /* real impl: vkAllocateMemory + vkCmdCopyBuffer */
    if (!e->gpu_ptr) return VAIST_OUT_OF_MEMORY;
    memcpy(e->gpu_ptr, e->host_staging, e->byte_size);

    cache->resident_bytes += e->byte_size;
    e->refcount = 1;
    _lru_push_front(cache, idx);
    *gpu_ptr_out = e->gpu_ptr;
    return VAIST_OK;
}

VAIST_API VaistStatus vaist_gguf_tensor_evict_gpu(
    VaistStreamingCache *cache,
    const char *tensor_name)
{
    if (!cache || !tensor_name) return VAIST_INVALID_ARGUMENT;
    uint32_t idx = _find_entry_by_name(cache, tensor_name, 1);
    if (idx == UINT32_MAX) return VAIST_INVALID_ARGUMENT;
    VaistCacheEntry *e = &cache->entries[idx];
    e->refcount--;
    if (e->refcount > 0) return VAIST_OK;
    if (e->gpu_ptr) {
        free(e->gpu_ptr);
        e->gpu_ptr = NULL;
        cache->resident_bytes -= e->byte_size;
    }
    _lru_remove(cache, idx);
    return VAIST_OK;
}

/* ---- High-level read (prefetch + dequant) ---- */

VAIST_API VaistStatus vaist_gguf_streaming_tensor_read(
    VaistStreamingCache *cache,
    const char *tensor_name,
    float *dst_f32,
    size_t dst_cap)
{
    if (!cache || !tensor_name || !dst_f32) return VAIST_INVALID_ARGUMENT;

    uint32_t idx = _find_entry_by_name(cache, tensor_name, 0);
    if (idx == UINT32_MAX) return VAIST_INVALID_ARGUMENT;

    VaistCacheEntry *e = &cache->entries[idx];
    if (cache->handle && cache->handle->mapped_base && !e->host_staging) {
        /* Lazy-load if not prefetched */
        const char *src = (const char *)cache->handle->mapped_base + e->offset;
        e->host_staging = (float *)malloc(e->byte_size);
        if (!e->host_staging) return VAIST_OUT_OF_MEMORY;
        memcpy(e->host_staging, src, e->byte_size);
    }

    if (!e->host_staging) return VAIST_IO_ERROR;

    /* Determine quant type and dequantize */
    /* The dtype field in VaistTensorDesc stores GGUF_DTYPE_OFFSET + ggml_type */
    VaistQuantType qt;
    switch (e->byte_size) {
        default:
            /* Assume f32 passthrough if byte_size matches dst_cap * sizeof(float) */
            if (e->byte_size == dst_cap * sizeof(float)) {
                memcpy(dst_f32, e->host_staging, e->byte_size);
                return VAIST_OK;
            }
            /* Fall through to dequant for quantized types */
            qt = VAIST_Q4_0; /* placeholder — real impl uses desc->dtype */
            break;
    }
    return vaist_dequantize_f32(qt, e->host_staging, e->byte_size, dst_f32, dst_cap);
}
