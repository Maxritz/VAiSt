/**
 * \file vaist_model_streaming.h
 * \brief SSD-offload streaming layer for GGUF models.
 *
 * Implements the Edge0 pattern: byte-range mmap for direct tensor access,
 * LRU-backed GPU resident cache with async prefetch from SSD.
 * Mirrors SharedExpertCache (LRU) + PrefetchBuffer from edge0/streaming/cache.py.
 */
#ifndef VAIST_MODEL_STREAMING_H
#define VAIST_MODEL_STREAMING_H
#include "vaist_model.h"
#include "vaist_core.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Streaming handle: one GGUF file mapped via mmap (Windows: CreateFileMapping,
 * Linux: mmap). Tensor offsets are resolved without full materialization.
 */
typedef struct VaistStreamingHandle VaistStreamingHandle;

/**
 * Resident cache entry: a tensor slab currently resident in GPU VRAM.
 * LRU ordering is maintained via a doubly-linked list per cache instance.
 */
typedef struct VaistCacheEntry {
    char     name[256];
    uint64_t offset;          /**< byte offset in the GGUF file */
    uint64_t byte_size;       /**< raw stored size */
    VaistQuantType quant_type; /**< quantization format (f32, q4_0, nvfp4, etc.) */
    void    *gpu_ptr;         /**< device pointer (NULL if evicted) */
    float   *host_staging;    /**< pinned host staging buffer */
    uint32_t lru_prev;        /**< previous entry in LRU list (index) */
    uint32_t lru_next;        /**< next entry in LRU list (index) */
    uint32_t refcount;        /**< active references (0 = evictable) */
    int      dirty;           /**< write-back needed */
} VaistCacheEntry;

/**
 * Streaming cache with bounded GPU memory budget.
 * Uses async prefetch to overlap SSD read + H2D DMA with compute.
 */
typedef struct VaistStreamingCache {
    VaistStreamingHandle *handle;  /**< backing mmap'd GGUF */
    size_t max_entries;            /**< max resident tensors */
    size_t resident_bytes;         /**< current GPU memory usage */
    size_t max_bytes;              /**< GPU memory budget */
    VaistCacheEntry *entries;      /**< flat array of cache slots */
    uint32_t head;                 /**< MRU head (index) */
    uint32_t tail;                 /**< LRU tail (index) */
    void *stream;                  /**< Vulkan command stream for async DMA */
} VaistStreamingCache;

/**
 * \brief Open a GGUF file for streaming access.
 *
 * Opens the file and creates an mmap (or equivalent) view over its entire
 * contents. No tensor data is loaded into memory yet.
 */
VAIST_API VaistStatus vaist_gguf_streaming_open(
    const char *path,
    VaistStreamingHandle **out);

/**
 * \brief Create a streaming cache bound to an already-opened GGUF handle.
 *
 * max_bytes is the total GPU VRAM budget for resident weights; max_entries
 * caps the number of concurrently resident tensor slabs.
 */
VAIST_API VaistStatus vaist_streaming_cache_create(
    VaistStreamingHandle *handle,
    size_t max_bytes,
    size_t max_entries,
    VaistStreamingCache **out);

/**
 * \brief Prefetch a tensor from SSD into host staging (async, non-blocking).
 *
 * Reads the tensor's raw bytes from the mmap into a pinned host buffer.
 * Does not transfer to GPU. Returns immediately; completion must be
 * waited on via vaist_streaming_prefetch_wait().
 */
VAIST_API VaistStatus vaist_gguf_tensor_prefetch(
    VaistStreamingCache *cache,
    const char *tensor_name,
    uint64_t byte_offset,
    size_t raw_bytes,
    VaistQuantType quant_type);

/**
 * \brief Wait for all pending prefetch operations to complete.
 */
VAIST_API VaistStatus vaist_streaming_prefetch_wait(
    VaistStreamingCache *cache);

/**
 * \brief Transfer a prefetched tensor from host staging to GPU VRAM.
 *
 * If the GPU cache is full, evicts the LRU entry first.
 * Returns a pointer to the GPU-resident data.
 */
VAIST_API VaistStatus vaist_gguf_tensor_to_gpu(
    VaistStreamingCache *cache,
    const char *tensor_name,
    void **gpu_ptr_out);

/**
 * \brief Evict a tensor from GPU VRAM (back to LRU state).
 */
VAIST_API VaistStatus vaist_gguf_tensor_evict_gpu(
    VaistStreamingCache *cache,
    const char *tensor_name);

/**
 * \brief Read+optional-dequantize a tensor, streaming from SSD if needed.
 *
 * High-level convenience: handles prefetch + H2D + dequant in one call.
 * For the zero-copy path, callers using GPU kernels can query gpu_ptr
 * directly via vaist_gguf_tensor_to_gpu.
 */
VAIST_API VaistStatus vaist_gguf_streaming_tensor_read(
    VaistStreamingCache *cache,
    const char *tensor_name,
    float *dst_f32,
    size_t dst_cap);

/**
 * \brief Close the streaming handle and release resources.
 */
VAIST_API VaistStatus vaist_gguf_streaming_close(VaistStreamingHandle *handle);

/**
 * \brief Destroy the streaming cache.
 */
VAIST_API VaistStatus vaist_streaming_cache_destroy(VaistStreamingCache *cache);

#ifdef __cplusplus
}
#endif
#endif
