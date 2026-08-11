/**
 * \file vkcompress.h
 * \brief Vulkan-native compressed buffer store (agonistic VRAM+RAM compression).
 *
 * Provides transparent compression/decompression for GPU buffers via a
 * tag-based catalog. Any application can register buffers with a string tag
 * and read/write them compressed. A persistent .catalog file per tag prefix
 * allows reuse across runs.
 */
#ifndef VKCOMPRESS_H
#define VKCOMPRESS_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VkCompressContext VkCompressContext;

typedef uint64_t vkcomp_buffer_id_t;
#define VKCOMP_INVALID_ID 0

typedef enum {
    VKCOMP_ALGO_LZ4   = 0,
    VKCOMP_ALGO_STORED = 1,
} vkcomp_algo_t;

/**
 * \brief Create a VkCompressContext bound to a VkDevice.
 * \param physicalDevice Physical device (for cap queries).
 * \param device          Logical device.
 * \param pContext        Receives new context.
 */
VkResult vkcompress_create_context(VkPhysicalDevice physicalDevice,
                                   VkDevice device,
                                   VkCompressContext** pContext);

void vkcompress_destroy_context(VkCompressContext* ctx);

/**
 * \brief Register a buffer for compressed storage under a tag.
 * Returns a buffer id used in subsequent write/read calls.
 * \param ctx  valid context
 * \param size  uncompressed size in bytes
 * \param tag  string tag for identification (e.g. "model:abc123:tensor:0")
 * \param compression_level  GPU compression level: 0 = fastest, 9 = highest ratio
 * \return buffer id, or VKCOMP_INVALID_ID on error
 */
vkcomp_buffer_id_t vkcompress_register_buffer(VkCompressContext* ctx,
                                              VkDeviceSize size,
                                              const char* tag,
                                               int compression_level); /* 0-9, 0 = fast, 9 = highest ratio */

/**
 * \brief Register a large (>VRAM) buffer as multiple chunks (virtual VRAM).
 * Each chunk is compressed/decompressed independently. Use
 * vkcompress_stream_write/read to process one chunk at a time.
 * \param ctx  valid context
 * \param total_size  total uncompressed size in bytes
 * \param chunk_size  per-chunk size (recommended: 256MB-1GB)
 * \param tag  string tag prefix
 * \param compression_level  GPU compression level (0-9)
 * \return buffer id (chunk 0), or VKCOMP_INVALID_ID on error
 */
vkcomp_buffer_id_t vkcompress_register_buffer_streaming(VkCompressContext* ctx,
                                                         VkDeviceSize total_size,
                                                         VkDeviceSize chunk_size,
                                                         const char* tag,
                                                         int compression_level);

/**
 * \brief Stream-write a chunk to a buffer registered with streaming mode.
 * Processes one chunk_index at a time, records compression dispatch into cmd.
 * \param chunk_index  which chunk (0-based)
 * \param chunk_size   actual bytes to write (may be less than registered chunk_size for last chunk)
 */
VkResult vkcompress_stream_write(VkCompressContext* ctx, VkCommandBuffer cmd,
                                  vkcomp_buffer_id_t id, uint32_t chunk_index,
                                  VkBuffer src, VkDeviceSize chunk_size, VkDeviceSize offset);

/**
 * \brief Stream-read a chunk from a buffer registered with streaming mode.
 * Decompresses one chunk_index from GPU-compressed storage back to dst.
 */
VkResult vkcompress_stream_read(VkCompressContext* ctx, VkCommandBuffer cmd,
                                 vkcomp_buffer_id_t id, uint32_t chunk_index,
                                 VkBuffer dst, VkDeviceSize chunk_size, VkDeviceSize offset);

/**
 * \brief Get the number of chunks for a streaming-registered buffer.
 */
uint32_t vkcompress_get_chunk_count(VkCompressContext* ctx, vkcomp_buffer_id_t id);

/**
 * \brief Write data to a compressed buffer (compresses on GPU).
 * Records a compression compute pass into cmd.
 */
VkResult vkcompress_write(VkCompressContext* ctx, VkCommandBuffer cmd,
                          vkcomp_buffer_id_t id,
                          VkBuffer src, VkDeviceSize size, VkDeviceSize offset);

/**
 * \brief Read data from a compressed buffer (decompresses on GPU).
 * Records a decompression compute pass into cmd.
 */
VkResult vkcompress_read(VkCompressContext* ctx, VkCommandBuffer cmd,
                         vkcomp_buffer_id_t id,
                         VkBuffer dst, VkDeviceSize size, VkDeviceSize offset);

/**
 * \brief Load a persistent catalog file (from a previous run with same tag prefix).
 * Allows pre-allocation and knowing compressed layout without recompressing.
 * \param ctx       valid context
 * \param tag_prefix e.g. "model:abc123def"
 * \param catalog_path path to .catalog file
 */
VkResult vkcompress_load_catalog(VkCompressContext* ctx,
                                 const char* tag_prefix,
                                 const char* catalog_path);

/**
 * \brief Save current catalog to disk for reuse.
 */
VkResult vkcompress_save_catalog(VkCompressContext* ctx,
                                 const char* catalog_path);

#ifdef __cplusplus
}
#endif
#endif /* VKCOMPRESS_H */
