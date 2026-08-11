/**
 * \file vkcompress_internal.h
 * \brief Internal VKCompress structures: context, buffer registry, pipeline cache.
 */
#ifndef VKCOMPRESS_INTERNAL_H
#define VKCOMPRESS_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkcompress.h"

#define VKCOMP_MAX_BUFFERS 64u
#define VKCOMP_MAX_TAG_LEN 128u
#define VKCOMP_WORKGROUP_SIZE 256u
#define VKCOMP_DEFAULT_CHUNK_SIZE (256u * 1024u * 1024u)  /* 256 MB */

typedef struct {
    vkcomp_buffer_id_t id;
    char tag[VKCOMP_MAX_TAG_LEN];
    VkDeviceSize uncompressed_size;
    VkDeviceSize compressed_size;     /* device-side after compression */
    VkDeviceSize compressed_offset;   /* offset in staging blob */
    int compression_level;
    VkBuffer src_buffer;
    VkBuffer comp_buffer;             /* GPU-side compressed buffer */
    VkBuffer meta_buffer;             /* GPU-side compression metadata */
    VkDeviceMemory src_memory;
    VkDeviceMemory comp_memory;
    VkDeviceMemory meta_memory;
    VkPipeline pipeline_write;        /* compress */
    VkPipeline pipeline_read;         /* decompress */
    VkShaderModule module_write;
    VkShaderModule module_read;
    VkDescriptorSet desc_set;         /* one per entry */
    uint32_t chunk_count;             /* for streaming buffers */
    uint8_t valid;
} vkcomp_buffer_entry_t;

typedef struct VkCompressContext {
    VkDevice device;
    VkPhysicalDevice physical_device;
    VkPipelineLayout pipeline_layout;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool descriptor_pool;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    VkDeviceSize staging_size;
    VkDeviceSize staging_offset;      /* cursor into staging */
    vkcomp_buffer_entry_t buffers[VKCOMP_MAX_BUFFERS];
    uint64_t next_id;
    uint32_t buffer_count;
    VkBool32 has_subgroup;
} VkCompressContext;

/* Push constant (std140, 16 bytes) */
typedef struct {
    uint32_t num_blocks;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
} vkcomp_push_constants_t;

#endif /* VKCOMPRESS_INTERNAL_H */
