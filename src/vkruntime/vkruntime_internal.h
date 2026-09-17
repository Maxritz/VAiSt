/**
 * \file vkruntime_internal.h
 * \brief Internal VKRuntime structures: pooled allocator, capability cache, context.
 *
 * The pooled allocator keeps one linked list of large VkDeviceMemory blocks
 * per memory class (device-local, host-visible/coherent). Each block bump
 * allocates with a coalescing free list so vkr_free() regions are reused
 * without per-free vkFreeMemory syscalls. A small open-addressing table maps
 * live VkBuffer handles back to their (memory, offset, size) records.
 */
#ifndef VKRUNTIME_INTERNAL_H
#define VKRUNTIME_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"

/* ── Memory pool classes ───────────────────────────────────────────────── */

#define VKR_POOL_COUNT          2u  /**< device-local + host-visible.          */
#define VKR_POOL_DEVICE_LOCAL   0u  /**< GPU-resident buffers (hipMalloc).     */
#define VKR_POOL_HOST_VISIBLE   1u  /**< host-visible/coherent staging.        */

#define VKR_DEFAULT_BLOCK_SIZE  ((VkDeviceSize)16u * 1024u * 1024u) /**< 16 MiB. */

/* ── Free list ──────────────────────────────────────────────────────────── *
 * A block owns a dynamic array of free byte regions (offset, size). Regions
 * are coalesced on free and first-fit on alloc. Offsets are relative to the
 * start of the block's VkDeviceMemory. ──────────────────────────────────── */

typedef struct vkr_block_t vkr_block_t;

struct vkr_block_t {
    VkDeviceMemory memory;       /**< One Vulkan memory allocation.          */
    VkDeviceSize   size;         /**< Allocated size of the block.           */
    VkDeviceSize   cursor;       /**< Bump pointer into unused tail.         */
    void          *mapped;       /**< Host mapping (host-visible blocks only). */
    VkDeviceSize  *free_offs;    /**< Free region start offsets.             */
    VkDeviceSize  *free_sz;      /**< Free region sizes.                     */
    uint32_t       free_count;   /**< Number of live free regions.           */
    uint32_t       free_cap;     /**< Capacity of the free region arrays.    */
    vkr_block_t   *next;         /**< Next block in the pool.                */
};

/* ── Pool ──────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t      memory_type_index; /**< VkMemoryType used by all blocks.  */
    VkDeviceSize  alignment;         /**< Max observed buffer alignment.    */
    vkr_block_t  *blocks;            /**< Head of the block list.           */
} vkr_pool_t;

/* ── Allocation record table ────────────────────────────────────────────── *
 * Open addressing, linear probing. A slot is empty when buffer is null.
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    VkBuffer       buffer;  /**< Handle (key).                              */
    VkDeviceMemory memory;  /**< Block memory backing the region.           */
    VkDeviceSize   offset;  /**< Region offset within the block memory.     */
    VkDeviceSize   size;    /**< Region size (aligned up).                  */
} vkr_alloc_slot_t;

/* ── Capability cache ────────────────────────────────────────────────────
 * The runtime caches the same public VkRuntimeCaps struct filled by
 * vkr_detect_capabilities(). The push_desc_fn member mirrors the runtime's
 * own device-fn field so the getters can read caps directly.
 * ──────────────────────────────────────────────────────────────────────── */

typedef VkRuntimeCaps vkr_caps_t;

/* ── Runtime ───────────────────────────────────────────────────────────── */

struct VkRuntime {
    VkDevice         device;
    VkPhysicalDevice physical_device;
    VkQueue          queue;

    VkPhysicalDeviceMemoryProperties mem_props;

    vkr_caps_t caps;

    /* pooled allocator */
    vkr_pool_t pools[VKR_POOL_COUNT];
    vkr_alloc_slot_t *alloc_table;   /**< Open-addressing buffer->record map. */
    uint32_t  alloc_capacity;        /**< Table capacity (power of two).      */
    uint32_t  alloc_count;           /**< Live allocation count.              */

    /* push descriptor availability (detected, not used by this library) */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn;
};

#endif /* VKRUNTIME_INTERNAL_H */
