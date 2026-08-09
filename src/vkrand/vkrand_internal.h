/**
 * \file vkrand_internal.h
 * \brief Internal VKRAND structures: push constants, pipeline cache, context.
 *
 * Mirrors the design of vkmath_internal.h: same pipeline-caching strategy,
 * same tier detection pattern, same embedded-SPIR-V pattern. Only a baseline
 * tier shader exists, so the active tier always resolves to baseline.
 */
#ifndef VKRAND_INTERNAL_H
#define VKRAND_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"
#include "vkrand.h"

/* ── Push constant block (std140, must match GLSL layout exactly) ──────── */

typedef struct {
    uint32_t count;  /* offset  0: number of f32 values to generate */
    uint32_t seed;   /* offset  4: deterministic seed                */
    uint32_t _pad0;  /* offset  8: padding                           */
    uint32_t _pad1;  /* offset 12: padding                           */
} vkrand_push_constants_t;

/* Static assert (C99-compatible): struct must be exactly 16 bytes */
typedef char vkrand_pc_static_assert[sizeof(vkrand_push_constants_t) == 16 ? 1 : -1];

/* ── Kernel types ──────────────────────────────────────────────────────── */

#define VKRAND_KERNEL_UNIFORM_F32        0
#define VKRAND_KERNEL_THREEFRY_F32       1
#define VKRAND_KERNEL_NORMAL_F32         2
#define VKRAND_KERNEL_UNIFORM_UINT32     3
#define VKRAND_KERNEL_COUNT              4

/* ── Capability tiers ──────────────────────────────────────────────────── */

typedef enum {
    VKRAND_TIER_BASELINE = 0,   /* only tier with a compiled shader today */
} VkRandTier_t;

/* ── Pipeline cache ────────────────────────────────────────────────────── */

#define VKRAND_MAX_PIPELINES 256

typedef struct {
    uint64_t key;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    uint32_t kernel;
    uint32_t tier;
    uint8_t  valid;
} vkrand_pipeline_entry_t;

/* ── Context ───────────────────────────────────────────────────────────── */

struct VkRandContext {
    VkDevice device;
    VkPipelineCache pipeline_cache;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;

    /* capability flags */
    VkBool32 has_subgroup;
    VkBool32 has_shader_int64;
    uint32_t max_subgroup_size;
    uint32_t max_compute_workgroup_size[3];
    VkBool32 has_push_descriptor;    /* active tier (highest supported) */
    VkRandTier_t active_tier;

    /* push descriptor function (loaded via vkGetDeviceProcAddr) */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn;

    /* pipeline cache storage (open-addressing, linear probing) */
    vkrand_pipeline_entry_t pipelines[VKRAND_MAX_PIPELINES];
    uint32_t pipeline_count;
};

/* ── Internal functions ────────────────────────────────────────────────── */

VkResult vkrand_load_shader_module(VkDevice device,
                                   const uint32_t* spirv,
                                   size_t spirv_words,
                                   VkShaderModule* out_module);

VkResult vkrand_ensure_pipeline(VkRandContext* ctx,
                                uint32_t kernel,
                                VkPipeline* out_pipeline);

VkPipelineLayout vkrand_get_pipeline_layout(VkRandContext* ctx);

void vkrand_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                    const vkrand_push_constants_t* pc);

VkResult vkrand_alloc_descriptor_set(VkRandContext* ctx,
                                     VkDescriptorSet* out_ds);

uint64_t vkrand_hash_key(uint32_t kernel, uint32_t tier);

const uint32_t* vkrand_select_spirv(uint32_t kernel, uint32_t tier,
                                    size_t* out_size);

#endif /* VKRAND_INTERNAL_H */
