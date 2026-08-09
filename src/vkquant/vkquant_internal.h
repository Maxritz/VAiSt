/**
 * \file vkquant_internal.h
 * \brief Internal VKQuant structures: push constants, pipeline cache, context.
 *
 * Mirrors the design of vkmath_internal.h: same pipeline-caching strategy,
 * same tier detection, same embedded-SPIR-V pattern. Only baseline shaders
 * exist, so the active tier always falls back to baseline.
 */
#ifndef VKQUANT_INTERNAL_H
#define VKQUANT_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"
#include "vkquant.h"

/* ── Push constant block (std140, must match GLSL layout exactly) ──────── */

typedef struct {
    uint32_t num_blocks; /* offset  0: number of quantized blocks           */
    uint32_t _pad0;      /* offset  4: reserved (uint32_t only, no int64)   */
    uint32_t _pad1;      /* offset  8: reserved                             */
    uint32_t _pad2;      /* offset 12: reserved                             */
} vkquant_push_constants_t;

/* Static assert (C99-compatible): struct must be exactly 16 bytes */
typedef char vkquant_pc_static_assert[sizeof(vkquant_push_constants_t) == 16 ? 1 : -1];

/* ── Data types ────────────────────────────────────────────────────────── */

#define VKQUANT_DTYPE_F32 0

/* ── Kernel types ──────────────────────────────────────────────────────── */

#define VKQUANT_KERNEL_Q8_0_DEQUANT   0
#define VKQUANT_KERNEL_Q4_0_DEQUANT   1
#define VKQUANT_KERNEL_Q4K_DEQUANT    2
#define VKQUANT_KERNEL_Q6K_DEQUANT    3
#define VKQUANT_KERNEL_IQ4XS_DEQUANT  4
#define VKQUANT_KERNEL_Q8_0_QUANT     5
#define VKQUANT_KERNEL_Q4_0_QUANT     6
#define VKQUANT_KERNEL_Q4_1_DEQUANT   7
#define VKQUANT_KERNEL_Q5_0_DEQUANT   8
#define VKQUANT_KERNEL_Q5_1_DEQUANT   9
#define VKQUANT_KERNEL_Q8_1_DEQUANT   10
#define VKQUANT_KERNEL_Q2K_DEQUANT    11
#define VKQUANT_KERNEL_Q3K_DEQUANT    12
#define VKQUANT_KERNEL_Q5K_DEQUANT    13
#define VKQUANT_KERNEL_IQ4NL_DEQUANT  14
#define VKQUANT_KERNEL_IQ1S_DEQUANT   15
#define VKQUANT_KERNEL_IQ1M_DEQUANT   16
#define VKQUANT_KERNEL_IQ2XS_DEQUANT  17
#define VKQUANT_KERNEL_IQ2S_DEQUANT   18
#define VKQUANT_KERNEL_IQ2XXS_DEQUANT 19
#define VKQUANT_KERNEL_IQ3S_DEQUANT   20
#define VKQUANT_KERNEL_IQ3XXS_DEQUANT 21
#define VKQUANT_KERNEL_TQ1_0_DEQUANT  22
#define VKQUANT_KERNEL_TQ2_0_DEQUANT  23
#define VKQUANT_KERNEL_Q4_1_QUANT     24
#define VKQUANT_KERNEL_Q5_0_QUANT     25
#define VKQUANT_KERNEL_Q5_1_QUANT     26
#define VKQUANT_KERNEL_Q8_1_QUANT     27
#define VKQUANT_KERNEL_Q2K_QUANT      28
#define VKQUANT_KERNEL_Q3K_QUANT      29
#define VKQUANT_KERNEL_Q4K_QUANT      30
#define VKQUANT_KERNEL_Q5K_QUANT      31
#define VKQUANT_KERNEL_Q6K_QUANT      32
#define VKQUANT_KERNEL_COUNT          33

/* ── Capability tiers ──────────────────────────────────────────────────── */

typedef enum {
    VKQUANT_TIER_BASELINE   = 0,
    VKQUANT_TIER_SUBGROUP   = 1,
    VKQUANT_TIER_COOPMATRIX = 2,
} VkQuantTier_t;

/* ── Pipeline cache ────────────────────────────────────────────────────── */

#define VKQUANT_MAX_PIPELINES 256

typedef struct {
    uint64_t key;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    uint32_t kernel;
    uint32_t data_type;
    uint32_t tier;
    uint8_t  valid;
} vkquant_pipeline_entry_t;

/* ── Context ───────────────────────────────────────────────────────────── */

struct VkQuantContext {
    VkDevice device;
    VkPipelineCache pipeline_cache;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;

    /* capability flags */
    VkBool32 has_subgroup;
    VkBool32 has_coop_matrix;
    uint32_t max_subgroup_size;
    uint32_t max_compute_workgroup_size[3];
    VkBool32 has_shader_int64;
    VkBool32 has_push_descriptor;    /* active tier (highest supported) */
    VkQuantTier_t active_tier;

    /* push descriptor function (loaded via vkGetDeviceProcAddr) */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn;

    /* pipeline cache storage (open-addressing, linear probing) */
    vkquant_pipeline_entry_t pipelines[VKQUANT_MAX_PIPELINES];
    uint32_t pipeline_count;
};

/* ── Internal functions ────────────────────────────────────────────────── */

VkResult vkquant_load_shader_module(VkDevice device,
                                    const uint32_t* spirv,
                                    size_t spirv_words,
                                    VkShaderModule* out_module);

VkResult vkquant_ensure_pipeline(VkQuantContext* ctx,
                                 uint32_t kernel,
                                 uint32_t data_type,
                                 VkPipeline* out_pipeline);

VkPipelineLayout vkquant_get_pipeline_layout(VkQuantContext* ctx);

void vkquant_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                     const vkquant_push_constants_t* pc);

VkResult vkquant_alloc_descriptor_set(VkQuantContext* ctx,
                                      VkDescriptorSet* out_ds);

uint64_t vkquant_hash_key(uint32_t kernel, uint32_t data_type, uint32_t tier);

const uint32_t* vkquant_select_spirv(uint32_t kernel, uint32_t data_type,
                                     uint32_t tier, size_t* out_size);

#endif /* VKQUANT_INTERNAL_H */
