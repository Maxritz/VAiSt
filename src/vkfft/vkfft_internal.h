/**
 * \file vkfft_internal.h
 * \brief Internal VKFFT structures: push constants, pipeline cache, plan.
 *
 * Mirrors the design of vkmath_internal.h: same open-addressing pipeline-cache
 * strategy, same tier detection, same embedded-SPIR-V pattern. VkFFTPlan is the
 * context (rocfft plan semantics) and adds the FFT size n and log2(n).
 */
#ifndef VKFFT_INTERNAL_H
#define VKFFT_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkfft.h"

/* ── Push constant block (std140, 16 bytes, must match GLSL exactly) ────── */

typedef struct {
    uint32_t n;        /* offset  0: FFT size (power of two, <= 256)       */
    uint32_t log2n;    /* offset  4: log2(n)                                */
    uint32_t _pad0;    /* offset  8: padding (no uint64_t in push constants) */
    uint32_t _pad1;    /* offset 12: padding                                */
} vkfft_push_constants_t;

/* Static assert (C99-compatible): struct must be exactly 16 bytes */
typedef char vkfft_pc_static_assert[sizeof(vkfft_push_constants_t) == 16 ? 1 : -1];

/* ── Data types ────────────────────────────────────────────────────────── */

#define VKFFT_DTYPE_F32 0

/* ── Kernel types ──────────────────────────────────────────────────────── */

#define VKFFT_KERNEL_FFT_F32 0
#define VKFFT_KERNEL_COUNT   1

/* ── Capability tiers ──────────────────────────────────────────────────── */

typedef enum {
    VKFFT_TIER_BASELINE   = 0,
    VKFFT_TIER_SUBGROUP   = 1,
    VKFFT_TIER_COOPMATRIX = 2,
} VkFFTTier_t;

/* ── Pipeline cache ────────────────────────────────────────────────────── */

#define VKFFT_MAX_PIPELINES 256

typedef struct {
    uint64_t key;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    uint32_t kernel;
    uint32_t data_type;
    uint32_t tier;
    uint8_t  valid;
} vkfft_pipeline_entry_t;

/* ── Plan (context) ────────────────────────────────────────────────────── */

struct VkFFTPlan {
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
    VkFFTTier_t active_tier;

    /* push descriptor function (loaded via vkGetDeviceProcAddr) */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn;

    /* pipeline cache storage (open-addressing, linear probing) */
    vkfft_pipeline_entry_t pipelines[VKFFT_MAX_PIPELINES];
    uint32_t pipeline_count;

    /* FFT size */
    uint32_t n;
    uint32_t log2n;
};

/* ── Internal functions ────────────────────────────────────────────────── */

VkResult vkfft_load_shader_module(VkDevice device,
                                  const uint32_t* spirv,
                                  size_t spirv_words,
                                  VkShaderModule* out_module);

VkResult vkfft_ensure_pipeline(VkFFTPlan* plan,
                               uint32_t kernel,
                               uint32_t data_type,
                               VkPipeline* out_pipeline);

void vkfft_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                   const vkfft_push_constants_t* pc);

VkResult vkfft_alloc_descriptor_set(VkFFTPlan* plan, VkDescriptorSet* out_ds);

uint64_t vkfft_hash_key(uint32_t kernel, uint32_t data_type, uint32_t tier);

const uint32_t* vkfft_select_spirv(uint32_t kernel, uint32_t data_type,
                                   uint32_t tier, size_t* out_size);

#endif /* VKFFT_INTERNAL_H */
