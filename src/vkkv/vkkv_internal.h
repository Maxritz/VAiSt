/**
 * \file vkkv_internal.h
 * \brief Internal VKKV structures: push constants, context, helper decls.
 *
 * Mirrors the vkquant/vkmath internal-header style: the context struct lives
 * here (opaque to the public API), the push-constant block is defined once to
 * match the GLSL layout exactly, and internal helpers are declared for the
 * .c file. Only the baseline apply shader exists, so there is no tier ladder.
 */
#ifndef VKKV_INTERNAL_H
#define VKKV_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"
#include "vkkv.h"

/* ── Push constant block (std140, 16 bytes, must match GLSL exactly) ───── */

typedef struct {
    uint32_t num_elements; /* offset  0: total output elements = n * tgt_dim */
    uint32_t tgt_dim;      /* offset  4: target feature dimension            */
    uint32_t src_dim;      /* offset  8: source feature dimension            */
    uint32_t w_offset;     /* offset 12: element offset of Wh in the W buffer */
} vkkv_push_constants_t;

/* C99 static assert: the struct must be exactly 16 bytes */
typedef char vkkv_pc_static_assert[sizeof(vkkv_push_constants_t) == 16 ? 1 : -1];

/* ── Workgroup size (matches apply.comp) ───────────────────────────────── */

#define VKKV_WORKGROUP_SIZE 256u

/* ── Context (opaque in the public API) ────────────────────────────────── */

struct VkKVTransfer {
    VkDevice device;
    VkPhysicalDevice pd;
    VkRuntime *rt;             /* internal runtime: W alloc + upload/download */
    VkQueue queue;             /* queue family 0 fetched at create            */
    VkCommandPool cmd_pool;    /* internal pool for vkr_upload/download       */
    VkCommandBuffer cmd;       /* internal cmd (owned; used by staging)       */

    uint32_t n_heads;
    uint32_t src_dim;
    uint32_t tgt_dim;
    float ridge_lambda;

    VkPipelineCache pipeline_cache;
    VkDescriptorPool descriptor_pool;   /* push-desc fallback only            */
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;                /* lazily created on first apply      */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn;
    VkBool32 has_push_descriptor;

    VkBuffer w_buffer;                  /* [n_heads][src_dim*tgt_dim] floats  */
    VkDeviceMemory w_memory;
    VkDeviceSize w_size;                /* bytes                              */
    VkBool32 fitted;                    /* set by vkkv_fit_cpu after upload   */
};

/* ── Internal helpers ──────────────────────────────────────────────────── */

VkResult vkkv_ensure_pipeline(VkKVTransfer *t);
VkResult vkkv_alloc_descriptor_set(VkKVTransfer *t, VkDescriptorSet *out);
const uint32_t *vkkv_select_spirv(size_t *out_size);

#endif /* VKKV_INTERNAL_H */
