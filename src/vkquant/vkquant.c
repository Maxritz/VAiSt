/**
 * \file vkquant.c
 * \brief VKQuant implementation: context lifecycle, pipeline caching, dispatch.
 *
 * Follows the contract in AGENTS.md — lazy pipeline creation with open-addressing
 * cache, push descriptors where available, embedded SPIR-V shader selection
 * with tier fallback. Only baseline dequant shaders exist, so every dispatch
 * resolves to the baseline blob regardless of detected capabilities.
 */
#include "vkquant_internal.h"
#include "shaders_spv.h"

#include <string.h>
#include <stdlib.h>

#define VKQUANT_WORKGROUP_SIZE 256u

/* ── Shader blob lookup table ────────────────────────────────────────────── *
 * Maps (kernel, data_type, tier) to the embedded SPIR-V array.
 * Only combinations that have a compiled .comp shader are present.
 * ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t kernel;
    uint32_t data_type;
    uint32_t tier;
    const uint32_t *spirv;
    size_t spirv_size;
} shader_blob_t;

static const shader_blob_t s_shader_table[] = {
    /* baseline tier — the only tier with dequant shaders */
    {VKQUANT_KERNEL_Q8_0_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q8_0, vkquant_spv_baseline_dequant_q8_0_size},
    {VKQUANT_KERNEL_Q4_0_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q4_0, vkquant_spv_baseline_dequant_q4_0_size},
    {VKQUANT_KERNEL_Q4K_DEQUANT,  VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q4k_f32, vkquant_spv_baseline_dequant_q4k_f32_size},
    {VKQUANT_KERNEL_Q6K_DEQUANT,  VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q6k_f32, vkquant_spv_baseline_dequant_q6k_f32_size},
    {VKQUANT_KERNEL_IQ4XS_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq4xs_f32, vkquant_spv_baseline_dequant_iq4xs_f32_size},
    {VKQUANT_KERNEL_Q8_0_QUANT,   VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q8_0_f32, vkquant_spv_baseline_quantize_q8_0_f32_size},
    {VKQUANT_KERNEL_Q4_0_QUANT,   VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q4_0_f32, vkquant_spv_baseline_quantize_q4_0_f32_size},
    /* legacy 32-elem dequant */
    {VKQUANT_KERNEL_Q4_1_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q4_1_f32, vkquant_spv_baseline_dequant_q4_1_f32_size},
    {VKQUANT_KERNEL_Q5_0_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q5_0_f32, vkquant_spv_baseline_dequant_q5_0_f32_size},
    {VKQUANT_KERNEL_Q5_1_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q5_1_f32, vkquant_spv_baseline_dequant_q5_1_f32_size},
    {VKQUANT_KERNEL_Q8_1_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q8_1_f32, vkquant_spv_baseline_dequant_q8_1_f32_size},
    /* K-quant dequant */
    {VKQUANT_KERNEL_Q2K_DEQUANT,  VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q2k_f32, vkquant_spv_baseline_dequant_q2k_f32_size},
    {VKQUANT_KERNEL_Q3K_DEQUANT,  VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q3k_f32, vkquant_spv_baseline_dequant_q3k_f32_size},
    {VKQUANT_KERNEL_Q5K_DEQUANT,  VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_q5k_f32, vkquant_spv_baseline_dequant_q5k_f32_size},
    /* IQ / TQ dequant */
    {VKQUANT_KERNEL_IQ4NL_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq4_nl_f32, vkquant_spv_baseline_dequant_iq4_nl_f32_size},
    {VKQUANT_KERNEL_IQ1S_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq1_s_f32, vkquant_spv_baseline_dequant_iq1_s_f32_size},
    {VKQUANT_KERNEL_IQ1M_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq1_m_f32, vkquant_spv_baseline_dequant_iq1_m_f32_size},
    {VKQUANT_KERNEL_IQ2XS_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq2_xs_f32, vkquant_spv_baseline_dequant_iq2_xs_f32_size},
    {VKQUANT_KERNEL_IQ2S_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq2_s_f32, vkquant_spv_baseline_dequant_iq2_s_f32_size},
    {VKQUANT_KERNEL_IQ2XXS_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq2_xxs_f32, vkquant_spv_baseline_dequant_iq2_xxs_f32_size},
    {VKQUANT_KERNEL_IQ3S_DEQUANT, VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq3_s_f32, vkquant_spv_baseline_dequant_iq3_s_f32_size},
    {VKQUANT_KERNEL_IQ3XXS_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_iq3_xxs_f32, vkquant_spv_baseline_dequant_iq3_xxs_f32_size},
    {VKQUANT_KERNEL_TQ1_0_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_tq1_0_f32, vkquant_spv_baseline_dequant_tq1_0_f32_size},
    {VKQUANT_KERNEL_TQ2_0_DEQUANT,VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_dequant_tq2_0_f32, vkquant_spv_baseline_dequant_tq2_0_f32_size},
    /* legacy 32-elem forward quant */
    {VKQUANT_KERNEL_Q4_1_QUANT,   VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q4_1_f32, vkquant_spv_baseline_quantize_q4_1_f32_size},
    {VKQUANT_KERNEL_Q5_0_QUANT,   VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q5_0_f32, vkquant_spv_baseline_quantize_q5_0_f32_size},
    {VKQUANT_KERNEL_Q5_1_QUANT,   VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q5_1_f32, vkquant_spv_baseline_quantize_q5_1_f32_size},
    {VKQUANT_KERNEL_Q8_1_QUANT,   VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q8_1_f32, vkquant_spv_baseline_quantize_q8_1_f32_size},
    /* K-quant forward quant */
    {VKQUANT_KERNEL_Q2K_QUANT,    VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q2k_f32, vkquant_spv_baseline_quantize_q2k_f32_size},
    {VKQUANT_KERNEL_Q3K_QUANT,    VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q3k_f32, vkquant_spv_baseline_quantize_q3k_f32_size},
    {VKQUANT_KERNEL_Q4K_QUANT,    VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q4k_f32, vkquant_spv_baseline_quantize_q4k_f32_size},
    {VKQUANT_KERNEL_Q5K_QUANT,    VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q5k_f32, vkquant_spv_baseline_quantize_q5k_f32_size},
    {VKQUANT_KERNEL_Q6K_QUANT,    VKQUANT_DTYPE_F32, VKQUANT_TIER_BASELINE, vkquant_spv_baseline_quantize_q6k_f32, vkquant_spv_baseline_quantize_q6k_f32_size},
};
#define SHADER_TABLE_COUNT (sizeof(s_shader_table) / sizeof(s_shader_table[0]))

/* ── Shader selection ────────────────────────────────────────────────────── */

const uint32_t *vkquant_select_spirv(uint32_t kernel, uint32_t data_type,
                                     uint32_t tier, size_t *out_size) {
    for (uint32_t i = 0; i < SHADER_TABLE_COUNT; i++) {
        if (s_shader_table[i].kernel == kernel &&
            s_shader_table[i].data_type == data_type &&
            s_shader_table[i].tier == tier) {
            if (out_size) *out_size = s_shader_table[i].spirv_size;
            return s_shader_table[i].spirv;
        }
    }
    if (out_size) *out_size = 0;
    return NULL;
}

/* ── Hash key ────────────────────────────────────────────────────────────── */

uint64_t vkquant_hash_key(uint32_t kernel, uint32_t data_type, uint32_t tier) {
    return ((uint64_t)tier << 32) | ((uint64_t)data_type << 16) | (uint64_t)kernel;
}

/* ── Shader module loading ───────────────────────────────────────────────── */

VkResult vkquant_load_shader_module(VkDevice device,
                                    const uint32_t *spirv,
                                    size_t spirv_words,
                                    VkShaderModule *out_module) {
    VkShaderModuleCreateInfo smci;
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.pNext = NULL;
    smci.flags = 0;
    smci.codeSize = spirv_words;
    smci.pCode = spirv;
    return vkCreateShaderModule(device, &smci, NULL, out_module);
}

/* ── Descriptor set allocation (fallback path) ──────────────────────────── */

VkResult vkquant_alloc_descriptor_set(VkQuantContext *ctx, VkDescriptorSet *out_ds) {
    if (!ctx->descriptor_pool) return VK_ERROR_INITIALIZATION_FAILED;
    VkDescriptorSetAllocateInfo dsai;
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.pNext = NULL;
    dsai.descriptorPool = ctx->descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &ctx->set_layout;
    return vkAllocateDescriptorSets(ctx->device, &dsai, out_ds);
}

/* ── Push constants ──────────────────────────────────────────────────────── */

void vkquant_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                     const vkquant_push_constants_t *pc) {
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(vkquant_push_constants_t), pc);
}

/* ── Pipeline layout getter ──────────────────────────────────────────────── */

VkPipelineLayout vkquant_get_pipeline_layout(VkQuantContext *ctx) {
    return ctx ? ctx->pipeline_layout : VK_NULL_HANDLE;
}

/* ── Pipeline cache (open-addressing, linear probing) ──────────────────── */

static vkquant_pipeline_entry_t *cache_lookup(vkquant_pipeline_entry_t *cache, uint64_t key) {
    uint32_t mask = VKQUANT_MAX_PIPELINES - 1; /* 255 */
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKQUANT_MAX_PIPELINES; probe++) {
        if (!cache[idx].valid) return NULL;
        if (cache[idx].key == key) return &cache[idx];
        idx = (idx + 1) & mask;
    }
    return NULL;
}

static vkquant_pipeline_entry_t *cache_insert(vkquant_pipeline_entry_t *cache,
                                              uint64_t key,
                                              uint32_t kernel, uint32_t dt, uint32_t tier,
                                              VkPipeline pipeline, VkPipelineLayout layout) {
    uint32_t mask = VKQUANT_MAX_PIPELINES - 1;
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKQUANT_MAX_PIPELINES; probe++) {
        if (!cache[idx].valid) {
            cache[idx].key = key;
            cache[idx].pipeline = pipeline;
            cache[idx].layout = layout;
            cache[idx].kernel = kernel;
            cache[idx].data_type = dt;
            cache[idx].tier = tier;
            cache[idx].valid = 1;
            return &cache[idx];
        }
        idx = (idx + 1) & mask;
    }
    return NULL;
}

/* ── Pipeline creation / cache ───────────────────────────────────────────── */

VkResult vkquant_ensure_pipeline(VkQuantContext *ctx,
                                 uint32_t kernel,
                                 uint32_t data_type,
                                 VkPipeline *out_pipeline) {
    /*
     * TRUTH TABLE: shader fallback
     * ─────────────────────────────────────────────────────────────────
     * active_tier | try tiers in order          | result
     * COOPMATRIX  | coop -> subgroup -> base    | baseline blob (only blob)
     * SUBGROUP    | subgroup -> baseline        | baseline blob (only blob)
     * BASELINE    | baseline                      | baseline blob
     * ─────────────────────────────────────────────────────────────────
     * If no blob found at any tried tier -> VK_ERROR_FEATURE_NOT_PRESENT.
     */
    static const uint32_t try_order[] = {
        VKQUANT_TIER_COOPMATRIX, VKQUANT_TIER_SUBGROUP, VKQUANT_TIER_BASELINE
    };

    uint64_t key = 0;
    const uint32_t *spirv = NULL;
    size_t spirv_size = 0;
    uint32_t selected_tier = 0;

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t t = try_order[i];
        if (t > ctx->active_tier) continue; /* skip tiers the device can't support */
        spirv = vkquant_select_spirv(kernel, data_type, t, &spirv_size);
        if (spirv) {
            selected_tier = t;
            key = vkquant_hash_key(kernel, data_type, t);
            break;
        }
    }

    if (spirv == NULL) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Cache hit? */
    vkquant_pipeline_entry_t *entry = cache_lookup(ctx->pipelines, key);
    if (entry) {
        *out_pipeline = entry->pipeline;
        return VK_SUCCESS;
    }

    /* Cache miss — create pipeline */
    VkShaderModule sh;
    VkResult r = vkquant_load_shader_module(ctx->device, spirv, spirv_size, &sh);
    if (r != VK_SUCCESS) return r;

    VkComputePipelineCreateInfo cpci;
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.pNext = NULL;
    cpci.flags = 0;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.pNext = NULL;
    cpci.stage.flags = 0;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sh;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = NULL;
    cpci.layout = ctx->pipeline_layout;

    r = vkCreateComputePipelines(ctx->device, ctx->pipeline_cache,
                                 1, &cpci, NULL, out_pipeline);
    vkDestroyShaderModule(ctx->device, sh, NULL);
    if (r != VK_SUCCESS) return r;

    entry = cache_insert(ctx->pipelines, key, kernel, data_type,
                         selected_tier, *out_pipeline, ctx->pipeline_layout);
    if (entry) {
        ctx->pipeline_count++;
    }
    return VK_SUCCESS;
}

/* ── Dispatch helper ─────────────────────────────────────────────────────── */

/*
 * Records pipeline bind + push constants + descriptor push + vkCmdDispatch.
 * The descriptor set layout declares exactly two bindings (0 = input read,
 * 2 = output write); both are written on every dispatch.
 */
static VkResult vkquant_cmd_dispatch(VkQuantContext *ctx, VkCommandBuffer cmd,
    uint32_t kernel,
    const vkquant_push_constants_t *pc,
    uint32_t dispatch_x,
    VkBuffer buf_in, VkBuffer buf_out) {

    VkPipeline pipeline;
    VkResult r = vkquant_ensure_pipeline(ctx, kernel, VKQUANT_DTYPE_F32, &pipeline);
    if (r != VK_SUCCESS) return r;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkquant_push_pc(cmd, ctx->pipeline_layout, pc);

    VkDescriptorBufferInfo info_in   = { buf_in,  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo info_out  = { buf_out, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[2];
    memset(writes, 0, sizeof(writes));
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info_in;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 2;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &info_out;

    if (ctx->push_desc_fn) {
        ctx->push_desc_fn(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx->pipeline_layout, 0,
                          2, writes);
    } else {
        VkDescriptorSet ds;
        r = vkquant_alloc_descriptor_set(ctx, &ds);
        if (r != VK_SUCCESS) return r;
        for (int i = 0; i < 2; i++) {
            writes[i].dstSet = ds; /* set dstSet BEFORE vkUpdateDescriptorSets */
        }
        vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ctx->pipeline_layout, 0, 1, &ds, 0, NULL);
    }

    if (dispatch_x == 0) dispatch_x = 1;
    vkCmdDispatch(cmd, dispatch_x, 1, 1);
    return VK_SUCCESS;
}

static uint32_t elem_to_groups(uint32_t num_elements) {
    uint32_t g = (num_elements + VKQUANT_WORKGROUP_SIZE - 1) / VKQUANT_WORKGROUP_SIZE;
    return g ? g : 1;
}

/* ── Capability detection ────────────────────────────────────────────────── */

VkResult vkquant_init_capabilities(VkQuantContext *ctx, VkPhysicalDevice pd) {
    if (!ctx) return VK_ERROR_INITIALIZATION_FAILED;

    /* Single VKRuntime implementation: shaderInt64 / subgroup / cooperative
       matrix / push-descriptor detection plus the tier ladder. */
    VkRuntimeCaps caps;
    VkResult r = vkr_detect_capabilities(pd, ctx->device, &caps);
    if (r != VK_SUCCESS) return r;

    ctx->has_shader_int64 = caps.has_shader_int64;
    ctx->has_coop_matrix  = caps.has_coop_matrix;
    ctx->has_subgroup     = caps.has_subgroup;
    ctx->max_subgroup_size = caps.subgroup_size;
    memcpy(ctx->max_compute_workgroup_size, caps.max_workgroup_size,
           sizeof(ctx->max_compute_workgroup_size));

    /* Highest supported tier (informational — only baseline shaders exist) */
    if (ctx->has_coop_matrix)
        ctx->active_tier = VKQUANT_TIER_COOPMATRIX;
    else if (ctx->has_subgroup)
        ctx->active_tier = VKQUANT_TIER_SUBGROUP;
    else
        ctx->active_tier = VKQUANT_TIER_BASELINE;

    return VK_SUCCESS;
}

/* ── Context lifecycle ──────────────────────────────────────────────────── */

VkResult vkquant_create_context(VkPhysicalDevice pd, VkDevice device,
                                VkQuantContext **pp_ctx) {
    if (!pp_ctx) return VK_ERROR_INITIALIZATION_FAILED;

    VkQuantContext *ctx = (VkQuantContext *)calloc(1, sizeof(VkQuantContext));
    if (!ctx) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ctx->device = device;

    /* Capability detection + push-desc fn load via VKRuntime (single
       implementation; was five inline copies). */
    VkRuntimeCaps caps;
    VkResult r = vkr_detect_capabilities(pd, device, &caps);
    if (r != VK_SUCCESS) { free(ctx); return r; }

    ctx->has_shader_int64 = caps.has_shader_int64;
    ctx->has_coop_matrix  = caps.has_coop_matrix;
    ctx->has_subgroup     = caps.has_subgroup;
    ctx->max_subgroup_size = caps.subgroup_size;
    memcpy(ctx->max_compute_workgroup_size, caps.max_workgroup_size,
           sizeof(ctx->max_compute_workgroup_size));
    ctx->push_desc_fn = caps.push_desc_fn;
    ctx->has_push_descriptor = caps.has_push_descriptor;

    /* Active tier from the vkr ladder (2=coopmatrix, 1=subgroup, 0=baseline) */
    if (caps.arch_index == 2)
        ctx->active_tier = VKQUANT_TIER_COOPMATRIX;
    else if (caps.arch_index == 1)
        ctx->active_tier = VKQUANT_TIER_SUBGROUP;
    else
        ctx->active_tier = VKQUANT_TIER_BASELINE;

    /* Descriptor set layout: 2 SSBO bindings (binding 0 read, binding 2 write) */
    VkDescriptorSetLayoutBinding bindings[2];
    memset(bindings, 0, sizeof(bindings));
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 2;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags;
    binding_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    binding_flags.pNext = NULL;
    binding_flags.bindingCount = 2;
    VkDescriptorBindingFlags flags[2] = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    };
    binding_flags.pBindingFlags = flags;

    VkDescriptorSetLayoutCreateInfo dslci;
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.pNext = ctx->has_push_descriptor ? (const void *)&binding_flags : NULL;
    dslci.flags = ctx->has_push_descriptor
        ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT : 0;
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;

    r = vkCreateDescriptorSetLayout(device, &dslci, NULL, &ctx->set_layout);
    if (r != VK_SUCCESS) { free(ctx); return r; }

    /* Pipeline layout: 1 descriptor set + 16-byte push constant range
       (push-constant range is lib-specific, stays local). */
    VkPushConstantRange pc_range;
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkquant_push_constants_t); /* 16 bytes */

    r = vkr_create_pipeline_layout(device, ctx->set_layout, 1, &pc_range,
                                   &ctx->pipeline_layout);
    if (r != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, ctx->set_layout, NULL);
        free(ctx);
        return r;
    }

    /* Pipeline cache (driver-level, accelerates lazy pipeline creation) */
    r = vkr_create_pipeline_cache(device, &ctx->pipeline_cache);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, ctx->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device, ctx->set_layout, NULL);
        free(ctx);
        return r;
    }

    /* Descriptor pool only needed for non-push-descriptor fallback */
    if (!ctx->has_push_descriptor) {
        r = vkr_create_descriptor_pool(device, VKQUANT_MAX_PIPELINES,
                                       VKQUANT_MAX_PIPELINES * 2,
                                       &ctx->descriptor_pool);
        if (r != VK_SUCCESS) {
            vkDestroyPipelineCache(device, ctx->pipeline_cache, NULL);
            vkDestroyPipelineLayout(device, ctx->pipeline_layout, NULL);
            vkDestroyDescriptorSetLayout(device, ctx->set_layout, NULL);
            free(ctx);
            return r;
        }
    }

    *pp_ctx = ctx;
    return VK_SUCCESS;
}

void vkquant_destroy_context(VkQuantContext *ctx) {
    if (!ctx) return;
    vkquant_flush_pipelines(ctx);
    if (ctx->descriptor_pool)
        vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
    if (ctx->set_layout)
        vkDestroyDescriptorSetLayout(ctx->device, ctx->set_layout, NULL);
    if (ctx->pipeline_layout)
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
    if (ctx->pipeline_cache)
        vkDestroyPipelineCache(ctx->device, ctx->pipeline_cache, NULL);
    free(ctx);
}

void vkquant_flush_pipelines(VkQuantContext *ctx) {
    if (!ctx) return;
    for (uint32_t i = 0; i < VKQUANT_MAX_PIPELINES; i++) {
        if (ctx->pipelines[i].valid) {
            vkDestroyPipeline(ctx->device, ctx->pipelines[i].pipeline, NULL);
            ctx->pipelines[i].valid = 0;
        }
    }
    ctx->pipeline_count = 0;
}

/* ── Arch queries ────────────────────────────────────────────────────────── */

uint32_t vkquant_get_arch_index(VkQuantContext *ctx) {
    return ctx ? (uint32_t)ctx->active_tier : 0;
}

const char *vkquant_get_arch_name(VkQuantContext *ctx) {
    if (!ctx) return "unknown";
    static const char *names[3] = { "baseline", "subgroup", "coopmatrix" };
    return names[(uint32_t)ctx->active_tier];
}

/* ── Public API: dequantization (f32) ───────────────────────────────────── */

VkResult vkquant_dequant_q8_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_Q8_0_DEQUANT,
        &pc, elem_to_groups(num_blocks * 32u),
        input, output);
}

VkResult vkquant_dequant_q4_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_Q4_0_DEQUANT,
        &pc, elem_to_groups(num_blocks * 32u),
        input, output);
}

/* ── Public API: Q4_K / Q6_K / IQ4_XS dequantization (f32) ────────────── */

VkResult vkquant_dequant_q4k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                 uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_Q4K_DEQUANT,
        &pc, num_blocks,
        input, output);
}

VkResult vkquant_dequant_q6k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                 uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_Q6K_DEQUANT,
        &pc, num_blocks,
        input, output);
}

VkResult vkquant_dequant_iq4xs_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_IQ4XS_DEQUANT,
        &pc, num_blocks,
        input, output);
}

/* ── Public API: legacy 32-elem dequantization (f16-scale ggml layouts) ──
 * Block sizes: Q4_1 = 20 B, Q5_0 = 22 B, Q5_1 = 24 B, Q8_1 = 36 B.
 * One thread per output element (32 elems/block). */

static VkResult vkquant_dequant_legacy(VkQuantContext *ctx, VkCommandBuffer cmd,
    uint32_t num_blocks, uint32_t kernel, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, kernel,
        &pc, elem_to_groups(num_blocks * 32u),
        input, output);
}

VkResult vkquant_dequant_q4_1_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_legacy(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q4_1_DEQUANT, input, output);
}

VkResult vkquant_dequant_q5_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_legacy(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q5_0_DEQUANT, input, output);
}

VkResult vkquant_dequant_q5_1_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_legacy(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q5_1_DEQUANT, input, output);
}

VkResult vkquant_dequant_q8_1_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_legacy(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q8_1_DEQUANT, input, output);
}

/* ── Public API: K-quant dequantization (f32, 256 elems/block) ────────── */

static VkResult vkquant_dequant_k(VkQuantContext *ctx, VkCommandBuffer cmd,
    uint32_t num_blocks, uint32_t kernel, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, kernel,
        &pc, num_blocks,
        input, output);
}

VkResult vkquant_dequant_q2k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                 uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q2K_DEQUANT, input, output);
}

VkResult vkquant_dequant_q3k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                 uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q3K_DEQUANT, input, output);
}

VkResult vkquant_dequant_q5k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                 uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q5K_DEQUANT, input, output);
}

/* ── Public API: IQ / TQ dequantization (f32) ────────────────────────────
 * IQ4_NL uses 32-elem blocks (18 B); the other IQ/TQ formats use 256-elem
 * super-blocks. */

VkResult vkquant_dequant_iq4_nl_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                    uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_legacy(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ4NL_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq1_s_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ1S_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq1_m_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ1M_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq2_xs_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                    uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ2XS_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq2_s_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ2S_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq2_xxs_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                     uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ2XXS_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq3_s_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ3S_DEQUANT, input, output);
}

VkResult vkquant_dequant_iq3_xxs_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                     uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_IQ3XXS_DEQUANT, input, output);
}

VkResult vkquant_dequant_tq1_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_TQ1_0_DEQUANT, input, output);
}

VkResult vkquant_dequant_tq2_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_dequant_k(ctx, cmd, num_blocks, VKQUANT_KERNEL_TQ2_0_DEQUANT, input, output);
}

/* ── Public API: forward quantization (f32 -> block) ────────────────────
 * Each new quantizer handles one block per workgroup (32-elem for legacy,
 * 256-elem for K-quants), so dispatch groups = num_blocks. */

static VkResult vkquant_quantize_one(VkQuantContext *ctx, VkCommandBuffer cmd,
    uint32_t num_blocks, uint32_t kernel, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, kernel,
        &pc, num_blocks ? num_blocks : 1u,
        input, output);
}

VkResult vkquant_quantize_q4_1_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q4_1_QUANT, input, output);
}

VkResult vkquant_quantize_q5_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q5_0_QUANT, input, output);
}

VkResult vkquant_quantize_q5_1_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q5_1_QUANT, input, output);
}

VkResult vkquant_quantize_q8_1_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q8_1_QUANT, input, output);
}

VkResult vkquant_quantize_q2k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q2K_QUANT, input, output);
}

VkResult vkquant_quantize_q3k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q3K_QUANT, input, output);
}

VkResult vkquant_quantize_q4k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q4K_QUANT, input, output);
}

VkResult vkquant_quantize_q5k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q5K_QUANT, input, output);
}

VkResult vkquant_quantize_q6k_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                  uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    return vkquant_quantize_one(ctx, cmd, num_blocks, VKQUANT_KERNEL_Q6K_QUANT, input, output);
}

/* ── Public API: forward quantization (f32 -> Q8_0 / Q4_0) ──────────────
 * Our Q8_0/Q4_0 formats use an f32 scale (matching our existing dequant
 * shaders). Each workgroup of 256 threads handles 8 blocks of 32 elements.
 */

VkResult vkquant_quantize_q8_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_Q8_0_QUANT,
        &pc, (num_blocks + 7u) / 8u,
        input, output);
}

VkResult vkquant_quantize_q4_0_f32(VkQuantContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_blocks, VkBuffer input, VkBuffer output) {
    vkquant_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_blocks = num_blocks;
    return vkquant_cmd_dispatch(ctx, cmd, VKQUANT_KERNEL_Q4_0_QUANT,
        &pc, (num_blocks + 7u) / 8u,
        input, output);
}
