/**
 * \file vkmath.c
 * \brief VKMath implementation: context lifecycle, pipeline caching, op dispatch.
 *
 * Follows the contract in AGENTS.md — lazy pipeline creation with open-addressing
 * cache, push descriptors where available, embedded SPIR-V shader selection
 * with tier fallback (coopmatrix -> subgroup -> baseline).
 */
#include "vkmath_internal.h"
#include "shaders_spv.h"

#include <string.h>
#include <stdlib.h>

#define VKMATH_WORKGROUP_SIZE 256u

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
    /* baseline tier — all kernels with baseline .comp files */
    {VKMATH_KERNEL_RELU,       VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_relu_f32,         vkmath_spv_baseline_relu_f32_size},
    {VKMATH_KERNEL_RELU,       VKMATH_DTYPE_F16, VKMATH_TIER_BASELINE, vkmath_spv_baseline_relu_f16,         vkmath_spv_baseline_relu_f16_size},
    {VKMATH_KERNEL_SILU,       VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_silu_f32,         vkmath_spv_baseline_silu_f32_size},
    {VKMATH_KERNEL_SILU,       VKMATH_DTYPE_F16, VKMATH_TIER_BASELINE, vkmath_spv_baseline_silu_f16,         vkmath_spv_baseline_silu_f16_size},
    {VKMATH_KERNEL_GELU,       VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_gelu_f32,         vkmath_spv_baseline_gelu_f32_size},
    {VKMATH_KERNEL_GELU,       VKMATH_DTYPE_F16, VKMATH_TIER_BASELINE, vkmath_spv_baseline_gelu_f16,         vkmath_spv_baseline_gelu_f16_size},
    {VKMATH_KERNEL_TANH,       VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_tanh_f32,         vkmath_spv_baseline_tanh_f32_size},
    {VKMATH_KERNEL_TANH,       VKMATH_DTYPE_F16, VKMATH_TIER_BASELINE, vkmath_spv_baseline_tanh_f16,         vkmath_spv_baseline_tanh_f16_size},
    {VKMATH_KERNEL_SIGMOID,    VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_sigmoid_f32,      vkmath_spv_baseline_sigmoid_f32_size},
    {VKMATH_KERNEL_SIGMOID,    VKMATH_DTYPE_F16, VKMATH_TIER_BASELINE, vkmath_spv_baseline_sigmoid_f16,      vkmath_spv_baseline_sigmoid_f16_size},
    {VKMATH_KERNEL_ADD,        VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_add_f32,          vkmath_spv_baseline_add_f32_size},
    {VKMATH_KERNEL_MUL,        VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_mul_f32,          vkmath_spv_baseline_mul_f32_size},
    {VKMATH_KERNEL_ADD_MUL,    VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_add_mul_f32,      vkmath_spv_baseline_add_mul_f32_size},
    {VKMATH_KERNEL_ADD_MUL,    VKMATH_DTYPE_F16, VKMATH_TIER_BASELINE, vkmath_spv_baseline_add_mul_f16,      vkmath_spv_baseline_add_mul_f16_size},
    {VKMATH_KERNEL_SCALE,      VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_scale_f32,        vkmath_spv_baseline_scale_f32_size},
    {VKMATH_KERNEL_MAX_REDUCE, VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_max_reduce_dim_f32, vkmath_spv_baseline_max_reduce_dim_f32_size},
    {VKMATH_KERNEL_SUM_REDUCE, VKMATH_DTYPE_F32, VKMATH_TIER_BASELINE, vkmath_spv_baseline_sum_reduce_dim_f32, vkmath_spv_baseline_sum_reduce_dim_f32_size},
    /* subgroup tier — only kernels with subgroup .comp files */
    {VKMATH_KERNEL_SILU,       VKMATH_DTYPE_F32, VKMATH_TIER_SUBGROUP, vkmath_spv_subgroup_silu_f32,          vkmath_spv_subgroup_silu_f32_size},
    {VKMATH_KERNEL_SILU,       VKMATH_DTYPE_F16, VKMATH_TIER_SUBGROUP, vkmath_spv_subgroup_silu_f16,          vkmath_spv_subgroup_silu_f16_size},
    {VKMATH_KERNEL_GELU,       VKMATH_DTYPE_F32, VKMATH_TIER_SUBGROUP, vkmath_spv_subgroup_gelu_f32,          vkmath_spv_subgroup_gelu_f32_size},
    {VKMATH_KERNEL_SIGMOID,    VKMATH_DTYPE_F32, VKMATH_TIER_SUBGROUP, vkmath_spv_subgroup_sigmoid_f32,       vkmath_spv_subgroup_sigmoid_f32_size},
    {VKMATH_KERNEL_MAX_REDUCE, VKMATH_DTYPE_F32, VKMATH_TIER_SUBGROUP, vkmath_spv_subgroup_max_reduce_dim_f32, vkmath_spv_subgroup_max_reduce_dim_f32_size},
    {VKMATH_KERNEL_SUM_REDUCE, VKMATH_DTYPE_F32, VKMATH_TIER_SUBGROUP, vkmath_spv_subgroup_sum_reduce_dim_f32,  vkmath_spv_subgroup_sum_reduce_dim_f32_size},
};
#define SHADER_TABLE_COUNT (sizeof(s_shader_table) / sizeof(s_shader_table[0]))

/* ── Shader selection ────────────────────────────────────────────────────── */

const uint32_t *vkmath_select_spirv(uint32_t kernel, uint32_t data_type,
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

uint64_t vkmath_hash_key(uint32_t kernel, uint32_t data_type, uint32_t tier) {
    return ((uint64_t)tier << 32) | ((uint64_t)data_type << 16) | (uint64_t)kernel;
}

/* ── Shader module loading ───────────────────────────────────────────────── */

VkResult vkmath_load_shader_module(VkDevice device,
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

VkResult vkmath_alloc_descriptor_set(VkMathContext *ctx, VkDescriptorSet *out_ds) {
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

void vkmath_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                     const vkmath_push_constants_t *pc) {
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(vkmath_push_constants_t), pc);
}

/* ── Pipeline layout getter ──────────────────────────────────────────────── */

VkPipelineLayout vkmath_get_pipeline_layout(VkMathContext *ctx) {
    return ctx ? ctx->pipeline_layout : VK_NULL_HANDLE;
}

/* ── Pipeline cache (open-addressing, linear probing) ──────────────────── */

static vkmath_pipeline_entry_t *cache_lookup(vkmath_pipeline_entry_t *cache, uint64_t key) {
    uint32_t mask = VKMATH_MAX_PIPELINES - 1; /* 255 */
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKMATH_MAX_PIPELINES; probe++) {
        if (!cache[idx].valid) return NULL;
        if (cache[idx].key == key) return &cache[idx];
        idx = (idx + 1) & mask;
    }
    return NULL;
}

static vkmath_pipeline_entry_t *cache_insert(vkmath_pipeline_entry_t *cache,
                                              uint64_t key,
                                              uint32_t kernel, uint32_t dt, uint32_t tier,
                                              VkPipeline pipeline, VkPipelineLayout layout) {
    uint32_t mask = VKMATH_MAX_PIPELINES - 1;
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKMATH_MAX_PIPELINES; probe++) {
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

VkResult vkmath_ensure_pipeline(VkMathContext *ctx,
                                 uint32_t kernel,
                                 uint32_t data_type,
                                 VkPipeline *out_pipeline) {
    /*
     * TRUTH TABLE: shader fallback
     * ─────────────────────────────────────────────────────────────────
     * active_tier | try tiers in order          | result
     * COOPMATRIX  | coop -> subgroup -> base    | subgroup or baseline blob
     * SUBGROUP    | subgroup -> baseline        | subgroup or baseline blob
     * BASELINE    | baseline                      | baseline blob
     * ─────────────────────────────────────────────────────────────────
     * If no blob found at any tried tier -> VK_ERROR_FEATURE_NOT_PRESENT.
     */
    static const uint32_t try_order[] = {
        VKMATH_TIER_COOPMATRIX, VKMATH_TIER_SUBGROUP, VKMATH_TIER_BASELINE
    };

    uint64_t key = 0;
    const uint32_t *spirv = NULL;
    size_t spirv_size = 0;
    uint32_t selected_tier = 0;

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t t = try_order[i];
        if (t > ctx->active_tier) continue; /* skip tiers the device can't support */
        spirv = vkmath_select_spirv(kernel, data_type, t, &spirv_size);
        if (spirv) {
            selected_tier = t;
            key = vkmath_hash_key(kernel, data_type, t);
            break;
        }
    }

    if (spirv == NULL) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Cache hit? */
    vkmath_pipeline_entry_t *entry = cache_lookup(ctx->pipelines, key);
    if (entry) {
        *out_pipeline = entry->pipeline;
        return VK_SUCCESS;
    }

    /* Cache miss — create pipeline */
    VkShaderModule sh;
    VkResult r = vkmath_load_shader_module(ctx->device, spirv, spirv_size, &sh);
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
 * num_inputs=1 for unary/reduce (buf_b unused, reused as dummy),
 * num_inputs=2 for binary/ternary.
 */
static VkResult vkmath_cmd_dispatch(VkMathContext *ctx, VkCommandBuffer cmd,
    uint32_t kernel, uint32_t data_type,
    const vkmath_push_constants_t *pc,
    uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z,
    VkBuffer buf_a, VkBuffer buf_b, VkBuffer buf_out,
    uint32_t num_inputs) {

    VkPipeline pipeline;
    VkResult r = vkmath_ensure_pipeline(ctx, kernel, data_type, &pipeline);
    if (r != VK_SUCCESS) return r;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkmath_push_pc(cmd, ctx->pipeline_layout, pc);

    /* Build descriptor writes — always write all 3 bindings so the set
       layout (which declares all 3) is fully satisfied.  For unary ops
       buf_b is VK_NULL_HANDLE and we reuse buf_a as a harmless placeholder
       for binding 1. */
    VkDescriptorBufferInfo info_a   = { buf_a,  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo info_b   = { buf_b ? buf_b : buf_a, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo info_out = { buf_out, 0, VK_WHOLE_SIZE };
    const VkDescriptorBufferInfo *infos[3] = { &info_a, &info_b, &info_out };

    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = infos[i];
    }

    if (ctx->push_desc_fn) {
        ctx->push_desc_fn(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx->pipeline_layout, 0,
                          num_inputs >= 2 ? 3 : 2, writes);
    } else {
        VkDescriptorSet ds;
        r = vkmath_alloc_descriptor_set(ctx, &ds);
        if (r != VK_SUCCESS) return r;
        for (int i = 0; i < 3; i++) {
            writes[i].dstSet = ds;
        }
        vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ctx->pipeline_layout, 0, 1, &ds, 0, NULL);
    }

    if (dispatch_x == 0) dispatch_x = 1;
    vkCmdDispatch(cmd, dispatch_x, dispatch_y, dispatch_z);
    return VK_SUCCESS;
}

static uint32_t elem_to_groups(uint32_t num_elements) {
    uint32_t g = (num_elements + VKMATH_WORKGROUP_SIZE - 1) / VKMATH_WORKGROUP_SIZE;
    return g ? g : 1;
}
/* ── Capability detection ────────────────────────────────────────────────── */

VkResult vkmath_init_capabilities(VkMathContext *ctx, VkPhysicalDevice pd) {
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

    /* Highest supported tier */
    if (ctx->has_coop_matrix)
        ctx->active_tier = VKMATH_TIER_COOPMATRIX;
    else if (ctx->has_subgroup)
        ctx->active_tier = VKMATH_TIER_SUBGROUP;
    else
        ctx->active_tier = VKMATH_TIER_BASELINE;

    return VK_SUCCESS;
}

/* ── Context lifecycle ──────────────────────────────────────────────────── */

VkResult vkmath_create_context(VkPhysicalDevice pd, VkDevice device,
                               VkMathContext **pp_ctx) {
    if (!pp_ctx) return VK_ERROR_INITIALIZATION_FAILED;

    VkMathContext *ctx = (VkMathContext *)calloc(1, sizeof(VkMathContext));
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
        ctx->active_tier = VKMATH_TIER_COOPMATRIX;
    else if (caps.arch_index == 1)
        ctx->active_tier = VKMATH_TIER_SUBGROUP;
    else
        ctx->active_tier = VKMATH_TIER_BASELINE;

    /* Descriptor set layout: 3 SSBO bindings (read, read, write) */
    VkDescriptorSetLayoutBinding bindings[3];
    memset(bindings, 0, sizeof(bindings));
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags;
    binding_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    binding_flags.pNext = NULL;
    binding_flags.bindingCount = 3;
    VkDescriptorBindingFlags flags[3] = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    };
    binding_flags.pBindingFlags = flags;

    VkDescriptorSetLayoutCreateInfo dslci;
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.pNext = ctx->has_push_descriptor ? (const void *)&binding_flags : NULL;
    dslci.flags = ctx->has_push_descriptor
        ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT : 0;
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;

    r = vkCreateDescriptorSetLayout(device, &dslci, NULL, &ctx->set_layout);
    if (r != VK_SUCCESS) { free(ctx); return r; }

    /* Pipeline layout: 1 descriptor set + 72-byte push constant range
       (push-constant range is lib-specific, stays local). */
    VkPushConstantRange pc_range;
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkmath_push_constants_t); /* 72 bytes */

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
        r = vkr_create_descriptor_pool(device, VKMATH_MAX_PIPELINES,
                                       VKMATH_MAX_PIPELINES * 3,
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

void vkmath_destroy_context(VkMathContext *ctx) {
    if (!ctx) return;
    vkmath_flush_pipelines(ctx);
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

void vkmath_flush_pipelines(VkMathContext *ctx) {
    if (!ctx) return;
    for (uint32_t i = 0; i < VKMATH_MAX_PIPELINES; i++) {
        if (ctx->pipelines[i].valid) {
            vkDestroyPipeline(ctx->device, ctx->pipelines[i].pipeline, NULL);
            ctx->pipelines[i].valid = 0;
        }
    }
    ctx->pipeline_count = 0;
}

/* ── Arch queries ────────────────────────────────────────────────────────── */

uint32_t vkmath_get_arch_index(VkMathContext *ctx) {
    return ctx ? (uint32_t)ctx->active_tier : 0;
}

const char *vkmath_get_arch_name(VkMathContext *ctx) {
    if (!ctx) return "unknown";
    static const char *names[3] = { "baseline", "subgroup", "coopmatrix" };
    return names[(uint32_t)ctx->active_tier];
}
/* ── Public API: elementwise unary (f32) ────────────────────────────────── */

VkResult vkmath_relu_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_RELU, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_silu_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SILU, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_gelu_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_GELU, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_tanh_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_TANH, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_sigmoid_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                            uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SIGMOID, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

/* ── Public API: elementwise unary (f16) ────────────────────────────────── */

VkResult vkmath_relu_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_RELU, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_silu_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SILU, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_gelu_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_GELU, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_tanh_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                         uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_TANH, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_sigmoid_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                            uint32_t num_elements, VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SIGMOID, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

/* ── Public API: binary ops (f32) ───────────────────────────────────────── */

VkResult vkmath_add_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                        uint32_t num_elements, VkBuffer a, VkBuffer b, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_ADD, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        a, b, output, 2);
}

VkResult vkmath_mul_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                        uint32_t num_elements, VkBuffer a, VkBuffer b, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_MUL, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        a, b, output, 2);
}

VkResult vkmath_add_mul_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                            uint32_t num_elements, VkBuffer a, VkBuffer b,
                            float alpha, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    pc.alpha = alpha;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_ADD_MUL, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        a, b, output, 2);
}

VkResult vkmath_scale_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                          uint32_t num_elements, float alpha,
                          VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    pc.alpha = alpha;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SCALE, VKMATH_DTYPE_F32,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

/* ── Public API: binary ops (f16) ───────────────────────────────────────── */

VkResult vkmath_add_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                        uint32_t num_elements, VkBuffer a, VkBuffer b, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_ADD, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        a, b, output, 2);
}

VkResult vkmath_mul_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                        uint32_t num_elements, VkBuffer a, VkBuffer b, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_MUL, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        a, b, output, 2);
}

VkResult vkmath_add_mul_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                            uint32_t num_elements, VkBuffer a, VkBuffer b,
                            float alpha, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    pc.alpha = alpha;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_ADD_MUL, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        a, b, output, 2);
}

VkResult vkmath_scale_f16(VkMathContext *ctx, VkCommandBuffer cmd,
                          uint32_t num_elements, float alpha,
                          VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = num_elements;
    pc.alpha = alpha;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SCALE, VKMATH_DTYPE_F16,
        &pc, elem_to_groups(num_elements), 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

/* ── Public API: reductions (f32) ───────────────────────────────────────── */

VkResult vkmath_max_reduce_dim_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_rows, uint32_t num_cols,
                                   VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_rows = num_rows;
    pc.num_cols = num_cols;
    pc.num_elements = num_rows * num_cols;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_MAX_REDUCE, VKMATH_DTYPE_F32,
        &pc, num_rows ? num_rows : 1, 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}

VkResult vkmath_sum_reduce_dim_f32(VkMathContext *ctx, VkCommandBuffer cmd,
                                   uint32_t num_rows, uint32_t num_cols,
                                   VkBuffer input, VkBuffer output) {
    vkmath_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_rows = num_rows;
    pc.num_cols = num_cols;
    pc.num_elements = num_rows * num_cols;
    return vkmath_cmd_dispatch(ctx, cmd, VKMATH_KERNEL_SUM_REDUCE, VKMATH_DTYPE_F32,
        &pc, num_rows ? num_rows : 1, 1, 1,
        input, VK_NULL_HANDLE, output, 1);
}
