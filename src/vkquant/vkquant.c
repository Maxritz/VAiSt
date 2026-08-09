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
    /* shaderInt64 + cooperative matrix features (pNext chain) */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_features;
    coop_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coop_features.pNext = NULL;

    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &coop_features;
    vkGetPhysicalDeviceFeatures2(pd, &features2);

    ctx->has_shader_int64 = features2.features.shaderInt64 ? VK_TRUE : VK_FALSE;
    ctx->has_coop_matrix  = coop_features.cooperativeMatrix ? VK_TRUE : VK_FALSE;

    /* Subgroup properties via pNext chain */
    VkPhysicalDeviceSubgroupProperties subgroup_props;
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_props.pNext = NULL;

    VkPhysicalDeviceProperties2 props2;
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(pd, &props2);

    ctx->max_subgroup_size = subgroup_props.subgroupSize;
    memcpy(ctx->max_compute_workgroup_size,
           props2.properties.limits.maxComputeWorkGroupSize,
           sizeof(ctx->max_compute_workgroup_size));

    /* Subgroup is a core Vulkan 1.3+ feature; supportedStages gates compute */
    ctx->has_subgroup = (subgroup_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
                        ? VK_TRUE : VK_FALSE;

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

    VkResult r = vkquant_init_capabilities(ctx, pd);
    if (r != VK_SUCCESS) { free(ctx); return r; }

    /* Load push descriptor function pointer */
    ctx->push_desc_fn = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    ctx->has_push_descriptor = ctx->push_desc_fn ? VK_TRUE : VK_FALSE;

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

    /* Pipeline layout: 1 descriptor set + 16-byte push constant range */
    VkPushConstantRange pc_range;
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkquant_push_constants_t); /* 16 bytes */

    VkPipelineLayoutCreateInfo plci;
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pNext = NULL;
    plci.flags = 0;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &ctx->set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc_range;

    r = vkCreatePipelineLayout(device, &plci, NULL, &ctx->pipeline_layout);
    if (r != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, ctx->set_layout, NULL);
        free(ctx);
        return r;
    }

    /* Pipeline cache (driver-level, accelerates lazy pipeline creation) */
    VkPipelineCacheCreateInfo pcci;
    pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pcci.pNext = NULL;
    pcci.flags = 0;
    pcci.initialDataSize = 0;
    pcci.pInitialData = NULL;
    r = vkCreatePipelineCache(device, &pcci, NULL, &ctx->pipeline_cache);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, ctx->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device, ctx->set_layout, NULL);
        free(ctx);
        return r;
    }

    /* Descriptor pool only needed for non-push-descriptor fallback */
    if (!ctx->has_push_descriptor) {
        VkDescriptorPoolSize pool_sizes[1];
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[0].descriptorCount = VKQUANT_MAX_PIPELINES * 2;

        VkDescriptorPoolCreateInfo dpci;
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.pNext = NULL;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = VKQUANT_MAX_PIPELINES;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = pool_sizes;

        r = vkCreateDescriptorPool(device, &dpci, NULL, &ctx->descriptor_pool);
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
