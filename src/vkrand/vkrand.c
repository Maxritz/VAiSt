/**
 * \file vkrand.c
 * \brief VKRAND implementation: context lifecycle, pipeline caching, op dispatch.
 *
 * Follows the contract in AGENTS.md — lazy pipeline creation with open-addressing
 * cache, push descriptors where available, embedded SPIR-V shader selection.
 * Only a baseline tier shader exists today, so the active tier always
 * resolves to baseline and ensure_pipeline falls back accordingly.
 */
#include "vkrand_internal.h"
#include "shaders_spv.h"

#include <string.h>
#include <stdlib.h>

#define VKRAND_WORKGROUP_SIZE 256u

/* ── Shader blob lookup table ────────────────────────────────────────────── *
 * Maps (kernel, tier) to the embedded SPIR-V array.
 * Only combinations that have a compiled .comp shader are present.
 * ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t kernel;
    uint32_t tier;
    const uint32_t *spirv;
    size_t spirv_size;
} shader_blob_t;

static const shader_blob_t s_shader_table[] = {
    /* baseline tier — the only tier with a Philox shader today */
    {VKRAND_KERNEL_UNIFORM_F32, VKRAND_TIER_BASELINE, vkrand_spv_baseline_uniform_f32, vkrand_spv_baseline_uniform_f32_size},
};
#define SHADER_TABLE_COUNT (sizeof(s_shader_table) / sizeof(s_shader_table[0]))

/* ── Shader selection ────────────────────────────────────────────────────── */

const uint32_t *vkrand_select_spirv(uint32_t kernel, uint32_t tier,
                                    size_t *out_size) {
    for (uint32_t i = 0; i < SHADER_TABLE_COUNT; i++) {
        if (s_shader_table[i].kernel == kernel &&
            s_shader_table[i].tier == tier) {
            if (out_size) *out_size = s_shader_table[i].spirv_size;
            return s_shader_table[i].spirv;
        }
    }
    if (out_size) *out_size = 0;
    return NULL;
}

/* ── Hash key ────────────────────────────────────────────────────────────── */

uint64_t vkrand_hash_key(uint32_t kernel, uint32_t tier) {
    return ((uint64_t)tier << 32) | (uint64_t)kernel;
}

/* ── Shader module loading ───────────────────────────────────────────────── */

VkResult vkrand_load_shader_module(VkDevice device,
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

VkResult vkrand_alloc_descriptor_set(VkRandContext *ctx, VkDescriptorSet *out_ds) {
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

void vkrand_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                    const vkrand_push_constants_t *pc) {
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(vkrand_push_constants_t), pc);
}

/* ── Pipeline layout getter ──────────────────────────────────────────────── */

VkPipelineLayout vkrand_get_pipeline_layout(VkRandContext *ctx) {
    return ctx ? ctx->pipeline_layout : VK_NULL_HANDLE;
}

/* ── Pipeline cache (open-addressing, linear probing) ──────────────────── */

static vkrand_pipeline_entry_t *cache_lookup(vkrand_pipeline_entry_t *cache, uint64_t key) {
    uint32_t mask = VKRAND_MAX_PIPELINES - 1; /* 255 */
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKRAND_MAX_PIPELINES; probe++) {
        if (!cache[idx].valid) return NULL;
        if (cache[idx].key == key) return &cache[idx];
        idx = (idx + 1) & mask;
    }
    return NULL;
}

static vkrand_pipeline_entry_t *cache_insert(vkrand_pipeline_entry_t *cache,
                                             uint64_t key,
                                             uint32_t kernel, uint32_t tier,
                                             VkPipeline pipeline, VkPipelineLayout layout) {
    uint32_t mask = VKRAND_MAX_PIPELINES - 1;
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKRAND_MAX_PIPELINES; probe++) {
        if (!cache[idx].valid) {
            cache[idx].key = key;
            cache[idx].pipeline = pipeline;
            cache[idx].layout = layout;
            cache[idx].kernel = kernel;
            cache[idx].tier = tier;
            cache[idx].valid = 1;
            return &cache[idx];
        }
        idx = (idx + 1) & mask;
    }
    return NULL;
}

/* ── Pipeline creation / cache ───────────────────────────────────────────── */

VkResult vkrand_ensure_pipeline(VkRandContext *ctx,
                                uint32_t kernel,
                                VkPipeline *out_pipeline) {
    /*
     * TRUTH TABLE: shader fallback
     * ─────────────────────────────────────────────────────────────────
     * active_tier | try tiers in order   | result
     * BASELINE    | baseline             | baseline blob
     * ─────────────────────────────────────────────────────────────────
     * Only a baseline shader exists today, so the highest tier the device
     * supports is clamped to baseline and no fallback chain is exercised.
     * If no blob found at any tried tier -> VK_ERROR_FEATURE_NOT_PRESENT.
     */
    static const uint32_t try_order[] = {
        VKRAND_TIER_BASELINE
    };

    uint64_t key = 0;
    const uint32_t *spirv = NULL;
    size_t spirv_size = 0;
    uint32_t selected_tier = 0;

    for (uint32_t i = 0; i < 1; i++) {
        uint32_t t = try_order[i];
        if (t > ctx->active_tier) continue; /* skip tiers the device can't support */
        spirv = vkrand_select_spirv(kernel, t, &spirv_size);
        if (spirv) {
            selected_tier = t;
            key = vkrand_hash_key(kernel, t);
            break;
        }
    }

    if (spirv == NULL) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Cache hit? */
    vkrand_pipeline_entry_t *entry = cache_lookup(ctx->pipelines, key);
    if (entry) {
        *out_pipeline = entry->pipeline;
        return VK_SUCCESS;
    }

    /* Cache miss — create pipeline */
    VkShaderModule sh;
    VkResult r = vkrand_load_shader_module(ctx->device, spirv, spirv_size, &sh);
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

    entry = cache_insert(ctx->pipelines, key, kernel,
                         selected_tier, *out_pipeline, ctx->pipeline_layout);
    if (entry) {
        ctx->pipeline_count++;
    }
    return VK_SUCCESS;
}

/* ── Dispatch helper ─────────────────────────────────────────────────────── */

/*
 * Records pipeline bind + push constants + descriptor push + vkCmdDispatch.
 * The descriptor set layout declares a single SSBO write binding (binding 2);
 * exactly that one write is issued on every dispatch.
 */
static VkResult vkrand_cmd_dispatch(VkRandContext *ctx, VkCommandBuffer cmd,
    uint32_t kernel,
    const vkrand_push_constants_t *pc,
    uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z,
    VkBuffer buf_out) {

    VkPipeline pipeline;
    VkResult r = vkrand_ensure_pipeline(ctx, kernel, &pipeline);
    if (r != VK_SUCCESS) return r;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkrand_push_pc(cmd, ctx->pipeline_layout, pc);

    /* Descriptor write: binding 2 = output SSBO (write) */
    VkDescriptorBufferInfo info_out = { buf_out, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet write;
    memset(&write, 0, sizeof(write));
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = 2;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info_out;

    if (ctx->push_desc_fn) {
        ctx->push_desc_fn(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx->pipeline_layout, 0,
                          1, &write);
    } else {
        VkDescriptorSet ds;
        r = vkrand_alloc_descriptor_set(ctx, &ds);
        if (r != VK_SUCCESS) return r;
        write.dstSet = ds;  /* CRITICAL: set dstSet BEFORE vkUpdateDescriptorSets */
        vkUpdateDescriptorSets(ctx->device, 1, &write, 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ctx->pipeline_layout, 0, 1, &ds, 0, NULL);
    }

    if (dispatch_x == 0) dispatch_x = 1;
    vkCmdDispatch(cmd, dispatch_x, dispatch_y, dispatch_z);
    return VK_SUCCESS;
}

static uint32_t elem_to_groups(uint32_t count) {
    uint32_t g = (count + VKRAND_WORKGROUP_SIZE - 1) / VKRAND_WORKGROUP_SIZE;
    return g ? g : 1;
}

/* ── Capability detection ────────────────────────────────────────────────── */

VkResult vkrand_init_capabilities(VkRandContext *ctx, VkPhysicalDevice pd) {
    /* shaderInt64 feature (queried for parity; the shaders do not use it) */
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = NULL;
    vkGetPhysicalDeviceFeatures2(pd, &features2);

    ctx->has_shader_int64 = features2.features.shaderInt64 ? VK_TRUE : VK_FALSE;

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

    /* Only a baseline shader exists; clamp the active tier to baseline. */
    ctx->active_tier = VKRAND_TIER_BASELINE;

    return VK_SUCCESS;
}

/* ── Context lifecycle ──────────────────────────────────────────────────── */

VkResult vkrand_create_context(VkPhysicalDevice pd, VkDevice device,
                               VkRandContext **pp_ctx) {
    if (!pp_ctx) return VK_ERROR_INITIALIZATION_FAILED;

    VkRandContext *ctx = (VkRandContext *)calloc(1, sizeof(VkRandContext));
    if (!ctx) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ctx->device = device;

    VkResult r = vkrand_init_capabilities(ctx, pd);
    if (r != VK_SUCCESS) { free(ctx); return r; }

    /* Load push descriptor function pointer */
    ctx->push_desc_fn = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    ctx->has_push_descriptor = ctx->push_desc_fn ? VK_TRUE : VK_FALSE;

    /* Descriptor set layout: 1 SSBO binding (write) at binding 2 — matches
       the shader exactly (layout declares what the shader uses). */
    VkDescriptorSetLayoutBinding bindings[1];
    memset(bindings, 0, sizeof(bindings));
    bindings[0].binding = 2;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci;
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.pNext = NULL;
    dslci.flags = ctx->has_push_descriptor
        ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT : 0;
    dslci.bindingCount = 1;
    dslci.pBindings = bindings;

    r = vkCreateDescriptorSetLayout(device, &dslci, NULL, &ctx->set_layout);
    if (r != VK_SUCCESS) { free(ctx); return r; }

    /* Pipeline layout: 1 descriptor set + 16-byte push constant range */
    VkPushConstantRange pc_range;
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkrand_push_constants_t); /* 16 bytes */

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
        pool_sizes[0].descriptorCount = VKRAND_MAX_PIPELINES;

        VkDescriptorPoolCreateInfo dpci;
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.pNext = NULL;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = VKRAND_MAX_PIPELINES;
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

void vkrand_destroy_context(VkRandContext *ctx) {
    if (!ctx) return;
    vkrand_flush_pipelines(ctx);
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

void vkrand_flush_pipelines(VkRandContext *ctx) {
    if (!ctx) return;
    for (uint32_t i = 0; i < VKRAND_MAX_PIPELINES; i++) {
        if (ctx->pipelines[i].valid) {
            vkDestroyPipeline(ctx->device, ctx->pipelines[i].pipeline, NULL);
            ctx->pipelines[i].valid = 0;
        }
    }
    ctx->pipeline_count = 0;
}

/* ── Arch queries ────────────────────────────────────────────────────────── */

uint32_t vkrand_get_arch_index(VkRandContext *ctx) {
    return ctx ? (uint32_t)ctx->active_tier : 0;
}

const char *vkrand_get_arch_name(VkRandContext *ctx) {
    if (!ctx) return "unknown";
    static const char *names[1] = { "baseline" };
    return names[(uint32_t)ctx->active_tier];
}

/* ── Public API: uniform f32 ────────────────────────────────────────────── */

VkResult vkrand_uniform_f32(VkRandContext *ctx, VkCommandBuffer cmd,
                            uint32_t seed, uint32_t count, VkBuffer output) {
    if (!ctx) return VK_ERROR_INITIALIZATION_FAILED;

    vkrand_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.count = count;
    pc.seed = seed;
    return vkrand_cmd_dispatch(ctx, cmd, VKRAND_KERNEL_UNIFORM_F32,
        &pc, elem_to_groups(count), 1, 1,
        output);
}
