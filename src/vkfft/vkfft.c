/**
 * \file vkfft.c
 * \brief VKFFT implementation: plan lifecycle, pipeline caching, FFT dispatch.
 *
 * Follows the contract in AGENTS.md — lazy pipeline creation with open-addressing
 * cache, push descriptors where available, embedded SPIR-V shader selection
 * with tier fallback. Only a baseline shader exists today, so ensure_pipeline
 * always falls back to the baseline blob.
 */
#include "vkfft_internal.h"
#include "shaders_spv.h"

#include <string.h>
#include <stdlib.h>

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
    /* baseline tier — the only FFT shaders compiled today. Direction is
       carried in the push constants (pc.direction), so each precision has a
       single shader serving both forward and inverse. */
    {VKFFT_KERNEL_FFT, VKFFT_DTYPE_F32, VKFFT_TIER_BASELINE, vkfft_spv_baseline_fft_f32, vkfft_spv_baseline_fft_f32_size},
    {VKFFT_KERNEL_FFT, VKFFT_DTYPE_F16, VKFFT_TIER_BASELINE, vkfft_spv_baseline_fft_f16, vkfft_spv_baseline_fft_f16_size},
};
#define SHADER_TABLE_COUNT (sizeof(s_shader_table) / sizeof(s_shader_table[0]))

/* ── Shader selection ────────────────────────────────────────────────────── */

const uint32_t *vkfft_select_spirv(uint32_t kernel, uint32_t data_type,
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

uint64_t vkfft_hash_key(uint32_t kernel, uint32_t data_type, uint32_t tier) {
    return ((uint64_t)tier << 32) | ((uint64_t)data_type << 16) | (uint64_t)kernel;
}

/* ── Shader module loading ───────────────────────────────────────────────── */

VkResult vkfft_load_shader_module(VkDevice device,
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

VkResult vkfft_alloc_descriptor_set(VkFFTPlan *plan, VkDescriptorSet *out_ds) {
    if (!plan->descriptor_pool) return VK_ERROR_INITIALIZATION_FAILED;
    VkDescriptorSetAllocateInfo dsai;
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.pNext = NULL;
    dsai.descriptorPool = plan->descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &plan->set_layout;
    return vkAllocateDescriptorSets(plan->device, &dsai, out_ds);
}

/* ── Push constants ──────────────────────────────────────────────────────── */

void vkfft_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                   const vkfft_push_constants_t *pc) {
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(vkfft_push_constants_t), pc);
}

/* ── Pipeline cache (open-addressing, linear probing) ──────────────────── */

static vkfft_pipeline_entry_t *cache_lookup(vkfft_pipeline_entry_t *cache, uint64_t key) {
    uint32_t mask = VKFFT_MAX_PIPELINES - 1; /* 255 */
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKFFT_MAX_PIPELINES; probe++) {
        if (!cache[idx].valid) return NULL;
        if (cache[idx].key == key) return &cache[idx];
        idx = (idx + 1) & mask;
    }
    return NULL;
}

static vkfft_pipeline_entry_t *cache_insert(vkfft_pipeline_entry_t *cache,
                                            uint64_t key,
                                            uint32_t kernel, uint32_t dt, uint32_t tier,
                                            VkPipeline pipeline, VkPipelineLayout layout) {
    uint32_t mask = VKFFT_MAX_PIPELINES - 1;
    uint32_t idx = (uint32_t)(key & mask);
    for (uint32_t probe = 0; probe < VKFFT_MAX_PIPELINES; probe++) {
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

VkResult vkfft_ensure_pipeline(VkFFTPlan *plan,
                               uint32_t kernel,
                               uint32_t data_type,
                               VkPipeline *out_pipeline) {
    /*
     * TRUTH TABLE: shader fallback
     * ─────────────────────────────────────────────────────────────────
     * active_tier | try tiers in order          | result
     * COOPMATRIX  | coop -> subgroup -> base    | only baseline blob exists
     * SUBGROUP    | subgroup -> baseline        | baseline blob
     * BASELINE    | baseline                    | baseline blob
     * ─────────────────────────────────────────────────────────────────
     * If no blob found at any tried tier -> VK_ERROR_FEATURE_NOT_PRESENT.
     */
    static const uint32_t try_order[] = {
        VKFFT_TIER_COOPMATRIX, VKFFT_TIER_SUBGROUP, VKFFT_TIER_BASELINE
    };

    uint64_t key = 0;
    const uint32_t *spirv = NULL;
    size_t spirv_size = 0;
    uint32_t selected_tier = 0;

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t t = try_order[i];
        if (t > plan->active_tier) continue; /* skip tiers the device can't support */
        spirv = vkfft_select_spirv(kernel, data_type, t, &spirv_size);
        if (spirv) {
            selected_tier = t;
            key = vkfft_hash_key(kernel, data_type, t);
            break;
        }
    }

    if (spirv == NULL) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    /* Cache hit? */
    vkfft_pipeline_entry_t *entry = cache_lookup(plan->pipelines, key);
    if (entry) {
        *out_pipeline = entry->pipeline;
        return VK_SUCCESS;
    }

    /* Cache miss — create pipeline */
    VkShaderModule sh;
    VkResult r = vkfft_load_shader_module(plan->device, spirv, spirv_size, &sh);
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
    cpci.layout = plan->pipeline_layout;

    r = vkCreateComputePipelines(plan->device, plan->pipeline_cache,
                                 1, &cpci, NULL, out_pipeline);
    vkDestroyShaderModule(plan->device, sh, NULL);
    if (r != VK_SUCCESS) return r;

    entry = cache_insert(plan->pipelines, key, kernel, data_type,
                         selected_tier, *out_pipeline, plan->pipeline_layout);
    if (entry) {
        plan->pipeline_count++;
    }
    return VK_SUCCESS;
}

/* ── Dispatch helper ─────────────────────────────────────────────────────── */

/*
 * Records pipeline bind + push constants + descriptor push + vkCmdDispatch.
 * Descriptor set layout is set=0 binding=0 (input, read) and binding=2
 * (output, write). Push-descriptor path writes both; the fallback path sets
 * writes[i].dstSet = ds BEFORE vkUpdateDescriptorSets.
 */
static VkResult vkfft_cmd_dispatch(VkFFTPlan *plan, VkCommandBuffer cmd,
    uint32_t kernel, uint32_t data_type,
    const vkfft_push_constants_t *pc,
    uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z,
    VkBuffer buf_in, VkBuffer buf_out) {

    VkPipeline pipeline;
    VkResult r = vkfft_ensure_pipeline(plan, kernel, data_type, &pipeline);
    if (r != VK_SUCCESS) return r;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkfft_push_pc(cmd, plan->pipeline_layout, pc);

    VkDescriptorBufferInfo info_in   = { buf_in,  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo info_out  = { buf_out, 0, VK_WHOLE_SIZE };
    const VkDescriptorBufferInfo *infos[2] = { &info_in, &info_out };

    VkWriteDescriptorSet writes[2];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = (i == 0) ? 0u : 2u;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = infos[i];
    }

    if (plan->push_desc_fn) {
        plan->push_desc_fn(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           plan->pipeline_layout, 0, 2, writes);
    } else {
        VkDescriptorSet ds;
        r = vkfft_alloc_descriptor_set(plan, &ds);
        if (r != VK_SUCCESS) return r;
        for (int i = 0; i < 2; i++) {
            writes[i].dstSet = ds; /* dstSet set BEFORE vkUpdateDescriptorSets */
        }
        vkUpdateDescriptorSets(plan->device, 2, writes, 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                plan->pipeline_layout, 0, 1, &ds, 0, NULL);
    }

    if (dispatch_x == 0) dispatch_x = 1;
    vkCmdDispatch(cmd, dispatch_x, dispatch_y, dispatch_z);
    return VK_SUCCESS;
}

/* ── Capability detection ────────────────────────────────────────────────── */

static VkResult vkfft_init_capabilities(VkFFTPlan *plan, VkPhysicalDevice pd) {
    /* shaderInt64 + cooperative matrix features (pNext chain) */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_features;
    coop_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coop_features.pNext = NULL;

    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &coop_features;
    vkGetPhysicalDeviceFeatures2(pd, &features2);

    plan->has_shader_int64 = features2.features.shaderInt64 ? VK_TRUE : VK_FALSE;
    plan->has_coop_matrix  = coop_features.cooperativeMatrix ? VK_TRUE : VK_FALSE;

    /* Subgroup properties via pNext chain */
    VkPhysicalDeviceSubgroupProperties subgroup_props;
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_props.pNext = NULL;

    VkPhysicalDeviceProperties2 props2;
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(pd, &props2);

    plan->max_subgroup_size = subgroup_props.subgroupSize;
    memcpy(plan->max_compute_workgroup_size,
           props2.properties.limits.maxComputeWorkGroupSize,
           sizeof(plan->max_compute_workgroup_size));

    /* Subgroup is a core Vulkan 1.3+ feature; supportedStages gates compute */
    plan->has_subgroup = (subgroup_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
                         ? VK_TRUE : VK_FALSE;

    /* Highest supported tier */
    if (plan->has_coop_matrix)
        plan->active_tier = VKFFT_TIER_COOPMATRIX;
    else if (plan->has_subgroup)
        plan->active_tier = VKFFT_TIER_SUBGROUP;
    else
        plan->active_tier = VKFFT_TIER_BASELINE;

    return VK_SUCCESS;
}

/* ── Plan lifecycle ──────────────────────────────────────────────────────── */

VkResult vkfft_create_plan(VkPhysicalDevice pd, VkDevice device, uint32_t n,
                           VkFFTPlan **pp_plan) {
    if (!pp_plan) return VK_ERROR_INITIALIZATION_FAILED;

    /* Validate n: power of two in [2, 1024]. The shader covers the whole FFT
       with a single 256-thread workgroup via a strided loop, so n is bounded
       by the shared-memory array (1024 elements = 8 KiB, well inside the
       workgroup LDS limit), not by maxComputeWorkGroupInvocations. */
    if (n < 2u || n > 1024u || (n & (n - 1u)) != 0u) {
        return VKFFT_ERROR_INVALID_ARGUMENT;
    }

    VkFFTPlan *plan = (VkFFTPlan *)calloc(1, sizeof(VkFFTPlan));
    if (!plan) return VK_ERROR_OUT_OF_HOST_MEMORY;
    plan->device = device;
    plan->n = n;

    uint32_t log2n = 0;
    uint32_t t = n;
    while (t > 1u) { t >>= 1u; log2n++; }
    plan->log2n = log2n;

    VkResult r = vkfft_init_capabilities(plan, pd);
    if (r != VK_SUCCESS) { free(plan); return r; }

    /* Load push descriptor function pointer */
    plan->push_desc_fn = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    plan->has_push_descriptor = plan->push_desc_fn ? VK_TRUE : VK_FALSE;

    /* Descriptor set layout: binding 0 = input SSBO (read), binding 2 = output
       SSBO (write). Only those two bindings. */
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
    dslci.pNext = plan->has_push_descriptor ? (const void *)&binding_flags : NULL;
    dslci.flags = plan->has_push_descriptor
        ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT : 0;
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;

    r = vkCreateDescriptorSetLayout(device, &dslci, NULL, &plan->set_layout);
    if (r != VK_SUCCESS) { free(plan); return r; }

    /* Pipeline layout: 1 descriptor set + 16-byte push constant range */
    VkPushConstantRange pc_range;
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkfft_push_constants_t); /* 16 bytes */

    VkPipelineLayoutCreateInfo plci;
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pNext = NULL;
    plci.flags = 0;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &plan->set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc_range;

    r = vkCreatePipelineLayout(device, &plci, NULL, &plan->pipeline_layout);
    if (r != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, plan->set_layout, NULL);
        free(plan);
        return r;
    }

    /* Pipeline cache (driver-level, accelerates lazy pipeline creation) */
    VkPipelineCacheCreateInfo pcci;
    pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pcci.pNext = NULL;
    pcci.flags = 0;
    pcci.initialDataSize = 0;
    pcci.pInitialData = NULL;
    r = vkCreatePipelineCache(device, &pcci, NULL, &plan->pipeline_cache);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, plan->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device, plan->set_layout, NULL);
        free(plan);
        return r;
    }

    /* Descriptor pool only needed for non-push-descriptor fallback */
    if (!plan->has_push_descriptor) {
        VkDescriptorPoolSize pool_sizes[1];
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[0].descriptorCount = VKFFT_MAX_PIPELINES * 2; /* 2 bindings */

        VkDescriptorPoolCreateInfo dpci;
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.pNext = NULL;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = VKFFT_MAX_PIPELINES;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = pool_sizes;

        r = vkCreateDescriptorPool(device, &dpci, NULL, &plan->descriptor_pool);
        if (r != VK_SUCCESS) {
            vkDestroyPipelineCache(device, plan->pipeline_cache, NULL);
            vkDestroyPipelineLayout(device, plan->pipeline_layout, NULL);
            vkDestroyDescriptorSetLayout(device, plan->set_layout, NULL);
            free(plan);
            return r;
        }
    }

    *pp_plan = plan;
    return VK_SUCCESS;
}

void vkfft_destroy_plan(VkFFTPlan *plan) {
    if (!plan) return;
    for (uint32_t i = 0; i < VKFFT_MAX_PIPELINES; i++) {
        if (plan->pipelines[i].valid) {
            vkDestroyPipeline(plan->device, plan->pipelines[i].pipeline, NULL);
            plan->pipelines[i].valid = 0;
        }
    }
    plan->pipeline_count = 0;
    if (plan->descriptor_pool)
        vkDestroyDescriptorPool(plan->device, plan->descriptor_pool, NULL);
    if (plan->set_layout)
        vkDestroyDescriptorSetLayout(plan->device, plan->set_layout, NULL);
    if (plan->pipeline_layout)
        vkDestroyPipelineLayout(plan->device, plan->pipeline_layout, NULL);
    if (plan->pipeline_cache)
        vkDestroyPipelineCache(plan->device, plan->pipeline_cache, NULL);
    free(plan);
}

/* ── Arch queries ────────────────────────────────────────────────────────── */

uint32_t vkfft_get_arch_index(VkFFTPlan *plan) {
    return plan ? (uint32_t)plan->active_tier : 0;
}

const char *vkfft_get_arch_name(VkFFTPlan *plan) {
    if (!plan) return "unknown";
    static const char *names[3] = { "baseline", "subgroup", "coopmatrix" };
    return names[(uint32_t)plan->active_tier];
}

uint32_t vkfft_get_size(VkFFTPlan *plan) {
    return plan ? plan->n : 0;
}

/* ── Public API: execute ─────────────────────────────────────────────────── */

/*
 * Shared dispatch path: builds the push constants for the given direction and
 * records one workgroup (256 threads, n active) per FFT; batch = 1.
 */
static VkResult vkfft_execute_dir(VkFFTPlan *plan, VkCommandBuffer cmd,
                                  uint32_t data_type, uint32_t direction,
                                  VkBuffer input, VkBuffer output) {
    if (!plan) return VK_ERROR_INITIALIZATION_FAILED;

    vkfft_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n = plan->n;
    pc.log2n = plan->log2n;
    pc.direction = direction;

    return vkfft_cmd_dispatch(plan, cmd, VKFFT_KERNEL_FFT, data_type,
        &pc, 1, 1, 1, input, output);
}

VkResult vkfft_execute_f32(VkFFTPlan *plan, VkCommandBuffer cmd,
                           VkBuffer input, VkBuffer output) {
    return vkfft_execute_dir(plan, cmd, VKFFT_DTYPE_F32, VKFFT_DIR_FORWARD,
                             input, output);
}

VkResult vkfft_execute_inverse_f32(VkFFTPlan *plan, VkCommandBuffer cmd,
                                   VkBuffer input, VkBuffer output) {
    return vkfft_execute_dir(plan, cmd, VKFFT_DTYPE_F32, VKFFT_DIR_INVERSE,
                             input, output);
}

VkResult vkfft_execute_f16(VkFFTPlan *plan, VkCommandBuffer cmd,
                           VkBuffer input, VkBuffer output) {
    return vkfft_execute_dir(plan, cmd, VKFFT_DTYPE_F16, VKFFT_DIR_FORWARD,
                             input, output);
}

VkResult vkfft_execute_inverse_f16(VkFFTPlan *plan, VkCommandBuffer cmd,
                                   VkBuffer input, VkBuffer output) {
    return vkfft_execute_dir(plan, cmd, VKFFT_DTYPE_F16, VKFFT_DIR_INVERSE,
                             input, output);
}
