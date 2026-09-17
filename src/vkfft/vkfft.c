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

/* ── Plan lifecycle ──────────────────────────────────────────────────────── */

/*
 * Shared 1D/2D plan creation. is_2d = 0 selects the 1D path, is_2d = 1 the
 * N x N separable 2D path; both validate n as a power of two in [2, 1024]
 * (the 2D FFT length equals the row/column dimension). The pipeline layout,
 * descriptor set layout, and pipeline cache are identical for both, so the
 * same 16-byte push-constant block (with the `mode` field) drives every
 * kernel.
 */
static VkResult vkfft_create_plan_internal(VkPhysicalDevice pd, VkDevice device,
                                           uint32_t n, uint32_t is_2d,
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
    plan->is_2d = is_2d;

    uint32_t log2n = 0;
    uint32_t t = n;
    while (t > 1u) { t >>= 1u; log2n++; }
    plan->log2n = log2n;

    /* Capability detection + push-desc fn load via VKRuntime (single
       implementation; was five inline copies). */
    VkRuntimeCaps caps;
    VkResult r = vkr_detect_capabilities(pd, device, &caps);
    if (r != VK_SUCCESS) { free(plan); return r; }

    plan->has_shader_int64 = caps.has_shader_int64;
    plan->has_coop_matrix  = caps.has_coop_matrix;
    plan->has_subgroup     = caps.has_subgroup;
    plan->max_subgroup_size = caps.subgroup_size;
    memcpy(plan->max_compute_workgroup_size, caps.max_workgroup_size,
           sizeof(plan->max_compute_workgroup_size));
    plan->push_desc_fn = caps.push_desc_fn;
    plan->has_push_descriptor = caps.has_push_descriptor;

    /* Highest supported tier from the vkr ladder (2=coopmatrix, 1=subgroup,
       0=baseline). Only baseline shaders exist, so ensure_pipeline falls back. */
    if (caps.arch_index == 2)
        plan->active_tier = VKFFT_TIER_COOPMATRIX;
    else if (caps.arch_index == 1)
        plan->active_tier = VKFFT_TIER_SUBGROUP;
    else
        plan->active_tier = VKFFT_TIER_BASELINE;

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

    /* Pipeline layout: 1 descriptor set + 16-byte push constant range
       (push-constant range is lib-specific, stays local). */
    VkPushConstantRange pc_range;
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkfft_push_constants_t); /* 16 bytes */

    r = vkr_create_pipeline_layout(device, plan->set_layout, 1, &pc_range,
                                   &plan->pipeline_layout);
    if (r != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, plan->set_layout, NULL);
        free(plan);
        return r;
    }

    /* Pipeline cache (driver-level, accelerates lazy pipeline creation) */
    r = vkr_create_pipeline_cache(device, &plan->pipeline_cache);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, plan->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device, plan->set_layout, NULL);
        free(plan);
        return r;
    }

    /* Descriptor pool only needed for non-push-descriptor fallback */
    if (!plan->has_push_descriptor) {
        r = vkr_create_descriptor_pool(device, VKFFT_MAX_PIPELINES,
                                       VKFFT_MAX_PIPELINES * 2,
                                       &plan->descriptor_pool);
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

VkResult vkfft_create_plan(VkPhysicalDevice pd, VkDevice device, uint32_t n,
                           VkFFTPlan **pp_plan) {
    return vkfft_create_plan_internal(pd, device, n, 0, pp_plan);
}

VkResult vkfft_create_plan_2d(VkPhysicalDevice pd, VkDevice device, uint32_t n,
                              VkFFTPlan **pp_plan) {
    return vkfft_create_plan_internal(pd, device, n, 1, pp_plan);
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

/* ── Public API: execute 2D (separable N x N) ───────────────────────────── */

/*
 * 2D separable FFT: N x N interleaved complex buffer, N = plan->n (power of
 * two). Records two passes into the same command buffer:
 *
 *   pass 1 (rows): dispatch N workgroups, mode=ROW. Workgroup r transforms
 *                  row r of `input` (elements r*N .. r*N+N-1, stride 1) and
 *                  writes it to `output`.
 *   barrier:       compute -> compute, SHADER_WRITE -> SHADER_READ so pass 2
 *                  sees pass 1's writes.
 *   pass 2 (cols): dispatch N workgroups, mode=COL. Workgroup c transforms
 *                  column c of `output` (elements c, c+N, ..., stride N) in
 *                  place.
 *
 * Twiddle sign follows pc.direction exactly like the 1D path (forward = -2pi,
 * inverse = +2pi, unnormalized), so forward-then-inverse over the two axes
 * recovers the original signal scaled by N*N. The buffer holds 2*N*N floats.
 */
static VkResult vkfft_execute_dir_2d(VkFFTPlan *plan, VkCommandBuffer cmd,
                                     uint32_t direction,
                                     VkBuffer input, VkBuffer output) {
    if (!plan) return VK_ERROR_INITIALIZATION_FAILED;
    if (!plan->is_2d) return VK_ERROR_INITIALIZATION_FAILED;

    vkfft_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n = plan->n;
    pc.log2n = plan->log2n;
    pc.direction = direction;

    /* Pass 1: rows. One workgroup per row; base = wg*n, stride 1. */
    pc.mode = VKFFT_MODE_ROW;
    VkResult r = vkfft_cmd_dispatch(plan, cmd, VKFFT_KERNEL_FFT, VKFFT_DTYPE_F32,
                                    &pc, plan->n, 1, 1, input, output);
    if (r != VK_SUCCESS) return r;

    /* Pass 2 reads pass 1 writes: compute -> compute barrier. */
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);

    /* Pass 2: columns, in place on `output`. base = wg, stride = n. */
    pc.mode = VKFFT_MODE_COL;
    return vkfft_cmd_dispatch(plan, cmd, VKFFT_KERNEL_FFT, VKFFT_DTYPE_F32,
                              &pc, plan->n, 1, 1, output, output);
}

VkResult vkfft_execute_2d_f32(VkFFTPlan *plan, VkCommandBuffer cmd,
                              VkBuffer input, VkBuffer output) {
    return vkfft_execute_dir_2d(plan, cmd, VKFFT_DIR_FORWARD, input, output);
}

VkResult vkfft_execute_2d_inverse_f32(VkFFTPlan *plan, VkCommandBuffer cmd,
                                      VkBuffer input, VkBuffer output) {
    return vkfft_execute_dir_2d(plan, cmd, VKFFT_DIR_INVERSE, input, output);
}
