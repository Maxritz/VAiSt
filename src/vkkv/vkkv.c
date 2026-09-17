/**
 * \file vkkv.c
 * \brief VKKV implementation: context lifecycle, host-side ridge fit, GPU apply.
 *
 * Cross-model KV cache transfer (arXiv:2608.03893). The fit runs on the host
 * in C99 double precision (calibration is offline); the apply runs on the GPU
 * as a single compute dispatch of the embedded `apply.comp` shader.
 *
 * Dependencies: vkruntime only (pooled allocator, staging upload/download,
 * pipeline helpers). No vkmath/vkblas: the fit is a host solve and the apply
 * is one self-contained baseline shader.
 */
#include "vkkv_internal.h"
#include "shaders_spv.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Shader blob selection ──────────────────────────────────────────────── */

const uint32_t *vkkv_select_spirv(size_t *out_size)
{
    if (out_size) *out_size = vkkv_spv_baseline_apply_size;
    return vkkv_spv_baseline_apply;
}

/* ── Descriptor set allocation (non-push-descriptor fallback) ──────────── */

VkResult vkkv_alloc_descriptor_set(VkKVTransfer *t, VkDescriptorSet *out)
{
    if (!t->descriptor_pool) return VK_ERROR_INITIALIZATION_FAILED;
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = t->descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &t->set_layout;
    return vkAllocateDescriptorSets(t->device, &dsai, out);
}

/* ── Pipeline creation (lazy, cached; single shader → no hashmap needed) ─ */

VkResult vkkv_ensure_pipeline(VkKVTransfer *t)
{
    if (t->pipeline) return VK_SUCCESS;

    size_t spv_size = 0;
    const uint32_t *spv = vkkv_select_spirv(&spv_size);
    if (!spv) return VK_ERROR_FEATURE_NOT_PRESENT;

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_size;
    smci.pCode = spv;

    VkShaderModule sh;
    VkResult r = vkCreateShaderModule(t->device, &smci, NULL, &sh);
    if (r != VK_SUCCESS) return r;

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sh;
    cpci.stage.pName = "main";
    cpci.layout = t->pipeline_layout;

    r = vkCreateComputePipelines(t->device, t->pipeline_cache, 1, &cpci,
                                 NULL, &t->pipeline);
    vkDestroyShaderModule(t->device, sh, NULL);
    return r;
}

/* ── Host-side ridge solve ──────────────────────────────────────────────── */

/**
 * \brief Solve G W = B in place by Gauss-Jordan elimination with partial
 *        pivoting.
 *
 * \param a Row-major augmented matrix [G | B], d rows x (d + m) columns.
 *          On return the left block is the identity and the right block holds
 *          W.
 * \param d Row/column count of the (square) system.
 * \param m Number of right-hand-side columns.
 * \retval 0 On success.
 * \retval -1 Singular system (no nonzero pivot found).
 */
static int solve_linear(double *a, uint32_t d, uint32_t m)
{
    const uint32_t width = d + m;

    for (uint32_t col = 0; col < d; col++) {
        uint32_t pivot = col;
        double best = fabs(a[(size_t)col * width + col]);
        for (uint32_t r = col + 1; r < d; r++) {
            double v = fabs(a[(size_t)r * width + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-300) return -1; /* singular */

        if (pivot != col) {
            for (uint32_t c = 0; c < width; c++) {
                double tmp = a[(size_t)col * width + c];
                a[(size_t)col * width + c] = a[(size_t)pivot * width + c];
                a[(size_t)pivot * width + c] = tmp;
            }
        }

        double piv = a[(size_t)col * width + col];
        for (uint32_t c = 0; c < width; c++) {
            a[(size_t)col * width + c] /= piv;
        }

        for (uint32_t r = 0; r < d; r++) {
            if (r == col) continue;
            double f = a[(size_t)r * width + col];
            if (f == 0.0) continue;
            for (uint32_t c = 0; c < width; c++) {
                a[(size_t)r * width + c] -= f * a[(size_t)col * width + c];
            }
        }
    }
    return 0;
}

/* ── Context lifecycle ──────────────────────────────────────────────────── */

VkResult vkkv_create_transfer(VkPhysicalDevice pd, VkDevice dev,
                              uint32_t n_heads, uint32_t src_dim,
                              uint32_t tgt_dim, float ridge_lambda,
                              VkKVTransfer **pT)
{
    if (!pT) return VK_ERROR_INITIALIZATION_FAILED;
    *pT = NULL;
    if (n_heads == 0 || src_dim == 0 || tgt_dim == 0 || ridge_lambda < 0.0f)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (pd == VK_NULL_HANDLE || dev == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    if (!queue) return VK_ERROR_INITIALIZATION_FAILED;

    VkKVTransfer *t = (VkKVTransfer *)calloc(1, sizeof(VkKVTransfer));
    if (!t) return VK_ERROR_OUT_OF_HOST_MEMORY;
    t->pd = pd;
    t->device = dev;
    t->queue = queue;
    t->n_heads = n_heads;
    t->src_dim = src_dim;
    t->tgt_dim = tgt_dim;
    t->ridge_lambda = ridge_lambda;

    VkResult r = vkr_create_runtime(pd, dev, queue, &t->rt);
    if (r != VK_SUCCESS) { free(t); return r; }

    r = vkr_create_command_pool(t->rt, 0, &t->cmd_pool);
    if (r != VK_SUCCESS) goto fail;

    {
        VkCommandBufferAllocateInfo cbai;
        memset(&cbai, 0, sizeof(cbai));
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = t->cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        r = vkAllocateCommandBuffers(dev, &cbai, &t->cmd);
        if (r != VK_SUCCESS) goto fail;
    }

    VkRuntimeCaps caps;
    r = vkr_detect_capabilities(pd, dev, &caps);
    if (r != VK_SUCCESS) goto fail;
    t->push_desc_fn = caps.push_desc_fn;
    t->has_push_descriptor = caps.has_push_descriptor;

    /* Descriptor set layout: 3 SSBO bindings (0 src read, 1 W read, 2 dst write) */
    VkDescriptorSetLayoutBinding bindings[3];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t b = 0; b < 3; b++) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags;
    memset(&binding_flags, 0, sizeof(binding_flags));
    binding_flags.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    {
        VkDescriptorBindingFlags flags[3] = {
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        };
        binding_flags.bindingCount = 3;
        binding_flags.pBindingFlags = flags;

        VkDescriptorSetLayoutCreateInfo dslci;
        memset(&dslci, 0, sizeof(dslci));
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.pNext = t->has_push_descriptor ? (const void *)&binding_flags : NULL;
        dslci.flags = t->has_push_descriptor
            ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT : 0;
        dslci.bindingCount = 3;
        dslci.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &t->set_layout);
        if (r != VK_SUCCESS) goto fail;
    }

    /* Pipeline layout: 1 descriptor set + 16-byte push-constant range */
    VkPushConstantRange pc_range;
    memset(&pc_range, 0, sizeof(pc_range));
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(vkkv_push_constants_t); /* 16 bytes */

    r = vkr_create_pipeline_layout(dev, t->set_layout, 1, &pc_range,
                                   &t->pipeline_layout);
    if (r != VK_SUCCESS) goto fail;

    r = vkr_create_pipeline_cache(dev, &t->pipeline_cache);
    if (r != VK_SUCCESS) goto fail;

    if (!t->has_push_descriptor) {
        r = vkr_create_descriptor_pool(dev, 8, 24, &t->descriptor_pool);
        if (r != VK_SUCCESS) goto fail;
    }

    /* W device buffer: [n_heads][src_dim*tgt_dim] floats */
    t->w_size = (VkDeviceSize)n_heads * src_dim * tgt_dim * sizeof(float);
    r = vkr_malloc(t->rt, t->w_size,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   &t->w_buffer, &t->w_memory);
    if (r != VK_SUCCESS) goto fail;

    *pT = t;
    return VK_SUCCESS;

fail:
    vkkv_destroy_transfer(t);
    return r;
}

void vkkv_destroy_transfer(VkKVTransfer *t)
{
    if (!t) return;
    if (t->pipeline) vkDestroyPipeline(t->device, t->pipeline, NULL);
    if (t->descriptor_pool)
        vkDestroyDescriptorPool(t->device, t->descriptor_pool, NULL);
    if (t->set_layout)
        vkDestroyDescriptorSetLayout(t->device, t->set_layout, NULL);
    if (t->pipeline_layout)
        vkDestroyPipelineLayout(t->device, t->pipeline_layout, NULL);
    if (t->pipeline_cache)
        vkDestroyPipelineCache(t->device, t->pipeline_cache, NULL);
    if (t->cmd && t->cmd_pool)
        vkFreeCommandBuffers(t->device, t->cmd_pool, 1, &t->cmd);
    if (t->cmd_pool)
        vkDestroyCommandPool(t->device, t->cmd_pool, NULL);
    if (t->w_buffer)
        vkr_free(t->rt, t->w_buffer, t->w_memory);
    if (t->rt) vkr_destroy_runtime(t->rt);
    free(t);
}

/* ── Public API: fit (host-side) ────────────────────────────────────────── */

VkResult vkkv_fit_cpu(VkKVTransfer *t, const float *const *X,
                      const float *const *Y, uint32_t n_samples)
{
    if (!t || !X || !Y || n_samples == 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!t->w_buffer) return VK_ERROR_INITIALIZATION_FAILED;

    const uint32_t d = t->src_dim;
    const uint32_t m = t->tgt_dim;
    const uint32_t width = d + m;

    double *aug = (double *)calloc((size_t)d * width, sizeof(double));
    float *w_host = (float *)malloc(t->w_size);
    if (!aug || !w_host) {
        free(aug);
        free(w_host);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VkResult r = VK_SUCCESS;

    for (uint32_t h = 0; h < t->n_heads; h++) {
        const float *Xh = X[h];
        const float *Yh = Y[h];

        memset(aug, 0, (size_t)d * width * sizeof(double));

        /* G = X^T X  (d x d) */
        for (uint32_t i = 0; i < d; i++) {
            for (uint32_t j = 0; j < d; j++) {
                double s = 0.0;
                for (uint32_t k = 0; k < n_samples; k++) {
                    s += (double)Xh[(size_t)k * d + i]
                       * (double)Xh[(size_t)k * d + j];
                }
                aug[(size_t)i * width + j] = s;
            }
        }

        /* B = X^T Y  (d x m), stored in the right block */
        for (uint32_t i = 0; i < d; i++) {
            for (uint32_t j = 0; j < m; j++) {
                double s = 0.0;
                for (uint32_t k = 0; k < n_samples; k++) {
                    s += (double)Xh[(size_t)k * d + i]
                       * (double)Yh[(size_t)k * m + j];
                }
                aug[(size_t)i * width + d + j] = s;
            }
        }

        /* Ridge: G = G + lambda * I */
        for (uint32_t i = 0; i < d; i++) {
            aug[(size_t)i * width + i] += (double)t->ridge_lambda;
        }

        if (solve_linear(aug, d, m) != 0) {
            r = VK_ERROR_INITIALIZATION_FAILED; /* singular G */
            break;
        }

        /* Copy W = G^-1 B (right block) into the head's block of w_host */
        for (uint32_t i = 0; i < d; i++) {
            for (uint32_t j = 0; j < m; j++) {
                w_host[(size_t)(h * d + i) * m + j] =
                    (float)aug[(size_t)i * width + d + j];
            }
        }
    }

    if (r == VK_SUCCESS) {
        r = vkr_upload(t->rt, t->cmd, t->queue, w_host, t->w_buffer, 0,
                       t->w_size);
    }

    free(aug);
    free(w_host);
    if (r != VK_SUCCESS) return r;

    t->fitted = VK_TRUE;
    return VK_SUCCESS;
}

/* ── Public API: apply (GPU) ────────────────────────────────────────────── */

VkResult vkkv_apply(VkKVTransfer *t, VkCommandBuffer cmd, uint32_t h,
                    VkBuffer src, uint32_t n, VkBuffer dst)
{
    if (!t || !cmd || !src || !dst)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (h >= t->n_heads) return VK_ERROR_INITIALIZATION_FAILED;
    if (n == 0) return VK_ERROR_INITIALIZATION_FAILED;
    if (!t->fitted) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult r = vkkv_ensure_pipeline(t);
    if (r != VK_SUCCESS) return r;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, t->pipeline);

    vkkv_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.num_elements = n * t->tgt_dim;
    pc.tgt_dim = t->tgt_dim;
    pc.src_dim = t->src_dim;
    pc.w_offset = h * t->src_dim * t->tgt_dim;
    vkCmdPushConstants(cmd, t->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    VkDescriptorBufferInfo src_info = { src, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo w_info   = { t->w_buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo dst_info = { dst, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &src_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &w_info;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &dst_info;

    if (t->push_desc_fn) {
        t->push_desc_fn(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        t->pipeline_layout, 0, 3, writes);
    } else {
        VkDescriptorSet ds;
        r = vkkv_alloc_descriptor_set(t, &ds);
        if (r != VK_SUCCESS) return r;
        for (int i = 0; i < 3; i++) writes[i].dstSet = ds;
        vkUpdateDescriptorSets(t->device, 3, writes, 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                t->pipeline_layout, 0, 1, &ds, 0, NULL);
    }

    uint64_t total = (uint64_t)n * t->tgt_dim;
    uint32_t groups = (uint32_t)((total + VKKV_WORKGROUP_SIZE - 1u)
                                 / VKKV_WORKGROUP_SIZE);
    if (groups == 0) groups = 1;
    vkCmdDispatch(cmd, groups, 1, 1);
    return VK_SUCCESS;
}
