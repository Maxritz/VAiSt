/**
 * \file vaist_vkblas.c
 * \brief VkBLAS C99 dispatch: quantized GEMM kernels + speculative verify.
 *
 * Implements vkblas_create_context/destroy_context + vkblas_qgemm_*,
 * vkblas_gemm_spec_verify_f32.
 *
 * When VAIST_HAVE_VK_HDR is set, uses real Vulkan types and SPIR-V shaders.
 * When unset, provides stubs returning VAIST_UNSUPPORTED/VK_NOT_READY.
 */
#include "vaist_blas.h"
#include "vaist_core.h"
#include "vaist_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Opaque VkBLASContext handle (defined in vkblas.h, but we avoid including it
 * to keep the build decoupled from the Vulkan SDK in stub mode). */
typedef struct VkBLASContext VkBLASContext;

#define VKBLAS_MAGIC 0x564B424C

#ifdef VAIST_HAVE_VK_HDR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#ifdef VAIST_SPIRV_EMBEDDED
#include "vaist_blas_spv.h"
#endif

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) return _r; } while(0)

/* ---- Pipeline slot: 3-binding qgemm (Wq, x, y) ---- */
typedef struct {
    VkDescriptorSetLayout dsl;
    VkDescriptorPool pool;
    VkDescriptorSet dset;
    VkPipelineLayout pipe_layout;
    VkPipeline pipeline;
    VkCommandPool cmdpool;
    VkCommandBuffer cb;
    PFN_vkCmdBindPipeline           CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets     CmdBindDescriptorSets;
    PFN_vkCmdPushConstants           CmdPushConstants;
    PFN_vkCmdDispatch                CmdDispatch;
    PFN_vkCmdPipelineBarrier         CmdPipelineBarrier;
    PFN_vkResetCommandBuffer        ResetCommandBuffer;
    PFN_vkBeginCommandBuffer        BeginCommandBuffer;
    PFN_vkEndCommandBuffer          EndCommandBuffer;
    PFN_vkCreateCommandPool          CreateCommandPool;
    PFN_vkDestroyCommandPool         DestroyCommandPool;
    PFN_vkAllocateCommandBuffers     AllocateCommandBuffers;
    PFN_vkCreateDescriptorPool       CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets     AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets       UpdateDescriptorSets;
    PFN_vkCreateDescriptorSetLayout  CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreateShaderModule         CreateShaderModule;
    PFN_vkDestroyShaderModule        DestroyShaderModule;
    PFN_vkCreatePipelineLayout       CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout      DestroyPipelineLayout;
    PFN_vkCreateComputePipelines     CreateComputePipelines;
    PFN_vkDestroyPipeline            DestroyPipeline;
    PFN_vkCreateFence                CreateFence;
    PFN_vkDestroyFence               DestroyFence;
    PFN_vkWaitForFences              WaitForFences;
    PFN_vkQueueSubmit                QueueSubmit;
    PFN_vkDestroyDescriptorPool      DestroyDescriptorPool;
} qgemm_slot;

typedef qgemm_slot spec_slot;

/* ---- Push constants ---- */
typedef struct {
    float alpha; float beta;
    int32_t m, n, k, ldw, ldx, ldy;
    uint32_t flags;
} vkblas_qgemm_pc;

typedef struct {
    float scale;
    uint32_t head_dim, num_heads, num_kv_heads, seqlen;
    uint32_t max_blocks, block_size, verify_mask, spec_depth, reserved;
} vkblas_spec_pc;

/* ---- Internal VkBLASContext (full definition) ---- */
#undef VkBLASContext
struct VkBLASContext {
    uint32_t magic;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t qfamily;
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkGetDeviceProcAddr    gpda;
    qgemm_slot q8_0;
    qgemm_slot nvfp4;
    qgemm_slot t2_0;
    spec_slot spec_verify;
};

/* ---- Set up qgemm pipeline (3 SSBO bindings) ---- */
static VkResult _init_qgemm_slot(VkBLASContext *ctx, qgemm_slot *s,
    const uint32_t *spv, size_t spv_len)
{
    VkResult r;
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 3; dlci.pBindings = b;
    VK_CHECK(ctx->CreateDescriptorSetLayout(ctx->device, &dlci, NULL, &s->dsl));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vkblas_qgemm_pc)};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &s->dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VK_CHECK(ctx->CreatePipelineLayout(ctx->device, &plci, NULL, &s->pipe_layout));

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv_len * 4; smci.pCode = spv;
    VK_CHECK(ctx->CreateShaderModule(ctx->device, &smci, NULL, &s->pipeline));

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = s->pipeline;
    cpci.stage.pName = "main";
    cpci.layout = s->pipe_layout;
    r = ctx->CreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &s->pipeline);
    if (r) return r;

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx->qfamily;
    VK_CHECK(ctx->CreateCommandPool(ctx->device, &pci, NULL, &s->cmdpool));

    VkCommandBufferAllocateInfo cba = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = s->cmdpool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VK_CHECK(ctx->AllocateCommandBuffers(ctx->device, &cba, &s->cb));

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VK_CHECK(ctx->CreateDescriptorPool(ctx->device, &dpci, NULL, &s->pool));

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = s->pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &s->dsl;
    return ctx->AllocateDescriptorSets(ctx->device, &dsai, &s->dset);
}

static VkResult _init_spec_slot(VkBLASContext *ctx, spec_slot *s,
    const uint32_t *spv, size_t spv_len)
{
    VkDescriptorSetLayoutBinding b[7];
    for (int i = 0; i < 7; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 7; dlci.pBindings = b;
    VK_CHECK(ctx->CreateDescriptorSetLayout(ctx->device, &dlci, NULL, &s->dsl));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vkblas_spec_pc)};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &s->dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VK_CHECK(ctx->CreatePipelineLayout(ctx->device, &plci, NULL, &s->pipe_layout));

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv_len * 4; smci.pCode = spv;
    VK_CHECK(ctx->CreateShaderModule(ctx->device, &smci, NULL, &s->pipeline));

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = s->pipeline;
    cpci.stage.pName = "main";
    cpci.layout = s->pipe_layout;
    VK_CHECK(ctx->CreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &s->pipeline));

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx->qfamily;
    VK_CHECK(ctx->CreateCommandPool(ctx->device, &pci, NULL, &s->cmdpool));

    VkCommandBufferAllocateInfo cba = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = s->cmdpool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VK_CHECK(ctx->AllocateCommandBuffers(ctx->device, &cba, &s->cb));

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VK_CHECK(ctx->CreateDescriptorPool(ctx->device, &dpci, NULL, &s->pool));

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = s->pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &s->dsl;
    return ctx->AllocateDescriptorSets(ctx->device, &dsai, &s->dset);
}

static VkResult _dispatch(VkBLASContext *ctx, void *slot_v, int n_bindings,
    VkBuffer *bufs, void *pc, size_t pc_size, uint32_t gx, uint32_t gy)
{
    qgemm_slot *s = (qgemm_slot *)slot_v;
    VkWriteDescriptorSet wds[7];
    VkDescriptorBufferInfo infos[7];
    for (int i = 0; i < n_bindings; i++) {
        infos[i] = (VkDescriptorBufferInfo){bufs[i], 0, VK_WHOLE_SIZE};
        wds[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL,
            s->dset, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &infos[i], NULL};
    }
    ctx->UpdateDescriptorSets(ctx->device, (uint32_t)n_bindings, wds, 0, NULL);

    s->ResetCommandBuffer(s->cb, 0);
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(s->BeginCommandBuffer(s->cb, &cbbi));

    s->CmdBindPipeline(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
    s->CmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe_layout,
        0, 1, &s->dset, 0, NULL);
    s->CmdPushConstants(s->cb, s->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, (size_t)pc_size, pc);
    s->CmdDispatch(s->cb, gx, gy, 1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    s->CmdPipelineBarrier(s->cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
    VK_CHECK(s->EndCommandBuffer(s->cb));

    VkFence f = VK_NULL_HANDLE;
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    ctx->CreateFence(ctx->device, &fci, NULL, &f);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &s->cb;
    VkResult r = ctx->QueueSubmit(ctx->queue, 1, &si, f);
    if (r) { ctx->DestroyFence(ctx->device, f, NULL); return r; }
    ctx->WaitForFences(ctx->device, 1, &f, VK_TRUE, UINT64_MAX);
    ctx->DestroyFence(ctx->device, f, NULL);
    return VK_SUCCESS;
}

/* ---- Public API ---- */

VkResult vkblas_create_context(VkInstance instance,
    VkPhysicalDevice physicalDevice, VkDevice device, VkBLASContext** pContext)
{
    if (!pContext) return VK_ERROR_INITIALIZATION_FAILED;
    *pContext = NULL;
    VkBLASContext *ctx = (VkBLASContext *)calloc(1, sizeof(VkBLASContext));
    if (!ctx) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ctx->magic = VKBLAS_MAGIC;
    ctx->instance = instance;
    ctx->physical_device = physicalDevice;
    ctx->device = device;

    /* Resolve device procs */
    PFN_vkGetDeviceProcAddr gpda = (PFN_vkGetDeviceProcAddr)
        ((PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr ? 
         vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr") : NULL);
    if (gpda) {
        ctx->gpda = gpda;
        ctx->CmdBindPipeline           = (PFN_vkCmdBindPipeline)gpda(device, "vkCmdBindPipeline");
        ctx->CmdBindDescriptorSets     = (PFN_vkCmdBindDescriptorSets)gpda(device, "vkCmdBindDescriptorSets");
        ctx->CmdPushConstants          = (PFN_vkCmdPushConstants)gpda(device, "vkCmdPushConstants");
        ctx->CmdDispatch               = (PFN_vkCmdDispatch)gpda(device, "vkCmdDispatch");
        ctx->CmdPipelineBarrier        = (PFN_vkCmdPipelineBarrier)gpda(device, "vkCmdPipelineBarrier");
    }
    if (vkGetInstanceProcAddr) {
        ctx->gipa = vkGetInstanceProcAddr;
        ctx->CreateCommandPool          = (PFN_vkCreateCommandPool)vkGetInstanceProcAddr(instance, "vkCreateCommandPool");
        ctx->DestroyCommandPool         = (PFN_vkDestroyCommandPool)vkGetInstanceProcAddr(instance, "vkDestroyCommandPool");
        ctx->AllocateCommandBuffers     = (PFN_vkAllocateCommandBuffers)vkGetInstanceProcAddr(instance, "vkAllocateCommandBuffers");
        ctx->CreateDescriptorPool       = (PFN_vkCreateDescriptorPool)vkGetInstanceProcAddr(instance, "vkCreateDescriptorPool");
        ctx->AllocateDescriptorSets     = (PFN_vkAllocateDescriptorSets)vkGetInstanceProcAddr(instance, "vkAllocateDescriptorSets");
        ctx->UpdateDescriptorSets       = (PFN_vkUpdateDescriptorSets)vkGetInstanceProcAddr(instance, "vkUpdateDescriptorSets");
        ctx->CreateDescriptorSetLayout  = (PFN_vkCreateDescriptorSetLayout)vkGetInstanceProcAddr(instance, "vkCreateDescriptorSetLayout");
        ctx->DestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)vkGetInstanceProcAddr(instance, "vkDestroyDescriptorSetLayout");
        ctx->CreateShaderModule         = (PFN_vkCreateShaderModule)vkGetInstanceProcAddr(instance, "vkCreateShaderModule");
        ctx->DestroyShaderModule        = (PFN_vkDestroyShaderModule)vkGetInstanceProcAddr(instance, "vkDestroyShaderModule");
        ctx->CreatePipelineLayout       = (PFN_vkCreatePipelineLayout)vkGetInstanceProcAddr(instance, "vkCreatePipelineLayout");
        ctx->DestroyPipelineLayout      = (PFN_vkDestroyPipelineLayout)vkGetInstanceProcAddr(instance, "vkDestroyPipelineLayout");
        ctx->CreateComputePipelines     = (PFN_vkCreateComputePipelines)vkGetInstanceProcAddr(instance, "vkCreateComputePipelines");
        ctx->DestroyPipeline            = (PFN_vkDestroyPipeline)vkGetInstanceProcAddr(instance, "vkDestroyPipeline");
        ctx->CreateFence                = (PFN_vkCreateFence)vkGetInstanceProcAddr(instance, "vkCreateFence");
        ctx->DestroyFence               = (PFN_vkDestroyFence)vkGetInstanceProcAddr(instance, "vkDestroyFence");
        ctx->WaitForFences              = (PFN_vkWaitForFences)vkGetInstanceProcAddr(instance, "vkWaitForFences");
        ctx->QueueSubmit                = (PFN_vkQueueSubmit)vkGetInstanceProcAddr(instance, "vkQueueSubmit");
    }

    PFN_vkGetDeviceQueue get_q = (PFN_vkGetDeviceQueue)vkGetDeviceQueue;
    if (get_q && device)
        get_q(device, 0, 0, &ctx->queue);

    *pContext = ctx;
    return VK_SUCCESS;
}

void vkblas_destroy_context(VkBLASContext* context) {
    if (!context || context->magic != VKBLAS_MAGIC) return;
    /* Cleanup pipelines, descriptor pools, command pools */
    if (context->q8_0.pool)      context->DestroyDescriptorPool(context->device, context->q8_0.pool, NULL);
    if (context->q8_0.dsl)       context->DestroyDescriptorSetLayout(context->device, context->q8_0.dsl, NULL);
    if (context->q8_0.pipe_layout) context->DestroyPipelineLayout(context->device, context->q8_0.pipe_layout, NULL);
    if (context->q8_0.pipeline)  context->DestroyPipeline(context->device, context->q8_0.pipeline, NULL);
    if (context->q8_0.cmdpool)   context->DestroyCommandPool(context->device, context->q8_0.cmdpool, NULL);
    if (context->q8_0.shader)    context->DestroyShaderModule(context->device, context->q8_0.shader, NULL);

    free(context);
}

VkResult vkblas_qgemm_q8_0_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
    int32_t m, int32_t n, int32_t k,
    const float* alpha, VkBuffer Wq, int32_t ldw,
    VkBuffer x, int32_t ldx,
    const float* beta, VkBuffer y, int32_t ldy)
{
    (void)cmd;
    if (!ctx || ctx->magic != VKBLAS_MAGIC) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->q8_0.pipe_layout) {
#ifdef VAIST_SPIRV_EMBEDDED
        VkResult r = _init_qgemm_slot(ctx, &ctx->q8_0,
            vaist_spv_qgemm_q8_0, vaist_spv_qgemm_q8_0_len);
        if (r) return r;
#else
        return VK_NOT_READY;
#endif
    }
    vkblas_qgemm_pc pc;
    pc.alpha = alpha ? *alpha : 1.0f;
    pc.beta  = beta  ? *beta  : 0.0f;
    pc.m = m; pc.n = n; pc.k = k;
    pc.ldw = ldw; pc.ldx = ldx; pc.ldy = ldy;
    pc.flags = VKBLAS_QGEMM_Q8_0;
    VkBuffer bufs[3] = {Wq, x, y};
    uint32_t gx = (uint32_t)((n + 63) / 64);
    uint32_t gy = (uint32_t)((m + 63) / 64);
    return _dispatch(ctx, &ctx->q8_0, 3, bufs, &pc, sizeof(pc), gx, gy);
}

VkResult vkblas_qgemm_nvfp4_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
    int32_t m, int32_t n, int32_t k,
    const float* alpha, VkBuffer Wq, int32_t ldw,
    VkBuffer x, int32_t ldx,
    const float* beta, VkBuffer y, int32_t ldy)
{
    (void)cmd;
    if (!ctx || ctx->magic != VKBLAS_MAGIC) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->nvfp4.pipe_layout) {
#ifdef VAIST_SPIRV_EMBEDDED
        VkResult r = _init_qgemm_slot(ctx, &ctx->nvfp4,
            vaist_spv_qgemm_nvfp4, vaist_spv_qgemm_nvfp4_len);
        if (r) return r;
#else
        return VK_NOT_READY;
#endif
    }
    vkblas_qgemm_pc pc;
    pc.alpha = alpha ? *alpha : 1.0f;
    pc.beta  = beta  ? *beta  : 0.0f;
    pc.m = m; pc.n = n; pc.k = k;
    pc.ldw = ldw; pc.ldx = ldx; pc.ldy = ldy;
    pc.flags = VKBLAS_QGEMM_NVFP4;
    VkBuffer bufs[3] = {Wq, x, y};
    uint32_t gx = (uint32_t)((n + 15) / 64);
    uint32_t gy = (uint32_t)((m + 15) / 16);
    return _dispatch(ctx, &ctx->nvfp4, 3, bufs, &pc, sizeof(pc), gx, gy);
}

VkResult vkblas_qgemm_t2_0_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
    int32_t m, int32_t n, int32_t k,
    const float* alpha, VkBuffer Wq, int32_t ldw,
    VkBuffer x, int32_t ldx,
    const float* beta, VkBuffer y, int32_t ldy)
{
    (void)cmd;
    if (!ctx || ctx->magic != VKBLAS_MAGIC) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->t2_0.pipe_layout) {
#ifdef VAIST_SPIRV_EMBEDDED
        VkResult r = _init_qgemm_slot(ctx, &ctx->t2_0,
            vaist_spv_qgemm_t2_0, vaist_spv_qgemm_t2_0_len);
        if (r) return r;
#else
        return VK_NOT_READY;
#endif
    }
    vkblas_qgemm_pc pc;
    pc.alpha = alpha ? *alpha : 1.0f;
    pc.beta  = beta  ? *beta  : 0.0f;
    pc.m = m; pc.n = n; pc.k = k;
    pc.ldw = ldw; pc.ldx = ldx; pc.ldy = ldy;
    pc.flags = VKBLAS_QGEMM_T2_0;
    VkBuffer bufs[3] = {Wq, x, y};
    uint32_t gx = (uint32_t)((n + 255) / 256);
    uint32_t gy = (uint32_t)((m + 15) / 16);
    return _dispatch(ctx, &ctx->t2_0, 3, bufs, &pc, sizeof(pc), gx, gy);
}

VkResult vkblas_gemm_spec_verify_f32(VkBLASContext* ctx, VkCommandBuffer cmd,
    int32_t batch, int32_t num_heads, int32_t head_dim,
    int32_t seqlen,
    VkBuffer q_gpu, VkBuffer k_cache, VkBuffer v_cache,
    VkBuffer block_table, VkBuffer draft_tokens,
    VkBuffer active_tokens, VkBuffer out,
    uint32_t verify_mask, uint32_t spec_depth, float scale)
{
    (void)cmd;
    if (!ctx || ctx->magic != VKBLAS_MAGIC) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->spec_verify.pipe_layout) {
#ifdef VAIST_SPIRV_EMBEDDED
        VkResult r = _init_spec_slot(ctx, &ctx->spec_verify,
            vaist_spv_attn_speculative, vaist_spv_attn_speculative_len);
        if (r) return r;
#else
        return VK_NOT_READY;
#endif
    }
    vkblas_spec_pc pc;
    pc.scale        = scale;
    pc.head_dim     = (uint32_t)head_dim;
    pc.num_heads    = (uint32_t)num_heads;
    pc.num_kv_heads = (uint32_t)num_heads;
    pc.seqlen       = (uint32_t)seqlen;
    pc.max_blocks   = 0;
    pc.block_size   = 0;
    pc.verify_mask  = verify_mask;
    pc.spec_depth   = spec_depth;
    pc.reserved     = 0;
    VkBuffer bufs[7] = {q_gpu, k_cache, v_cache, block_table, out, draft_tokens, active_tokens};
    return _dispatch(ctx, (void *)&ctx->spec_verify, 7, bufs, &pc, sizeof(pc),
        (uint32_t)batch, (uint32_t)num_heads);
}

#else  /* !VAIST_HAVE_VK_HDR: stubs (no Vulkan headers available) */

#define VkResult int
#define VK_SUCCESS 0
#define VK_NOT_READY 1
#define VK_ERROR_INITIALIZATION_FAILED -2
#define VK_ERROR_OUT_OF_HOST_MEMORY -3

/* Full definition for stub path */
struct VkBLASContext { uint32_t magic; };

int vkblas_create_context(void* i, void* p, void* d, VkBLASContext** o) {
    (void)i; (void)p; (void)d;
    if (!o) return VK_ERROR_INITIALIZATION_FAILED;
    *o = (VkBLASContext *)calloc(1, sizeof(VkBLASContext));
    if (!*o) return VK_ERROR_OUT_OF_HOST_MEMORY;
    (*o)->magic = VKBLAS_MAGIC;
    return 0;
}
void vkblas_destroy_context(VkBLASContext* c) { if (c) free(c); }

int vkblas_qgemm_q8_0_f32(VkBLASContext* ctx, void* cmd,
    int32_t m, int32_t n, int32_t k, const float* a, void* Wq, int32_t ldw,
    void* x, int32_t ldx, const float* beta, void* y, int32_t ldy) {
    (void)ctx;(void)cmd;(void)m;(void)n;(void)k;(void)a;(void)Wq;(void)ldw;
    (void)x;(void)ldx;(void)beta;(void)y;(void)ldy; return VK_NOT_READY;
}
int vkblas_qgemm_nvfp4_f32(VkBLASContext* ctx, void* cmd,
    int32_t m, int32_t n, int32_t k, const float* a, void* Wq, int32_t ldw,
    void* x, int32_t ldx, const float* beta, void* y, int32_t ldy) {
    (void)ctx;(void)cmd;(void)m;(void)n;(void)k;(void)a;(void)Wq;(void)ldw;
    (void)x;(void)ldx;(void)beta;(void)y;(void)ldy; return VK_NOT_READY;
}
int vkblas_qgemm_t2_0_f32(VkBLASContext* ctx, void* cmd,
    int32_t m, int32_t n, int32_t k, const float* a, void* Wq, int32_t ldw,
    void* x, int32_t ldx, const float* beta, void* y, int32_t ldy) {
    (void)ctx;(void)cmd;(void)m;(void)n;(void)k;(void)a;(void)Wq;(void)ldw;
    (void)x;(void)ldx;(void)beta;(void)y;(void)ldy; return VK_NOT_READY;
}
int vkblas_gemm_spec_verify_f32(VkBLASContext* ctx, void* cmd,
    int32_t batch, int32_t num_heads, int32_t head_dim, int32_t seqlen,
    void* q_gpu, void* k_cache, void* v_cache,
    void* block_table, void* draft_tokens,
    void* active_tokens, void* out,
    uint32_t verify_mask, uint32_t spec_depth, float scale) {
    (void)ctx;(void)cmd;(void)batch;(void)num_heads;(void)head_dim;(void)seqlen;
    (void)q_gpu;(void)k_cache;(void)v_cache;(void)block_table;(void)draft_tokens;
    (void)active_tokens;(void)out;(void)verify_mask;(void)spec_depth;(void)scale;
    return VK_NOT_READY;
}
#undef VkResult
#endif
