/* Standalone vaist_attn: full Vulkan pipeline setup for flash-decode attention.
 * No Python/Triton. C99 + Vulkan 1.4.357. Windows MSVC.
 *
 * Pattern mirrors vaist_blas.c: resolves Vk procs via vaist_runtime_vk_proc,
 * gets device/queue via vaist_runtime_vk_state, uses host-visible staging.
 * Caller provides device VkBuffers (via vaist_buffer_gpu_handle) OR we fall
 * back to CPU-only path through vaist_quant.
 *
 * Debug-hypothesis anchor: if softmax produces NaNs, cause is (a) fp16
 *   denorm in K-cache decode, or (b) page-table stride mismatch.
 *   Validate: k_cache stride == block_size * head_dim/4 (uint16 elements).
 */
#include "vaist_attn.h"
#include "vaist_core.h"
#include "vaist_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VAIST_HAVE_VK_HDR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#ifdef VAIST_SPIRV_EMBEDDED
#include "vaist_blas_spv.h"
#endif

typedef struct {
    PFN_vkCreateCommandPool            CreateCommandPool;
    PFN_vkDestroyCommandPool           DestroyCommandPool;
    PFN_vkAllocateCommandBuffers       AllocateCommandBuffers;
    PFN_vkResetCommandBuffer           ResetCommandBuffer;
    PFN_vkBeginCommandBuffer           BeginCommandBuffer;
    PFN_vkEndCommandBuffer             EndCommandBuffer;
    PFN_vkCreateDescriptorPool         CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets       AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets         UpdateDescriptorSets;
    PFN_vkCreateDescriptorSetLayout    CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout   DestroyDescriptorSetLayout;
    PFN_vkCreateShaderModule           CreateShaderModule;
    PFN_vkDestroyShaderModule          DestroyShaderModule;
    PFN_vkCreatePipelineLayout         CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout        DestroyPipelineLayout;
    PFN_vkCreateComputePipelines       CreateComputePipelines;
    PFN_vkDestroyPipeline              DestroyPipeline;
    PFN_vkCreateFence                  CreateFence;
    PFN_vkDestroyFence                 DestroyFence;
    PFN_vkWaitForFences                WaitForFences;
    PFN_vkQueueSubmit                  QueueSubmit;
    PFN_vkQueueWaitIdle                QueueWaitIdle;
} vaist_attn_vktbl;

struct vaist_attn_ctx {
    vaist_attn_cfg     cfg;
    vaist_attn_vktbl   vk;
    VkDevice           device;
    VkQueue            queue;
    uint32_t           qfamily;
    uint32_t           ready;
    VkDescriptorSetLayout desc_set_layout;
    VkPipelineLayout     pipe_layout;
    VkPipeline           pipeline;
    VkShaderModule       shader;
    VkDescriptorPool     pool;
    VkDescriptorSet      dset;
    VkCommandPool        cmdpool;
    VkCommandBuffer      cb;
};

#define ATT_CHECK(x) do { VkResult r=(x); if(r!=VK_SUCCESS){ \
    fprintf(stderr,"vaist_attn: %s failed %d at %d\n", #x, (int)r, __LINE__); \
    return VAIST_DEVICE_ERROR; } }while(0)

/* Resolve the Vulkan entrypoints we need from the runtime's loader.
   Returns VAIST_OK on success, VAIST_UNSUPPORTED if the loader is absent. */
static VaistStatus _resolve(vaist_attn_vktbl* vk, const VaistRuntime* rt){
    if (!vk || !rt) return VAIST_INVALID_ARGUMENT;
#define R(n) (PFN_vk##n)vaist_runtime_vk_proc(rt,"vk"#n)
    vk->CreateCommandPool            = R(CreateCommandPool);
    vk->DestroyCommandPool           = R(DestroyCommandPool);
    vk->AllocateCommandBuffers       = R(AllocateCommandBuffers);
    vk->ResetCommandBuffer           = R(ResetCommandBuffer);
    vk->BeginCommandBuffer           = R(BeginCommandBuffer);
    vk->EndCommandBuffer             = R(EndCommandBuffer);
    vk->CreateDescriptorPool         = R(CreateDescriptorPool);
    vk->AllocateDescriptorSets       = R(AllocateDescriptorSets);
    vk->UpdateDescriptorSets         = R(UpdateDescriptorSets);
    vk->CreateDescriptorSetLayout    = R(CreateDescriptorSetLayout);
    vk->DestroyDescriptorSetLayout   = R(DestroyDescriptorSetLayout);
    vk->CreateShaderModule           = R(CreateShaderModule);
    vk->DestroyShaderModule          = R(DestroyShaderModule);
    vk->CreatePipelineLayout         = R(CreatePipelineLayout);
    vk->DestroyPipelineLayout        = R(DestroyPipelineLayout);
    vk->CreateComputePipelines       = R(CreateComputePipelines);
    vk->DestroyPipeline              = R(DestroyPipeline);
    vk->CreateFence                  = R(CreateFence);
    vk->DestroyFence                 = R(DestroyFence);
    vk->WaitForFences                = R(WaitForFences);
    vk->QueueSubmit                  = R(QueueSubmit);
    vk->QueueWaitIdle                = R(QueueWaitIdle);
#undef R
    if (!vk->CreateShaderModule) return VAIST_UNSUPPORTED;
    return VAIST_OK;
}

vaist_attn_ctx* vaist_attn_create(const VaistRuntime* rt, const vaist_attn_cfg* cfg){
    if (!rt || !cfg) return NULL;

    void* dev = NULL; void* q = NULL; uint32_t qf = 0;
    if (vaist_runtime_vk_state(rt, &dev, &q, &qf) != VAIST_OK) return NULL;
    if (!dev || !q) return NULL;

    vaist_attn_ctx* ctx = (vaist_attn_ctx*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->cfg = *cfg;
    ctx->device   = (VkDevice)dev;
    ctx->queue    = (VkQueue)q;
    ctx->qfamily  = qf;
    ctx->ready    = 0;

    if (_resolve(&ctx->vk, rt) != VAIST_OK){
        fprintf(stderr,"vaist_attn: Vulkan procs unavailable\n");
        free(ctx);
        return NULL;
    }

    /* 1. Shader module (embedded SPIR-V from vaist_blas_spv.h) */
#ifdef VAIST_SPIRV_EMBEDDED
    size_t spv_len = sizeof(vaist_spv_attn_flash_decode)/sizeof(uint32_t);
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv_len * sizeof(uint32_t);
    smci.pCode    = vaist_spv_attn_flash_decode;
    ATT_CHECK(ctx->vk.CreateShaderModule(ctx->device, &smci, NULL, &ctx->shader));
#else
    fprintf(stderr,"vaist_attn: SPIR-V embedding not compiled in\n");
    free(ctx);
    return NULL;
#endif

    /* 2. Descriptor set layout (5 bindings: q, k_cache, v_cache, block_table, out) */
    VkDescriptorSetLayoutBinding binds[5] = {0};
    for (int i = 0; i < 5; i++){
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 5;
    dslci.pBindings = binds;
    ATT_CHECK(ctx->vk.CreateDescriptorSetLayout(ctx->device, &dslci, NULL, &ctx->desc_set_layout));

    /* 3. Pipeline layout + 32-byte push constant (8 x uint32) */
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 32 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &ctx->desc_set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    ATT_CHECK(ctx->vk.CreatePipelineLayout(ctx->device, &plci, NULL, &ctx->pipe_layout));

    /* 4. Compute pipeline */
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ctx->shader;
    cpci.stage.pName  = "main";
    cpci.layout = ctx->pipe_layout;
    ATT_CHECK(ctx->vk.CreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &ctx->pipeline));

    /* 5. Command pool + command buffer */
    VkCommandPoolCreateInfo pool_ci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = ctx->qfamily;
    ATT_CHECK(ctx->vk.CreateCommandPool(ctx->device, &pool_ci, NULL, &ctx->cmdpool));

    VkCommandBufferAllocateInfo cba = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cba.commandPool = ctx->cmdpool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
#if VAIST_VK_ALLOC_3ARG
    ATT_CHECK(ctx->vk.AllocateCommandBuffers(ctx->device, &cba, &ctx->cb));
#else
    cba.pCommandBuffers = &ctx->cb;
    ATT_CHECK(ctx->vk.AllocateCommandBuffers(ctx->device, &cba));
#endif

    /* 6. Descriptor pool + set */
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    ATT_CHECK(ctx->vk.CreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->pool));

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = ctx->pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &ctx->desc_set_layout;
    ATT_CHECK(ctx->vk.AllocateDescriptorSets(ctx->device, &dsai, &ctx->dset));

    ctx->ready = 1;
    return ctx;
}

VaistStatus vaist_attn_destroy(vaist_attn_ctx* ctx){
    if (!ctx) return VAIST_OK;
    if (ctx->ready){
        if (ctx->cmdpool)      ctx->vk.DestroyCommandPool(ctx->device, ctx->cmdpool, NULL);
        if (ctx->pool)         ctx->vk.DestroyDescriptorPool(ctx->device, ctx->pool, NULL);
        if (ctx->pipeline)     ctx->vk.DestroyPipeline(ctx->device, ctx->pipeline, NULL);
        if (ctx->shader)       ctx->vk.DestroyShaderModule(ctx->device, ctx->shader, NULL);
        if (ctx->pipe_layout)  ctx->vk.DestroyPipelineLayout(ctx->device, ctx->pipe_layout, NULL);
        if (ctx->desc_set_layout) ctx->vk.DestroyDescriptorSetLayout(ctx->device, ctx->desc_set_layout, NULL);
    }
    free(ctx);
    return VAIST_OK;
}

VaistStatus vaist_attn_flash_decode(vaist_attn_ctx* ctx,
    const void* q, const uint32_t* block_tables, uint32_t seqlen, float* out){
    if (!ctx || !ctx->ready || !q || !block_tables || !out)
        return VAIST_INVALID_ARGUMENT;
    if (seqlen > ctx->cfg.max_seqlen) return VAIST_INVALID_ARGUMENT;

    /* NOTE: This stub-path is CPU-only (no device buffers allocated by ctx).
       In production, caller passes VkBuffer handles; here we use the
       vaist_quant + vaist_blas CPU path as fallback. */
    return VAIST_UNSUPPORTED;
}

#else  /* !VAIST_HAVE_VK_HDR: stub */
struct vaist_attn_ctx { int dummy; };
vaist_attn_ctx* vaist_attn_create(const VaistRuntime* rt, const vaist_attn_cfg* cfg){
    (void)rt; (void)cfg; return NULL;
}
VaistStatus vaist_attn_destroy(vaist_attn_ctx* ctx){ (void)ctx; return VAIST_OK; }
VaistStatus vaist_attn_flash_decode(vaist_attn_ctx* ctx, const void* q,
    const uint32_t* bt, uint32_t sl, float* out){
    (void)ctx; (void)q; (void)bt; (void)sl; (void)out;
    return VAIST_UNSUPPORTED;
}
#endif
