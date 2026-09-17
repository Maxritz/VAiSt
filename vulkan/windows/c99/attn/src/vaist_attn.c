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
    PFN_vkDestroyDescriptorPool        DestroyDescriptorPool;
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
    PFN_vkCmdBindPipeline              CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets        CmdBindDescriptorSets;
    PFN_vkCmdPushConstants             CmdPushConstants;
    PFN_vkCmdDispatch                  CmdDispatch;
    PFN_vkCmdPipelineBarrier           CmdPipelineBarrier;
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
    /* Device-side buffers managed by ctx (host-visible for staging). */
    PFN_vkCreateBuffer         CreateBuffer;
    PFN_vkDestroyBuffer        DestroyBuffer;
    PFN_vkAllocateMemory       AllocateMemory;
    PFN_vkFreeMemory           FreeMemory;
    PFN_vkBindBufferMemory     BindBufferMemory;
    PFN_vkMapMemory            MapMemory;
    PFN_vkUnmapMemory          UnmapMemory;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    VkBuffer           buf_q;
    VkBuffer           buf_bt;
    VkBuffer           buf_out;
    VkDeviceMemory     mem_q;
    VkDeviceMemory     mem_bt;
    VkDeviceMemory     mem_out;
    size_t             sz_q;
    size_t             sz_bt;
    size_t             sz_out;
};

#define ATT_CHECK(x) do { VkResult r=(x); if(r!=VK_SUCCESS){ \
    fprintf(stderr,"vaist_attn: %s failed %d at %d\n", #x, (int)r, __LINE__); \
    return NULL; } }while(0)
#define ATT_CHECK_STATUS(x) do { VkResult r=(x); if(r!=VK_SUCCESS){ \
    fprintf(stderr,"vaist_attn: %s failed %d at %d\n", #x, (int)r, __LINE__); \
    return VAIST_DEVICE_ERROR; } }while(0)

static VaistStatus _resolve(vaist_attn_ctx* ctx, const VaistRuntime* rt){
    if (!ctx || !rt) return VAIST_INVALID_ARGUMENT;
#define R(n) (PFN_vk##n)vaist_runtime_vk_proc(rt,"vk"#n)
    ctx->vk.CreateCommandPool            = R(CreateCommandPool);
    ctx->vk.DestroyCommandPool           = R(DestroyCommandPool);
    ctx->vk.AllocateCommandBuffers       = R(AllocateCommandBuffers);
    ctx->vk.ResetCommandBuffer           = R(ResetCommandBuffer);
    ctx->vk.BeginCommandBuffer           = R(BeginCommandBuffer);
    ctx->vk.EndCommandBuffer             = R(EndCommandBuffer);
    ctx->vk.CreateDescriptorPool         = R(CreateDescriptorPool);
    ctx->vk.AllocateDescriptorSets       = R(AllocateDescriptorSets);
    ctx->vk.UpdateDescriptorSets         = R(UpdateDescriptorSets);
    ctx->vk.CreateDescriptorSetLayout    = R(CreateDescriptorSetLayout);
    ctx->vk.DestroyDescriptorSetLayout   = R(DestroyDescriptorSetLayout);
    ctx->vk.CreateShaderModule           = R(CreateShaderModule);
    ctx->vk.DestroyShaderModule          = R(DestroyShaderModule);
    ctx->vk.CreatePipelineLayout         = R(CreatePipelineLayout);
    ctx->vk.DestroyPipelineLayout        = R(DestroyPipelineLayout);
    ctx->vk.CreateComputePipelines       = R(CreateComputePipelines);
    ctx->vk.DestroyPipeline              = R(DestroyPipeline);
    ctx->vk.CreateFence                  = R(CreateFence);
    ctx->vk.DestroyFence                 = R(DestroyFence);
    ctx->vk.WaitForFences                = R(WaitForFences);
    ctx->vk.QueueSubmit                  = R(QueueSubmit);
    ctx->vk.QueueWaitIdle                = R(QueueWaitIdle);
    ctx->vk.DestroyDescriptorPool       = R(DestroyDescriptorPool);
    ctx->vk.CmdBindPipeline              = R(CmdBindPipeline);
    ctx->vk.CmdBindDescriptorSets        = R(CmdBindDescriptorSets);
    ctx->vk.CmdPushConstants             = R(CmdPushConstants);
    ctx->vk.CmdDispatch                  = R(CmdDispatch);
    ctx->vk.CmdPipelineBarrier           = R(CmdPipelineBarrier);
    ctx->CreateBuffer                    = R(CreateBuffer);
    ctx->DestroyBuffer             = R(DestroyBuffer);
    ctx->AllocateMemory            = R(AllocateMemory);
    ctx->FreeMemory                = R(FreeMemory);
    ctx->BindBufferMemory          = R(BindBufferMemory);
    ctx->MapMemory                 = R(MapMemory);
    ctx->UnmapMemory               = R(UnmapMemory);
    ctx->GetBufferMemoryRequirements = R(GetBufferMemoryRequirements);
#undef R
    if (!ctx->vk.CreateShaderModule) return VAIST_UNSUPPORTED;
    if (!ctx->CreateBuffer) return VAIST_UNSUPPORTED;
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

    if (_resolve(ctx, rt) != VAIST_OK){
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

    /* 7. Device buffers: staging (host-visible) for Q, block_table, output.
       Note: K/V cache buffers are owned by the caller (via vaist_buffer_create),
       not by attn — they're persistent across calls.
       Resolve memory props through the same loader-proc mechanism vaist_blas.c uses. */
    {
        PFN_vkGetPhysicalDeviceMemoryProperties gmem;
        gmem = (PFN_vkGetPhysicalDeviceMemoryProperties)
            vaist_runtime_vk_proc(rt, "vkGetPhysicalDeviceMemoryProperties");
        if (!gmem){
            /* Runtime may not have Vulkan available (weak GPU box). */
            fprintf(stderr,"vaist_attn: no vkGetPhysicalDeviceMemoryProperties\n");
            goto create_fail;
        }
        VkPhysicalDevice pd = VK_NULL_HANDLE;
        {
            void* _d = NULL; void* _q = NULL; uint32_t _qf = 0;
            if (vaist_runtime_vk_state(rt, &_d, &_q, &_qf) != VAIST_OK || !_d){
                goto create_fail;
            }
            pd = (VkPhysicalDevice)_d;
        }
        VkPhysicalDeviceMemoryProperties mprops;
        gmem(pd, &mprops);

        uint32_t mt_host = 0;
        int found = 0;
        for (uint32_t t = 0; t < mprops.memoryTypeCount; t++){
            if ((mprops.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mprops.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
                mt_host = t; found = 1; break;
            }
        }
        if (!found){
            for (uint32_t t = 0; t < mprops.memoryTypeCount; t++){
                if (mprops.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT){
                    mt_host = t; found = 1; break;
                }
            }
        }
        if (!found){
            fprintf(stderr,"vaist_attn: no host-visible memory type\n");
            goto create_fail;
        }

        /* Allocate Q buffer */
        ctx->sz_q = cfg->num_q_heads * cfg->head_dim * sizeof(float);
        {
            VkBufferCreateInfo bc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bc.size = ctx->sz_q;
            bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ATT_CHECK(ctx->CreateBuffer(ctx->device, &bc, NULL, &ctx->buf_q));
            VkMemoryRequirements mr;
            ctx->GetBufferMemoryRequirements(ctx->device, ctx->buf_q, &mr);
            VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = mt_host;
            ATT_CHECK(ctx->AllocateMemory(ctx->device, &mai, NULL, &ctx->mem_q));
            ATT_CHECK(ctx->BindBufferMemory(ctx->device, ctx->buf_q, ctx->mem_q, 0));
        }
        /* Allocate block_table buffer */
        ctx->sz_bt = cfg->num_kv_heads * cfg->max_blocks * sizeof(uint32_t);
        {
            VkBufferCreateInfo bc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bc.size = ctx->sz_bt;
            bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ATT_CHECK(ctx->CreateBuffer(ctx->device, &bc, NULL, &ctx->buf_bt));
            VkMemoryRequirements mr;
            ctx->GetBufferMemoryRequirements(ctx->device, ctx->buf_bt, &mr);
            VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = mt_host;
            ATT_CHECK(ctx->AllocateMemory(ctx->device, &mai, NULL, &ctx->mem_bt));
            ATT_CHECK(ctx->BindBufferMemory(ctx->device, ctx->buf_bt, ctx->mem_bt, 0));
        }
        /* Allocate output buffer */
        ctx->sz_out = cfg->num_q_heads * cfg->head_dim * sizeof(float);
        {
            VkBufferCreateInfo bc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bc.size = ctx->sz_out;
            bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ATT_CHECK(ctx->CreateBuffer(ctx->device, &bc, NULL, &ctx->buf_out));
            VkMemoryRequirements mr;
            ctx->GetBufferMemoryRequirements(ctx->device, ctx->buf_out, &mr);
            VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = mt_host;
            ATT_CHECK(ctx->AllocateMemory(ctx->device, &mai, NULL, &ctx->mem_out));
            ATT_CHECK(ctx->BindBufferMemory(ctx->device, ctx->buf_out, ctx->mem_out, 0));
        }
    }

    ctx->ready = 1;
    return ctx;

create_fail:
    vaist_attn_destroy(ctx);
    return NULL;
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
        if (ctx->buf_out)      ctx->DestroyBuffer(ctx->device, ctx->buf_out, NULL);
        if (ctx->mem_out)      ctx->FreeMemory(ctx->device, ctx->mem_out, NULL);
        if (ctx->buf_bt)       ctx->DestroyBuffer(ctx->device, ctx->buf_bt, NULL);
        if (ctx->mem_bt)       ctx->FreeMemory(ctx->device, ctx->mem_bt, NULL);
        if (ctx->buf_q)        ctx->DestroyBuffer(ctx->device, ctx->buf_q, NULL);
        if (ctx->mem_q)        ctx->FreeMemory(ctx->device, ctx->mem_q, NULL);
    }
    free(ctx);
    return VAIST_OK;
}

VaistStatus vaist_attn_flash_decode(vaist_attn_ctx* ctx,
    const void* q, void* k_cache_gpu, void* v_cache_gpu,
    const uint32_t* block_tables, uint32_t seqlen, float* out){
    if (!ctx || !ctx->ready || !q || !k_cache_gpu || !v_cache_gpu || !block_tables || !out)
        return VAIST_INVALID_ARGUMENT;
    if (seqlen > ctx->cfg.max_seqlen) return VAIST_INVALID_ARGUMENT;

    /* 1. Upload Q to staging buffer */
    void* p = NULL;
    ATT_CHECK_STATUS(ctx->MapMemory(ctx->device, ctx->mem_q, 0, ctx->sz_q, 0, &p));
    memcpy(p, q, ctx->sz_q);
    ctx->UnmapMemory(ctx->device, ctx->mem_q);

    /* 2. Upload block tables */
    ATT_CHECK_STATUS(ctx->MapMemory(ctx->device, ctx->mem_bt, 0, ctx->sz_bt, 0, &p));
    memcpy(p, block_tables, ctx->sz_bt);
    ctx->UnmapMemory(ctx->device, ctx->mem_bt);

    /* 3. Update descriptors (binding 0=q, 1=k_cache, 2=v_cache, 3=bt, 4=out) */
    VkDescriptorBufferInfo q_info  = { ctx->buf_q,   0, (VkDeviceSize)ctx->sz_q };
    VkDescriptorBufferInfo k_info  = { (VkBuffer)k_cache_gpu, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo v_info  = { (VkBuffer)v_cache_gpu, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bt_info = { ctx->buf_bt, 0, (VkDeviceSize)ctx->sz_bt };
    VkDescriptorBufferInfo o_info  = { ctx->buf_out, 0, (VkDeviceSize)ctx->sz_out };
    VkDescriptorBufferInfo infos[5] = { q_info, k_info, v_info, bt_info, o_info };
    VkWriteDescriptorSet wds[5];
    for (int i = 0; i < 5; i++){
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = ctx->dset;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &infos[i];
    }
    ctx->vk.UpdateDescriptorSets(ctx->device, 5, wds, 0, NULL);

    /* 4. Push constants: scale, head_dim, num_heads(kv), kv_head_idx,
       seqlen, max_blocks, block_size, q_head_idx */
    struct { float scale; uint32_t a,b,c,d,e,f2,g,h; } pc;
    pc.scale  = ctx->cfg.scale;
    pc.a      = ctx->cfg.head_dim;
    pc.b      = ctx->cfg.num_kv_heads;
    pc.c      = 0;            /* kv_head_idx (single-head decode) */
    pc.d      = seqlen;
    pc.e      = ctx->cfg.max_blocks;
    pc.f2     = ctx->cfg.block_size;
    pc.g      = 0;            /* q_head_idx */

    /* 5. Record command buffer */
    ATT_CHECK_STATUS(ctx->vk.ResetCommandBuffer(ctx->cb, 0));
    VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    ATT_CHECK_STATUS(ctx->vk.BeginCommandBuffer(ctx->cb, &cbbi));
    ctx->vk.CmdBindPipeline(ctx->cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    ctx->vk.CmdBindDescriptorSets(ctx->cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe_layout,
        0, 1, &ctx->dset, 0, NULL);
    ctx->vk.CmdPushConstants(ctx->cb, ctx->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, &pc);
    ctx->vk.CmdDispatch(ctx->cb, ctx->cfg.num_q_heads, 1, 1);

    /* Memory barrier: ensure shader writes finish before readback */
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    ctx->vk.CmdPipelineBarrier(ctx->cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    ATT_CHECK_STATUS(ctx->vk.EndCommandBuffer(ctx->cb));

    /* 6. Submit + wait */
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &ctx->cb;
    VkFence f = VK_NULL_HANDLE;
    ATT_CHECK_STATUS(ctx->vk.CreateFence(ctx->device, &(VkFenceCreateInfo){VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL, &f));
    ATT_CHECK_STATUS(ctx->vk.QueueSubmit(ctx->queue, 1, &si, f));
    ATT_CHECK_STATUS(ctx->vk.WaitForFences(ctx->device, 1, &f, VK_TRUE, UINT64_MAX));
    ctx->vk.DestroyFence(ctx->device, f, NULL);

    /* 7. Read output */
    ATT_CHECK_STATUS(ctx->MapMemory(ctx->device, ctx->mem_out, 0, ctx->sz_out, 0, &p));
    memcpy(out, p, ctx->sz_out);
    ctx->UnmapMemory(ctx->device, ctx->mem_out);

    return VAIST_OK;
}

#else  /* !VAIST_HAVE_VK_HDR: stub */
struct vaist_attn_ctx { int dummy; };
vaist_attn_ctx* vaist_attn_create(const VaistRuntime* rt, const vaist_attn_cfg* cfg){
    (void)rt; (void)cfg; return NULL;
}
VaistStatus vaist_attn_destroy(vaist_attn_ctx* ctx){ (void)ctx; return VAIST_OK; }
VaistStatus vaist_attn_flash_decode(vaist_attn_ctx* ctx, const void* q,
    void* kg, void* vg, const uint32_t* bt, uint32_t sl, float* out){
    (void)ctx; (void)q; (void)kg; (void)vg; (void)bt; (void)sl; (void)out;
    return VAIST_UNSUPPORTED;
}
#endif
