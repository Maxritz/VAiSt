#include "vaist_blas.h"
#include "vaist_compute.h"
#include "vaist_quant.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

/* ---- CPU kernels: scalar fallback, Zen3-friendly (no SIMD forced here;
 *      callers can swap in vaist_compute SIMD for the dense fp32 path). ---- */

static VaistStatus gemm_scalar(const float*A,const float*B,float*C,
        size_t m,size_t k,size_t n){
    size_t i,j,p;
    if(!A||!B||!C||!m||!k||!n||m>SIZE_MAX/k||m*k>SIZE_MAX/n||k>SIZE_MAX/n||n>SIZE_MAX/k)
        return VAIST_INVALID_ARGUMENT;
    for(i=0;i<m;i++)for(j=0;j<n;j++){
        float s=0.0f;
        for(p=0;p<k;p++) s += A[i*k+p]*B[p*n+j];
        C[i*n+j] += s;
    }
    return VAIST_OK;
}

/* ---- MatMul-free: ternary {-1,0,+1} (2406.02528, 2608.03142) ---- */
static VaistStatus vaist_blas_matvec_ternary_cpu(const int8_t*wsign,const float*scale,
        size_t k,size_t n,const float*x,float*y){
    size_t i,j;
    if(!wsign||!scale||!x||!y||!k||!n) return VAIST_INVALID_ARGUMENT;
    for(j=0;j<n;j++){
        float acc=0.0f; const int8_t*w=wsign+j*k;
        for(i=0;i<k;i++){ int8_t s=w[i]; if(s>0) acc+=x[i]; else if(s<0) acc-=x[i]; }
        y[j]=acc*scale[j];
    }
    return VAIST_OK;
}

/* ---- MatMul-free: 1-bit xnor + popcount (2608.01528, 2608.00860) ----
 *   wbit packs 1 weight per bit, MSB first across each byte, row-major (n rows x k).
 *   y[j] = scale[j] * (k - 2*popcount(xor(w, x_sign_plane)))  => xor + popcount,
 *   no multiplies, no inner-loop branch. Sign plane of x = bit (|x|>=0 ? 1 : 0).
 *   Host pads k to a multiple of 8 (VAIST_BLAS requires this for the tile). */
static inline uint32_t vaist_blas_pop8(uint8_t v){
    v-=(v>>1)&0x55u; v=(v&0x33u)+((v>>2)&0x33u); return (uint32_t)((v+(v>>4))&0x0fu);
}
static VaistStatus vaist_blas_matvec_binary_cpu(const uint8_t*wbit,const float*scale,
        size_t k,size_t n,const float*x,float*y){
    size_t i,j; size_t k8=k/8u;
    if(!wbit||!scale||!x||!y||!k||!n||(k%8u)) return VAIST_INVALID_ARGUMENT;
    for(j=0;j<n;j++){
        const uint8_t*w=wbit+(j*k8);
        int32_t agree=0;
        for(i=0;i<k8;i++){
            uint8_t xbits=0u; const float*xp=x+i*8u; uint32_t b=0u;
            for(uint32_t t=0u;t<8u;t++) b |= (xp[t]>=0.0f?1u:0u) << (7u-t);
            xbits=(uint8_t)b;
            agree += (int)(8 - 2*vaist_blas_pop8((uint8_t)(w[i]^xbits))); /* agree-disagree */
        }
        y[j]= (float)agree * scale[j];
    }
    return VAIST_OK;
}

/* ---- dispatch ---- */
VAIST_API VaistComputePath vaist_blas_best_path(const VaistRuntime*rt,
        size_t m,size_t k,size_t n,VaistWeightFormat wfmt){
    (void)m;(void)k;(void)n;
    if(wfmt==VAIST_W_TERNARY_I8||wfmt==VAIST_W_BINARY_I1) return VAIST_PATH_MATMUL_FREE;
    return vaist_compute_best_path(rt,m,k,n,0);
}

#if VAIST_HAVE_VK_HDR
#include <vulkan/vulkan.h>
#include "vaist_blas_spv.h"   /* generated SPIR-V embeddings (Vulkan builds only) */

/* Resolved entrypoints. Kept additive: resolved lazily from the runtime's
 * loader via vaist_runtime_vk_proc, so the blas library never links libvulkan. */
typedef struct {
    PFN_vkCreateCommandPool            CreateCommandPool;
    PFN_vkAllocateCommandBuffers       AllocateCommandBuffers;
    PFN_vkResetCommandBuffer           ResetCommandBuffer;
    PFN_vkBeginCommandBuffer           BeginCommandBuffer;
    PFN_vkEndCommandBuffer             EndCommandBuffer;
    PFN_vkCreateDescriptorPool         CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets     AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets       UpdateDescriptorSets;
    PFN_vkCreateDescriptorSetLayout    CreateDescriptorSetLayout;
    PFN_vkCreateShaderModule           CreateShaderModule;
    PFN_vkCreatePipelineLayout         CreatePipelineLayout;
    PFN_vkCreateComputePipelines       CreateComputePipelines;
    PFN_vkCmdBindDescriptorSets        CmdBindDescriptorSets;
    PFN_vkCmdBindPipeline            CmdBindPipeline;
    PFN_vkCmdPushConstants           CmdPushConstants;
    PFN_vkCmdDispatch                CmdDispatch;
    PFN_vkCmdPipelineBarrier          CmdPipelineBarrier;
    PFN_vkQueueSubmit                QueueSubmit;
    PFN_vkQueueWaitIdle              QueueWaitIdle;
} vaist_blas_vktbl;

enum { VK_GEMM=0, VK_TER=1, VK_BIN=2 };

typedef struct {
    int tried;
    int ready;
    VkDevice device;
    uint32_t qfamily;
    VkQueue queue;
    vaist_blas_vktbl vk;
    VkCommandPool cmdpool;
    VkDescriptorPool descpool;
    VkDescriptorSetLayout layout3;   /* gemm: 3 SSBOs  */
    VkDescriptorSetLayout layout4;   /* matvec: 4 SSBOs */
    VkShaderModule sh[3];
    VkPipelineLayout pl[3];
    VkPipeline pipe[3];
    VkDescriptorSet set[3];
    VkCommandBuffer cb;
} vaist_blas_ctx;
static vaist_blas_ctx g_ctx;

static int vk_resolve(vaist_blas_vktbl*vk, const VaistRuntime*rt){
    if(!rt) return 0;
#define R(n) (PFN_vk##n)vaist_runtime_vk_proc(rt,"vk"#n)
    vk->CreateCommandPool          = R(CreateCommandPool);
    vk->AllocateCommandBuffers     = R(AllocateCommandBuffers);
    vk->ResetCommandBuffer         = R(ResetCommandBuffer);
    vk->BeginCommandBuffer         = R(BeginCommandBuffer);
    vk->EndCommandBuffer           = R(EndCommandBuffer);
    vk->CreateDescriptorPool       = R(CreateDescriptorPool);
    vk->AllocateDescriptorSets     = R(AllocateDescriptorSets);
    vk->UpdateDescriptorSets       = R(UpdateDescriptorSets);
    vk->CreateDescriptorSetLayout  = R(CreateDescriptorSetLayout);
    vk->CreateShaderModule         = R(CreateShaderModule);
    vk->CreatePipelineLayout       = R(CreatePipelineLayout);
    vk->CreateComputePipelines     = R(CreateComputePipelines);
    vk->CmdBindDescriptorSets      = R(CmdBindDescriptorSets);
    vk->CmdBindPipeline            = R(CmdBindPipeline);
    vk->CmdPushConstants           = R(CmdPushConstants);
    vk->CmdDispatch                = R(CmdDispatch);
    vk->CmdPipelineBarrier         = R(CmdPipelineBarrier);
    vk->QueueSubmit                = R(QueueSubmit);
    vk->QueueWaitIdle              = R(QueueWaitIdle);
#undef R
    return vk->CreateCommandPool && vk->AllocateCommandBuffers &&
           vk->ResetCommandBuffer && vk->BeginCommandBuffer &&
           vk->EndCommandBuffer && vk->CreateDescriptorPool &&
           vk->AllocateDescriptorSets && vk->UpdateDescriptorSets &&
           vk->CreateDescriptorSetLayout && vk->CreateShaderModule &&
           vk->CreatePipelineLayout && vk->CreateComputePipelines &&
           vk->CmdBindDescriptorSets && vk->CmdBindPipeline &&
           vk->CmdPushConstants && vk->CmdDispatch &&
           vk->CmdPipelineBarrier && vk->QueueSubmit && vk->QueueWaitIdle;
}

/* Push-constant structs mirror the GLSL `layout(push_constant) uniform PC` blocks. */
typedef struct { int32_t m,n,k; uint32_t a_off,b_off,c_off; uint32_t flags; } vaist_gemm_pc;
typedef struct { int32_t i0,n; uint32_t w_off,s_off,x_off,y_off; } vaist_mv_pc;
/* ternary i0=k ; binary i0=k8=k/8. Unused *_off fields kept zero (shaders ignore them). */

static int vk_init_once(const VaistRuntime*rt){
    void *dev=NULL,*q=NULL; uint32_t qf=0;
    if(g_ctx.ready) return 1;
    if(g_ctx.tried) return 0;
    g_ctx.tried=1;
    if(vaist_runtime_vk_state(rt,&dev,&q,&qf)!=VAIST_OK) return 0;
    if(!dev||!q) return 0;
    if(!vk_resolve(&g_ctx.vk,rt)) return 0;
    g_ctx.device=(VkDevice)dev; g_ctx.qfamily=qf; g_ctx.queue=(VkQueue)q;
    {
        PFN_vkCreateCommandPool P=g_ctx.vk.CreateCommandPool;
        VkCommandPoolCreateInfo cp={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,NULL,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, g_ctx.qfamily};
        if(P(g_ctx.device,&cp,NULL,&g_ctx.cmdpool)!=VK_SUCCESS) return 0;
    }
    {
        PFN_vkCreateDescriptorPool P=g_ctx.vk.CreateDescriptorPool;
        VkDescriptorPoolSize sz={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,16};
        VkDescriptorPoolCreateInfo dc={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,NULL,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,3,1,&sz};
        if(P(g_ctx.device,&dc,NULL,&g_ctx.descpool)!=VK_SUCCESS) return 0;
    }
    {   /* set layouts: 3-binding (gemm) and 4-binding (matvec) */
        PFN_vkCreateDescriptorSetLayout P=g_ctx.vk.CreateDescriptorSetLayout;
        uint32_t i;
        VkDescriptorSetLayoutBinding b; memset(&b,0,sizeof(b));
        b.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b.descriptorCount=1;
        b.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        /* layout3: bindings 0..2 */
        {
            VkDescriptorSetLayoutBinding bs[3]; 
            for(i=0;i<3;i++){ bs[i]=b; bs[i].binding=i; }
            VkDescriptorSetLayoutCreateInfo lc={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,NULL,0,3,bs};
            if(P(g_ctx.device,&lc,NULL,&g_ctx.layout3)!=VK_SUCCESS) return 0;
        }
        /* layout4: bindings 0..3 */
        {
            VkDescriptorSetLayoutBinding bs[4];
            for(i=0;i<4;i++){ bs[i]=b; bs[i].binding=i; }
            VkDescriptorSetLayoutCreateInfo lc={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,NULL,0,4,bs};
            if(P(g_ctx.device,&lc,NULL,&g_ctx.layout4)!=VK_SUCCESS) return 0;
        }
    }
    {   /* shader modules */
        PFN_vkCreateShaderModule P=g_ctx.vk.CreateShaderModule;
        const uint32_t*code[3]={vaist_spv_gemm_f32_tiled,vaist_spv_matvec_ternary,vaist_spv_matvec_binar};
        size_t bytes[3]={vaist_spv_gemm_f32_tiled_len,vaist_spv_matvec_ternary_len,vaist_spv_matvec_binar_len};
        uint32_t i;
        for(i=0;i<3;i++){
            VkShaderModuleCreateInfo sc; memset(&sc,0,sizeof(sc));
            sc.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            sc.codeSize=bytes[i]; sc.pCode=code[i];
            if(P(g_ctx.device,&sc,NULL,&g_ctx.sh[i])!=VK_SUCCESS) return 0;
        }
    }
    {   /* pipeline layouts + compute pipelines */
        PFN_vkCreatePipelineLayout PL=g_ctx.vk.CreatePipelineLayout;
        PFN_vkCreateComputePipelines CP=g_ctx.vk.CreateComputePipelines;
        uint32_t i;
        for(i=0;i<3;i++){
            VkDescriptorSetLayout lay = (i==VK_GEMM)?g_ctx.layout3:g_ctx.layout4;
            VkPushConstantRange pc; memset(&pc,0,sizeof(pc));
            pc.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
            pc.offset=0;
            pc.size=(i==VK_GEMM) ? (uint32_t)sizeof(vaist_gemm_pc) : (uint32_t)sizeof(vaist_mv_pc);
            VkPipelineLayoutCreateInfo plci={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,NULL,0,1,&lay,1,&pc};
            if(PL(g_ctx.device,&plci,NULL,&g_ctx.pl[i])!=VK_SUCCESS) return 0;
            {
                VkComputePipelineCreateInfo pci;
                memset(&pci,0,sizeof(pci));
                pci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                pci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
                pci.stage.module=g_ctx.sh[i];
                pci.stage.pName="main";
                pci.layout=g_ctx.pl[i];
                if(CP(g_ctx.device,VK_NULL_HANDLE,1,&pci,NULL,&g_ctx.pipe[i])!=VK_SUCCESS) return 0;
            }
        }
    }
    {   /* one command buffer (reused) + one descriptor set per pipeline */
        PFN_vkAllocateCommandBuffers PB=g_ctx.vk.AllocateCommandBuffers;
        PFN_vkAllocateDescriptorSets AS=g_ctx.vk.AllocateDescriptorSets;
        VkCommandBufferAllocateInfo cba; memset(&cba,0,sizeof(cba));
        cba.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cba.pNext=NULL;
        cba.commandPool=g_ctx.cmdpool; cba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cba.commandBufferCount=1;
#if VAIST_VK_ALLOC_3ARG
        /* Defective SDKs (struct lacks pCommandBuffers, PFN is 3-arg): pass it explicitly. */
        if(PB(g_ctx.device,&cba,&g_ctx.cb)!=VK_SUCCESS) return 0;
#else
        /* Vulkan 1.4 (modern): pCommandBuffers lives in the struct, 2-arg PFN. */
        cba.pCommandBuffers=&g_ctx.cb;
        if(PB(g_ctx.device,&cba)!=VK_SUCCESS) return 0;
#endif
        {
            VkDescriptorSetLayout lays[3]={g_ctx.layout3,g_ctx.layout4,g_ctx.layout4};
            VkDescriptorSetAllocateInfo asi;
            memset(&asi,0,sizeof(asi));
            asi.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            asi.descriptorPool=g_ctx.descpool;
            asi.descriptorSetCount=3;
            asi.pSetLayouts=lays;
            if(AS(g_ctx.device,&asi,g_ctx.set)!=VK_SUCCESS) return 0;
        }
    }
    g_ctx.ready=1;
    return 1;
}

/* Record + submit one compute dispatch. bufs[i] bound to shader binding i.
 * Returns 1 on success, 0 if the GPU path could not be completed. */
static int vk_run(const VaistRuntime*rt,int which,const VkBuffer*bufs,uint32_t nbufs,
        const void*pc,size_t pcsize,uint32_t gx,uint32_t gy,uint32_t gz){
    (void)rt;
    vaist_blas_ctx *c=&g_ctx;
    PFN_vkResetCommandBuffer Reset=c->vk.ResetCommandBuffer;
    PFN_vkBeginCommandBuffer Begin=c->vk.BeginCommandBuffer;
    PFN_vkEndCommandBuffer End=c->vk.EndCommandBuffer;
    PFN_vkCmdBindPipeline BindP=c->vk.CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets BindDS=c->vk.CmdBindDescriptorSets;
    PFN_vkCmdPushConstants Push=c->vk.CmdPushConstants;
    PFN_vkCmdDispatch Dispatch=c->vk.CmdDispatch;
    PFN_vkCmdPipelineBarrier Barrier=c->vk.CmdPipelineBarrier;
    PFN_vkUpdateDescriptorSets Upd=c->vk.UpdateDescriptorSets;
    PFN_vkQueueSubmit Submit=c->vk.QueueSubmit;
    PFN_vkQueueWaitIdle Wait=c->vk.QueueWaitIdle;
    VkDescriptorBufferInfo dbs[4];
    VkWriteDescriptorSet writes[4];
    VkMemoryBarrier mb;
    VkCommandBuffer cb=c->cb;
    VkDescriptorSet set=c->set[which];
    VkPipelineLayout pl=c->pl[which];
    uint32_t i;
    if(!c->ready) return 0;
    if(!Reset||!Begin||!End||!BindP||!BindDS||!Push||!Dispatch||!Barrier||!Upd||!Submit||!Wait) return 0;
    if(nbufs>4) return 0;
    if(Reset(cb,0)!=VK_SUCCESS) return 0;
    {
        VkCommandBufferBeginInfo bi;
        bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.pNext=NULL; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if(Begin(cb,&bi)!=VK_SUCCESS) return 0;
    }
    for(i=0;i<nbufs;i++){
        dbs[i].buffer=bufs[i]; dbs[i].offset=0;
        dbs[i].range=VK_WHOLE_SIZE;
        memset(&writes[i],0,sizeof(writes[i]));
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet=set; writes[i].dstBinding=i;
        writes[i].descriptorCount=1;
        writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo=&dbs[i];
    }
    Upd(c->device,nbufs,writes,0,0);
    BindP(cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->pipe[which]);
    BindDS(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&set,0,0);
    if(pc && pcsize){
        Push(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,(uint32_t)pcsize,pc);
    }
    Dispatch(cb,gx,gy,gz);
    /* ponytail: one global memory barrier (src=shader-write, dst=host-read) is
     * the minimal sync for single-queue dispatch->waitIdle; upgrade to
     * per-buffer barriers only if aliasing/overlap grows. */
    mb.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER; mb.pNext=VK_NULL_HANDLE;
    mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    Barrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0,1,&mb,0,0,0,0);
    End(cb);
    {
        VkSubmitInfo si;
        si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; si.pNext=NULL;
        si.waitSemaphoreCount=0; si.pWaitSemaphores=NULL; si.pWaitDstStageMask=NULL;
        si.commandBufferCount=1; si.pCommandBuffers=&cb;
        si.signalSemaphoreCount=0; si.pSignalSemaphores=NULL;
        if(Submit(c->queue,1,&si,VK_NULL_HANDLE)!=VK_SUCCESS) return 0;
    }
    Wait(c->queue);
    return 1;
}

static VaistStatus vaist_blas_vk_gemm(const VaistRuntime*rt,
        const float*A,const float*B,float*C,size_t m,size_t k,size_t n){
    VaistStatus st; VaistBuffer *ab=NULL,*bb=NULL,*cb=NULL;
    void *ph; VkBuffer bufs[3]; size_t an=m*k, bn=k*n, cn=m*n;
    float *tmp;
    if(!rt||!A||!B||!C) return VAIST_DEVICE_ERROR;
    if(vaist_buffer_create((VaistRuntime*)rt,an*sizeof(float),&ab)!=VAIST_OK) return VAIST_DEVICE_ERROR;
    if(vaist_buffer_create((VaistRuntime*)rt,bn*sizeof(float),&bb)!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; }
    if(vaist_buffer_create((VaistRuntime*)rt,cn*sizeof(float),&cb)!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; }
    if(vaist_buffer_upload(ab,A,an*sizeof(float))!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; }
    if(vaist_buffer_upload(bb,B,bn*sizeof(float))!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; }
    if(vaist_buffer_gpu_handle(ab,&ph)!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; } bufs[0]=(VkBuffer)ph;
    if(vaist_buffer_gpu_handle(bb,&ph)!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; } bufs[1]=(VkBuffer)ph;
    if(vaist_buffer_gpu_handle(cb,&ph)!=VAIST_OK){ st=VAIST_DEVICE_ERROR; goto out; } bufs[2]=(VkBuffer)ph;
    {
        vaist_gemm_pc pc;
        pc.m=(int32_t)m; pc.n=(int32_t)n; pc.k=(int32_t)k;
        pc.a_off=0; pc.b_off=0; pc.c_off=0; pc.flags=0;
        if(!vk_run(rt,VK_GEMM,bufs,3,&pc,sizeof(pc),
                (uint32_t)((m+15u)>>4u),(uint32_t)((n+15u)>>4u),1)){
            st=VAIST_DEVICE_ERROR; goto out;
        }
    }
    tmp=(float*)malloc(cn*sizeof(float));
    if(!tmp){ st=VAIST_OUT_OF_MEMORY; goto out; }
    if(vaist_buffer_download(cb,tmp,cn*sizeof(float))!=VAIST_OK){ free(tmp); st=VAIST_DEVICE_ERROR; goto out; }
    for(size_t i=0;i<cn;i++) C[i]+=tmp[i];
    free(tmp);
    st=VAIST_OK;
out:
    if(cb) vaist_buffer_destroy(cb);
    if(bb) vaist_buffer_destroy(bb);
    if(ab) vaist_buffer_destroy(ab);
    return st;
}

static VaistStatus vaist_blas_vk_matvec(const VaistRuntime*rt,int which,
        const void*w, const float*scale, size_t k, size_t n, const float*x, float*y){
    VaistStatus st=VAIST_DEVICE_ERROR;
    VaistBuffer *wb=NULL,*sb=NULL,*xb=NULL,*yb=NULL;
    void *ph; VkBuffer bufs[4]; size_t wn, sn=n, xn=k, yn=n;
    if(!rt||!w||!scale||!x||!y||!k||!n) return VAIST_INVALID_ARGUMENT;
    if(which==VK_TER){
        wn=k*n;
        if(vaist_buffer_create((VaistRuntime*)rt,wn*sizeof(int8_t),&wb)!=VAIST_OK) return VAIST_DEVICE_ERROR;
    } else {
        size_t k8=k/8u; if(k%8u) return VAIST_INVALID_ARGUMENT; wn=k8*n;
        if(vaist_buffer_create((VaistRuntime*)rt,wn*sizeof(uint8_t),&wb)!=VAIST_OK) return VAIST_DEVICE_ERROR;
    }
    if(vaist_buffer_create((VaistRuntime*)rt,sn*sizeof(float),&sb)!=VAIST_OK) goto out;
    if(vaist_buffer_create((VaistRuntime*)rt,xn*sizeof(float),&xb)!=VAIST_OK) goto out;
    if(vaist_buffer_create((VaistRuntime*)rt,yn*sizeof(float),&yb)!=VAIST_OK) goto out;
    if(vaist_buffer_upload(wb,w,wn*(which==VK_TER?sizeof(int8_t):sizeof(uint8_t)))!=VAIST_OK) goto out;
    if(vaist_buffer_upload(sb,scale,sn*sizeof(float))!=VAIST_OK) goto out;
    if(vaist_buffer_upload(xb,x,xn*sizeof(float))!=VAIST_OK) goto out;
    if(vaist_buffer_gpu_handle(wb,&ph)!=VAIST_OK) goto out; bufs[0]=(VkBuffer)ph;
    if(vaist_buffer_gpu_handle(sb,&ph)!=VAIST_OK) goto out; bufs[1]=(VkBuffer)ph;
    if(vaist_buffer_gpu_handle(xb,&ph)!=VAIST_OK) goto out; bufs[2]=(VkBuffer)ph;
    if(vaist_buffer_gpu_handle(yb,&ph)!=VAIST_OK) goto out; bufs[3]=(VkBuffer)ph;
    {
        vaist_mv_pc pc;
        pc.n=(int32_t)n;
        pc.i0=(which==VK_TER)?(int32_t)k:(int32_t)(k/8u);
        pc.w_off=0; pc.s_off=0; pc.x_off=0; pc.y_off=0;
        if(!vk_run(rt,which,bufs,4,&pc,sizeof(pc),(uint32_t)((n+7u)>>3u),1,1)){
            goto out;  /* stays DEVICE_ERROR */
        }
    }
    st=VAIST_OK;
out:
    if(yb){ if(st==VAIST_OK) vaist_buffer_download(yb,y,yn*sizeof(float)); vaist_buffer_destroy(yb); }
    if(xb) vaist_buffer_destroy(xb);
    if(sb) vaist_buffer_destroy(sb);
    if(wb) vaist_buffer_destroy(wb);
    return st;
}
#endif /* VAIST_HAVE_VK_HDR */

VAIST_API VaistStatus vaist_blas_gemm(const VaistRuntime*rt,
        const float*A,const float*B,float*C,size_t m,size_t k,size_t n,
        const void*w,VaistWeightFormat wfmt){
    VaistComputePath p=vaist_blas_best_path(rt,m,k,n,wfmt);
    (void)w;   /* consumed only by the Vulkan tile path (VAIST_HAVE_VK_HDR) */
    if(p==VAIST_PATH_MATMUL_FREE){
        /* Ternary/binary dense GEMM is only meaningful on the GPU tile path
         * (requires a per-row scale blob alongside w). On a CPU-only build the
         * caller should have used vaist_blas_matvec_ternary/binary instead,
         * so the dense GEMM falls back to the scalar fp32 path. */
        if(wfmt!=VAIST_W_FP16) return VAIST_UNSUPPORTED;
    }
    if((p==VAIST_PATH_VULKAN_DOT||p==VAIST_PATH_VULKAN_TILE) && rt){
#if VAIST_HAVE_VK_HDR
        VaistRuntimeInfo info;
        if(vaist_runtime_info(rt,&info)==VAIST_OK && info.backend==VAIST_BACKEND_VULKAN){
            if(vaist_blas_vk_gemm(rt,A,B,C,m,k,n)==VAIST_OK) return VAIST_OK;
        }
#endif
        (void)p;
    }
    if(p==VAIST_PATH_SIMD){
        /* TODO: AVX2 (Zen3) / AMX (Intel P-core) dispatch via vaist_compute SIMD. */
        DBG_TRACE("trap: SIMD path selected but unimplemented -> scalar-fallback m=%lu k=%lu n=%lu",(unsigned long)m,(unsigned long)k,(unsigned long)n);
    }
    return gemm_scalar(A,B,C,m,k,n);
}

VAIST_API VaistStatus vaist_blas_matvec_ternary(const VaistRuntime*rt,
        const int8_t*wsign,const float*scale,size_t k,size_t n,
        const float*x,float*y){
    VaistComputePath p=vaist_blas_best_path(rt,k,k,n,VAIST_W_TERNARY_I8);
    if(p==VAIST_PATH_MATMUL_FREE && rt){
#if VAIST_HAVE_VK_HDR
        VaistRuntimeInfo info;
        if(vaist_runtime_info(rt,&info)==VAIST_OK && info.backend==VAIST_BACKEND_VULKAN){
            if(vaist_blas_vk_matvec(rt,VK_TER,wsign,scale,k,n,x,y)==VAIST_OK) return VAIST_OK;
        }
#endif
        (void)p;
    }
    return vaist_blas_matvec_ternary_cpu(wsign,scale,k,n,x,y);
}

VAIST_API VaistStatus vaist_blas_matvec_binary(const VaistRuntime*rt,
        const uint8_t*wbit,const float*scale,size_t k,size_t n,
        const float*x,float*y){
    VaistComputePath p=vaist_blas_best_path(rt,k,k,n,VAIST_W_BINARY_I1);
    if(p==VAIST_PATH_MATMUL_FREE && rt){
#if VAIST_HAVE_VK_HDR
        VaistRuntimeInfo info;
        if(vaist_runtime_info(rt,&info)==VAIST_OK && info.backend==VAIST_BACKEND_VULKAN){
            if(vaist_blas_vk_matvec(rt,VK_BIN,wbit,scale,k,n,x,y)==VAIST_OK) return VAIST_OK;
        }
#endif
        (void)p;
    }
     return vaist_blas_matvec_binary_cpu(wbit,scale,k,n,x,y);
}

/* ---- Quantized matmul (dequantize → dense).
 * 'w' = GGUF weight blob, row-major (k rows × n cols), block-padded per qtype.
 * CPU fallback: dequantize full W to fp32, then gemm_scalar(A, W_dec, C).
 * TODO: Vulkan fused dequant+gemm tile (avoids materializing full W_dec). */
VAIST_API VaistStatus vaist_blas_mul_mat_q(const VaistRuntime*rt,
        const float*A,float*C,size_t m,size_t k,size_t n,
        const void*w,VaistQuantType qtype){
    if(!A||!C||!w||!k||!n) return VAIST_INVALID_ARGUMENT;
    (void)rt;
    size_t bs=vaist_quant_block_bytes(qtype);
    if(bs==0) return VAIST_UNSUPPORTED;
    size_t elem_blk=vaist_quant_block_size(qtype);
    if(elem_blk==0) return VAIST_UNSUPPORTED;
    /* dequant W (k×n row-major) -> temp */
    float*Wd=NULL;
    VaistStatus st=VAIST_OK;
    size_t need=k*n*sizeof(float);
    if(need/k/n!=sizeof(float)) return VAIST_INVALID_ARGUMENT;
    Wd=(float*)malloc(need);
    if(!Wd) return VAIST_DEVICE_ERROR;
    size_t row_blks=(n+elem_blk-1)/elem_blk;
    size_t row_bytes=row_blks*bs;
    size_t off=0;
    for(size_t r=0;r<k;r++){
        st=vaist_dequantize_f32(qtype, (const unsigned char*)w+off, row_bytes, Wd+r*n, n);
        if(st!=VAIST_OK) goto done;
        off += row_bytes;
    }
    st=gemm_scalar(A,Wd,C,m,k,n);
done:
    free(Wd);
    return st;
}

/* ---- MoE top-1 routing (CPU fallback).  GPU path dispatched to moe_route.comp.
 * x: (m x k) row-major, gate: (num_experts x k_gate) row-major.
 * indices[r] = argmax_e dot(x[r], gate[e]); scores[r] = max dot. */
VAIST_API VaistStatus vaist_blas_moe_route(const VaistRuntime*rt,
        const float*x,size_t m,size_t k,size_t k_gate,
        const float*gate,size_t num_experts,
        uint32_t*indices,float*scores){
    if(!x||!gate||!indices||!scores||!k||!k_gate||!num_experts||!m) return VAIST_INVALID_ARGUMENT;
    size_t kk=(k<k_gate)?k:k_gate;
    for(size_t r=0;r<m;r++){
        float best=-1e30f; uint32_t be=0;
        for(size_t e=0;e<num_experts;e++){
            float dot=0;
            const float*xr=x+r*k; const float*ge=gate+e*k_gate;
            for(size_t p=0;p<kk;p++) dot+=xr[p]*ge[p];
            if(dot>best){ best=dot; be=(uint32_t)e; }
        }
        indices[r]=be; scores[r]=best;
    }
    return VAIST_OK;
}
