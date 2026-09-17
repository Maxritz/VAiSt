/**
 * \file vkblas_l1l2.c
 * \brief Vulkan-native BLAS Level 1 / Level 2 implementation (rocBLAS-style).
 *
 * Companion to VKBLAS. Reuses the opaque #VkBLASContext (design option (a)):
 * this translation unit links against libvkblas and drives the shared context's
 * descriptor pool, set layout, pipeline layout and pipeline cache via the
 * internal helpers declared in src/vkblas/vkblas_internal.h. L1/L2 pipelines
 * are inserted into the context's shared cache under hash keys carrying a
 * marker bit so they can never collide with GEMM pipeline keys.
 *
 * Ops: axpy, scal, dot, nrm2, asum, amax (Level 1) and gemv (Level 2).
 * f16 variants of axpy/scal/dot/gemv use integer bit-packing for the half
 * type (one half per uint32, low 16 bits) and therefore need no shaderFloat16
 * device feature. nrm2/asum/amax are f32-only (documented).
 *
 * Reductions (dot/nrm2/asum/amax) are two-stage: a partial-reduction kernel
 * writes one value per workgroup into a scratch region of the caller's result
 * buffer, then a single-workgroup finalize kernel reduces the partials into
 * result[0]. A compute-to-compute barrier is recorded between the stages.
 */
#include "vkblas_l1l2.h"
#include "vkblas_l1l2_internal.h"
#include "shaders_spv.h"

#include <string.h>

/* ── Embedded shader blob table ───────────────────────────────────────────── *
 * Maps (kernel, data_type) to the embedded SPIR-V array. Only combinations
 * that have a compiled baseline .comp shader are present. dot/nrm2/asum/amax
 * finalize kernels are shared as f32 regardless of the f16 input dtype.
 * ─────────────────────────────────────────────────────────────────────────── */

static const vkblas_l1l2_shader_blob_t s_shader_table[] = {
    {VKBLAS_L1L2_KERNEL_AXPY,   VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_axpy_f32,        vkblas_l1l2_spv_baseline_axpy_f32_size},
    {VKBLAS_L1L2_KERNEL_AXPY,   VKBLAS_DTYPE_F16, vkblas_l1l2_spv_baseline_axpy_f16,        vkblas_l1l2_spv_baseline_axpy_f16_size},
    {VKBLAS_L1L2_KERNEL_SCAL,   VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_scal_f32,        vkblas_l1l2_spv_baseline_scal_f32_size},
    {VKBLAS_L1L2_KERNEL_SCAL,   VKBLAS_DTYPE_F16, vkblas_l1l2_spv_baseline_scal_f16,        vkblas_l1l2_spv_baseline_scal_f16_size},
    {VKBLAS_L1L2_KERNEL_DOT_P,  VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_dot_partial_f32, vkblas_l1l2_spv_baseline_dot_partial_f32_size},
    {VKBLAS_L1L2_KERNEL_DOT_P,  VKBLAS_DTYPE_F16, vkblas_l1l2_spv_baseline_dot_partial_f16, vkblas_l1l2_spv_baseline_dot_partial_f16_size},
    {VKBLAS_L1L2_KERNEL_DOT_F,  VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_dot_finalize_f32, vkblas_l1l2_spv_baseline_dot_finalize_f32_size},
    {VKBLAS_L1L2_KERNEL_NRM2_P, VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_nrm2_partial_f32, vkblas_l1l2_spv_baseline_nrm2_partial_f32_size},
    {VKBLAS_L1L2_KERNEL_NRM2_F, VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_nrm2_finalize_f32, vkblas_l1l2_spv_baseline_nrm2_finalize_f32_size},
    {VKBLAS_L1L2_KERNEL_ASUM_P, VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_asum_partial_f32, vkblas_l1l2_spv_baseline_asum_partial_f32_size},
    {VKBLAS_L1L2_KERNEL_ASUM_F, VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_asum_finalize_f32, vkblas_l1l2_spv_baseline_asum_finalize_f32_size},
    {VKBLAS_L1L2_KERNEL_AMAX_P, VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_amax_partial_f32, vkblas_l1l2_spv_baseline_amax_partial_f32_size},
    {VKBLAS_L1L2_KERNEL_AMAX_F, VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_amax_finalize_f32, vkblas_l1l2_spv_baseline_amax_finalize_f32_size},
    {VKBLAS_L1L2_KERNEL_GEMV,   VKBLAS_DTYPE_F32, vkblas_l1l2_spv_baseline_gemv_f32,         vkblas_l1l2_spv_baseline_gemv_f32_size},
    {VKBLAS_L1L2_KERNEL_GEMV,   VKBLAS_DTYPE_F16, vkblas_l1l2_spv_baseline_gemv_f16,         vkblas_l1l2_spv_baseline_gemv_f16_size},
};
#define SHADER_TABLE_COUNT (sizeof(s_shader_table) / sizeof(s_shader_table[0]))

/* ── Hash key ─────────────────────────────────────────────────────────────── */

uint64_t vkblas_l1l2_hash_key(uint32_t kernel, uint32_t data_type)
{
    /* Marker bit 63 keeps L1/L2 keys disjoint from VKBLAS GEMM keys, which
     * are produced by vkblas_hash_key() and always fit below 2^32. */
    return (1ULL << 63) | ((uint64_t)data_type << 32) | (uint64_t)kernel;
}

/* ── Workgroup count ──────────────────────────────────────────────────────── */

uint32_t vkblas_l1l2_groups(uint32_t n)
{
    uint32_t g = (n + VKBLAS_L1L2_WORKGROUP_SIZE - 1) / VKBLAS_L1L2_WORKGROUP_SIZE;
    return g ? g : 1u;
}

/* ── Pipeline cache lookup ────────────────────────────────────────────────── */

static uint32_t vkblas_l1l2_hash_to_slot(uint64_t key)
{
    return (uint32_t)(key & (VKBLAS_MAX_PIPELINES - 1));
}

VkPipeline vkblas_l1l2_get_cached_pipeline(VkBLASContext* ctx,
                                           uint32_t kernel, uint32_t data_type)
{
    uint64_t key = vkblas_l1l2_hash_key(kernel, data_type);
    uint32_t slot = vkblas_l1l2_hash_to_slot(key);

    for (uint32_t i = 0; i < VKBLAS_MAX_PIPELINES; ++i) {
        uint32_t idx = (slot + i) & (VKBLAS_MAX_PIPELINES - 1);
        if (!ctx->pipelines[idx].valid)
            break;
        if (ctx->pipelines[idx].key == key)
            return ctx->pipelines[idx].pipeline;
    }
    return VK_NULL_HANDLE;
}

/* ── Pipeline creation / cache ────────────────────────────────────────────── */

VkResult vkblas_l1l2_ensure_pipeline(VkBLASContext* ctx,
                                     uint32_t kernel, uint32_t data_type,
                                     VkPipeline* out_pipeline)
{
    if (!ctx || !out_pipeline)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkPipeline cached = vkblas_l1l2_get_cached_pipeline(ctx, kernel, data_type);
    if (cached != VK_NULL_HANDLE) {
        *out_pipeline = cached;
        return VK_SUCCESS;
    }

    /* Locate the embedded SPIR-V blob for (kernel, data_type). */
    const uint32_t* spirv = NULL;
    size_t spirv_size = 0;
    for (uint32_t i = 0; i < SHADER_TABLE_COUNT; ++i) {
        if (s_shader_table[i].kernel == kernel &&
            s_shader_table[i].data_type == data_type) {
            spirv = s_shader_table[i].spirv;
            spirv_size = s_shader_table[i].spirv_size;
            break;
        }
    }
    if (spirv == NULL)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    VkShaderModule sm;
    VkResult r = vkblas_load_shader_module(ctx->device, spirv, spirv_size, &sm);
    if (r != VK_SUCCESS)
        return r;

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = NULL;  /* L1/L2 shaders use no spec consts */
    cpci.layout = ctx->pipeline_layout;

    VkPipeline pipeline;
    r = vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1, &cpci,
                                 NULL, &pipeline);
    vkDestroyShaderModule(ctx->device, sm, NULL);
    if (r != VK_SUCCESS)
        return r;

    /* Insert into the context's shared pipeline cache (open addressing,
     * linear probing — never leaves a hole, so GEMM lookups stay correct). */
    uint64_t key = vkblas_l1l2_hash_key(kernel, data_type);
    uint32_t slot = vkblas_l1l2_hash_to_slot(key);
    for (uint32_t i = 0; i < VKBLAS_MAX_PIPELINES; ++i) {
        uint32_t idx = (slot + i) & (VKBLAS_MAX_PIPELINES - 1);
        if (!ctx->pipelines[idx].valid) {
            ctx->pipelines[idx].key        = key;
            ctx->pipelines[idx].pipeline   = pipeline;
            ctx->pipelines[idx].layout     = ctx->pipeline_layout;
            ctx->pipelines[idx].data_type  = data_type;
            ctx->pipelines[idx].transA     = 0;
            ctx->pipelines[idx].transB     = 0;
            ctx->pipelines[idx].is_strided = 0;
            ctx->pipelines[idx].tier       = (uint32_t)ctx->active_tier;
            ctx->pipelines[idx].valid      = 1;
            ctx->pipeline_count++;
            break;
        }
    }

    *out_pipeline = pipeline;
    return VK_SUCCESS;
}

/* ── Dispatch helper ──────────────────────────────────────────────────────── *
 * Records pipeline bind + push constants + one descriptor set (all four
 * bindings written via vkblas_write_descriptor_set, matching VKBLAS) + dispatch.
 * b2 is a dummy read binding for the three-buffer ops; b0..b3 must be valid
 * VkBuffer handles (VK_WHOLE_SIZE ranges are written for every binding).
 * ─────────────────────────────────────────────────────────────────────────── */

VkResult vkblas_l1l2_dispatch(VkBLASContext* ctx, VkCommandBuffer cmd,
                              uint32_t kernel, uint32_t data_type,
                              const vkblas_push_constants_t* pc,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              VkBuffer b0, VkBuffer b1, VkBuffer b2,
                              VkBuffer b3)
{
    VkPipeline pipeline;
    VkResult r = vkblas_l1l2_ensure_pipeline(ctx, kernel, data_type, &pipeline);
    if (r != VK_SUCCESS)
        return r;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkblas_push_pc(cmd, ctx->pipeline_layout, pc);

    VkDescriptorSet ds;
    r = vkblas_alloc_descriptor_set(ctx, &ds);
    if (r != VK_SUCCESS)
        return r;

    vkblas_write_descriptor_set(ctx, ds, b0, b1, b2, b3);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &ds, 0, NULL);

    vkCmdDispatch(cmd, gx ? gx : 1u, gy ? gy : 1u, gz ? gz : 1u);
    return VK_SUCCESS;
}

/* ── Compute-to-compute barrier (reduction stage 1 -> stage 2) ────────────── */

void vkblas_l1l2_cmd_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}

/* ===========================================================================
 * BLAS Level 1 — f32
 * ========================================================================== */

VkResult vkblas_l1_axpy(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, const float* alpha,
                        VkBuffer x, int32_t incx,
                        VkBuffer y, int32_t incy)
{
    if (!ctx || !cmd || !alpha ||
        x == VK_NULL_HANDLE || y == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0 || incy <= 0)
        return VK_SUCCESS;

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n     = (uint32_t)n;
    pc.alpha = *alpha;
    pc.lda   = (uint32_t)incx;   /* incx */
    pc.ldb   = (uint32_t)incy;   /* incy */

    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_AXPY, VKBLAS_DTYPE_F32,
                                &pc, vkblas_l1l2_groups((uint32_t)n), 1, 1,
                                x, y, x, x);
}

VkResult vkblas_l1_scal(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, const float* alpha,
                        VkBuffer x, int32_t incx)
{
    if (!ctx || !cmd || !alpha || x == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0)
        return VK_SUCCESS;

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n     = (uint32_t)n;
    pc.alpha = *alpha;
    pc.lda   = (uint32_t)incx;   /* incx */

    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_SCAL, VKBLAS_DTYPE_F32,
                                &pc, vkblas_l1l2_groups((uint32_t)n), 1, 1,
                                x, x, x, x);
}

VkResult vkblas_l1_dot(VkBLASContext* ctx, VkCommandBuffer cmd,
                       int32_t n, VkBuffer x, int32_t incx,
                       VkBuffer y, int32_t incy, VkBuffer result)
{
    if (!ctx || !cmd || x == VK_NULL_HANDLE || y == VK_NULL_HANDLE ||
        result == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0 || incy <= 0)
        return VK_SUCCESS;

    uint32_t G = vkblas_l1l2_groups((uint32_t)n);

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n   = (uint32_t)n;
    pc.lda = (uint32_t)incx;
    pc.ldb = (uint32_t)incy;

    VkResult r = vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_DOT_P,
                                      VKBLAS_DTYPE_F32, &pc, G, 1, 1,
                                      x, y, x, result);
    if (r != VK_SUCCESS)
        return r;

    vkblas_l1l2_cmd_barrier(cmd);

    memset(&pc, 0, sizeof(pc));
    pc.batchCount = G;
    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_DOT_F,
                                VKBLAS_DTYPE_F32, &pc, 1, 1, 1,
                                result, result, result, result);
}

VkResult vkblas_l1_nrm2(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, VkBuffer x, int32_t incx, VkBuffer result)
{
    if (!ctx || !cmd || x == VK_NULL_HANDLE || result == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0)
        return VK_SUCCESS;

    uint32_t G = vkblas_l1l2_groups((uint32_t)n);

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n   = (uint32_t)n;
    pc.lda = (uint32_t)incx;

    VkResult r = vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_NRM2_P,
                                      VKBLAS_DTYPE_F32, &pc, G, 1, 1,
                                      x, x, x, result);
    if (r != VK_SUCCESS)
        return r;

    vkblas_l1l2_cmd_barrier(cmd);

    memset(&pc, 0, sizeof(pc));
    pc.batchCount = G;
    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_NRM2_F,
                                VKBLAS_DTYPE_F32, &pc, 1, 1, 1,
                                result, result, result, result);
}

VkResult vkblas_l1_asum(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, VkBuffer x, int32_t incx, VkBuffer result)
{
    if (!ctx || !cmd || x == VK_NULL_HANDLE || result == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0)
        return VK_SUCCESS;

    uint32_t G = vkblas_l1l2_groups((uint32_t)n);

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n   = (uint32_t)n;
    pc.lda = (uint32_t)incx;

    VkResult r = vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_ASUM_P,
                                      VKBLAS_DTYPE_F32, &pc, G, 1, 1,
                                      x, x, x, result);
    if (r != VK_SUCCESS)
        return r;

    vkblas_l1l2_cmd_barrier(cmd);

    memset(&pc, 0, sizeof(pc));
    pc.batchCount = G;
    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_ASUM_F,
                                VKBLAS_DTYPE_F32, &pc, 1, 1, 1,
                                result, result, result, result);
}

VkResult vkblas_l1_amax(VkBLASContext* ctx, VkCommandBuffer cmd,
                        int32_t n, VkBuffer x, int32_t incx, VkBuffer result)
{
    if (!ctx || !cmd || x == VK_NULL_HANDLE || result == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0)
        return VK_SUCCESS;

    uint32_t G = vkblas_l1l2_groups((uint32_t)n);

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n   = (uint32_t)n;
    pc.lda = (uint32_t)incx;

    VkResult r = vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_AMAX_P,
                                      VKBLAS_DTYPE_F32, &pc, G, 1, 1,
                                      x, x, x, result);
    if (r != VK_SUCCESS)
        return r;

    vkblas_l1l2_cmd_barrier(cmd);

    memset(&pc, 0, sizeof(pc));
    pc.batchCount = G;
    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_AMAX_F,
                                VKBLAS_DTYPE_F32, &pc, 1, 1, 1,
                                result, result, result, result);
}

/* ===========================================================================
 * BLAS Level 1 — f16 (half stored one-per-uint32, low 16 bits)
 * ========================================================================== */

VkResult vkblas_l1_axpy_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                            int32_t n, const float* alpha,
                            VkBuffer x, int32_t incx,
                            VkBuffer y, int32_t incy)
{
    if (!ctx || !cmd || !alpha ||
        x == VK_NULL_HANDLE || y == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0 || incy <= 0)
        return VK_SUCCESS;

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n     = (uint32_t)n;
    pc.alpha = *alpha;
    pc.lda   = (uint32_t)incx;
    pc.ldb   = (uint32_t)incy;

    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_AXPY, VKBLAS_DTYPE_F16,
                                &pc, vkblas_l1l2_groups((uint32_t)n), 1, 1,
                                x, y, x, x);
}

VkResult vkblas_l1_scal_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                            int32_t n, const float* alpha,
                            VkBuffer x, int32_t incx)
{
    if (!ctx || !cmd || !alpha || x == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0)
        return VK_SUCCESS;

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n     = (uint32_t)n;
    pc.alpha = *alpha;
    pc.lda   = (uint32_t)incx;

    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_SCAL, VKBLAS_DTYPE_F16,
                                &pc, vkblas_l1l2_groups((uint32_t)n), 1, 1,
                                x, x, x, x);
}

VkResult vkblas_l1_dot_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                           int32_t n, VkBuffer x, int32_t incx,
                           VkBuffer y, int32_t incy, VkBuffer result)
{
    if (!ctx || !cmd || x == VK_NULL_HANDLE || y == VK_NULL_HANDLE ||
        result == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (n <= 0 || incx <= 0 || incy <= 0)
        return VK_SUCCESS;

    uint32_t G = vkblas_l1l2_groups((uint32_t)n);

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.n   = (uint32_t)n;
    pc.lda = (uint32_t)incx;
    pc.ldb = (uint32_t)incy;

    /* Partial kernel consumes f16 inputs; finalize reuses the f32 kernel
     * because partials and the result scalar are stored as f32. */
    VkResult r = vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_DOT_P,
                                      VKBLAS_DTYPE_F16, &pc, G, 1, 1,
                                      x, y, x, result);
    if (r != VK_SUCCESS)
        return r;

    vkblas_l1l2_cmd_barrier(cmd);

    memset(&pc, 0, sizeof(pc));
    pc.batchCount = G;
    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_DOT_F,
                                VKBLAS_DTYPE_F32, &pc, 1, 1, 1,
                                result, result, result, result);
}

/* ===========================================================================
 * BLAS Level 2 — f32 and f16
 * ========================================================================== */

static VkResult vkblas_l2_gemv_common(VkBLASContext* ctx, VkCommandBuffer cmd,
                                      VkBLASOperation_t transA,
                                      int32_t m, int32_t n,
                                      const float* alpha,
                                      VkBuffer A, int32_t lda,
                                      VkBuffer x, int32_t incx,
                                      const float* beta,
                                      VkBuffer y, int32_t incy,
                                      uint32_t data_type)
{
    if (!ctx || !cmd || !alpha || !beta ||
        A == VK_NULL_HANDLE || x == VK_NULL_HANDLE || y == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (m <= 0 || n <= 0 || lda <= 0 || incx <= 0 || incy <= 0)
        return VK_SUCCESS;

    uint32_t tA = (transA == VKBLAS_OP_N) ? 0 : 1;

    vkblas_push_constants_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.m     = (uint32_t)m;
    pc.n     = (uint32_t)n;
    pc.alpha = *alpha;
    pc.beta  = *beta;
    pc.lda   = (uint32_t)lda;
    pc.ldb   = (uint32_t)incx;
    pc.ldd   = (uint32_t)incy;
    pc.transA = (int32_t)tA;

    return vkblas_l1l2_dispatch(ctx, cmd, VKBLAS_L1L2_KERNEL_GEMV, data_type,
                                &pc, vkblas_l1l2_groups((uint32_t)m), 1, 1,
                                A, x, A, y);
}

VkResult vkblas_l2_gemv(VkBLASContext* ctx, VkCommandBuffer cmd,
                        VkBLASOperation_t transA,
                        int32_t m, int32_t n,
                        const float* alpha,
                        VkBuffer A, int32_t lda,
                        VkBuffer x, int32_t incx,
                        const float* beta,
                        VkBuffer y, int32_t incy)
{
    return vkblas_l2_gemv_common(ctx, cmd, transA, m, n, alpha, A, lda,
                                 x, incx, beta, y, incy, VKBLAS_DTYPE_F32);
}

VkResult vkblas_l2_gemv_f16(VkBLASContext* ctx, VkCommandBuffer cmd,
                            VkBLASOperation_t transA,
                            int32_t m, int32_t n,
                            const float* alpha,
                            VkBuffer A, int32_t lda,
                            VkBuffer x, int32_t incx,
                            const float* beta,
                            VkBuffer y, int32_t incy)
{
    return vkblas_l2_gemv_common(ctx, cmd, transA, m, n, alpha, A, lda,
                                 x, incx, beta, y, incy, VKBLAS_DTYPE_F16);
}
