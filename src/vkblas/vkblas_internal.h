/**
 * \file vkblas_internal.h
 * \brief Internal VKBLAS structures: push constants, pipeline cache, context.
 */
#ifndef VKBLAS_INTERNAL_H
#define VKBLAS_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"
#include "vkblas.h"  /* public types: VkBLASPointerMode_t, etc. */

/* ── Data type constants ────────────────────────────────────────────────── */
#define VKBLAS_DTYPE_F32 0
#define VKBLAS_DTYPE_F16 1
#define VKBLAS_DTYPE_BF16 2
#define VKBLAS_DTYPE_F64 4

/* ── Fused quantized-GEMM kernel codes ──────────────────────────────────── *
 * Distinct from every plain-GEMM data type so pipeline-cache hash keys
 * never collide with the f32/f16/bf16/i8/f64 GEMMs. The output of both
 * kernels is f32; the code only selects the dequant-in-matmul shader. */
#define VKBLAS_DTYPE_QGEMM_Q8_0 5
#define VKBLAS_DTYPE_QGEMM_Q4K  6
#define VKBLAS_DTYPE_QGEMM_Q4_0 7
#define VKBLAS_DTYPE_QGEMM_Q5K  8
#define VKBLAS_DTYPE_QGEMM_Q6K  9
#define VKBLAS_DTYPE_QGEMM_Q3K  10
#define VKBLAS_DTYPE_QGEMM_IQ4XS 11

/* ── BLAS L2/L3 extended-op dtype codes ─────────────────────────────────── *
 * Each extended op gets its own (f32, f16) dtype pair so the pipeline cache
 * never collides with GEMM, QGEMM, or qgemm-f16 codes.  Baseline tier only
 * for now: syrk/herk/symm/hemm are plain dot-product formulations, and the
 * triangular solves run one warp per RHS column; subgroup/coopmatrix tiers
 * are follow-up work.
 */
#define VKBLAS_DTYPE_TRSV_F32 39
#define VKBLAS_DTYPE_TRSV_F16 40
#define VKBLAS_DTYPE_TRSM_F32 41
#define VKBLAS_DTYPE_TRSM_F16 42
#define VKBLAS_DTYPE_SYMV_F32 43
#define VKBLAS_DTYPE_SYMV_F16 44
#define VKBLAS_DTYPE_HEMV_F32 45
#define VKBLAS_DTYPE_HEMV_F16 46
#define VKBLAS_DTYPE_SYMM_F32 47
#define VKBLAS_DTYPE_SYMM_F16 48
#define VKBLAS_DTYPE_HEMM_F32 49
#define VKBLAS_DTYPE_HEMM_F16 50
#define VKBLAS_DTYPE_SYRK_F32 51
#define VKBLAS_DTYPE_SYRK_F16 52
#define VKBLAS_DTYPE_HERK_F32 53
#define VKBLAS_DTYPE_HERK_F16 54

/* ── Push constant block (must match GLSL push_constant layout, std140) ── */

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    float    alpha;
    float    beta;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t ldd;
    int32_t  transA;
    int32_t  transB;
    int32_t  beta_is_zero;
    int32_t  _pad0;
    uint32_t strideA;
    uint32_t strideB;
    uint32_t strideC;
    uint32_t strideD;
    uint32_t batchCount;
    uint32_t _pad1;
    uint32_t _pad2;
    uint32_t _pad3;
} vkblas_push_constants_t;

/* Static assert: size must be <= 128 (Vulkan min maxPushConstantsSize) */
typedef char vkblas_pc_static_assert[sizeof(vkblas_push_constants_t) <= 128 ? 1 : -1];

/* ── Capability tiers ────────────────────────────────────────────── */

typedef enum {
    VKBLAS_TIER_BASELINE   = 0,
    VKBLAS_TIER_SUBGROUP   = 1,
    VKBLAS_TIER_COOPMATRIX = 2,
} VkBLASTier_t;

#define VKBLAS_MAX_PIPELINES 256

/* ── Pipeline cache entry ────────────────────────────────────────── */

typedef struct {
    uint64_t key;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    uint32_t   data_type;
    uint32_t   transA     : 1;
    uint32_t   transB     : 1;
    uint32_t   is_strided : 1;
    uint32_t   tier       : 4;
    uint32_t   _reserved  : 25;
    uint8_t    valid;
} vkblas_pipeline_entry_t;

/* ── Context ─────────────────────────────────────────────────────── */

struct VkBLASContext {
    VkInstance instance;
    VkDevice device;
    VkPipelineCache pipeline_cache;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;

    /* capability flags */
    VkBool32 has_subgroup;
    VkBool32 has_coop_matrix;
    /* Driver-guarded cooperative-matrix path. Default OFF: the AMD 26.7.1
       Windows driver hard-crashes (0xE06D7363) inside vkCreateComputePipelines
       for any module containing coopMatMulAddKHR. Only enabled when the
       VAIT_COOPMATRIX env var is set AND the device advertises the feature. */
    VkBool32 use_coopmat;
    uint32_t max_subgroup_size;
    uint32_t max_compute_workgroup_size[3];

    /* active tier (highest supported) */
    VkBLASTier_t active_tier;

    /* pointer mode for alpha/beta (host vs device) */
    VkBLASPointerMode_t pointer_mode;

    /* pipeline cache storage */
    vkblas_pipeline_entry_t pipelines[VKBLAS_MAX_PIPELINES];
    uint32_t pipeline_count;
};

/* ── Internal functions ──────────────────────────────────────────── */

VkResult vkblas_load_shader_module(VkDevice device,
                                   const uint32_t* spirv, size_t spirv_words,
                                   VkShaderModule* out_module);

VkResult vkblas_ensure_pipeline(VkBLASContext* ctx,
                                uint32_t data_type,
                                uint32_t transA, uint32_t transB,
                                uint32_t is_strided,
                                VkPipeline* out_pipeline);

VkPipelineLayout vkblas_get_pipeline_layout(VkBLASContext* ctx);

VkResult vkblas_alloc_descriptor_set(VkBLASContext* ctx, VkDescriptorSet* out);

void vkblas_write_descriptor_set(VkBLASContext* ctx, VkDescriptorSet ds,
                                 VkBuffer A, VkBuffer B, VkBuffer C, VkBuffer D);

void vkblas_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                    const vkblas_push_constants_t* pc);

uint64_t vkblas_hash_key(uint32_t dt, uint32_t tA, uint32_t tB,
                         uint32_t strided, uint32_t tier);

#endif /* VKBLAS_INTERNAL_H */
