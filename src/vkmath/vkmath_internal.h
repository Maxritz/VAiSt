/**
 * \file vkmath_internal.h
 * \brief Internal VKMath structures: push constants, pipeline cache, context.
 *
 * Mirrors the design of vkblas_internal.h: same pipeline-caching strategy,
 * same tier detection, same embedded-SPIR-V pattern.
 */
#ifndef VKMATH_INTERNAL_H
#define VKMATH_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"
#include "vkmath.h"

/* ── Push constant block (std140, must match GLSL layout exactly) ──────── */

typedef struct {
    uint32_t num_elements; /* offset  0: element count for 1D elementwise */
    uint32_t num_rows;     /* offset  4: row count for reductions          */
    uint32_t num_cols;     /* offset  8: col count for reductions          */
    float    alpha;        /* offset 12: scalar coefficient (scale, add)   */
    float    beta;         /* offset 16: scalar coefficient                  */
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
    uint64_t stride_a;     /* offset 32: batch stride for input A           */
    uint64_t stride_b;     /* offset 40: batch stride for input B           */
    uint64_t stride_out;   /* offset 48: batch stride for output            */
    uint64_t batch_count;  /* offset 56: number of batches                   */
    uint64_t _pad3;        /* offset 64: padding                             */
} vkmath_push_constants_t;

/* Static assert (C99-compatible): struct must be exactly 72 bytes */
typedef char vkmath_pc_static_assert[sizeof(vkmath_push_constants_t) == 72 ? 1 : -1];

/* ── Data types ────────────────────────────────────────────────────────── */

#define VKMATH_DTYPE_F32 0
#define VKMATH_DTYPE_F16 1
#define VKMATH_DTYPE_BF16 2

/* ── Kernel types ──────────────────────────────────────────────────────── */

#define VKMATH_KERNEL_RELU          0
#define VKMATH_KERNEL_SILU          1
#define VKMATH_KERNEL_GELU          2
#define VKMATH_KERNEL_TANH          3
#define VKMATH_KERNEL_SIGMOID       4
#define VKMATH_KERNEL_ADD           5
#define VKMATH_KERNEL_MUL           6
#define VKMATH_KERNEL_ADD_MUL       7
#define VKMATH_KERNEL_SCALE         8
#define VKMATH_KERNEL_MAX_REDUCE    9
#define VKMATH_KERNEL_SUM_REDUCE    10
#define VKMATH_KERNEL_SOFTMAX       11
#define VKMATH_KERNEL_RMS_NORM      12
#define VKMATH_KERNEL_LAYERNORM     13
#define VKMATH_KERNEL_ARGMAX        14
#define VKMATH_KERNEL_ARGMIN        15
#define VKMATH_KERNEL_CUMSUM        16
#define VKMATH_KERNEL_CLIP          17
#define VKMATH_KERNEL_ABS           18
#define VKMATH_KERNEL_SIGN          19
#define VKMATH_KERNEL_EXP           20
#define VKMATH_KERNEL_LOG           21
#define VKMATH_KERNEL_SQRT          22
#define VKMATH_KERNEL_RSQRT         23
#define VKMATH_KERNEL_POW           24
#define VKMATH_KERNEL_CAST_F32_TO_BF16  25
#define VKMATH_KERNEL_CAST_BF16_TO_F32  26
#define VKMATH_KERNEL_RELU_BF16         27
#define VKMATH_KERNEL_SILU_BF16         28
#define VKMATH_KERNEL_GELU_BF16         29
#define VKMATH_KERNEL_SIGMOID_BF16      30
#define VKMATH_KERNEL_TANH_BF16         31
#define VKMATH_KERNEL_CONV2D_F32          32
#define VKMATH_KERNEL_POOL2D_F32          33
#define VKMATH_KERNEL_BATCHNORM_F32       34
#define VKMATH_KERNEL_TRANSPOSE_F32       35
#define VKMATH_KERNEL_COUNT               36

/* ── Capability tiers ──────────────────────────────────────────────────── */

typedef enum {
    VKMATH_TIER_BASELINE   = 0,
    VKMATH_TIER_SUBGROUP   = 1,
    VKMATH_TIER_COOPMATRIX = 2,
} VkMathTier_t;

/* ── Pipeline cache ────────────────────────────────────────────────────── */

#define VKMATH_MAX_PIPELINES 256

typedef struct {
    uint64_t key;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    uint32_t kernel;
    uint32_t data_type;
    uint32_t tier;
    uint8_t  valid;
} vkmath_pipeline_entry_t;

/* ── Context ───────────────────────────────────────────────────────────── */

struct VkMathContext {
    VkDevice device;
    VkPipelineCache pipeline_cache;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;

     /* capability flags */
    VkBool32 has_subgroup;
    VkBool32 has_coop_matrix;
    uint32_t max_subgroup_size;
    uint32_t max_compute_workgroup_size[3];
    VkBool32 has_shader_int64;
    VkBool32 has_push_descriptor;    /* active tier (highest supported) */
    VkMathTier_t active_tier;

    /* push descriptor function (loaded via vkGetDeviceProcAddr) */
    PFN_vkCmdPushDescriptorSetKHR push_desc_fn;


    /* pipeline cache storage (open-addressing, linear probing) */
    vkmath_pipeline_entry_t pipelines[VKMATH_MAX_PIPELINES];
    uint32_t pipeline_count;
};

/* ── Internal functions ────────────────────────────────────────────────── */

VkResult vkmath_load_shader_module(VkDevice device,
                                   const uint32_t* spirv,
                                   size_t spirv_words,
                                   VkShaderModule* out_module);

VkResult vkmath_ensure_pipeline(VkMathContext* ctx,
                                uint32_t kernel,
                                uint32_t data_type,
                                VkPipeline* out_pipeline);

VkPipelineLayout vkmath_get_pipeline_layout(VkMathContext* ctx);

void vkmath_push_pc(VkCommandBuffer cmd, VkPipelineLayout layout,
                    const vkmath_push_constants_t* pc);

VkResult vkmath_alloc_descriptor_set(VkMathContext* ctx,
                                     VkDescriptorSet* out_ds);

uint64_t vkmath_hash_key(uint32_t kernel, uint32_t data_type, uint32_t tier);

const uint32_t* vkmath_select_spirv(uint32_t kernel, uint32_t data_type,
                                    uint32_t tier, size_t* out_size);

#endif /* VKMATH_INTERNAL_H */
