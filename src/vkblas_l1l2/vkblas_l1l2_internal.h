/**
 * \file vkblas_l1l2_internal.h
 * \brief Internal vkblas_l1l2 structures: kernel codes, pipeline cache helpers.
 *
 * Design choice (a): this library REUSES the opaque #VkBLASContext from VKBLAS.
 * There is no vkblas_l1l2-owned context, descriptor pool or pipeline layout.
 * The helpers declared here drive the shared VkBLASContext's pipeline cache,
 * descriptor-set allocator and push-constant path; implementations live in
 * vkblas_l1l2.c and link against libvkblas.
 */
#ifndef VKBLAS_L1L2_INTERNAL_H
#define VKBLAS_L1L2_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkblas.h"            /* VkBLASContext, VkBLASOperation_t */
#include "vkblas_internal.h"   /* vkblas_push_constants_t, VKBLAS_MAX_PIPELINES,
                                  VKBLAS_DTYPE_F32/F16, vkblas_* helpers */

/* ── Workgroup size for every L1/L2 kernel ─────────────────────────────────── */
#define VKBLAS_L1L2_WORKGROUP_SIZE 256u

/* ── Kernel codes (hash-key identity) ──────────────────────────────────────── */
enum {
    VKBLAS_L1L2_KERNEL_AXPY     = 0,
    VKBLAS_L1L2_KERNEL_SCAL     = 1,
    VKBLAS_L1L2_KERNEL_DOT_P    = 2, /* partial reduction   */
    VKBLAS_L1L2_KERNEL_DOT_F    = 3, /* finalize            */
    VKBLAS_L1L2_KERNEL_NRM2_P   = 4,
    VKBLAS_L1L2_KERNEL_NRM2_F   = 5,
    VKBLAS_L1L2_KERNEL_ASUM_P   = 6,
    VKBLAS_L1L2_KERNEL_ASUM_F   = 7,
    VKBLAS_L1L2_KERNEL_AMAX_P   = 8,
    VKBLAS_L1L2_KERNEL_AMAX_F   = 9,
    VKBLAS_L1L2_KERNEL_GEMV     = 10,
};

/* ── Shader blob table entry ───────────────────────────────────────────────── */
typedef struct {
    uint32_t       kernel;
    uint32_t       data_type;      /* VKBLAS_DTYPE_F32 / VKBLAS_DTYPE_F16 */
    const uint32_t* spirv;         /* embedded SPIR-V array               */
    size_t         spirv_size;     /* byte size (matches _size symbol)    */
} vkblas_l1l2_shader_blob_t;

/* ── Helpers implemented in vkblas_l1l2.c ─────────────────────────────────── */

/** Look up (kernel, data_type) in the context's shared pipeline cache. */
VkPipeline vkblas_l1l2_get_cached_pipeline(VkBLASContext* ctx,
                                           uint32_t kernel, uint32_t data_type);

/** Lazily create + cache the compute pipeline for (kernel, data_type). */
VkResult vkblas_l1l2_ensure_pipeline(VkBLASContext* ctx,
                                     uint32_t kernel, uint32_t data_type,
                                     VkPipeline* out_pipeline);

/**
 * Record a full dispatch: pipeline bind, push constants, one descriptor set
 * (bindings b0..b3 via vkblas_write_descriptor_set), vkCmdDispatch.
 */
VkResult vkblas_l1l2_dispatch(VkBLASContext* ctx, VkCommandBuffer cmd,
                              uint32_t kernel, uint32_t data_type,
                              const vkblas_push_constants_t* pc,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              VkBuffer b0, VkBuffer b1, VkBuffer b2,
                              VkBuffer b3);

/** Compute-to-compute memory barrier (reduction stage 1 -> stage 2). */
void vkblas_l1l2_cmd_barrier(VkCommandBuffer cmd);

/** ceil(n / 256), clamped to >= 1. */
uint32_t vkblas_l1l2_groups(uint32_t n);

/** vkblas_l1l2 pipeline hash key: kernel + dtype, marker bit set so L1/L2
 *  keys never collide with VKBLAS GEMM keys (which are < 2^32). */
uint64_t vkblas_l1l2_hash_key(uint32_t kernel, uint32_t data_type);

#endif /* VKBLAS_L1L2_INTERNAL_H */
