/**
 * \file vkisa.h
 * \brief ISA-level control and introspection for AMD RDNA2/RDNA4
 *
 * Covers Vulkan-exposed gaps from GAP_ANALYSIS.md:
 * - Cache control (S_DCACHE_INV, temporal hints) - stub
 * - Barrier state introspection (S_GET_BARRIER_STATE) - stub
 * - VGPR lifecycle (DEALLOC_VGPRS) - HIP backend
 * - Context switch state (RTN_GET_TMA) - stub
 * - Per-lane permute (V_PERMLANE_*) - stub
 * - Extended float atomics (MIN/MAX_NUM_F32) - stub
 */
#ifndef VKISA_H
#define VKISA_H

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VKISA_CACHE_HINT_DEFAULT = 0,
    VKISA_CACHE_HINT_STREAMING = 1,
    VKISA_CACHE_HINT_PERSISTENT = 2,
} VkISACacheHint;

typedef enum {
    VKISA_BARRIER_COMPLETE = 0,
    VKISA_BARRIER_TRAP = 1,
} VkISABarrierState;

typedef enum {
    VKISA_ATOMIC_MIN_NUM_F32 = 0,
    VKISA_ATOMIC_MAX_NUM_F32 = 1,
    VKISA_ATOMIC_IMAGE_MIN_F32 = 2,
    VKISA_ATOMIC_IMAGE_MAX_F32 = 3,
    VKISA_ATOMIC_PK_ADD_F16 = 4,
    VKISA_ATOMIC_PK_ADD_BF16 = 5,
} VkISAAtomicExtOp;

typedef struct {
    VkPhysicalDevice pdevice;
    VkDevice device;
    VkBool32 hip_available;
    VkBool32 device_coherent_supported;
    VkBool32 debug_enabled;
} VkISAContext;

typedef struct {
    VkResult result;
    const char* gap_name;
    VkBool32 hip_fallback_used;
} VkISAResult;

VKSTREAM_EXPORT VkISAContext* vkisa_create(
    VkPhysicalDevice pdevice,
    VkDevice device
);

VKSTREAM_EXPORT void vkisa_destroy(VkISAContext* ctx);

VKSTREAM_EXPORT VkISAResult vkisa_cache_control(
    VkISAContext* ctx,
    VkCmdBuffer cmdBuf,
    VkBuffer buffer,
    VkISACacheHint hint
);

VKSTREAM_EXPORT VkISAResult vkisa_barrier_state(
    VkISAContext* ctx,
    VkISABarrierState* pState
);

VKSTREAM_EXPORT VkISAResult vkisa_vgpr_dealloc(
    VkISAContext* ctx,
    VkCommandBuffer cmdBuf,
    uint32_t wave_id
);

VKSTREAM_EXPORT VkISAResult vkisa_context_query(
    VkISAContext* ctx,
    uint32_t* pTrapLevel,
    VkDeviceSize* pTmaAddress
);

VKSTREAM_EXPORT VkISAResult vkisa_permlane(
    VkISAContext* ctx,
    uint32_t* pData,
    uint32_t lane_mask
);

VKSTREAM_EXPORT VkISAResult vkisa_float_atomic_ext(
    VkISAContext* ctx,
    VkISAAtomicExtOp op,
    VkBuffer buffer,
    uint32_t offset,
    float value
);

VKSTREAM_EXPORT VkBool32 vkisa_is_hip_available(VkISAContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* VKISA_H */
