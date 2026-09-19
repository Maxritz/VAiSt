#ifndef VAIST_RUNTIME_H
#define VAIST_RUNTIME_H
#include "vaist_core.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum VaistBackend { VAIST_BACKEND_AUTO=0, VAIST_BACKEND_CPU=1, VAIST_BACKEND_VULKAN=2 } VaistBackend;
typedef struct VaistRuntime VaistRuntime;
typedef struct VaistBuffer VaistBuffer;
typedef struct VaistStream VaistStream;

/* Existing info is unchanged (backward compatible). */
typedef struct VaistRuntimeInfo {
    VaistBackend backend;
    uint32_t vulkan_available;
    uint32_t vulkan_api_version;
    uint64_t allocated_bytes;
} VaistRuntimeInfo;

/*
 * Device capabilities queried from the Vulkan physical device.
 * This is what the MatMul-free / dot-product capability ladder keys off.
 * (Your C:\temp/SKILL.md "capability ladder" + portability notes.)
 *
 * Tiling model: workgroups are the unit of tile ownership (never a fixed
 * subgroup width). Integer dot-product is used ONLY when
 * `integer_dot_product_8bit_accelerated` reports true.
 */
typedef struct VaistDeviceCaps {
    uint32_t vulkan_api_version;      /* VK_API_VERSION_1_4 family */
    uint32_t vendor_id;               /* 0x1002 AMD, 0x10DE NV, 0x8086 Intel ... */
    uint32_t device_id;
    uint32_t subgroup_size;           /* queried min/max; do not assume 32/64 */
    uint32_t subgroup_supported_stages;
    uint32_t max_compute_workgroup_size[3];
    uint32_t max_compute_workgroups;
    uint32_t max_shared_mem_bytes;
    /* Integer dot-product (SPV_KHR_integer_dot_product) acceleration: */
    uint32_t integer_dot_product_8bit_supported;
    uint32_t integer_dot_product_8bit_accelerated; /* true => OpSDotKHR is real */
    uint32_t integer_dot_product_4bit_supported;
    uint32_t integer_dot_product_4bit_accelerated;
    /* Cooperative matrices; treated as optional (RDNA2/3 unreliable in practice). */
    uint32_t cooperative_matrix_supported;
    uint32_t cooperative_matrix_accelerated;
    /* AMD architecture classification (for workgroup tuning, coopmat disable). */
    uint32_t amd_rdna_gen;      /* 0=none, 1=RDNA1, 2=RDNA2, 3=RDNA3, 4=RDNA4 */
    uint32_t is_uma;            /* 1 if unified memory (APU), 0 if discrete */
} VaistDeviceCaps;

VAIST_API VaistStatus vaist_runtime_create(VaistBackend requested, VaistRuntime **out);
VAIST_API void vaist_runtime_destroy(VaistRuntime *rt);
VAIST_API VaistStatus vaist_runtime_info(const VaistRuntime *rt, VaistRuntimeInfo *out);
VAIST_API VaistStatus vaist_runtime_device_caps(const VaistRuntime *rt, VaistDeviceCaps *out);

VAIST_API VaistStatus vaist_buffer_create(VaistRuntime *rt,size_t size,VaistBuffer **out);
VAIST_API void vaist_buffer_destroy(VaistBuffer *b);
VAIST_API void *vaist_buffer_data(VaistBuffer *b);
VAIST_API size_t vaist_buffer_size(const VaistBuffer *b);
VAIST_API VaistStatus vaist_buffer_upload(VaistBuffer *b,const void*src,size_t n);  /* host->dev (or memcpy for CPU) */
VAIST_API VaistStatus vaist_buffer_download(VaistBuffer *b,void*dst,size_t n);
VAIST_API VaistStatus vaist_stream_create(VaistRuntime *rt,VaistStream **out);
VAIST_API void vaist_stream_destroy(VaistStream *s);
VAIST_API VaistStatus vaist_stream_synchronize(VaistStream *s);

/*
 * Additive ABI surface for the Vulkan dispatch path (vaist_blas).
 *
 * vaist_buffer_gpu_handle: returns the opaque VkBuffer backing `b` as a
 *   void* when the buffer is resident on the Vulkan backend; sets *handle to
 *   NULL and returns VAIST_UNSUPPORTED for CPU buffers / no-Vulkan builds.
 *
 * vaist_runtime_vk_state: fills the opaque Vulkan handles the BLAS layer
 *   needs to record commands against the runtime's device. Returns
 *   VAIST_UNSUPPORTED when the runtime has no real Vulkan device (e.g. weak
 *   GPU box or VAIST_ENABLE_Vulkan off). Handles are opaque (VkXxx_T*) so this
 *   header never transitively pulls in <vulkan/vulkan.h>.
 *
 * vaist_runtime_vk_physdev: returns the opaque VkPhysicalDevice selected at
 *   device-creation time (NULL when no Vulkan device). Needed for physical-
 *   device queries (memory properties, feature checks); vk_state's device
 *   handle is a VkDevice and must never be cast to VkPhysicalDevice.
 *
 * vaist_runtime_vk_proc: resolves any Vulkan entrypoint from the runtime's
 *   loader handle (works for both instance and device functions), so callers
 *   need never link libvulkan directly. Returns NULL on no-loader builds.
 */
VAIST_API VaistStatus vaist_buffer_gpu_handle(VaistBuffer*b,void**handle);
VAIST_API VaistStatus vaist_runtime_vk_state(const VaistRuntime*rt,
    void**device,void**queue,uint32_t*queue_family);
VAIST_API void* vaist_runtime_vk_physdev(const VaistRuntime*rt);
VAIST_API void* vaist_runtime_vk_proc(const VaistRuntime*rt,const char*name);
#ifdef __cplusplus
}
#endif
#endif
