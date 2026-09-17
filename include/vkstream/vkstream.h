/**
 * \file vkstream.h
 * \brief Async model streaming with Resizable BAR support
 */
#ifndef VKSTREAM_H
#define VKSTREAM_H

#include <vulkan/vulkan.h>
#include <vkmodel/vkmodel.h>

#ifdef _WIN32
    #ifdef VKSTREAM_EXPORTS
        #define VKSTREAM_EXPORT __declspec(dllexport)
    #else
        #define VKSTREAM_EXPORT
    #endif
#else
    #define VKSTREAM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VKSTREAM_PATH_STAGING,
    VKSTREAM_PATH_BAR_ZERO_COPY,
    VKSTREAM_PATH_COMPRESSED
} vkstream_path_t;

typedef struct vkstream_request vkstream_request_t;
typedef void (*vkstream_complete_cb)(vkstream_request_t*, void* user_data);

typedef struct {
    VkDevice device;
    VkPhysicalDevice pdevice;
    VkQueue transfer_queue;
    uint32_t transfer_queue_family;
    VkQueue compute_queue;
    uint32_t compute_queue_family;

    VkDeviceSize bar_aperture_size;
    VkBool32 bar_supported;
    VkBool32 external_memory_host;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    VkDeviceSize staging_size;
    VkDeviceSize staging_offset;

    VkFence* fences;
    uint32_t fence_count;
    uint32_t fence_free_head;

    vkstream_request_t* requests;
    uint32_t request_count;
} vkstream_context_t;

typedef struct {
    const char* filepath;
    VkBuffer dest_buffer;
    VkDeviceSize dest_offset;
    VkDeviceSize size;
    vkstream_path_t preferred_path;
} vkstream_load_desc_t;

typedef struct {
    vkstream_path_t actual_path;
    VkDeviceSize bytes_transferred;
    VkResult result;
} vkstream_result_t;

VKSTREAM_EXPORT vkstream_context_t* vkstream_create(
    VkDevice device,
    VkPhysicalDevice pdevice,
    uint32_t transfer_qfamily,
    uint32_t compute_qfamily
);

VKSTREAM_EXPORT void vkstream_destroy(vkstream_context_t* ctx);

VKSTREAM_EXPORT vkstream_request_t* vkstream_load_tensor(
    vkstream_context_t* ctx,
    const vkstream_load_desc_t* desc,
    vkstream_complete_cb callback,
    void* user_data
);

VKSTREAM_EXPORT void vkstream_poll(vkstream_context_t* ctx);
VKSTREAM_EXPORT void vkstream_flush(vkstream_context_t* ctx);
VKSTREAM_EXPORT vkstream_result_t vkstream_get_result(vkstream_request_t* req);

VKSTREAM_EXPORT VkBool32 vkstream_is_bar_available(vkstream_context_t* ctx);
VKSTREAM_EXPORT VkDeviceSize vkstream_get_bar_aperture(vkstream_context_t* ctx);
VKSTREAM_EXPORT VkBool32 vkstream_is_external_memory_host_supported(vkstream_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* VKSTREAM_H */
