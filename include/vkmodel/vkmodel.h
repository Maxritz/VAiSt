/**
 * \file vkmodel.h
 * \brief Vulkan-native GGUF model loader (parse + tensor upload).
 *
 * VKModel is the model-loading entry point of the Vulkan AI stack. It parses
 * a GGUF model file (metadata key-value pairs + tensor infos) on the host and
 * uploads every tensor's raw bytes into a device buffer through vkruntime
 * (vkr_malloc + vkr_upload). The result is a ready-to-use model object: host
 * metadata lookups plus a per-tensor VkBuffer holding the exact quantized
 * bytes straight from the file.
 *
 * The loader does NOT dequantize. Each tensor's ggml_type is reported so the
 * consumer (VKQuant) can dispatch the right dequant kernel; the device buffer
 * always holds the raw, file-verbatim bytes.
 *
 * Format coverage (GGUF v1/v2/v3 header, little-endian):
 *   - magic "GGUF", version, tensor_count, metadata_kv_count
 *   - metadata value types: UINT8, INT8, UINT16, INT16, UINT32, INT32,
 *     FLOAT32, BOOL, STRING, ARRAY (of any of the above, nestable), UINT64,
 *     INT64, FLOAT64
 *   - tensor infos: name, n_dims, dims[], ggml_type, 32-byte-aligned data
 *     offset (the offset field is used verbatim; the data region starts at
 *     the header end rounded up to `general.alignment`, default 32).
 */
#ifndef VKMODEL_H
#define VKMODEL_H

#include <vulkan/vulkan.h>
#include <stdint.h>

#include "../vkruntime/vkruntime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================================================================
 * Types
 * ========================================================================== */

/**
 * \brief An opaque, loaded GGUF model.
 *
 * Owns the host-side metadata + tensor-info arrays and the device buffers
 * backing every tensor. Created by vkmodel_load() and released by
 * vkmodel_destroy(). Thread-unsafe; callers must serialize access.
 */
typedef struct VkModel VkModel;

/* ===========================================================================
 * Lifecycle
 * ========================================================================== */

/**
 * \brief Parse a GGUF file and upload all tensor data to the GPU.
 *
 * Opens \p path, validates magic/version, parses the metadata key-value pairs
 * (all 13 GGUF value types, including nested arrays, are copied into host
 * storage) and the tensor infos (name, dims, ggml_type, offset), then uploads
 * each tensor's raw bytes into a dedicated device buffer. Tensor data is read
 * from the file in bounded streaming chunks (never the whole file in RAM) and
 * uploaded with vkr_upload() through one model-owned command pool/buffer that
 * is reset and reused across tensors.
 *
 * The returned model owns every device buffer and frees it in
 * vkmodel_destroy(). On failure *pModel is set to NULL and all partial state
 * is released.
 *
 * \param rt     Runtime providing device/queue/allocator. Its queue is
 *               assumed to live on queue family 0 (the stack-wide single
 *               queue convention used by every test harness).
 * \param path   Path to the .gguf file to load.
 * \param pModel Receives the loaded model on success.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Bad magic/version, malformed
 *         metadata/tensor section, or unreadable file.
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY Host bookkeeping allocation failed.
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY Device buffer / staging allocation
 *         failed (propagated from vkr_malloc / vkr_upload).
 * \retval VK_ERROR_FEATURE_NOT_PRESENT Staging buffer could not be mapped.
 */
VkResult vkmodel_load(VkRuntime* rt, const char* path, VkModel** pModel);

/**
 * \brief Destroy a loaded model and release every resource it owns.
 *
 * Frees the host metadata/tensor arrays and vkr_free()s every tensor device
 * buffer, then destroys the model-owned command pool/buffer. \p model may be
 * NULL.
 *
 * \param model Model to destroy (may be NULL).
 */
void vkmodel_destroy(VkModel* model);

/* ===========================================================================
 * Metadata access
 * ========================================================================== */

/**
 * \brief Return the number of metadata key-value pairs parsed from the file.
 *
 * \param m Valid model.
 * \return Number of metadata entries.
 */
uint32_t vkmodel_get_kv_count(VkModel* m);

/**
 * \brief Return the key string of the i-th metadata entry.
 *
 * Keys are returned in file order. The returned pointer is owned by the model
 * and stays valid until vkmodel_destroy().
 *
 * \param m Valid model.
 * \param i Entry index in [0, vkmodel_get_kv_count()).
 * \return NUL-terminated key, or NULL if \p i is out of range.
 */
const char* vkmodel_get_kv_key(VkModel* m, uint32_t i);

/**
 * \brief Look up a metadata string value by key.
 *
 * If the key is present and its value is a GGUF STRING, the stored
 * NUL-terminated copy is returned. If the key is present but holds a
 * non-string value, NULL is returned (the value type does not match). If the
 * key is absent, \p fallback is returned.
 *
 * \param m        Valid model.
 * \param key      Metadata key to look up.
 * \param fallback Returned when the key is absent (may be NULL).
 * \return The string value, NULL for a present non-string key, or
 *         \p fallback when the key is missing.
 */
const char* vkmodel_get_kv_string(VkModel* m, const char* key,
                                  const char* fallback);

/* ===========================================================================
 * Tensor access
 * ========================================================================== */

/**
 * \brief Return the number of tensors parsed from the file.
 *
 * \param m Valid model.
 * \return Number of tensors.
 */
uint32_t vkmodel_get_tensor_count(VkModel* m);

/**
 * \brief Return the name of the i-th tensor.
 *
 * The returned pointer is owned by the model and stays valid until
 * vkmodel_destroy().
 *
 * \param m Valid model.
 * \param i Tensor index in [0, vkmodel_get_tensor_count()).
 * \return NUL-terminated tensor name, or NULL if \p i is out of range.
 */
const char* vkmodel_get_tensor_name(VkModel* m, uint32_t i);

/**
 * \brief Return the ggml_type enum value of the i-th tensor.
 *
 * Values follow the GGUF spec: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1,
 * 8=Q8_0, 9=Q8_1, 10=Q2_K ... 14=Q6_K, 15=Q8_K, 16..23 IQ2..IQ4, 24=I8,
 * 25=I16, 26=I32, 27=I64, 28=F64, 29=IQ1_M, 30=BF16, 34=TQ1_0, 35=TQ2_0,
 * 39=MXFP4. See vkmodel_block_elems() for the corresponding block size.
 *
 * \param m Valid model.
 * \param i Tensor index.
 * \return ggml_type enum value, or 0 if \p i is out of range.
 */
uint32_t vkmodel_get_tensor_dtype(VkModel* m, uint32_t i);

/**
 * \brief Return the total element count of the i-th tensor (product of dims).
 *
 * \param m Valid model.
 * \param i Tensor index.
 * \return Element count, or 0 if \p i is out of range.
 */
uint32_t vkmodel_get_tensor_nelems(VkModel* m, uint32_t i);

/**
 * \brief Return the device buffer holding the i-th tensor's raw bytes.
 *
 * The buffer is created via vkr_malloc() with STORAGE | TRANSFER_SRC/DST
 * usage, so it can be fed straight into VKQuant dequant dispatch or copied
 * further. The model owns the buffer; it is destroyed by vkmodel_destroy().
 *
 * \param m Valid model.
 * \param i Tensor index.
 * \return VkBuffer handle, or VK_NULL_HANDLE if \p i is out of range.
 */
VkBuffer vkmodel_get_tensor_buffer(VkModel* m, uint32_t i);

/**
 * \brief Return the byte size of the i-th tensor (exact bytes uploaded).
 *
 * Computed from the tensor's ggml_type and element count; matches the data
 * written by the GGUF writer for the corresponding block layout.
 *
 * \param m Valid model.
 * \param i Tensor index.
 * \return Byte size, or 0 if \p i is out of range.
 */
VkDeviceSize vkmodel_get_tensor_size(VkModel* m, uint32_t i);

/* ===========================================================================
 * Type mapping (ready-to-use with vkquant)
 * ========================================================================== */

/**
 * \brief Map a ggml_type enum value to its elements-per-block count.
 *
 * Returns 32 for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/IQ4_NL, 256 for the K-format
 * super-blocks (Q2_K..Q8_K, IQ2_XXS/IQ2_XS/IQ3_XXS/IQ1_S/IQ3_S/IQ2_S/IQ4_XS/
 * IQ1_M, TQ1_0/TQ2_0), the MXFP4/NVFP4/Q1_0/Q2_0 block sizes, and 1 for the
 * non-blocked types (F32/F16/BF16/I8/I16/I32/I64/F64). Deprecated or unknown
 * enum values return 0.
 *
 * \param ggml_type The ggml_type enum value.
 * \return Elements per block, or 0 if the type is not a valid block type.
 */
uint32_t vkmodel_block_elems(uint32_t ggml_type);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKMODEL_H */
