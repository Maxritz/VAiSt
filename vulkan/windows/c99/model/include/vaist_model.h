/**
 * \file vaist_model.h
 * \brief Model weight loading: SafeTensors and GGUF format parsers.
 */
#ifndef VAIST_MODEL_H
#define VAIST_MODEL_H
#include "vaist_core.h"
#include "vaist_tensor.h"
#include "vaist_quant.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum VaistModelFormat {
    VAIST_MODEL_UNKNOWN = 0,
    VAIST_MODEL_GGUF    = 1,
    VAIST_MODEL_SAFETENSORS = 2,
    VAIST_MODEL_ONNX  = 3,
    VAIST_MODEL_OPENVINO = 4
} VaistModelFormat;

typedef struct VaistModelInfo {
    VaistModelFormat format;
    uint64_t file_size;
    uint32_t version;
    uint32_t tensor_count;
} VaistModelInfo;

/* ---- Tensor descriptor ---- */

#pragma pack(push, 1)
typedef struct {
    char     name[256];       /**< Tensor name (e.g. "model.layers.0.attention.wq.weight") */
    VaistDType dtype;         /**< Data type: VAIST_F32, VAIST_F16=4(VAIST_U8), etc. */
    uint32_t rank;            /**< Number of dimensions */
    uint64_t shape[8];        /**< Dimension sizes (row-major: shape[0] is outermost) */
    uint64_t offset;          /**< Byte offset in the weight file */
    uint64_t byte_size;       /**< Size in bytes of the stored data */
} VaistTensorDesc;
#pragma pack(pop)

/* ---- SafeTensors metadata ---- */

#pragma pack(push, 1)
typedef struct {
    char     name[128];       /**< Tensor name */
    char     data_type[32];   /**< SafeTensors dtype string (e.g. "F32", "F16") */
    uint64_t data_size;       /**< Size in bytes of the tensor data */
    uint64_t offset;          /**< Byte offset in the file */
} VaistSafetensorInfo;
#pragma pack(pop)

/* ---- Existing API (unchanged) ---- */
VAIST_API VaistStatus vaist_model_inspect(const char *path, VaistModelInfo *out);
VAIST_API const char *vaist_model_format_string(VaistModelFormat f);

/* ---- SafeTensors loading ---- */

/**
 * \brief Load tensor metadata from a SafeTensors file or index file.
 *
 * Parses the 8-byte little-endian header-length prefix, reads the JSON header,
 * and extracts tensor name, dtype, shape, and data_offsets for each tensor.
 * For sharded checkpoints, pass the .index.json path and offsets will point
 * into the appropriate .safetensors shard (caller must resolve shard path).
 *
 * \param path     Path to model.safetensors or model.safetensors.index.json.
 * \param tensors  Receives a dynamically allocated array of VaistTensorDesc.
 *                 Caller frees with free(). NULL if count == 0.
 * \param count    Receives the number of tensors parsed.
 * \param capacity Receives the allocated capacity (>= count; caller may realloc).
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT if path is NULL or output params are NULL.
 * \retval VAIST_IO_ERROR if the file cannot be opened.
 * \retval VAIST_MODEL_ERROR if the format is invalid or unsupported.
 * \retval VAIST_OUT_OF_MEMORY if allocation fails.
 */
VAIST_API VaistStatus vaist_model_load_safetensors(
    const char *path,
    VaistTensorDesc **tensors,
    size_t *count,
    size_t *capacity);

/* ---- GGUF loading ---- */

/**
 * \brief Load tensor metadata from a GGUF file.
 *
 * Parses the GGUF header (magic, version, tensor_count, metadata_kv_count,
 * metadata key-value pairs), then iterates the tensor info section to extract
 * name, dimensions, and ggml_type for each tensor.
 *
 * \param path    Path to the .gguf file.
 * \param tensors Receives a dynamically allocated array of VaistTensorDesc.
 *                Caller frees with free(). NULL if count == 0.
 * \param count   Receives the number of tensors parsed.
 * \param capacity Receives the allocated capacity (>= count).
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT if path is NULL or output params are NULL.
 * \retval VAIST_IO_ERROR if the file cannot be opened.
 * \retval VAIST_MODEL_ERROR if the format is invalid.
 * \retval VAIST_OUT_OF_MEMORY if allocation fails.
 */
VAIST_API VaistStatus vaist_model_load_gguf(
    const char *path,
    VaistTensorDesc **tensors,
    size_t *count,
    size_t *capacity);

/* ---- Tensor reading ---- */

/**
 * \brief Read a tensor's data from a file, dequantizing if needed.
 *
 * Reads raw bytes at the tensor's offset in the file. If the dtype is a
 * quantized type (VAIST_Q8_0, VAIST_Q4_0, etc.), uses vaist_dequantize_f32
 * to produce float32 output. If the dtype is VAIST_F32, a direct memcpy is
 * performed.
 *
 * \param path     Path to the weight file.
 * \param desc     Tensor descriptor (from load_safetensors or load_gguf).
 * \param dst      Destination buffer for float32 data.
 * \param dst_cap  Capacity of dst in float elements.
 * \retval VAIST_OK on success.
 * \retval VAIST_INVALID_ARGUMENT for NULL params or insufficient capacity.
 * \retval VAIST_IO_ERROR if the file cannot be read at the offset.
 * \retval VAIST_UNSUPPORTED for unknown dtypes.
 */
VAIST_API VaistStatus vaist_model_tensor_read(
    const char *path,
    const VaistTensorDesc *desc,
    float *dst,
    size_t dst_cap);

/**
 * \brief List all tensors (for debugging).
 * \param tensors Array of tensor descriptors.
 * \param count   Number of descriptors.
 */
VAIST_API VaistStatus vaist_model_tensor_list(
    VaistTensorDesc **tensors,
    size_t count);

#ifdef __cplusplus
}
#endif
#endif
