/**
 * \file vkmodel_internal.h
 * \brief VKModel internal layout: host metadata/tensor storage and parse state.
 *
 * This header is private to src/vkmodel/. The public VkModel handle is
 * defined as a pointer to the VkModel struct below.
 */
#ifndef VKMODEL_INTERNAL_H
#define VKMODEL_INTERNAL_H

#include <stdio.h>
#include <stdint.h>

#include "../../include/vkmodel/vkmodel.h"

/* ===========================================================================
 * Constants
 * ========================================================================== */

#define VKMODEL_MAGIC_LE     0x46554747u   /**< "GGUF" little-endian.         */
#define VKMODEL_GGUF_VERSION 3u            /**< Current GGUF format version.  */
#define VKMODEL_DEFAULT_ALIGN 32u          /**< Fallback tensor alignment.    */
#define VKMODEL_STREAM_CHUNK (64u << 20)   /**< 64 MiB read/upload slices.    */
#define VKMODEL_MAX_DIMS     4u            /**< GGML_MAX_DIMS.                */
#define VKMODEL_MAX_STRING   (64u << 20)   /**< Sanity cap on string/array.   */

/* Sentinel ggml_type for safetensors dtypes that have no 1:1 ggml mapping
 * (unsigned ints, BOOL, F8). Stored in VkModelTensor::dtype and reported by
 * vkmodel_get_tensor_dtype(); vkmodel_get_tensor_dtype_name() still returns
 * the safetensors name via VkModelTensor::dtype_name. */
#define VKMODEL_DTYPE_UNKNOWN 0xFFFFFFFFu

/* GGUF metadata value types */
#define VKMODEL_VAL_UINT8   0u
#define VKMODEL_VAL_INT8    1u
#define VKMODEL_VAL_UINT16  2u
#define VKMODEL_VAL_INT16   3u
#define VKMODEL_VAL_UINT32  4u
#define VKMODEL_VAL_INT32   5u
#define VKMODEL_VAL_FLOAT32 6u
#define VKMODEL_VAL_BOOL    7u
#define VKMODEL_VAL_STRING  8u
#define VKMODEL_VAL_ARRAY   9u
#define VKMODEL_VAL_UINT64  10u
#define VKMODEL_VAL_INT64   11u
#define VKMODEL_VAL_FLOAT64 12u

/* ===========================================================================
 * Metadata value storage
 * ========================================================================== */

/**
 * \brief One typed GGUF metadata value (copied into host storage).
 *
 * Scalars live in u64 (all int widths + bool) or f64 (f32 promoted to f64 /
 * f64 as-is); strings are NUL-terminated heap copies; arrays are heap arrays
 * of recursively-parsed VkModelValue (nesting supported).
 */
typedef struct VkModelValue {
    uint32_t type;             /**< VKMODEL_VAL_* value type.               */
    union {
        uint64_t u64;          /**< uint8..64 / int8..64 / bool.            */
        double   f64;          /**< float32 (promoted) / float64.           */
        char    *str;          /**< STRING: NUL-terminated copy.            */
        struct {
            uint32_t elem_type;    /**< Element value type.                 */
            uint64_t count;        /**< Element count (not bytes).          */
            struct VkModelValue *elems; /**< count entries.                 */
        } array;               /**< ARRAY (nestable).                       */
    } v;
} VkModelValue;

/**
 * \brief One metadata key-value pair.
 */
typedef struct {
    char         *key;         /**< NUL-terminated key copy.                */
    VkModelValue  value;       /**< Typed value.                            */
} VkModelKV;

/* ===========================================================================
 * Tensor storage
 * ========================================================================== */

/**
 * \brief One parsed tensor + its uploaded device buffer.
 */
typedef struct {
    char        *name;         /**< NUL-terminated name copy.               */
    uint32_t     dtype;        /**< ggml_type enum value (or DTYPE_UNKNOWN).*/
    const char  *dtype_name;   /**< Safetensors dtype name (static) / NULL.*/
    uint32_t     n_dims;       /**< Number of dimensions.                   */
    uint64_t     dims[VKMODEL_MAX_DIMS]; /**< Dimension extents.            */
    uint64_t     nelems;       /**< Product of dims.                        */
    uint64_t     offset;       /**< File-relative data offset (verbatim).   */
    VkDeviceSize size;         /**< Byte size uploaded (exact).             */
    VkBuffer     buffer;       /**< Device buffer (model-owned).            */
    VkDeviceMemory memory;     /**< Backing block memory for vkr_free.      */
} VkModelTensor;

/* ===========================================================================
 * Model context
 * ========================================================================== */

struct VkModel {
    VkRuntime     *rt;         /**< Runtime owning device/queue/allocator.  */
    VkCommandPool  upload_pool;/**< One pool for the upload command buffer. */
    VkCommandBuffer upload_cmd;/**< Reused across every tensor upload.      */

    uint64_t       alignment;  /**< general.alignment (default 32).         */
    uint64_t       data_offset;/**< File offset of tensor_data[0].          */

    uint32_t       kv_count;   /**< Metadata entry count.                   */
    VkModelKV     *kv;         /**< kv_count entries.                       */

    uint32_t       tensor_count; /**< Tensor count.                         */
    VkModelTensor *tensors;    /**< tensor_count entries.                   */
};

/* ===========================================================================
 * Internal helpers (shared by vkmodel.c)
 * ========================================================================== */

/**
 * \brief Release every resource owned by a parsed value.
 *
 * \param value Value to free (strings and arrays released).
 */
void vkmodel_value_free(VkModelValue* value);

#endif /* VKMODEL_INTERNAL_H */
