/**
 * \file vaist_model_state.hpp
 * \brief C++ model state and configuration types for VAiSt inference models.
 *
 * Provides the base VaistModelConfig struct describing transformer
 * hyperparameters, the VaistTensorRef lazy tensor reference, and the
 * VaistModel struct that bundles loaded weights, KV cache, and runtime.
 * All architecture classes (LlamaModel, Qwen3Model, DeepSeekModel) build
 * on top of these types.
 */
#ifndef VAIST_CPP_MODEL_STATE_HPP
#define VAIST_CPP_MODEL_STATE_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <memory>

#include "vaist_core.h"
#include "vaist_model.h"
#include "vaist_runtime.h"
#include "vaist_llm.h"
#include "vaist_quant.h"
#include "vaist_tensor.h"
#include "vaist_blas.h"
#include "vaist_compute.h"

namespace vaist {

/** \brief RoPE scaling strategy used by the model. */
enum class VaistRopeScalingType : uint32_t {
    kNone      = 0,  /**< No scaling; uses raw rope_theta. */
    kLinear    = 1,  /**< Linear scaling: rope_theta /= factor. */
    kDynamic   = 2,  /**< Dynamic NTK scaling (per-position factor). */
    kYarn       = 3,  /**< YaRN scaling (interpolates between base and sqrt dim). */
    kLlama3     = 4,  /**< Llama-3 style factor scaling. */
};

/** \brief Activation function enum for MLP blocks. */
enum class VaistActivationType : uint32_t {
    kSilu = 0,   /**< Sigmoid-weighted linear (SwiGLU): gate * silu(gate). */
    kGelu = 1,   /**< Gaussian Error Linear Unit. */
    kRelu = 2,   /**< Rectified Linear Unit. */
};

/** \brief Quantization type for a single weight tensor. */
enum class VaistWeightDtype : uint32_t {
    kF32    = 0,  /**< 32-bit float (no quantization). */
    kF16    = 1,  /**< 16-bit float. */
    kBf16   = 2,  /**< 16-bit bfloat. */
    kQ4_0   = 3,  /**< GGUF q4_0: 4-bit symmetric, block size 32. */
    kQ8_0   = 4,  /**< GGUF q8_0: 8-bit, per-block scale, block size 32. */
    kQ4_K   = 5,  /**< GGUF q4_K: 4-bit with per-block scale+min. */
    kQ5_K   = 6,  /**< GGUF q5_K: 5-bit with per-block scale+min. */
    kQ6_K   = 7,  /**< GGUF q6_K: 6-bit with per-row scale. */
    kQ3_K   = 8,  /**< GGUF q3_K: 3-bit mixed-bit. */
    kQ2_K   = 9,  /**< GGUF q2_K: 2-bit. */
    kIq4Nl  = 10, /**< GGUF IQ4_NL: nonlinearity-aware 4-bit. */
    kIq4Xs  = 11, /**< GGUF IQ4_XS: 4-bit extra-small with scale. */
    kIq2Xs  = 12, /**< GGUF IQ2_XS: 2-bit extra-small. */
    kIq1S   = 13, /**< GGUF IQ1_S: 1-bit scale. */
    kIq1M   = 14, /**< GGUF IQ1_M: 1-bit M scale. */
};

/** \brief Convert a VaistQuantType to a VaistWeightDtype. */
inline VaistWeightDtype weight_dtype_from_quant(VaistQuantType qt) {
    switch (qt) {
        case VAIST_Q4_0:  return VaistWeightDtype::kQ4_0;
        case VAIST_Q8_0:  return VaistWeightDtype::kQ8_0;
        case VAIST_Q4_K:  return VaistWeightDtype::kQ4_K;
        case VAIST_Q5_K:  return VaistWeightDtype::kQ5_K;
        case VAIST_Q6_K:  return VaistWeightDtype::kQ6_K;
        case VAIST_Q3_K:  return VaistWeightDtype::kQ3_K;
        case VAIST_Q2_K:  return VaistWeightDtype::kQ2_K;
        case VAIST_IQ4_NL: return VaistWeightDtype::kIq4Nl;
        case VAIST_IQ4_XS: return VaistWeightDtype::kIq4Xs;
        case VAIST_IQ2_XS: return VaistWeightDtype::kIq2Xs;
        case VAIST_IQ1_S:  return VaistWeightDtype::kIq1S;
        case VAIST_IQ1_M:  return VaistWeightDtype::kIq1M;
        default:          return VaistWeightDtype::kF32;
    }
}

/**
 * \brief Model configuration: all hyperparameters needed to instantiate a
 *        transformer architecture and its KV cache.
 */
struct VaistModelConfig {
    uint32_t n_layers          = 0;       /**< Number of transformer layers. */
    uint32_t n_heads           = 0;       /**< Number of attention query heads. */
    uint32_t n_kv_heads        = 0;       /**< Number of key/value heads (for GQA). */
    uint32_t head_dim          = 0;       /**< Dimensionality per attention head. */
    uint32_t hidden_size       = 0;       /**< Model hidden dimension. */
    uint32_t intermediate_size = 0;       /**< MLP intermediate dimension. */
    uint32_t vocab_size        = 0;       /**< Vocabulary size. */
    uint32_t max_position_embeddings = 2048; /**< Original max position embeddings. */
    float    rope_theta        = 10000.f; /**< RoPE base frequency. */
    float    norm_eps          = 1e-5f;   /**< RMSNorm epsilon. */
    uint32_t rope_dim          = 0;       /**< RoPE dimension (0 = head_dim). */
    uint32_t num_experts       = 0;       /**< Number of MoE experts (0 = dense). */
    uint32_t num_experts_per_tok = 0;     /**< Top-k routing per token. */
    uint32_t n_group           = 1;       /**< Expert group count (for grouped top-k). */
    uint32_t topk_group        = 1;       /**< Top-k groups for MoE routing. */
    uint32_t shared_expert_intermediate_size = 0; /**< Shared expert FFN dim (DeepSeek). */
    uint32_t moe_intermediate_size = 0;    /**< MoE expert intermediate dim. */
    uint32_t decapoda_extend = 0;          /**< Padded vocab size (0 if none). */
    bool     qk_layernorm    = false;     /**< Apply RMSNorm on Q and K before rope. */
    bool     qk_no_scale     = false;     /**< Skip scaling QK dot product. */
    bool     residual_no_bias = false;     /**< Residual path has no bias. */
    bool     attn_bias       = false;     /**< Attention projection has bias. */
    bool     mlp_bias        = false;     /**< MLP projection has bias. */
    bool     rope_theta_is_1e4_scaled = false;
    bool     tie_word_embeddings = false; /**< Tied input/output embeddings. */
    bool     use_cache       = true;       /**< Use KV cache. */
    bool     use_sliding_window = false;   /**< Sliding window attention. */
    uint32_t sliding_window   = 0;         /**< Sliding window size. */
    bool     sliding_window_patterns = false; /**< Layer-pattern sliding window. */
    bool     mlp_smoe = false;            /**< MoeMLP shared+expert routing (Qwen3-Moe). */
    VaistRopeScalingType rope_scaling_type = VaistRopeScalingType::kNone;
    float    rope_scaling_factor = 1.f;   /**< Rope scaling factor. */
    float    low_freq_factor = 1.f;       /**< Yarn low freq factor. */
    float    high_freq_factor = 1.f;      /**< Yarn high freq factor. */
    VaistActivationType hidden_act = VaistActivationType::kSilu;
    uint32_t kv_cache_dtype = VAIST_F16;  /**< KV cache data type. */
    uint32_t block_size = 16;            /**< Paged KV block size. */
    bool     mtp = false;                /**< Multi-headed attention for V3 (MTP). */
    bool     use_mla = false;            /**< Multi-head Latent Attention (DeepSeek V2/V3). */
    uint32_t q_lora_rank = 0;            /**< LoRA rank for Q (0 = no LoRA). */
    uint32_t kv_lora_rank = 0;           /**< LoRA rank for KV (0 = no LoRA). */
    uint32_t q_head_dim = 0;             /**< Q projection head dim (MLA). */
    uint32_t kv_head_dim = 0;            /**< KV projection head dim (MLA). */
    uint32_t cache_dtypes = 0;           /**< Cache dtype bits. */
    uint32_t decoder_layers = 0;         /**< MTP decoder layer count. */
    uint32_t attn_backend = 0;           /**< Attention backend selector. */
    uint32_t mtp_num_layers = 0;         /**< Number of MTP layers. */
    uint32_t shared_expert_num = 0;      /**< Number of shared experts (DeepSeek V3). */
    bool     norm_topk_prob = false;     /**< Normalize top-k probabilities. */
    bool     seq_aux = false;            /**< Sequence-level auxiliary loss. */
    uint32_t n_routed_experts = 0;       /**< Number of routed experts. */
    bool     first_k_dense_local = false; /**< First-k dense layers are local. */
    uint32_t n_local_experts = 0;        /**< Local expert count per rank. */
    uint32_t n_shared_experts = 0;        /**< Number of shared experts. */
    uint32_t routed_experts = 0;          /**< Number of routed experts. */
    uint32_t mtp_loss_scale = 0;          /**< MTP loss scaling factor. */
};

/**
 * \brief Lazy tensor reference: holds metadata for a weight tensor and
 *        provides access to its dequantized float32 data on demand.
 *
 * When the model file is opened, only the tensor descriptor (name, dtype,
 * offset, shape) is loaded into memory. The raw quantized data is read
 * from disk lazily via get_data() and cached.
 */
struct VaistTensorRef {
    /** \brief Tensor name (e.g. "model.layers.0.self_attn.qkv_proj.weight"). */
    std::string name;

    /** \brief Storage data type. */
    VaistWeightDtype dtype = VaistWeightDtype::kF32;

    /** \brief Quant type if quantized (VAIST_Q8_0 etc.), VAIST_Q8_0 as sentinel for non-quant. */
    VaistQuantType quant_type = VAIST_Q8_0;

    /** \brief Shape (row-major, shape[0] is outermost). */
    std::vector<uint64_t> shape;

    /** \brief Byte offset in the weight file. */
    uint64_t offset = 0;

    /** \brief Size in bytes of the stored (quantized or raw) data. */
    uint64_t byte_size = 0;

    /** \brief Cached dequantized float32 data (populated by get_data()). */
    std::vector<float> data_cache;

    /** \brief Whether data_cache has been populated. */
    bool loaded = false;

    /** \brief Weight file path (shared across all tensors in a model). */
    std::string file_path;

    /** \brief Compute element count from shape. */
    size_t numel() const {
        size_t n = 1;
        for (uint64_t s : shape) n *= (size_t)s;
        return n;
    }

    /** \brief Return true if the tensor is quantized (not raw fp32/f16). */
    bool is_quantized() const {
        return dtype != VaistWeightDtype::kF32 && dtype != VaistWeightDtype::kF16 && dtype != VaistWeightDtype::kBf16;
    }

    /**
     * \brief Load and dequantize the tensor data on first access.
     * \return Pointer to the float32 data array (numel() elements).
     *         nullptr if loading failed.
     */
    const float* get_data() {
        if (loaded) return data_cache.data();
        if (file_path.empty()) return nullptr;
        size_t n = numel();
        data_cache.resize(n);
        VaistTensorDesc desc;
        desc.offset = offset;
        desc.byte_size = byte_size;
        desc.rank = (uint32_t)shape.size();
        for (size_t i = 0; i < shape.size() && i < 8; i++) desc.shape[i] = shape[i];
        std::strncpy(desc.name, name.c_str(), 255);
        if (is_quantized()) {
            desc.dtype = (VaistDType)(GGUF_DTYPE_OFFSET_BASE + (uint32_t)quant_type);
        } else if (dtype == VaistWeightDtype::kF16) {
            desc.dtype = VAIST_F16;
        } else if (dtype == VaistWeightDtype::kBf16) {
            desc.dtype = VAIST_U8;
        } else {
            desc.dtype = VAIST_F32;
        }
        VaistStatus st = vaist_model_tensor_read(file_path.c_str(), &desc, data_cache.data(), n);
        if (st != VAIST_OK) {
            data_cache.clear();
            return nullptr;
        }
        loaded = true;
        return data_cache.data();
    }

    /** \brief GGUF dtype offset base for quantized types. */
    static constexpr uint32_t GGUF_DTYPE_OFFSET_BASE = 100;
};

/**
 * \brief Model state: loaded configuration, tensor map, KV cache, and runtime.
 *
 * Each architecture class inherits from (or holds) a VaistModel instance
 * that owns the tensor map and KV cache lifetime.
 */
struct VaistModel {
    /** \brief Model hyperparameters. */
    VaistModelConfig config;

    /** \brief Tensor map (name -> VaistTensorRef). */
    std::unordered_map<std::string, VaistTensorRef> tensors;

    /** \brief Weight file path (SafeTensors or GGUF). */
    std::string weight_path;

    /** \brief Paged KV cache handle (opaque C99). */
    VaistKVCachePaged* kv_cache = nullptr;

    /** \brief Runtime handle (for GPU dispatch / CPU fallback). */
    VaistRuntime* rt = nullptr;

    /** \brief Model format (GGUF, SafeTensors, etc.). */
    VaistModelFormat format = VAIST_MODEL_UNKNOWN;

    /** \brief Current sequence length (tokens generated so far). */
    uint32_t seq_len = 0;

    /** \brief Current batch size (1 for single-sequence). */
    uint32_t batch_size = 1;

    /** \brief Destructor frees the KV cache and runtime. */
    ~VaistModel() {
        if (kv_cache) vaist_kv_paged_destroy(kv_cache);
        if (rt) vaist_runtime_destroy(rt);
    }

    /**
     * \brief Look up a tensor by name.
     * \param name  Tensor name (e.g. "model.layers.0.self_attn.qkv_proj.weight").
     * \return Pointer to the tensor ref, or nullptr if not found.
     */
    const VaistTensorRef* get_tensor(std::string_view name) const {
        auto it = tensors.find(std::string(name));
        return (it != tensors.end()) ? &it->second : nullptr;
    }

    /**
     * \brief Get or lazy-load tensor data.
     * \param name  Tensor name.
     * \return Pointer to float32 data (numel elements), or nullptr on failure.
     */
    const float* tensor_data(std::string_view name) {
        auto it = tensors.find(std::string(name));
        if (it == tensors.end()) return nullptr;
        return it->second.get_data();
    }

    /**
     * \brief Build a paged KV cache according to the model config.
     * \param max_blocks  Total blocks in the pool.
     * \return VAIST_OK on success.
     */
    VaistStatus init_kv_cache(uint32_t max_blocks = 4096) {
        if (kv_cache) {
            vaist_kv_paged_destroy(kv_cache);
            kv_cache = nullptr;
        }
        VaistKVCachePagedConfig cfg{};
        cfg.num_layers     = config.n_layers;
        cfg.num_kv_heads   = config.n_kv_heads;
        cfg.head_dim       = config.head_dim;
        cfg.block_size     = config.block_size;
        cfg.kv_dtype       = (uint16_t)config.kv_cache_dtype;
        cfg.max_blocks     = max_blocks;
        cfg.total_blocks_used = 0;
        return vaist_kv_paged_create(rt, &cfg, &kv_cache);
    }

    /**
     * \brief Allocate blocks for a new sequence and assign them.
     * \param seq_id      Sequence identifier.
     * \param num_blocks  Number of blocks to allocate.
     * \param out_ids     Output array of block IDs (size num_blocks).
     * \return VAIST_OK on success.
     */
    VaistStatus alloc_seq_blocks(uint32_t seq_id, uint32_t num_blocks, uint32_t* out_ids) {
        if (!kv_cache) return VAIST_INVALID_STATE;
        VaistStatus st = vaist_kv_paged_alloc_block(kv_cache, num_blocks, out_ids);
        if (st != VAIST_OK) return st;
        return vaist_kv_paged_assign_blocks(kv_cache, seq_id, out_ids, num_blocks);
    }

    /**
     * \brief Write a K or V tensor for a token at (block, layer, token_idx).
     * \param block_id    Block ID.
     * \param layer       Layer index.
     * \param token_idx   Token index within the block.
     * \param data        Source data (num_kv_heads * head_dim floats).
     * \return VAIST_OK on success.
     */
    VaistStatus write_kv_block(uint32_t block_id, uint32_t layer,
                               uint32_t token_idx, const float* data) {
        if (!kv_cache) return VAIST_INVALID_STATE;
        return vaist_kv_paged_write_tensor(kv_cache, block_id, layer,
                                           token_idx, data, sizeof(float));
    }
};

} // namespace vaist

#endif // VAIST_CPP_MODEL_STATE_HPP
