/**
 * \file vaist_deepseek.hpp
 * \brief DeepSeek-V2 and DeepSeek-V3 inference model in C++.
 *
 * DeepSeek-V2 introduces Multi-Head Latent Attention (MLA) with a
 * compressed cross-attention KV cache. DeepSeek-V3 extends this with:
 *   - Mixture-of-Experts (routed + shared experts)
 *   - MTP (Multi-Token Prediction) for V3+
 *   - Blockwise attention with sliding window
 *   - Shared expert normalization gates
 *
 * This header wraps vaist_blas, vaist_attn, and vaist_llm primitives.
 */
#ifndef VAIST_CPP_DEEPSEEK_HPP
#define VAIST_CPP_DEEPSEEK_HPP

#include "vaist_model_state.hpp"
#include "vaist_qwen3.hpp"
#include "vaist_attn.h"
#include <vector>
#include <cstring>

namespace vaist {

/**
 * \brief DeepSeek-V2/V3 causal language model.
 *
 * Supports both V2 (MLA attention, shared experts) and V3 (MTP,
 * blockwise attention, enhanced routing).
 *
 * Architecture:
 *   - MLA: compressed KV cache (q_lora_rank, kv_lora_rank)
 *   - MoE: routed experts + always-on shared expert
 *   - RoPE with Llama-3-style dynamic scaling
 *   - MTP: auxiliary prediction heads for multi-token prediction (V3)
 */
class DeepSeekModel {
public:
    /**
     * \brief Construct a DeepSeek model from a config and weight file.
     * \param cfg      Model configuration (hyperparameters).
     * \param weight_path  Path to the weight file (SafeTensors or GGUF).
     * \param rt      Optional runtime handle for GPU dispatch (nullptr = CPU).
     * \throws std::runtime_error on weight loading failure.
     */
    DeepSeekModel(const VaistModelConfig& cfg,
                  std::string_view weight_path,
                  VaistRuntime* rt = nullptr);

    /** \brief Destructor releases all owned resources. */
    ~DeepSeekModel() = default;

    /**
     * \brief Run a full forward pass (V2 or V3).
     * \param tokens  Input token IDs (batch_size * seq_len).
     * \param positions  Position indices for each token (seq_len).
     * \param seq_len  Number of tokens in this sequence.
     * \param batch_size  Number of sequences (default 1).
     * \param logits_out  Output logits buffer (batch * seq_len * vocab_size).
     */
    void forward(const uint32_t* tokens,
                 const uint32_t* positions,
                 uint32_t seq_len,
                 uint32_t batch_size,
                 float* logits_out);

    /**
     * \brief MLA attention: compressed QKV projection + blockwise attention.
     *
     * Uses separate Q, K, V projections with LoRA-like low-rank adapters.
     * The KV cache stores compressed (q_lora_rank + kv_lora_rank) representations.
     *
     * \param layer_idx  Layer index.
     * \param input  Input hidden states (batch_seq * hidden_size).
     * \param output  Output hidden states (batch_seq * hidden_size).
     * \param positions  Position IDs.
     * \param batch_seq  Flattened batch*seq dimension.
     * \param seqlen  Sequence length.
     */
    void mla_attention(uint32_t layer_idx,
                       const float* input,
                       float* output,
                       const uint32_t* positions,
                       uint32_t batch_seq,
                       uint32_t seqlen);

    /**
     * \brief RoPE embedding with Llama-3-style scaling.
     * \param buf   Input tensor (will be modified in-place).
     * \param n     Number of floats.
     * \param rope_dim  Dimension to apply RoPE.
     * \param pos   Absolute position.
     * \param base  RoPE base theta.
     * \param scale_factor  Llama-3 scale factor.
     */
    void rope_llama3(float* buf, uint32_t n, uint32_t rope_dim,
                     uint32_t pos, float base, float scale_factor);

    /**
     * \brief MoE MLP with routed + shared experts.
     * \param layer_idx  Layer index.
     * \param input  Input hidden states (batch_seq * hidden).
     * \param output  Output hidden states (batch_seq * hidden).
     * \param batch_seq  Flattened batch*seq dimension.
     *
     * Pipeline:
     *   1. vaist_blas_moe_topk — compute top-k experts + softmaxed scores
     *   2. vaist_blas_moe_dispatch — scatter tokens (weighted by score)
     *   3. Per-expert vaist_blas_mul_mat_q — W13 (gate_up) + silu_and_mul + W2 (down)
     *   4. vaist_blas_moe_combine — gather weighted outputs back
     *   5. Shared expert: always-on FFN, added with a learned gate
     */
    void mlp_moe(uint32_t layer_idx,
                 const float* input,
                 float* output,
                 uint32_t batch_seq);

    /**
     * \brief Shared expert MLP (always evaluated, output gated).
     * \param layer_idx  Layer index.
     * \param input  Input hidden states.
     * \param output  Output added to MoE result.
     * \param batch_seq  Flattened batch*seq dimension.
     */
    void mlp_shared(uint32_t layer_idx,
                    const float* input,
                    float* output,
                    uint32_t batch_seq);

    /** \brief Embed input token IDs. */
    void embed_tokens(const uint32_t* tokens, uint32_t n_tokens, float* out);

    /** \brief Final RMSNorm + LM head (with shared head normalization). */
    void norm_head(const float* hidden, float* logits, uint32_t batch);

    /**
     * \brief MTP (Multi-Token Prediction) forward for DeepSeek-V3.
     * \param hidden  Input hidden states.
     * \param output  Output (hidden + residual for next MTP step).
     * \param batch_seq  Flattened batch*seq dimension.
     * \param step  MTP prediction step (0 = first MTP head).
     */
    void mtp_forward(uint32_t step,
                     const float* hidden,
                     float* output,
                     uint32_t batch_seq);

    /** \brief Access the underlying model state. */
    VaistModel& state() { return state_; }
    const VaistModelConfig& config() const { return state_.config; }

private:
    VaistModel state_;
    VaistRuntime* rt_;

    /** \brief Flash-Decode attention context (for GPU path). */
    vaist_attn_ctx* attn_ctx_ = nullptr;

    /** \brief QK RMSNorm (per-head normalization before RoPE). */
    void qk_norm(const float* input, const float* weight,
                 float* output, uint32_t n, float eps);

    /** \brief RMSNorm helper. */
    void rmsnorm(const float* input, const float* weight,
                 float* output, uint32_t n, float eps);

    /** \brief SwiGLU activation. */
    void silu_and_mul(const float* gate_up, float* output,
                      uint32_t n, uint32_t intermediate);

    /** \brief Quantized matmul (handles all GGUF weight formats). */
    void qgemm(const float* A, const VaistTensorRef* W, float* C,
               size_t M, size_t K, size_t N);

    /** \brief Internal working buffers for MoE. */
    std::vector<float> moe_buf_;
    std::vector<uint32_t> topk_indices_;
    std::vector<float> topk_scores_;
    std::vector<float> normalized_scores_;
    std::vector<uint32_t> expert_blocks_;
    std::vector<uint32_t> expert_offsets_;

    /** \brief Initialize the flash-decode attention context. */
    void init_attn_ctx();

    /** \brief Destroy the flash-decode attention context. */
    void destroy_attn_ctx();
};

} // namespace vaist

#endif // VAIST_CPP_DEEPSEEK_HPP
