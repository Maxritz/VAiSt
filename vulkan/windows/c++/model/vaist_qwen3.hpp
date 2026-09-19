/**
 * \file vaist_qwen3.hpp
 * \brief Qwen3 and Qwen3-MoE inference model in C++.
 *
 * Extends the Llama architecture with:
 * - Qwen3-MoE: top-k expert routing via vaist_blas_moe_topk/dispatch/combine
 * - QK normalization (per-head RMSNorm before RoPE)
 * - qk_rope scaling (dynamic, yarn, llama3-style)
 * - Shared expert normalization (DeepSeek-style shared experts)
 * - RMSNorm on both Q and K projections
 */
#ifndef VAIST_CPP_QWEN3_HPP
#define VAIST_CPP_QWEN3_HPP

#include "vaist_llama.hpp"
#include <vector>

namespace vaist {

/**
 * \brief Qwen3 causal language model with MoE support.
 *
 * For dense (non-MoE) Qwen3 models, this is identical to LlamaModel.
 * For MoE models, replaces the MLP sub-layer with a Mixture-of-Experts
 * block that dispatches tokens to the top-k experts.
 */
class Qwen3Model {
public:
    /**
     * \brief Construct a Qwen3 model from a config and weight file.
     * \param cfg      Model configuration (hyperparameters).
     * \param weight_path  Path to the weight file (SafeTensors or GGUF).
     * \param rt      Optional runtime handle for GPU dispatch (nullptr = CPU).
     * \throws std::runtime_error on weight loading failure.
     */
    Qwen3Model(const VaistModelConfig& cfg,
               std::string_view weight_path,
               VaistRuntime* rt = nullptr);

    /** \brief Destructor releases all owned resources. */
    ~Qwen3Model() = default;

    /**
     * \brief Run a full forward pass.
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
     * \brief Qwen3 attention with QK norm.
     * \param layer_idx  Layer index.
     * \param input  Input hidden states.
     * \param output  Output hidden states.
     * \param positions  Position IDs.
     * \param batch_seq  Flattened batch*seq dimension.
     * \param seqlen  Sequence length.
     */
    void attention(uint32_t layer_idx,
                   const float* input,
                   float* output,
                   const uint32_t* positions,
                   uint32_t batch_seq,
                   uint32_t seqlen);

    /**
     * \brief MoE MLP: top-k routing + per-expert FFN + combine.
     * \param layer_idx  Layer index.
     * \param input  Input hidden states (batch_seq * hidden_size).
     * \param output  Output hidden states (batch_seq * hidden_size).
     * \param batch_seq  Flattened batch*seq dimension.
     *
     * Uses vaist_blas_moe_topk to compute routing, vaist_blas_moe_dispatch
     * to scatter, per-expert vaist_blas_mul_mat_q for GEMM, and
     * vaist_blas_moe_combine to gather.
     */
    void moe_mlp(uint32_t layer_idx,
                 const float* input,
                 float* output,
                 uint32_t batch_seq);

    /**
     * \brief Shared expert MLP (always-on expert in DeepSeek-style models).
     * \param layer_idx  Layer index.
     * \param input  Input hidden states.
     * \param output  Output (added to MoE output).
     * \param batch_seq  Flattened batch*seq dimension.
     */
    void shared_expert_mlp(uint32_t layer_idx,
                           const float* input,
                           float* output,
                           uint32_t batch_seq);

    /** \brief Embed input token IDs. */
    void embed_tokens(const uint32_t* tokens, uint32_t n_tokens, float* out);

    /** \brief Final RMSNorm + LM head. */
    void norm_head(const float* hidden, float* logits, uint32_t batch);

    /** \brief Access the underlying model state. */
    VaistModel& state() { return state_; }
    const VaistModelConfig& config() const { return state_.config; }

private:
    VaistModel state_;
    std::unique_ptr<LlamaModel> dense_model_; /**< Fallback for dense layers. */

    /** \brief QK RMSNorm: normalize per-head before RoPE. */
    void qk_norm(const float* input, const float* weight,
                 float* output, uint32_t n, float eps);

    /** \brief Apply RoPE with the configured scaling strategy. */
    void apply_rope_scaled(float* qkv, uint32_t n_tokens,
                           const uint32_t* positions,
                           bool interleaved);

    /** \brief SwiGLU activation (shared with dense models). */
    void silu_and_mul(const float* gate_up, float* output,
                      uint32_t n, uint32_t intermediate);

    /** \brief Quantized matmul. */
    void qgemm(const float* A, const VaistTensorRef* W, float* C,
               size_t M, size_t K, size_t N);

    /** \brief Internal working buffers for MoE dispatch/combine. */
    std::vector<float> moe_buf_;
    std::vector<uint32_t> topk_indices_;
    std::vector<float> topk_scores_;
    std::vector<float> normalized_scores_;
    std::vector<uint32_t> expert_blocks_;
    std::vector<uint32_t> expert_offsets_;
};

} // namespace vaist

#endif // VAIST_CPP_QWEN3_HPP
