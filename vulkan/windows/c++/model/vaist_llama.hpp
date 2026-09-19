/**
 * \file vaist_llama.hpp
 * \brief Full Llama inference model (Llama 2, 3, 3.1+) in C++.
 *
 * Wraps vaist_blas_gemm (matmul), vaist_compute (elementwise),
 * vaist_nn (RMSNorm, SiLU), and vaist_quant (dequantization) into a
 * complete causal-LM forward pass with GQA, RMSNorm, SiLU activation,
 * and rotary positional embeddings.
 */
#ifndef VAIST_CPP_LLAMA_HPP
#define VAIST_CPP_LLAMA_HPP

#include "vaist_model_state.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vaist {

/**
 * \brief Llama causal language model.
 *
 * Supports Llama 2, Llama 3, and Llama 3.1+ configurations.
 * The model uses:
 * - Grouped-Query Attention (GQA)
 * - RMSNorm (no mean subtraction)
 * - SwiGLU activation (SiLU * gate)
 * - RoPE (Rotary Positional Embedding)
 * - Tied or untied word embeddings
 */
class LlamaModel {
public:
    /**
     * \brief Construct a Llama model from a config and weight file.
     * \param cfg      Model configuration (hyperparameters).
     * \param weight_path  Path to the weight file (SafeTensors or GGUF).
     * \param rt      Optional runtime handle for GPU dispatch (nullptr = CPU).
     * \throws std::runtime_error on weight loading failure.
     */
    LlamaModel(const VaistModelConfig& cfg,
               std::string_view weight_path,
               VaistRuntime* rt = nullptr);

    /** \brief Destructor releases all owned resources. */
    ~LlamaModel() = default;

    /**
     * \brief Run a full forward pass.
     * \param tokens  Input token IDs (batch_size * seq_len).
     * \param positions  Position indices for each token (seq_len).
     * \param seq_len  Number of tokens in this sequence.
     * \param batch_size  Number of sequences (default 1).
     * \return Logits tensor (batch_size * seq_len * vocab_size), allocated by caller.
     *
     * The logits buffer must be pre-allocated with at least
     * batch_size * seq_len * config.vocab_size floats.
     */
    void forward(const uint32_t* tokens,
                 const uint32_t* positions,
                 uint32_t seq_len,
                 uint32_t batch_size,
                 float* logits_out);

    /**
     * \brief Embed input token IDs.
     * \param tokens  Token IDs (n_tokens).
     * \param n_tokens  Number of tokens.
     * \param out  Output embeddings (n_tokens * hidden_size).
     */
    void embed_tokens(const uint32_t* tokens, uint32_t n_tokens, float* out);

    /**
     * \brief RMSNorm + attention sub-layer.
     * \param layer_idx  Layer index (0-based).
     * \param input  Input hidden states (batch * hidden_size).
     * \param output  Output hidden states (batch * hidden_size).
     * \param positions  Position IDs (seq_len).
     * \param batch_seq  batch_size * seq_len (flattened).
     * \param seqlen  Sequence length.
     */
    void attention(uint32_t layer_idx,
                   const float* input,
                   float* output,
                   const uint32_t* positions,
                   uint32_t batch_seq,
                   uint32_t seqlen);

    /**
     * \brief MLP sub-layer (gate_up_proj + silu_and_mul + down_proj).
     * \param layer_idx  Layer index.
     * \param input  Input hidden states (batch * hidden_size).
     * \param output  Output hidden states (batch * hidden_size).
     * \param batch_seq  batch_size * seq_len (flattened).
     */
    void mlp(uint32_t layer_idx,
             const float* input,
             float* output,
             uint32_t batch_seq);

    /**
     * \brief Final RMSNorm + language model head.
     * \param hidden  Final hidden states (batch * hidden_size).
     * \param logits  Output logits (batch * vocab_size).
     * \param batch  Batch size.
     */
    void norm_head(const float* hidden,
                   float* logits,
                   uint32_t batch);

    /** \brief Access the underlying model state. */
    VaistModel& state() { return state_; }
    const VaistModelConfig& config() const { return state_.config; }

private:
    VaistModel state_;

    /** \brief Compute rotary embedding (sin/cos) for given positions. */
    void apply_rope(float* qkv, uint32_t n_tokens,
                    const uint32_t* positions,
                    bool interleaved);

    /** \brief RMSNorm helper: divide by sqrt(mean(x^2) + eps). */
    void rmsnorm(const float* input, const float* weight,
                 float* output, uint32_t n, float eps);

    /** \brief SwiGLU activation: output[i] = gate[i] * silu(up[i]). */
    void silu_and_mul(const float* gate_up, float* output,
                      uint32_t n, uint32_t intermediate);

    /** \brief Quantized matmul wrapper: handles q4_0, q8_0, f32, f16 weights. */
    void qgemm(const float* A, const VaistTensorRef* W, float* C,
               size_t M, size_t K, size_t N);
};

} // namespace vaist

#endif // VAIST_CPP_LLAMA_HPP
