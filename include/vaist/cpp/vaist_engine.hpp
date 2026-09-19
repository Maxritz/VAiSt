/**
 * \file vaist_engine.hpp
 * \brief Full inference engine: batched generation, KV cache management,
 *        and sampling (top-k, top-p, temperature, min-p, repetition penalties).
 *
 * Wraps vaist_model, vaist_blas, vaist_llm, and vaist_tokens into a
 * complete inference loop. Manages multiple model instances, paged KV
 * cache, and a sampler with full CFG support.
 */
#ifndef VAIST_CPP_ENGINE_HPP
#define VAIST_CPP_ENGINE_HPP

#include "vaist_model_state.hpp"
#include "vaist_llama.hpp"
#include "vaist_qwen3.hpp"
#include "vaist_deepseek.hpp"
#include "vaist_tokens.h"   /* VaistTokenizer, vaist_tokenize, vaist_detokenize */
#include <string>
#include <string_view>
#include <vector>
#include <memory>

namespace vaist {

/**
 * \brief Sampling configuration: controls token generation diversity.
 *
 * Mirrors the VaistSamplerConfig C struct, with additional fields for
 * min-p and full penalty support via vaist_sample_cfg.
 */
struct VaistSamplingConfig {
    float temperature          = 0.0f;   /**< 0 = greedy argmax. */
    float top_p                = 0.0f;   /**< Nucleus threshold (0 = disabled). */
    uint32_t top_k             = 0;     /**< Top-K filter (0 = disabled). */
    float min_p                = 0.0f;   /**< Min-p filter (0 = disabled). */
    float presence_penalty     = 0.0f;   /**< Presence penalty (per new token). */
    float frequency_penalty    = 0.0f;   /**< Per-occurrence penalty. */
    uint64_t seed              = 0;      /**< PRNG seed (0 = default). */
};

/**
 * \brief Generated token batch: sequences and their sampled IDs.
 */
struct VaistGeneratedBatch {
    std::vector<uint32_t> token_ids;  /**< Sampled token IDs (num_tokens). */
    std::vector<uint32_t> seq_ids;    /**< Sequence ID per token. */
    std::vector<uint32_t> positions;  /**< Position per token. */
    uint32_t num_tokens;               /**< Total tokens in this batch. */
};

/**
 * \brief Full inference engine.
 *
 * Manages one or more model instances (Llama / Qwen3-MoE / DeepSeek-V2/V3),
 * handles batched token generation with token-by-token scheduling,
 * paged KV cache allocation/free per sequence, and full sampling pipeline
 * via vaist_sample_cfg.
 *
 * Usage:
 * \code
 *   VaistEngine engine;
 *   engine.load_model(VaistEngine::ModelType::kLlama, "model.safetensors", rt);
 *   engine.generate("Hello world", 100, cfg);
 * \endcode
 */
class VaistEngine {
public:
    /** \brief Supported model architectures. */
    enum class ModelType : uint32_t {
        kLlama      = 0,  /**< Llama 2/3/3.1+. */
        kQwen3      = 1,  /**< Qwen3 (dense). */
        kQwen3Moe   = 2,  /**< Qwen3-MoE. */
        kDeepseekV2 = 3,  /**< DeepSeek-V2 (MLA). */
        kDeepseekV3 = 4,  /**< DeepSeek-V3 (MLA + MTP). */
    };

    /**
     * \brief Construct the engine.
     * \param rt  Optional runtime handle (nullptr = auto-create CPU backend).
     */
    explicit VaistEngine(VaistRuntime* rt = nullptr);

    /** \brief Destructor frees runtime + model. */
    ~VaistEngine();

    // Non-copyable, movable
    VaistEngine(const VaistEngine&) = delete;
    VaistEngine& operator=(const VaistEngine&) = delete;
    VaistEngine(VaistEngine&&) noexcept;
    VaistEngine& operator=(VaistEngine&&) noexcept;

    /**
     * \brief Load a model from weights + (optionally) a config JSON.
     * \param type     Model architecture type.
     * \param weight_path  Path to SafeTensors or GGUF file.
     * \param config_path  Optional JSON config (e.g. config.json). If NULL,
     *                     the config is inferred from tensor names.
     * \return VAIST_OK on success.
     */
    VaistStatus load_model(ModelType type,
                           std::string_view weight_path,
                           std::string_view config_path = {});

    /**
     * \brief Tokenize a prompt string.
     * \param text  Input text (UTF-8, NUL-terminated).
     * \return Token IDs.
     * \throws std::runtime_error if tokenizer is not initialized.
     */
    std::vector<uint32_t> tokenize(const char* text);

    /**
     * \brief Detokenize token IDs to text.
     * \param tokens  Token ID array.
     * \param n       Number of tokens.
     * \return UTF-8 string.
     * \throws std::runtime_error if tokenizer is not initialized.
     */
    std::string detokenize(const uint32_t* tokens, size_t n);

    /**
     * \brief Generate tokens from a prompt (single sequence).
     * \param prompt    Input text.
     * \param max_tokens  Maximum new tokens to generate.
     * \param cfg       Sampling configuration.
     * \param out_tokens  Output token IDs (appended to this vector).
     * \return VAIST_OK on success.
     */
    VaistStatus generate(const char* prompt,
                         uint32_t max_tokens,
                         const VaistSamplingConfig& cfg,
                         std::vector<uint32_t>* out_tokens);

    /**
     * \brief Generate tokens from pre-tokenized input.
     * \param input_ids   Input token IDs.
     * \param input_len   Number of input tokens.
     * \param max_tokens  Maximum new tokens to generate.
     * \param cfg         Sampling configuration.
     * \param out_tokens  Output token IDs (appended).
     * \return VAIST_OK on success.
     */
    VaistStatus generate(const uint32_t* input_ids,
                         size_t input_len,
                         uint32_t max_tokens,
                         const VaistSamplingConfig& cfg,
                         std::vector<uint32_t>* out_tokens);

    /**
     * \brief Access the underlying model.
     * \return Pointer to the active model (type-unsafe; cast based on load_model type).
     */
    void* model_ptr() const { return model_ptr_; }
    ModelType model_type() const { return model_type_; }

    /** \brief Current sequence length (tokens generated). */
    uint32_t current_seq_len() const { return seq_len_; }

    /** \brief Current vocab size. */
    uint32_t vocab_size() const { return vocab_size_; }

    /** \brief Access the tokenizer handle. */
    VaistTokenizer* tokenizer() const { return tokenizer_; }

    /** \brief Release the tokenizer handle (caller takes ownership). */
    VaistTokenizer* release_tokenizer() {
        VaistTokenizer* t = tokenizer_;
        tokenizer_ = nullptr;
        return t;
    }

    /** \brief Set tokenizer handle (takes ownership of existing). */
    void set_tokenizer(VaistTokenizer* t) { tokenizer_ = t; }

private:
    VaistRuntime* rt_ = nullptr;
    ModelType model_type_ = ModelType::kLlama;
    void* model_ptr_ = nullptr;
    VaistTokenizer* tokenizer_ = nullptr;
    uint32_t seq_len_ = 0;
    uint32_t vocab_size_ = 0;
    VaistSampler* sampler_ = nullptr;

    /**
     * \brief Run model forward for a single token (decode step).
     * \param token_id  Last generated token.
     * \param position  Absolute position of the token.
     * \param logits_out  Output logits (vocab_size).
     */
    void decode_step(uint32_t token_id, uint32_t position, float* logits_out);

    /**
     * \brief Run model forward for a prompt (prefill step).
     * \param input_ids  Prompt token IDs.
     * \param input_len  Number of prompt tokens.
     * \param logits_out  Output logits (input_len * vocab_size).
     */
    void prefill_step(const uint32_t* input_ids, size_t input_len,
                      const uint32_t* positions, float* logits_out);

    /** \brief Apply penalties and sample the next token. */
    uint32_t do_sample(const float* logits, uint32_t vocab,
                       const std::vector<uint32_t>& generated,
                       const VaistSamplingConfig& cfg);
};

} // namespace vaist

#endif // VAIST_CPP_ENGINE_HPP
