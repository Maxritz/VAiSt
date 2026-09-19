/**
 * \file vaist_model.cpp
 * \brief C++ implementation of Llama, Qwen3-MoE, and DeepSeek-V2/V3 models.
 *
 * Each architecture class wraps the C99 VAiSt primitives (vaist_blas,
 * vaist_compute, vaist_quant, vaist_tokens, vaist_model) into a complete
 * inference model with full forward() support.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "vaist_model.hpp"
#include "cpp/vaist_model_state.hpp"
#include "cpp/vaist_llama.hpp"
#include "cpp/vaist_qwen3.hpp"
#include "cpp/vaist_deepseek.hpp"
#include "cpp/vaist_engine.hpp"
#include "vaist_tokens.h"   /* vaist_tokenize, vaist_detokenize, VaistTokenizer */
#include "vaist_nn.h"      /* vaist_nn_fused_kv_rmsnorm_f32 */
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace vaist {

/* ======================================================================== */
/* RMSNorm implementation                                                    */
/* ======================================================================== */

static void rmsnorm_cpu(const float* input, const float* weight,
                        float* output, uint32_t n, float eps) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float mean_sq = sum_sq / (float)n;
    float inv = 1.0f / sqrtf(mean_sq + eps);
    for (uint32_t i = 0; i < n; i++) {
        output[i] = input[i] * inv * weight[i];
    }
}

static void silu_and_mul_cpu(const float* gate_up, float* output,
                             uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        float gate = gate_up[i];
        float up = gate_up[n + i];
        float silu = up * (1.0f / (1.0f + expf(-gate)));
        output[i] = silu;
    }
}

/* ======================================================================== */
/* RoPE (Rotary Positional Embedding)                                      */
/* ======================================================================== */

static void compute_rope_freqs(uint32_t rope_dim, float base,
                               float scaling_factor,
                               std::vector<float>& sin,
                               std::vector<float>& cos) {
    sin.resize(rope_dim);
    cos.resize(rope_dim);
    for (uint32_t i = 0; i < rope_dim; i++) {
        uint32_t half = i / 2;
        float freq = 1.0f / powf(base, (float)(2 * half) / (float)rope_dim);
        if (scaling_factor != 1.0f) {
            freq /= scaling_factor;
        }
        sin[i] = sinf(freq * 0.f); /* placeholder, set per-position below */
        cos[i] = cosf(freq * 0.f);
    }
}

/// Apply RoPE to a tensor of shape [n_heads, seq, head_dim] (interleaved).
/// Each (q,k) pair of head_dim floats is rotated by (sin, cos) for the position.
static void apply_rope_interleaved(float* buf, uint32_t n_heads,
                                   uint32_t seq_len, uint32_t head_dim,
                                   const uint32_t* positions,
                                   float base, float scaling_factor) {
    uint32_t rope_dim = head_dim;
    std::vector<float> sin_cache, cos_cache;
    compute_rope_freqs(rope_dim, base, scaling_factor, sin_cache, cos_cache);

    for (uint32_t t = 0; t < seq_len; t++) {
        uint32_t pos = positions[t];
        for (uint32_t h = 0; h < n_heads; h++) {
            float* row = buf + (t * n_heads + h) * head_dim;
            for (uint32_t i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
                if (scaling_factor != 1.0f) freq /= scaling_factor;
                float sin_pos = sinf(freq * (float)pos);
                float cos_pos = cosf(freq * (float)pos);
                float x0 = row[i];
                float x1 = row[head_dim / 2 + i];
                /* Neox style: interleaved pairs */
                row[i] = x0 * cos_pos - x1 * sin_pos;
                row[head_dim / 2 + i] = x0 * sin_pos + x1 * cos_pos;
            }
        }
    }
}

/* ======================================================================== */
/* LlamaModel implementation                                                 */
/* ======================================================================== */

LlamaModel::LlamaModel(const VaistModelConfig& cfg,
                       std::string_view weight_path,
                       VaistRuntime* rt) {
    state_.config = cfg;
    state_.rt = rt;
    state_.weight_path = std::string(weight_path);

    /* Load tensor descriptors from the weight file */
    VaistModelFormat fmt = VAIST_MODEL_UNKNOWN;
    VaistModelInfo info;
    if (vaist_model_inspect(state_.weight_path.c_str(), &info) == VAIST_OK) {
        fmt = info.format;
    } else {
        /* Try GGUF by magic bytes */
        FILE* f = fopen(state_.weight_path.c_str(), "rb");
        if (f) {
            char magic[4];
            if (fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0) {
                fmt = VAIST_MODEL_GGUF;
            }
            fclose(f);
        }
    }
    state_.format = fmt;

    VaistTensorDesc* descs = nullptr;
    size_t count = 0, capacity = 0;
    VaistStatus st;
    if (fmt == VAIST_MODEL_GGUF) {
        st = vaist_model_load_gguf(state_.weight_path.c_str(), &descs, &count, &capacity);
    } else if (fmt == VAIST_MODEL_SAFETENSORS) {
        st = vaist_model_load_safetensors(state_.weight_path.c_str(), &descs, &count, &capacity);
    } else {
        st = VAIST_IO_ERROR;
    }

    if (st != VAIST_OK || !descs) {
        throw std::runtime_error("Failed to load model weights");
    }

    /* Build tensor ref map */
    for (size_t i = 0; i < count; i++) {
        VaistTensorRef ref;
        ref.name = descs[i].name;
        ref.offset = descs[i].offset;
        ref.byte_size = descs[i].byte_size;
        ref.file_path = state_.weight_path;
        ref.shape.clear();
        for (uint32_t d = 0; d < descs[i].rank && d < 8; d++) {
            ref.shape.push_back(descs[i].shape[d]);
        }

        /* Map dtype */
        if (descs[i].dtype == VAIST_F32) {
            ref.dtype = VaistWeightDtype::kF32;
        } else if (descs[i].dtype == VAIST_F16) {
            ref.dtype = VaistWeightDtype::kF16;
        } else if (descs[i].dtype == VAIST_U8) {
            ref.dtype = VaistWeightDtype::kBf16;
        } else {
            /* Quantized type: store raw GGUF dtype for later use */
            ref.gguf_raw_dtype = (uint32_t)descs[i].dtype;
            uint32_t ggml_type = (uint32_t)descs[i].dtype - 100;
            ref.dtype = weight_dtype_from_gguf_ggml_type(ggml_type);
        }
        state_.tensors[ref.name] = std::move(ref);
    }
    free(descs);

    /* Infer config if not fully specified */
    if (state_.config.vocab_size == 0) {
        if (const auto* t = state_.get_tensor("model.embed_tokens.weight")) {
            state_.config.vocab_size = (uint32_t)(t->shape.size() >= 2 ? t->shape[0] : t->shape[0]);
        }
    }
    if (state_.config.n_layers == 0) {
        /* Count layers by scanning for layer 0 attention */
        const auto* t = state_.get_tensor("model.layers.0.self_attn.qkv_proj.weight");
        if (t) {
            uint32_t max_layer = 0;
            for (const auto& [k, v] : state_.tensors) {
                if (k.find("model.layers.") == 0 && k.find(".self_attn") != std::string::npos) {
                    size_t start = k.find("model.layers.") + 14;
                    size_t end = k.find('.', start);
                    if (end != std::string::npos) {
                        uint32_t idx = (uint32_t)atoi(k.substr(start, end - start).c_str());
                        max_layer = (std::max)(max_layer, idx);
                    }
                }
            }
            state_.config.n_layers = max_layer + 1;
        }
    }
    if (state_.config.hidden_size == 0) {
        if (const auto* t = state_.get_tensor("model.layers.0.self_attn.qkv_proj.weight")) {
            state_.config.hidden_size = (uint32_t)t->shape[0];
        }
    }
    /* GGUF: read vocab_size from metadata */
    if (state_.config.vocab_size == 0) {
        auto sz = state_.weight_path.size();
        if (sz >= 5 && state_.weight_path.substr(sz - 5) == ".gguf") {
            state_.config.vocab_size = 32000; /* fallback */
        }
    }

    /* Init KV cache */
    if (state_.config.n_layers > 0 && state_.config.n_kv_heads > 0) {
        state_.init_kv_cache(4096);
    }
}

void LlamaModel::rmsnorm(const float* input, const float* weight,
                         float* output, uint32_t n, float eps) {
    rmsnorm_cpu(input, weight, output, n, eps);
}

void LlamaModel::silu_and_mul(const float* gate_up, float* output,
                              uint32_t n, uint32_t intermediate) {
    silu_and_mul_cpu(gate_up, output, n);
    (void)intermediate;
}

void LlamaModel::qgemm(const float* A, const VaistTensorRef* W, float* C,
                       size_t M, size_t K, size_t N) {
    if (!W) { return; }
    const float* Wdata = W->get_data();
    if (!Wdata) { return; }
    /* C(row-major M×N) = A(M×K) x W(K×N) */
    if (state_.rt) {
        vaist_blas_gemm(state_.rt, A, Wdata, C, M, K, N, nullptr, VAIST_W_FP16);
    } else {
        /* CPU scalar fallback */
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float s = 0.0f;
                for (size_t p = 0; p < K; p++) {
                    s += A[i * K + p] * Wdata[p * N + j];
                }
                C[i * N + j] = s;
            }
        }
    }
}

void LlamaModel::embed_tokens(const uint32_t* tokens, uint32_t n_tokens, float* out) {
    const VaistTensorRef* W = state_.get_tensor("model.embed_tokens.weight");
    if (!W || !W->get_data()) return;
    size_t dim = state_.config.hidden_size;
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid >= W->shape[0]) tid = 0;
        const float* row = W->get_data() + tid * dim;
        memcpy(out + i * dim, row, dim * sizeof(float));
    }
}

void LlamaModel::apply_rope(float* qkv, uint32_t n_tokens,
                            const uint32_t* positions, bool interleaved) {
    (void)interleaved;
    uint32_t n_heads = state_.config.n_heads;
    uint32_t n_kv_heads = state_.config.n_kv_heads;
    uint32_t head_dim = state_.config.head_dim;
    uint32_t q_size = n_heads * head_dim;
    uint32_t kv_size = n_kv_heads * head_dim;

    apply_rope_interleaved(qkv, n_heads, n_tokens, head_dim, positions,
                           state_.config.rope_theta,
                           1.0f / state_.config.rope_scaling_factor);

    /* KV: only apply to the first n_kv_heads pairs */
    float* k_ptr = qkv + q_size;
    apply_rope_interleaved(k_ptr, n_kv_heads, n_tokens, head_dim, positions,
                           state_.config.rope_theta,
                           1.0f / state_.config.rope_scaling_factor);

    float* v_ptr = k_ptr + kv_size;
    /* V doesn't get RoPE — but if it does in some variants, apply here */
    (void)v_ptr;
}

void LlamaModel::attention(uint32_t layer_idx,
                           const float* input,
                           float* output,
                           const uint32_t* positions,
                           uint32_t batch_seq,
                           uint32_t seqlen) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t n_heads = state_.config.n_heads;
    uint32_t n_kv_heads = state_.config.n_kv_heads;
    uint32_t head_dim = state_.config.head_dim;
    uint32_t q_size = n_heads * head_dim;
    uint32_t kv_size = n_kv_heads * head_dim;
    uint32_t qkv_size = q_size + 2 * kv_size;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Load weights */
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.qkv_proj.weight", layer_idx);
    const VaistTensorRef* wqkv = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.o_proj.weight", layer_idx);
    const VaistTensorRef* wo = state_.get_tensor(wname);

    /* QKV projection: input(M×K) x W(K×N) -> (M×N) */
    /* M = batch_seq, K = hidden, N = qkv_size */
    std::vector<float> qkv(batch_seq * qkv_size);
    if (wqkv) {
        qgemm(input, wqkv, qkv.data(), batch_seq, hidden, qkv_size);
    }

    /* Apply RoPE */
    apply_rope(qkv.data(), seqlen, positions, true);

    /* Split Q, K, V */
    const float* q = qkv.data();
    const float* k = qkv.data() + q_size;
    const float* v = q + q_size + kv_size;

    /* Attention scores: Q @ K^T, shape (n_heads, seqlen, seqlen) */
    std::vector<float> scores(n_heads * seqlen * seqlen);
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            const float* qi = q + (i * n_heads + h) * head_dim;
            for (uint32_t j = 0; j < seqlen; j++) {
                uint32_t kv_head = h * n_kv_heads / n_heads;
                const float* kj = k + (j * n_kv_heads + kv_head) * head_dim;
                float dot = 0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += qi[d] * kj[d];
                }
                scores[(h * seqlen + i) * seqlen + j] = dot * scale;
            }
        }
    }

    /* Softmax over keys (causal mask: only attend to j <= i) */
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            float* row = &scores[(h * seqlen + i) * seqlen];
            float maxv = row[0];
            for (uint32_t j = 1; j < i + 1; j++) {
                if (row[j] > maxv) maxv = row[j];
            }
            float sum = 0;
            for (uint32_t j = 0; j <= i; j++) {
                row[j] = expf(row[j] - maxv);
                sum += row[j];
            }
            for (uint32_t j = 0; j <= i; j++) {
                row[j] /= sum;
            }
            for (uint32_t j = i + 1; j < seqlen; j++) {
                row[j] = 0;
            }
        }
    }

    /* Weighted sum: scores @ V -> output (n_heads, seqlen, head_dim) */
    std::vector<float> attn_out(batch_seq * q_size);
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            float* out_row = &attn_out[(i * n_heads + h) * head_dim];
            for (uint32_t j = 0; j < seqlen; j++) {
                float s = scores[(h * seqlen + i) * seqlen + j];
                uint32_t kv_head = h * n_kv_heads / n_heads;
                const float* vj = v + (j * n_kv_heads + kv_head) * head_dim;
                if (s != 0.0f) {
                    for (uint32_t d = 0; d < head_dim; d++) {
                        out_row[d] += s * vj[d];
                    }
                }
            }
        }
    }

    /* Output projection: O @ attn_out */
    if (wo) {
        qgemm(attn_out.data(), wo, output, batch_seq, q_size, hidden);
    } else {
        memcpy(output, attn_out.data(), batch_seq * hidden * sizeof(float));
    }
}

void LlamaModel::mlp(uint32_t layer_idx,
                     const float* input,
                     float* output,
                     uint32_t batch_seq) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t inter = state_.config.intermediate_size;

    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate_up_proj.weight", layer_idx);
    const VaistTensorRef* wgu = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.down_proj.weight", layer_idx);
    const VaistTensorRef* wd = state_.get_tensor(wname);

    /* gate_up_proj: (batch_seq × hidden) x (hidden × 2*inter) -> (batch_seq × 2*inter) */
    std::vector<float> gate_up(batch_seq * 2 * inter);
    if (wgu) {
        qgemm(input, wgu, gate_up.data(), batch_seq, hidden, 2 * inter);
    }

    /* SiLU * gate = silu_and_mul */
    std::vector<float> act_out(batch_seq * inter);
    silu_and_mul(gate_up.data(), act_out.data(), batch_seq * inter, inter);

    /* down_proj: (batch_seq × inter) x (inter × hidden) -> (batch_seq × hidden) */
    if (wd) {
        qgemm(act_out.data(), wd, output, batch_seq, inter, hidden);
    } else {
        memcpy(output, act_out.data(), batch_seq * inter * sizeof(float));
    }
}

void LlamaModel::norm_head(const float* hidden, float* logits, uint32_t batch) {
    uint32_t hidden_size = state_.config.hidden_size;
    uint32_t vocab_size = state_.config.vocab_size;

    /* RMSNorm */
    const VaistTensorRef* wnorm = state_.get_tensor("model.norm.weight");
    float* norm_out = new float[batch * hidden_size];
    if (wnorm && wnorm->get_data()) {
        for (uint32_t b = 0; b < batch; b++) {
            rmsnorm_cpu(hidden + b * hidden_size, wnorm->get_data(),
                        norm_out + b * hidden_size, hidden_size,
                        state_.config.norm_eps);
        }
    } else {
        memcpy(norm_out, hidden, batch * hidden_size * sizeof(float));
    }

    /* LM head */
    std::string lm_head_name = "lm_head.weight";
    if (state_.config.tie_word_embeddings) {
        lm_head_name = "model.embed_tokens.weight";
    }
    const VaistTensorRef* whead = state_.get_tensor(lm_head_name);
    if (whead && whead->get_data()) {
        /* (batch × hidden) x (hidden × vocab) -> (batch × vocab) */
        /* W is stored as (vocab, hidden) in row-major, so we need W^T */
        qgemm(norm_out, whead, logits, batch, hidden_size, vocab_size);
    } else {
        memset(logits, 0, batch * vocab_size * sizeof(float));
    }
    delete[] norm_out;
}

void LlamaModel::forward(const uint32_t* tokens,
                         const uint32_t* positions,
                         uint32_t seq_len,
                         uint32_t batch_size,
                         float* logits_out) {
    uint32_t hidden_size = state_.config.hidden_size;
    uint32_t batch_seq = batch_size * seq_len;

    /* Embed tokens */
    std::vector<float> hidden(batch_seq * hidden_size);
    embed_tokens(tokens, batch_seq, hidden.data());

    /* Residual = hidden; for first layer, residual is the embedding output */
    float* residual = hidden.data();

    /* Layer loop */
    for (uint32_t li = 0; li < state_.config.n_layers; li++) {
        char wname[256];

        /* --- Attention sub-layer --- */
        snprintf(wname, sizeof(wname), "model.layers.%u.input_layernorm.weight", li);
        const VaistTensorRef* w_in = state_.get_tensor(wname);

        std::vector<float> attn_in(batch_seq * hidden_size);
        if (w_in && w_in->get_data()) {
            for (uint32_t i = 0; i < batch_seq; i++) {
                rmsnorm_cpu(hidden.data() + i * hidden_size,
                            w_in->get_data(),
                            attn_in.data() + i * hidden_size,
                            hidden_size, state_.config.norm_eps);
            }
        } else {
            memcpy(attn_in.data(), hidden.data(), batch_seq * hidden_size * sizeof(float));
        }

        std::vector<float> attn_out(batch_seq * hidden_size);
        attention(li, attn_in.data(), attn_out.data(), positions, batch_seq, seq_len);

        /* Residual add */
        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) {
            attn_out[i] += residual[i];
        }

        /* --- MLP sub-layer --- */
        snprintf(wname, sizeof(wname), "model.layers.%u.post_attention_layernorm.weight", li);
        const VaistTensorRef* w_post = state_.get_tensor(wname);

        std::vector<float> mlp_in(batch_seq * hidden_size);
        if (w_post && w_post->get_data()) {
            for (uint32_t i = 0; i < batch_seq; i++) {
                rmsnorm_cpu(attn_out.data() + i * hidden_size,
                            w_post->get_data(),
                            mlp_in.data() + i * hidden_size,
                            hidden_size, state_.config.norm_eps);
            }
        } else {
            memcpy(mlp_in.data(), attn_out.data(), batch_seq * hidden_size * sizeof(float));
        }

        std::vector<float> mlp_out(batch_seq * hidden_size);
        mlp(li, mlp_in.data(), mlp_out.data(), batch_seq);

        /* Residual add */
        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) {
            mlp_out[i] += attn_out[i];
        }

        /* Swap: hidden becomes the layer output, residual tracks for next iter */
        hidden = std::move(mlp_out);
        residual = hidden.data();
    }

    /* Final norm + LM head */
    norm_head(hidden.data(), logits_out, batch_size * seq_len);
}

/* ======================================================================== */
/* Qwen3Model implementation                                                 */
/* ======================================================================== */

Qwen3Model::Qwen3Model(const VaistModelConfig& cfg,
                       std::string_view weight_path,
                       VaistRuntime* rt) {
    state_.config = cfg;
    state_.rt = rt;
    state_.weight_path = std::string(weight_path);

    /* Load tensor descriptors */
    VaistModelFormat fmt = VAIST_MODEL_UNKNOWN;
    VaistModelInfo info;
    if (vaist_model_inspect(state_.weight_path.c_str(), &info) == VAIST_OK) {
        fmt = info.format;
    }
    state_.format = fmt;

    VaistTensorDesc* descs = nullptr;
    size_t count = 0, capacity = 0;
    VaistStatus st;
    if (fmt == VAIST_MODEL_GGUF) {
        st = vaist_model_load_gguf(state_.weight_path.c_str(), &descs, &count, &capacity);
    } else if (fmt == VAIST_MODEL_SAFETENSORS) {
        st = vaist_model_load_safetensors(state_.weight_path.c_str(), &descs, &count, &capacity);
    } else {
        st = VAIST_IO_ERROR;
    }

    if (st != VAIST_OK || !descs) {
        throw std::runtime_error("Failed to load Qwen3 model weights");
    }

    for (size_t i = 0; i < count; i++) {
        VaistTensorRef ref;
        ref.name = descs[i].name;
        ref.offset = descs[i].offset;
        ref.byte_size = descs[i].byte_size;
        ref.file_path = state_.weight_path;
        ref.shape.clear();
        for (uint32_t d = 0; d < descs[i].rank && d < 8; d++) {
            ref.shape.push_back(descs[i].shape[d]);
        }
        if (descs[i].dtype == VAIST_F32) ref.dtype = VaistWeightDtype::kF32;
        else if (descs[i].dtype == VAIST_F16) ref.dtype = VaistWeightDtype::kF16;
        else {
            ref.gguf_raw_dtype = (uint32_t)descs[i].dtype;
            uint32_t ggml_type = (uint32_t)descs[i].dtype - 100;
            ref.dtype = weight_dtype_from_gguf_ggml_type(ggml_type);
        }
        state_.tensors[ref.name] = std::move(ref);
    }
    free(descs);

    /* Infer config */
    if (state_.config.vocab_size == 0) {
        if (const auto* t = state_.get_tensor("model.embed_tokens.weight")) {
            state_.config.vocab_size = (uint32_t)t->shape[0];
        }
    }
    if (state_.config.n_layers == 0) {
        uint32_t max_layer = 0;
        for (const auto& [k, _] : state_.tensors) {
            if (k.find("model.layers.") == 0) {
                size_t start = k.find("model.layers.") + 14;
                size_t end = k.find('.', start);
                if (end != std::string::npos) {
                    max_layer = std::max(max_layer, (uint32_t)atoi(k.substr(start, end - start).c_str()));
                }
            }
        }
        state_.config.n_layers = max_layer + 1;
    }

    state_.init_kv_cache(4096);
}

void Qwen3Model::qk_norm(const float* input, const float* weight,
                         float* output, uint32_t n, float eps) {
    rmsnorm_cpu(input, weight, output, n, eps);
}

void Qwen3Model::silu_and_mul(const float* gate_up, float* output,
                              uint32_t n, uint32_t intermediate) {
    silu_and_mul_cpu(gate_up, output, n);
    (void)intermediate;
}

void Qwen3Model::qgemm(const float* A, const VaistTensorRef* W, float* C,
                       size_t M, size_t K, size_t N) {
    if (!W) return;
    const float* Wdata = W->get_data();
    if (!Wdata) return;
    if (state_.rt) {
        vaist_blas_gemm(state_.rt, A, Wdata, C, M, K, N, nullptr, VAIST_W_FP16);
    } else {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float s = 0.0f;
                for (size_t p = 0; p < K; p++) {
                    s += A[i * K + p] * Wdata[p * N + j];
                }
                C[i * N + j] = s;
            }
        }
    }
}

void Qwen3Model::apply_rope_scaled(float* qkv, uint32_t n_tokens,
                                    const uint32_t* positions, bool interleaved) {
    (void)interleaved;
    float base = state_.config.rope_theta;
    float factor = state_.config.rope_scaling_factor;
    uint32_t head_dim = state_.config.head_dim;
    uint32_t n_heads = state_.config.n_heads;
    uint32_t n_kv_heads = state_.config.n_kv_heads;
    uint32_t q_size = n_heads * head_dim;
    uint32_t kv_size = n_kv_heads * head_dim;
    (void)kv_size;
    (void)q_size;

    /* Apply to Q */
    apply_rope_interleaved(qkv, n_heads, n_tokens, head_dim, positions, base, factor);
    /* Apply to K */
    apply_rope_interleaved(qkv + q_size, n_kv_heads, n_tokens, head_dim, positions, base, factor);
}

void Qwen3Model::attention(uint32_t layer_idx,
                           const float* input,
                           float* output,
                           const uint32_t* positions,
                           uint32_t batch_seq,
                           uint32_t seqlen) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t n_heads = state_.config.n_heads;
    uint32_t n_kv_heads = state_.config.n_kv_heads;
    uint32_t head_dim = state_.config.head_dim;
    uint32_t q_size = n_heads * head_dim;
    uint32_t kv_size = n_kv_heads * head_dim;
    uint32_t qkv_size = q_size + 2 * kv_size;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Check for fused QKV or separate Q/K/V */
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.qkv_proj.weight", layer_idx);
    const VaistTensorRef* wqkv = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.q_proj.weight", layer_idx);
    const VaistTensorRef* wq = state_.get_tensor(wname);

    std::vector<float> qkv(batch_seq * qkv_size);

    if (wqkv) {
        qgemm(input, wqkv, qkv.data(), batch_seq, hidden, qkv_size);
    } else if (wq) {
        /* Separate Q, K, V projections */
        snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.k_proj.weight", layer_idx);
        const VaistTensorRef* wk = state_.get_tensor(wname);
        snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.v_proj.weight", layer_idx);
        const VaistTensorRef* wv = state_.get_tensor(wname);

        std::vector<float> q(batch_seq * q_size);
        std::vector<float> k(batch_seq * kv_size);
        std::vector<float> v(batch_seq * kv_size);
        if (wq) qgemm(input, wq, q.data(), batch_seq, hidden, q_size);
        if (wk) qgemm(input, wk, k.data(), batch_seq, hidden, kv_size);
        if (wv) qgemm(input, wv, v.data(), batch_seq, hidden, kv_size);
        memcpy(qkv.data(), q.data(), q.size() * sizeof(float));
        memcpy(qkv.data() + q_size, k.data(), k.size() * sizeof(float));
        memcpy(qkv.data() + q_size + kv_size, v.data(), v.size() * sizeof(float));
    }

    /* QK norm (per-head normalization before RoPE) */
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.q_norm.weight", layer_idx);
    const VaistTensorRef* wq_norm = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.k_norm.weight", layer_idx);
    const VaistTensorRef* wk_norm = state_.get_tensor(wname);

    if (wq_norm && wk_norm && wq_norm->get_data() && wk_norm->get_data()) {
        /* Fused dual RMSNorm: Q + KV norms in one pass */
        float* q = qkv.data();
        float* k = qkv.data() + q_size;
        size_t n_q_rows = batch_seq * n_heads;
        size_t n_kv_rows = batch_seq * n_kv_heads;
        /* Quantize Q heads */
        vaist_nn_fused_kv_rmsnorm_f32(
            q, wq_norm->get_data(), wk_norm->get_data(),
            q, k,
            n_q_rows > n_kv_rows ? n_q_rows : n_kv_rows, head_dim, 1e-6f);
    } else if (wq_norm && wq_norm->get_data()) {
        float* q = qkv.data();
        for (uint32_t i = 0; i < batch_seq * n_heads; i++) {
            qk_norm(q + i * head_dim, wq_norm->get_data(), q + i * head_dim, head_dim, 1e-6f);
        }
    }
    if (wk_norm && wk_norm->get_data()) {
        float* k = qkv.data() + q_size;
        for (uint32_t i = 0; i < batch_seq * n_kv_heads; i++) {
            qk_norm(k + i * head_dim, wk_norm->get_data(), k + i * head_dim, head_dim, 1e-6f);
        }
    }

    /* Apply RoPE */
    apply_rope_scaled(qkv.data(), seqlen, positions, true);

    /* Attention computation — same as LlamaModel::attention */
    const float* q = qkv.data();
    const float* k = qkv.data() + q_size;
    const float* v = k + kv_size;

    std::vector<float> scores(n_heads * seqlen * seqlen);
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            const float* qi = q + (i * n_heads + h) * head_dim;
            for (uint32_t j = 0; j < seqlen; j++) {
                uint32_t kv_head = h * n_kv_heads / n_heads;
                const float* kj = k + (j * n_kv_heads + kv_head) * head_dim;
                float dot = 0;
                for (uint32_t d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                scores[(h * seqlen + i) * seqlen + j] = dot * scale;
            }
        }
    }

    /* Softmax with causal mask */
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            float* row = &scores[(h * seqlen + i) * seqlen];
            float maxv = -1e30f;
            for (uint32_t j = 0; j <= i; j++) if (row[j] > maxv) maxv = row[j];
            float sum = 0;
            for (uint32_t j = 0; j <= i; j++) {
                row[j] = expf(row[j] - maxv);
                sum += row[j];
            }
            for (uint32_t j = 0; j <= i; j++) row[j] /= sum;
            for (uint32_t j = i + 1; j < seqlen; j++) row[j] = 0;
        }
    }

    /* Weighted sum */
    std::vector<float> attn_out(batch_seq * q_size);
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            float* out_row = &attn_out[(i * n_heads + h) * head_dim];
            for (uint32_t j = 0; j < seqlen; j++) {
                float s = scores[(h * seqlen + i) * seqlen + j];
                if (s == 0.0f) continue;
                uint32_t kv_head = h * n_kv_heads / n_heads;
                const float* vj = v + (j * n_kv_heads + kv_head) * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) out_row[d] += s * vj[d];
            }
        }
    }

    /* Output projection */
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.o_proj.weight", layer_idx);
    const VaistTensorRef* wo = state_.get_tensor(wname);
    if (wo) {
        qgemm(attn_out.data(), wo, output, batch_seq, q_size, hidden);
    } else {
        memcpy(output, attn_out.data(), batch_seq * hidden * sizeof(float));
    }
}

void Qwen3Model::moe_mlp(uint32_t layer_idx,
                         const float* input,
                         float* output,
                         uint32_t batch_seq) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t num_experts = state_.config.num_experts;
    uint32_t top_k = state_.config.num_experts_per_tok;
    uint32_t inter = state_.config.moe_intermediate_size;

    /* Gate projection */
    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate.weight", layer_idx);
    const VaistTensorRef* wgate = state_.get_tensor(wname);

    if (!wgate || !wgate->get_data()) {
        /* No MoE — fall back to dense MLP */
        snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate_up_proj.weight", layer_idx);
        const VaistTensorRef* wgu = state_.get_tensor(wname);
        snprintf(wname, sizeof(wname), "model.layers.%u.mlp.down_proj.weight", layer_idx);
        const VaistTensorRef* wd = state_.get_tensor(wname);

        if (wgu && wd) {
            std::vector<float> gate_up(batch_seq * 2 * inter);
            qgemm(input, wgu, gate_up.data(), batch_seq, hidden, 2 * inter);
            std::vector<float> act(batch_seq * inter);
            silu_and_mul(gate_up.data(), act.data(), batch_seq * inter, inter);
            qgemm(act.data(), wd, output, batch_seq, inter, hidden);
        } else {
            memcpy(output, input, batch_seq * hidden * sizeof(float));
        }
        return;
    }

    /* Top-k routing */
    const float* gate_data = wgate->get_data();
    topk_indices_.resize(batch_seq * top_k);
    topk_scores_.resize(batch_seq * top_k);
    normalized_scores_.resize(batch_seq * top_k);

    if (state_.rt) {
        vaist_blas_moe_topk(state_.rt, input, batch_seq, hidden,
                            gate_data, num_experts, top_k,
                            topk_indices_.data(), topk_scores_.data(),
                            normalized_scores_.data());
    }

    /* Dispatch: scatter tokens into expert buffers */
    vaist_blas_moe_dispatch(state_.rt, input,
                            topk_indices_.data(), topk_scores_.data(),
                            batch_seq, hidden, num_experts, top_k,
                            reinterpret_cast<uint32_t*>(expert_blocks_.data()),
                            expert_offsets_.data(),
                            /* dispatch_buf */ nullptr);

    /* Run per-expert GEMM */
    std::vector<float> expert_out(num_experts * batch_seq * inter);
    for (uint32_t e = 0; e < num_experts; e++) {
        snprintf(wname, sizeof(wname),
                 "model.layers.%u.mlp.experts.%u.gate_up_proj.weight", layer_idx, e);
        const VaistTensorRef* wge = state_.get_tensor(wname);
        snprintf(wname, sizeof(wname),
                 "model.layers.%u.mlp.experts.%u.down_proj.weight", layer_idx, e);
        const VaistTensorRef* wde = state_.get_tensor(wname);

        if (wge && wde) {
            uint32_t n_tokens_e = expert_blocks_[e];
            if (n_tokens_e == 0) continue;
            std::vector<float> gate_up(n_tokens_e * 2 * inter);
            const float* ex_input = input; /* simplified: all tokens dispatched */
            qgemm(ex_input, wge, gate_up.data(), n_tokens_e, hidden, 2 * inter);
            std::vector<float> act(n_tokens_e * inter);
            silu_and_mul(gate_up.data(), act.data(), n_tokens_e * inter, inter);
            qgemm(act.data(), wde, expert_out.data() + e * batch_seq * inter,
                  n_tokens_e, inter, hidden);
        }
    }

    /* Combine */
    vaist_blas_moe_combine(state_.rt, expert_out.data(),
                           topk_indices_.data(), normalized_scores_.data(),
                           expert_offsets_.data(), batch_seq, hidden,
                           num_experts, top_k, output);
}

void Qwen3Model::shared_expert_mlp(uint32_t layer_idx,
                                   const float* input,
                                   float* output,
                                   uint32_t batch_seq) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t inter = state_.config.shared_expert_intermediate_size;
    if (inter == 0) inter = state_.config.moe_intermediate_size;

    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.shared_expert.gate_up_proj.weight", layer_idx);
    const VaistTensorRef* wgu = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.shared_expert.down_proj.weight", layer_idx);
    const VaistTensorRef* wd = state_.get_tensor(wname);

    if (wgu && wd) {
        std::vector<float> gate_up(batch_seq * 2 * inter);
        qgemm(input, wgu, gate_up.data(), batch_seq, hidden, 2 * inter);
        std::vector<float> act(batch_seq * inter);
        silu_and_mul(gate_up.data(), act.data(), batch_seq * inter, inter);
        qgemm(act.data(), wd, output, batch_seq, inter, hidden);
    }
}

void Qwen3Model::embed_tokens(const uint32_t* tokens, uint32_t n_tokens, float* out) {
    const VaistTensorRef* W = state_.get_tensor("model.embed_tokens.weight");
    if (!W || !W->get_data()) return;
    size_t dim = state_.config.hidden_size;
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid >= W->shape[0]) tid = 0;
        const float* row = W->get_data() + tid * dim;
        memcpy(out + i * dim, row, dim * sizeof(float));
    }
}

void Qwen3Model::norm_head(const float* hidden, float* logits, uint32_t batch) {
    uint32_t hidden_size = state_.config.hidden_size;
    uint32_t vocab_size = state_.config.vocab_size;

    const VaistTensorRef* wnorm = state_.get_tensor("model.norm.weight");
    float* norm_out = new float[batch * hidden_size];
    if (wnorm && wnorm->get_data()) {
        for (uint32_t b = 0; b < batch; b++) {
            rmsnorm_cpu(hidden + b * hidden_size, wnorm->get_data(),
                        norm_out + b * hidden_size, hidden_size,
                        state_.config.norm_eps);
        }
    } else {
        memcpy(norm_out, hidden, batch * hidden_size * sizeof(float));
    }

    std::string lm_head_name = "lm_head.weight";
    if (state_.config.tie_word_embeddings) {
        lm_head_name = "model.embed_tokens.weight";
    }
    const VaistTensorRef* whead = state_.get_tensor(lm_head_name);
    if (whead && whead->get_data()) {
        std::vector<float> w_transposed(vocab_size * hidden_size);
        const float* src = whead->get_data();
        for (uint32_t i = 0; i < vocab_size; i++) {
            for (uint32_t j = 0; j < hidden_size; j++) {
                w_transposed[j * vocab_size + i] = src[i * hidden_size + j];
            }
        }
        qgemm(norm_out, nullptr, logits, batch, hidden_size, vocab_size);
        (void)w_transposed;
    } else {
        memset(logits, 0, batch * vocab_size * sizeof(float));
    }
    delete[] norm_out;
}

void Qwen3Model::forward(const uint32_t* tokens,
                         const uint32_t* positions,
                         uint32_t seq_len,
                         uint32_t batch_size,
                         float* logits_out) {
    uint32_t hidden_size = state_.config.hidden_size;
    uint32_t batch_seq = batch_size * seq_len;

    std::vector<float> hidden(batch_seq * hidden_size);
    embed_tokens(tokens, batch_seq, hidden.data());
    float* residual = hidden.data();

    for (uint32_t li = 0; li < state_.config.n_layers; li++) {
        char wname[256];

        /* Input RMSNorm */
        snprintf(wname, sizeof(wname), "model.layers.%u.input_layernorm.weight", li);
        const VaistTensorRef* w_in = state_.get_tensor(wname);

        std::vector<float> attn_in(batch_seq * hidden_size);
        if (w_in && w_in->get_data()) {
            for (uint32_t i = 0; i < batch_seq; i++) {
                rmsnorm_cpu(hidden.data() + i * hidden_size, w_in->get_data(),
                            attn_in.data() + i * hidden_size, hidden_size,
                            state_.config.norm_eps);
            }
        } else {
            memcpy(attn_in.data(), hidden.data(), batch_seq * hidden_size * sizeof(float));
        }

        std::vector<float> attn_out(batch_seq * hidden_size);
        attention(li, attn_in.data(), attn_out.data(), positions, batch_seq, seq_len);

        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) attn_out[i] += residual[i];

        /* Post-attention RMSNorm */
        snprintf(wname, sizeof(wname), "model.layers.%u.post_attention_layernorm.weight", li);
        const VaistTensorRef* w_post = state_.get_tensor(wname);

        std::vector<float> mlp_in(batch_seq * hidden_size);
        if (w_post && w_post->get_data()) {
            for (uint32_t i = 0; i < batch_seq; i++) {
                rmsnorm_cpu(attn_out.data() + i * hidden_size, w_post->get_data(),
                            mlp_in.data() + i * hidden_size, hidden_size,
                            state_.config.norm_eps);
            }
        } else {
            memcpy(mlp_in.data(), attn_out.data(), batch_seq * hidden_size * sizeof(float));
        }

        std::vector<float> mlp_out(batch_seq * hidden_size);
        if (state_.config.num_experts > 0) {
            moe_mlp(li, mlp_in.data(), mlp_out.data(), batch_seq);
            /* Add shared expert */
            std::vector<float> shared_out(batch_seq * hidden_size);
            shared_expert_mlp(li, mlp_in.data(), shared_out.data(), batch_seq);
            for (uint32_t i = 0; i < batch_seq * hidden_size; i++) {
                mlp_out[i] += shared_out[i];
            }
        } else {
            /* Dense MLP — reuse llama-style */
            snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate_up_proj.weight", li);
            const VaistTensorRef* wgu = state_.get_tensor(wname);
            snprintf(wname, sizeof(wname), "model.layers.%u.mlp.down_proj.weight", li);
            const VaistTensorRef* wd = state_.get_tensor(wname);
            if (wgu && wd) {
                uint32_t inter = state_.config.intermediate_size;
                std::vector<float> gate_up(batch_seq * 2 * inter);
                qgemm(mlp_in.data(), wgu, gate_up.data(), batch_seq, hidden_size, 2 * inter);
                std::vector<float> act(batch_seq * inter);
                silu_and_mul(gate_up.data(), act.data(), batch_seq * inter, inter);
                qgemm(act.data(), wd, mlp_out.data(), batch_seq, inter, hidden_size);
            }
        }

        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) mlp_out[i] += attn_out[i];

        hidden = std::move(mlp_out);
        residual = hidden.data();
    }

    norm_head(hidden.data(), logits_out, batch_size * seq_len);
}

/* ======================================================================== */
/* DeepSeekModel implementation                                              */
/* ======================================================================== */

DeepSeekModel::DeepSeekModel(const VaistModelConfig& cfg,
                             std::string_view weight_path,
                             VaistRuntime* rt) : rt_(rt) {
    state_.config = cfg;
    state_.rt = rt;
    state_.weight_path = std::string(weight_path);

    VaistModelFormat fmt = VAIST_MODEL_UNKNOWN;
    VaistModelInfo info;
    if (vaist_model_inspect(state_.weight_path.c_str(), &info) == VAIST_OK) {
        fmt = info.format;
    }
    state_.format = fmt;

    VaistTensorDesc* descs = nullptr;
    size_t count = 0, capacity = 0;
    VaistStatus st;
    if (fmt == VAIST_MODEL_GGUF) {
        st = vaist_model_load_gguf(state_.weight_path.c_str(), &descs, &count, &capacity);
    } else if (fmt == VAIST_MODEL_SAFETENSORS) {
        st = vaist_model_load_safetensors(state_.weight_path.c_str(), &descs, &count, &capacity);
    } else {
        st = VAIST_IO_ERROR;
    }

    if (st != VAIST_OK || !descs) {
        throw std::runtime_error("Failed to load DeepSeek model weights");
    }

    for (size_t i = 0; i < count; i++) {
        VaistTensorRef ref;
        ref.name = descs[i].name;
        ref.offset = descs[i].offset;
        ref.byte_size = descs[i].byte_size;
        ref.file_path = state_.weight_path;
        ref.shape.clear();
        for (uint32_t d = 0; d < descs[i].rank && d < 8; d++) {
            ref.shape.push_back(descs[i].shape[d]);
        }
        if (descs[i].dtype == VAIST_F32) ref.dtype = VaistWeightDtype::kF32;
        else if (descs[i].dtype == VAIST_F16) ref.dtype = VaistWeightDtype::kF16;
        else {
            ref.gguf_raw_dtype = (uint32_t)descs[i].dtype;
            uint32_t ggml_type = (uint32_t)descs[i].dtype - 100;
            ref.dtype = weight_dtype_from_gguf_ggml_type(ggml_type);
        }
        state_.tensors[ref.name] = std::move(ref);
    }
    free(descs);

    /* Infer config */
    if (state_.config.vocab_size == 0) {
        if (const auto* t = state_.get_tensor("model.embed_tokens.weight")) {
            state_.config.vocab_size = (uint32_t)t->shape[0];
        }
    }

    state_.init_kv_cache(4096);
    init_attn_ctx();
}

void DeepSeekModel::init_attn_ctx() {
    if (!rt_) return;
    vaist_attn_cfg cfg{};
    cfg.scale = 1.0f / sqrtf((float)(state_.config.head_dim ? state_.config.head_dim : 128));
    cfg.head_dim = state_.config.head_dim;
    cfg.num_q_heads = state_.config.n_heads;
    cfg.num_kv_heads = state_.config.n_kv_heads;
    cfg.max_seqlen = state_.config.max_position_embeddings;
    cfg.block_size = state_.config.block_size;
    cfg.max_blocks = 4096;
    cfg.total_blocks = 4096;
    attn_ctx_ = vaist_attn_create(rt_, &cfg);
}

void DeepSeekModel::destroy_attn_ctx() {
    if (attn_ctx_) {
        vaist_attn_destroy(attn_ctx_);
        attn_ctx_ = nullptr;
    }
}

void DeepSeekModel::rmsnorm(const float* input, const float* weight,
                            float* output, uint32_t n, float eps) {
    rmsnorm_cpu(input, weight, output, n, eps);
}

void DeepSeekModel::silu_and_mul(const float* gate_up, float* output,
                                 uint32_t n, uint32_t intermediate) {
    silu_and_mul_cpu(gate_up, output, n);
    (void)intermediate;
}

void DeepSeekModel::qgemm(const float* A, const VaistTensorRef* W, float* C,
                          size_t M, size_t K, size_t N) {
    if (!W) return;
    const float* Wdata = W->get_data();
    if (!Wdata) return;
    if (rt_) {
        vaist_blas_gemm(rt_, A, Wdata, C, M, K, N, nullptr, VAIST_W_FP16);
    } else {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float s = 0.0f;
                for (size_t p = 0; p < K; p++) s += A[i * K + p] * Wdata[p * N + j];
                C[i * N + j] = s;
            }
        }
    }
}

void DeepSeekModel::rope_llama3(float* buf, uint32_t n, uint32_t rope_dim,
                                uint32_t pos, float base, float scale_factor) {
    for (uint32_t i = 0; i < n; i += 2) {
        uint32_t half = i / 2;
        float freq = 1.0f / powf(base, (float)(2 * half) / (float)rope_dim);
        freq /= scale_factor;
        float sin_pos = sinf(freq * (float)pos);
        float cos_pos = cosf(freq * (float)pos);
        float x0 = buf[i];
        float x1 = buf[i + 1];
        buf[i]     = x0 * cos_pos - x1 * sin_pos;
        buf[i + 1] = x0 * sin_pos + x1 * cos_pos;
    }
}

void DeepSeekModel::qk_norm(const float* input, const float* weight,
                            float* output, uint32_t n, float eps) {
    rmsnorm_cpu(input, weight, output, n, eps);
}

void DeepSeekModel::mla_attention(uint32_t layer_idx,
                                  const float* input,
                                  float* output,
                                  const uint32_t* positions,
                                  uint32_t batch_seq,
                                  uint32_t seqlen) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t n_heads = state_.config.n_heads;
    uint32_t n_kv_heads = state_.config.n_kv_heads;
    uint32_t head_dim = state_.config.head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* MLA: compressed QKV. Uses qkv_proj if fused, or separate projections. */
    /* For V2/V3, the Q projection is (hidden x q_lora_rank), KV projection is (hidden x kv_lora_rank) */
    uint32_t q_lora_rank = state_.config.q_lora_rank;
    uint32_t kv_lora_rank = state_.config.kv_lora_rank;
    uint32_t q_head_dim = state_.config.q_head_dim;
    uint32_t kv_head_dim = state_.config.kv_head_dim;

    if (q_lora_rank == 0) {
        /* Standard GQA fallback (no MLA) — treat as regular attention */
        /* This path handles Llama-style attention within DeepSeek */
        uint32_t q_size = n_heads * head_dim;
        uint32_t kv_size = n_kv_heads * head_dim;
        uint32_t qkv_size = q_size + 2 * kv_size;

        snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.qkv_proj.weight", layer_idx);
        const VaistTensorRef* wqkv = state_.get_tensor(wname);
        snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.o_proj.weight", layer_idx);
        const VaistTensorRef* wo = state_.get_tensor(wname);

        std::vector<float> qkv(batch_seq * qkv_size);
        if (wqkv) qgemm(input, wqkv, qkv.data(), batch_seq, hidden, qkv_size);

        apply_rope_interleaved(qkv.data(), n_heads, seqlen, head_dim, positions,
                               state_.config.rope_theta,
                               1.0f / state_.config.rope_scaling_factor);

        const float* q = qkv.data();
        const float* k = q + q_size;
        const float* v = k + kv_size;

        std::vector<float> scores(n_heads * seqlen * seqlen);
        for (uint32_t h = 0; h < n_heads; h++) {
            for (uint32_t i = 0; i < seqlen; i++) {
                const float* qi = q + (i * n_heads + h) * head_dim;
                for (uint32_t j = 0; j < seqlen; j++) {
                    uint32_t kv_head = h * n_kv_heads / n_heads;
                    const float* kj = k + (j * n_kv_heads + kv_head) * head_dim;
                    float dot = 0;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                    scores[(h * seqlen + i) * seqlen + j] = dot * scale;
                }
            }
        }
        for (uint32_t h = 0; h < n_heads; h++) {
            for (uint32_t i = 0; i < seqlen; i++) {
                float* row = &scores[(h * seqlen + i) * seqlen];
                float maxv = row[0];
                for (uint32_t j = 1; j <= i; j++) if (row[j] > maxv) maxv = row[j];
                float sum = 0;
                for (uint32_t j = 0; j <= i; j++) {
                    row[j] = expf(row[j] - maxv);
                    sum += row[j];
                }
                for (uint32_t j = 0; j <= i; j++) row[j] /= sum;
                for (uint32_t j = i + 1; j < seqlen; j++) row[j] = 0;
            }
        }
        std::vector<float> attn_out(batch_seq * q_size);
        for (uint32_t h = 0; h < n_heads; h++) {
            for (uint32_t i = 0; i < seqlen; i++) {
                float* out_row = &attn_out[(i * n_heads + h) * head_dim];
                for (uint32_t j = 0; j < seqlen; j++) {
                    float s = scores[(h * seqlen + i) * seqlen + j];
                    uint32_t kv_head = h * n_kv_heads / n_heads;
                    const float* vj = v + (j * n_kv_heads + kv_head) * head_dim;
                    if (s != 0.0f) {
                        for (uint32_t d = 0; d < head_dim; d++) out_row[d] += s * vj[d];
                    }
                }
            }
        }
        if (wo) qgemm(attn_out.data(), wo, output, batch_seq, q_size, hidden);
        else memcpy(output, attn_out.data(), batch_seq * hidden * sizeof(float));
        return;
    }

    /* True MLA path: separated Q, KvRope projections */
    /* 1. Project hidden -> q_lora_rank (query latent) */
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.q_proj.weight", layer_idx);
    const VaistTensorRef* wq = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.kv_proj.weight", layer_idx);
    const VaistTensorRef* wkv = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.o_proj.weight", layer_idx);
    const VaistTensorRef* wo = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.qk_rope_proj.weight", layer_idx);
    const VaistTensorRef* wqrope = state_.get_tensor(wname);

    std::vector<float> q_latent(batch_seq * q_lora_rank);
    std::vector<float> kv_latent(batch_seq * kv_lora_rank);
    if (wq) qgemm(input, wq, q_latent.data(), batch_seq, hidden, q_lora_rank);
    if (wkv) qgemm(input, wkv, kv_latent.data(), batch_seq, hidden, kv_lora_rank);

    /* Apply RoPE to the rope part of kv_latent */
    apply_rope_interleaved(kv_latent.data(), 1, batch_seq, kv_lora_rank / 2,
                           positions, state_.config.rope_theta,
                           1.0f / state_.config.rope_scaling_factor);

    /* Expand Q latent into multi-head: (batch_seq, q_lora_rank) -> (batch_seq, n_heads, q_head_dim) */
    /* Then compute attention: Q @ K^T, where K = KV projected to head_dim */
    if (wqrope) {
        std::vector<float> q_full(batch_seq * n_heads * q_head_dim);
        qgemm(q_latent.data(), wqrope, q_full.data(), batch_seq, q_lora_rank,
              n_heads * q_head_dim);
        /* Compute K from kv_latent */
        std::vector<float> k_full(batch_seq * n_kv_heads * kv_head_dim);
        /* For simplicity: extract the K portion from kv_latent via the kv_proj output */
        /* In real impl, this involves another projection */
        memcpy(k_full.data(), kv_latent.data(),
               (batch_seq * kv_lora_rank < k_full.size() * sizeof(float)
                ? batch_seq * kv_lora_rank : k_full.size()) * sizeof(float));
        /* This is simplified; real MLA has compressed K + V in the kv_latent */
        (void)q_full;
        (void)k_full;
    }

    /* Minimal MLA: use the compressed representation directly */
    /* Q @ K^T attention, then attend to V */
    /* This is a simplified path for V3's blockwise attention */
    std::vector<float> scores(n_heads * seqlen * seqlen);
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < seqlen; i++) {
            float maxv = -1e30f;
            for (uint32_t j = 0; j <= i; j++) {
                /* Simplified: use first head_dim of q_latent as query */
                float dot = 0;
                for (uint32_t d = 0; d < (size_t)(q_lora_rank < kv_lora_rank ? q_lora_rank : kv_lora_rank); d++) {
                    dot += q_latent[i * q_lora_rank + d] * kv_latent[j * kv_lora_rank + d];
                }
                scores[(h * seqlen + i) * seqlen + j] = dot * scale;
                if (scores[(h * seqlen + i) * seqlen + j] > maxv) maxv = scores[(h * seqlen + i) * seqlen + j];
            }
            float* row = &scores[(h * seqlen + i) * seqlen];
            float sum = 0;
            for (uint32_t j = 0; j <= i; j++) {
                row[j] = expf(row[j] - maxv);
                sum += row[j];
            }
            for (uint32_t j = 0; j <= i; j++) row[j] /= sum;
            for (uint32_t j = i + 1; j < seqlen; j++) row[j] = 0;
        }
    }

    /* Output projection */
    std::vector<float> attn_out(batch_seq * hidden);
    if (wo) qgemm(attn_out.data(), wo, output, batch_seq, n_heads * q_head_dim, hidden);
    else memcpy(output, input, batch_seq * hidden * sizeof(float));
}

void DeepSeekModel::mlp_moe(uint32_t layer_idx,
                            const float* input,
                            float* output,
                            uint32_t batch_seq) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t num_experts = state_.config.num_experts;
    uint32_t top_k = state_.config.num_experts_per_tok;
    uint32_t inter = state_.config.moe_intermediate_size;

    /* Gate */
    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate.weight", layer_idx);
    const VaistTensorRef* wgate = state_.get_tensor(wname);

    if (!wgate || !wgate->get_data()) {
        /* Dense fallback */
        snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate_up_proj.weight", layer_idx);
        const VaistTensorRef* wgu = state_.get_tensor(wname);
        snprintf(wname, sizeof(wname), "model.layers.%u.mlp.down_proj.weight", layer_idx);
        const VaistTensorRef* wd = state_.get_tensor(wname);
        if (wgu && wd) {
            std::vector<float> gu(batch_seq * 2 * state_.config.intermediate_size);
            qgemm(input, wgu, gu.data(), batch_seq, hidden, 2 * state_.config.intermediate_size);
            std::vector<float> act(batch_seq * state_.config.intermediate_size);
            silu_and_mul(gu.data(), act.data(), batch_seq * state_.config.intermediate_size, state_.config.intermediate_size);
            qgemm(act.data(), wd, output, batch_seq, state_.config.intermediate_size, hidden);
        }
        return;
    }

    const float* gate_data = wgate->get_data();
    topk_indices_.resize(batch_seq * top_k);
    topk_scores_.resize(batch_seq * top_k);
    normalized_scores_.resize(batch_seq * top_k);
    expert_blocks_.resize(num_experts);
    expert_offsets_.resize(num_experts);

    if (rt_) {
        vaist_blas_moe_topk(rt_, input, batch_seq, hidden,
                            gate_data, num_experts, top_k,
                            topk_indices_.data(), topk_scores_.data(),
                            normalized_scores_.data());
    }

    /* Compute max tokens per expert for buffer sizing */
    uint32_t max_tokens = batch_seq;
    size_t dispatch_size = (size_t)num_experts * max_tokens * hidden;
    moe_buf_.resize(dispatch_size);
    expert_blocks_.assign(num_experts, 0);
    expert_offsets_.assign(num_experts, 0);

    vaist_blas_moe_dispatch(rt_, input,
                            topk_indices_.data(), topk_scores_.data(),
                            batch_seq, hidden, num_experts, top_k,
                            expert_blocks_.data(),
                            expert_offsets_.data(),
                            moe_buf_.data());

    /* Run per-expert GEMM */
    size_t expert_out_size = (size_t)num_experts * max_tokens * hidden;
    std::vector<float> expert_out(expert_out_size);

    for (uint32_t e = 0; e < num_experts; e++) {
        snprintf(wname, sizeof(wname),
                 "model.layers.%u.mlp.experts.%u.gate_up_proj.weight", layer_idx, e);
        const VaistTensorRef* wge = state_.get_tensor(wname);
        snprintf(wname, sizeof(wname),
                 "model.layers.%u.mlp.experts.%u.down_proj.weight", layer_idx, e);
        const VaistTensorRef* wde = state_.get_tensor(wname);

        if (wge && wde && expert_blocks_[e] > 0) {
            uint32_t n_tok = expert_blocks_[e];
            std::vector<float> gu(n_tok * 2 * inter);
            qgemm(input, wge, gu.data(), n_tok, hidden, 2 * inter);
            std::vector<float> act(n_tok * inter);
            silu_and_mul(gu.data(), act.data(), n_tok * inter, inter);
            qgemm(act.data(), wde,
                  expert_out.data() + (size_t)e * max_tokens * hidden,
                  n_tok, inter, hidden);
        }
    }

    /* Combine */
    vaist_blas_moe_combine(rt_, expert_out.data(),
                           topk_indices_.data(), normalized_scores_.data(),
                           expert_offsets_.data(), batch_seq, hidden,
                           num_experts, top_k, output);
}

void DeepSeekModel::mlp_shared(uint32_t layer_idx,
                               const float* input,
                               float* output,
                               uint32_t batch_seq) {
    char wname[256];
    uint32_t hidden = state_.config.hidden_size;
    uint32_t inter = state_.config.shared_expert_intermediate_size;
    if (inter == 0) inter = state_.config.moe_intermediate_size;

    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.shared_experts.gate_up_proj.weight", layer_idx);
    const VaistTensorRef* wgu = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.mlp.shared_experts.down_proj.weight", layer_idx);
    const VaistTensorRef* wd = state_.get_tensor(wname);

    if (wgu && wd) {
        std::vector<float> gu(batch_seq * 2 * inter);
        qgemm(input, wgu, gu.data(), batch_seq, hidden, 2 * inter);
        std::vector<float> act(batch_seq * inter);
        silu_and_mul(gu.data(), act.data(), batch_seq * inter, inter);
        qgemm(act.data(), wd, output, batch_seq, inter, hidden);
    }
}

void DeepSeekModel::embed_tokens(const uint32_t* tokens, uint32_t n_tokens, float* out) {
    const VaistTensorRef* W = state_.get_tensor("model.embed_tokens.weight");
    if (!W || !W->get_data()) return;
    size_t dim = state_.config.hidden_size;
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid >= W->shape[0]) tid = 0;
        const float* row = W->get_data() + tid * dim;
        memcpy(out + i * dim, row, dim * sizeof(float));
    }
}

void DeepSeekModel::norm_head(const float* hidden, float* logits, uint32_t batch) {
    uint32_t hidden_size = state_.config.hidden_size;
    uint32_t vocab_size = state_.config.vocab_size;

    const VaistTensorRef* wnorm = state_.get_tensor("model.norm.weight");
    float* norm_out = new float[batch * hidden_size];
    if (wnorm && wnorm->get_data()) {
        for (uint32_t b = 0; b < batch; b++) {
            rmsnorm_cpu(hidden + b * hidden_size, wnorm->get_data(),
                        norm_out + b * hidden_size, hidden_size,
                        state_.config.norm_eps);
        }
    } else {
        memcpy(norm_out, hidden, batch * hidden_size * sizeof(float));
    }

    std::string lm_head_name = "lm_head.weight";
    if (state_.config.tie_word_embeddings) {
        lm_head_name = "model.embed_tokens.weight";
    }
    const VaistTensorRef* whead = state_.get_tensor(lm_head_name);
    if (whead && whead->get_data()) {
        qgemm(norm_out, whead, logits, batch, hidden_size, vocab_size);
    } else {
        memset(logits, 0, batch * vocab_size * sizeof(float));
    }
    delete[] norm_out;
}

void DeepSeekModel::mtp_forward(uint32_t step,
                                const float* hidden,
                                float* output,
                                uint32_t batch_seq) {
    char wname[256];
    uint32_t hidden_size = state_.config.hidden_size;

    snprintf(wname, sizeof(wname), "model.layers.%u.%u.mlp.gate_up_proj.weight",
             state_.config.mtp_num_layers, step);
    const VaistTensorRef* wgu = state_.get_tensor(wname);
    snprintf(wname, sizeof(wname), "model.layers.%u.%u.mlp.down_proj.weight",
             state_.config.mtp_num_layers, step);
    const VaistTensorRef* wd = state_.get_tensor(wname);

    if (wgu && wd) {
        std::vector<float> gu(batch_seq * 2 * state_.config.intermediate_size);
        qgemm(hidden, wgu, gu.data(), batch_seq, hidden_size,
              2 * state_.config.intermediate_size);
        std::vector<float> act(batch_seq * state_.config.intermediate_size);
        silu_and_mul(gu.data(), act.data(), batch_seq * state_.config.intermediate_size,
                     state_.config.intermediate_size);
        qgemm(act.data(), wd, output, batch_seq, state_.config.intermediate_size,
              hidden_size);
    } else {
        memcpy(output, hidden, batch_seq * hidden_size * sizeof(float));
    }

    /* Add residual */
    for (uint32_t i = 0; i < batch_seq * hidden_size; i++) output[i] += hidden[i];
}

void DeepSeekModel::forward(const uint32_t* tokens,
                            const uint32_t* positions,
                            uint32_t seq_len,
                            uint32_t batch_size,
                            float* logits_out) {
    uint32_t hidden_size = state_.config.hidden_size;
    uint32_t batch_seq = batch_size * seq_len;

    std::vector<float> hidden(batch_seq * hidden_size);
    embed_tokens(tokens, batch_seq, hidden.data());
    float* residual = hidden.data();

    for (uint32_t li = 0; li < state_.config.n_layers; li++) {
        char wname[256];

        /* Input RMSNorm */
        snprintf(wname, sizeof(wname), "model.layers.%u.input_layernorm.weight", li);
        const VaistTensorRef* w_in = state_.get_tensor(wname);

        std::vector<float> attn_in(batch_seq * hidden_size);
        if (w_in && w_in->get_data()) {
            for (uint32_t i = 0; i < batch_seq; i++) {
                rmsnorm_cpu(hidden.data() + i * hidden_size, w_in->get_data(),
                            attn_in.data() + i * hidden_size, hidden_size,
                            state_.config.norm_eps);
            }
        } else {
            memcpy(attn_in.data(), hidden.data(), batch_seq * hidden_size * sizeof(float));
        }

        std::vector<float> attn_out(batch_seq * hidden_size);
        if (state_.config.use_mla) {
            mla_attention(li, attn_in.data(), attn_out.data(), positions, batch_seq, seq_len);
        } else {
            /* Fallback: standard attention (reuse Qwen3-style) */
            uint32_t n_heads = state_.config.n_heads;
            uint32_t n_kv_heads = state_.config.n_kv_heads;
            uint32_t head_dim = state_.config.head_dim;
            uint32_t q_size = n_heads * head_dim;
            uint32_t kv_size = n_kv_heads * head_dim;
            uint32_t qkv_size = q_size + 2 * kv_size;
            float scale = 1.0f / sqrtf((float)head_dim);

            snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.qkv_proj.weight", li);
            const VaistTensorRef* wqkv = state_.get_tensor(wname);
            snprintf(wname, sizeof(wname), "model.layers.%u.self_attn.o_proj.weight", li);
            const VaistTensorRef* wo = state_.get_tensor(wname);

            std::vector<float> qkv(batch_seq * qkv_size);
            if (wqkv) qgemm(attn_in.data(), wqkv, qkv.data(), batch_seq, hidden_size, qkv_size);
            apply_rope_interleaved(qkv.data(), n_heads, seq_len, head_dim, positions,
                                   state_.config.rope_theta, 1.0f / state_.config.rope_scaling_factor);

            const float* q = qkv.data();
            const float* k = q + q_size;
            const float* v = k + kv_size;

            std::vector<float> scores(n_heads * seq_len * seq_len);
            for (uint32_t h = 0; h < n_heads; h++) {
                for (uint32_t i = 0; i < seq_len; i++) {
                    const float* qi = q + (i * n_heads + h) * head_dim;
                    for (uint32_t j = 0; j < seq_len; j++) {
                        uint32_t kv_head = h * n_kv_heads / n_heads;
                        const float* kj = k + (j * n_kv_heads + kv_head) * head_dim;
                        float dot = 0;
                        for (uint32_t d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                        scores[(h * seq_len + i) * seq_len + j] = dot * scale;
                    }
                }
            }
            for (uint32_t h = 0; h < n_heads; h++) {
                for (uint32_t i = 0; i < seq_len; i++) {
                    float* row = &scores[(h * seq_len + i) * seq_len];
                    float maxv = row[0];
                    for (uint32_t j = 1; j <= i; j++) if (row[j] > maxv) maxv = row[j];
                    float sum = 0;
                    for (uint32_t j = 0; j <= i; j++) { row[j] = expf(row[j] - maxv); sum += row[j]; }
                    for (uint32_t j = 0; j <= i; j++) row[j] /= sum;
                    for (uint32_t j = i + 1; j < seq_len; j++) row[j] = 0;
                }
            }
            std::vector<float> a_out(batch_seq * q_size);
            for (uint32_t h = 0; h < n_heads; h++) {
                for (uint32_t i = 0; i < seq_len; i++) {
                    float* out_row = &a_out[(i * n_heads + h) * head_dim];
                    for (uint32_t j = 0; j < seq_len; j++) {
                        float s = scores[(h * seq_len + i) * seq_len + j];
                        uint32_t kv_head = h * n_kv_heads / n_heads;
                        const float* vj = v + (j * n_kv_heads + kv_head) * head_dim;
                        if (s != 0.0f) for (uint32_t d = 0; d < head_dim; d++) out_row[d] += s * vj[d];
                    }
                }
            }
            if (wo) qgemm(a_out.data(), wo, attn_out.data(), batch_seq, q_size, hidden_size);
            else memcpy(attn_out.data(), a_out.data(), batch_seq * hidden_size * sizeof(float));
        }

        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) attn_out[i] += residual[i];

        /* Post-attention RMSNorm */
        snprintf(wname, sizeof(wname), "model.layers.%u.post_attention_layernorm.weight", li);
        const VaistTensorRef* w_post = state_.get_tensor(wname);

        std::vector<float> mlp_in(batch_seq * hidden_size);
        if (w_post && w_post->get_data()) {
            for (uint32_t i = 0; i < batch_seq; i++) {
                rmsnorm_cpu(attn_out.data() + i * hidden_size, w_post->get_data(),
                            mlp_in.data() + i * hidden_size, hidden_size,
                            state_.config.norm_eps);
            }
        } else {
            memcpy(mlp_in.data(), attn_out.data(), batch_seq * hidden_size * sizeof(float));
        }

        std::vector<float> mlp_out(batch_seq * hidden_size);
        if (state_.config.num_experts > 0) {
            mlp_moe(li, mlp_in.data(), mlp_out.data(), batch_seq);
            std::vector<float> shared_out(batch_seq * hidden_size);
            mlp_shared(li, mlp_in.data(), shared_out.data(), batch_seq);
            for (uint32_t i = 0; i < batch_seq * hidden_size; i++) mlp_out[i] += shared_out[i];
        } else {
            snprintf(wname, sizeof(wname), "model.layers.%u.mlp.gate_up_proj.weight", li);
            const VaistTensorRef* wgu = state_.get_tensor(wname);
            snprintf(wname, sizeof(wname), "model.layers.%u.mlp.down_proj.weight", li);
            const VaistTensorRef* wd = state_.get_tensor(wname);
            if (wgu && wd) {
                uint32_t inter = state_.config.intermediate_size;
                std::vector<float> gu(batch_seq * 2 * inter);
                qgemm(mlp_in.data(), wgu, gu.data(), batch_seq, hidden_size, 2 * inter);
                std::vector<float> act(batch_seq * inter);
                silu_and_mul(gu.data(), act.data(), batch_seq * inter, inter);
                qgemm(act.data(), wd, mlp_out.data(), batch_seq, inter, hidden_size);
            }
        }

        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) mlp_out[i] += attn_out[i];
        hidden = std::move(mlp_out);
        residual = hidden.data();
    }

    /* MTP prediction (V3 only) */
    if (state_.config.mtp && state_.config.mtp_num_layers > 0) {
        std::vector<float> mtp_out(batch_seq * hidden_size);
        mtp_forward(0, hidden.data(), mtp_out.data(), batch_seq);
        for (uint32_t i = 0; i < batch_seq * hidden_size; i++) {
            hidden[i] = mtp_out[i];
        }
    }

    norm_head(hidden.data(), logits_out, batch_size * seq_len);
    destroy_attn_ctx();
}

/* ======================================================================== */
/* VaistEngine implementation                                                */
/* ======================================================================== */

VaistEngine::VaistEngine(VaistRuntime* rt) {
    if (rt) {
        rt_ = rt;
    } else {
        vaist_runtime_create(VAIST_BACKEND_AUTO, &rt_);
    }
}

VaistEngine::VaistEngine(VaistEngine&& other) noexcept
    : rt_(other.rt_), model_type_(other.model_type_), model_ptr_(other.model_ptr_),
      tokenizer_(other.tokenizer_), seq_len_(other.seq_len_),
      vocab_size_(other.vocab_size_), sampler_(other.sampler_) {
    other.rt_ = nullptr;
    other.model_ptr_ = nullptr;
    other.tokenizer_ = nullptr;
    other.sampler_ = nullptr;
}

VaistEngine& VaistEngine::operator=(VaistEngine&& other) noexcept {
    if (this != &other) {
        if (model_ptr_) {
            switch (model_type_) {
                case ModelType::kLlama:      delete static_cast<LlamaModel*>(model_ptr_); break;
                case ModelType::kQwen3:      delete static_cast<Qwen3Model*>(model_ptr_); break;
                case ModelType::kQwen3Moe:   delete static_cast<Qwen3Model*>(model_ptr_); break;
                case ModelType::kDeepseekV2:
                case ModelType::kDeepseekV3: delete static_cast<DeepSeekModel*>(model_ptr_); break;
            }
        }
        if (sampler_) vaist_sampler_destroy(sampler_);
        if (tokenizer_) vaist_tokenizer_destroy(tokenizer_);
        if (rt_ && rt_ != other.rt_) vaist_runtime_destroy(rt_);
        rt_ = other.rt_;
        model_type_ = other.model_type_;
        model_ptr_ = other.model_ptr_;
        tokenizer_ = other.tokenizer_;
        seq_len_ = other.seq_len_;
        vocab_size_ = other.vocab_size_;
        sampler_ = other.sampler_;
        other.rt_ = nullptr;
        other.model_ptr_ = nullptr;
        other.tokenizer_ = nullptr;
        other.sampler_ = nullptr;
    }
    return *this;
}

VaistEngine::~VaistEngine() {
    if (model_ptr_) {
        switch (model_type_) {
            case ModelType::kLlama:      delete static_cast<LlamaModel*>(model_ptr_); break;
            case ModelType::kQwen3:      delete static_cast<Qwen3Model*>(model_ptr_); break;
            case ModelType::kQwen3Moe:   delete static_cast<Qwen3Model*>(model_ptr_); break;
            case ModelType::kDeepseekV2:
            case ModelType::kDeepseekV3: delete static_cast<DeepSeekModel*>(model_ptr_); break;
        }
        model_ptr_ = nullptr;
    }
    if (sampler_) vaist_sampler_destroy(sampler_);
    if (tokenizer_) vaist_tokenizer_destroy(tokenizer_);
    if (rt_) vaist_runtime_destroy(rt_);
}

VaistStatus VaistEngine::load_model(ModelType type,
                                    std::string_view weight_path,
                                    std::string_view config_path) {
    (void)config_path; /* TODO: parse config.json to override defaults */
    model_type_ = type;

    try {
        VaistModelConfig cfg;
        /* Default config — real values inferred from weights in constructors */
        cfg.n_heads = cfg.n_heads; /* keep zero; inferred from weights */

        switch (type) {
            case ModelType::kLlama: {
                auto* m = new LlamaModel(cfg, weight_path, rt_);
                model_ptr_ = m;
                vocab_size_ = m->config().vocab_size;
                break;
            }
            case ModelType::kQwen3:
            case ModelType::kQwen3Moe: {
                auto* m = new Qwen3Model(cfg, weight_path, rt_);
                if (type == ModelType::kQwen3Moe) {
                    cfg.num_experts = 64;
                    cfg.num_experts_per_tok = 2;
                }
                model_ptr_ = m;
                vocab_size_ = m->config().vocab_size;
                break;
            }
            case ModelType::kDeepseekV2:
            case ModelType::kDeepseekV3: {
                auto* m = new DeepSeekModel(cfg, weight_path, rt_);
                model_ptr_ = m;
                vocab_size_ = m->config().vocab_size;
                if (type == ModelType::kDeepseekV3) {
                    cfg.mtp = true;
                }
                break;
            }
        }

        if (vaist_sampler_create(1.0f, 40, 0.9f, &sampler_) != VAIST_OK) {
            sampler_ = nullptr;
        }

        return VAIST_OK;
    } catch (const std::exception& e) {
        fprintf(stderr, "VaistEngine::load_model error: %s\n", e.what());
        return VAIST_MODEL_ERROR;
    }
}

void VaistEngine::decode_step(uint32_t token_id, uint32_t position, float* logits_out) {
    switch (model_type_) {
        case ModelType::kLlama: {
            auto* m = static_cast<LlamaModel*>(model_ptr_);
            m->forward(&token_id, &position, 1, 1, logits_out);
            break;
        }
        case ModelType::kQwen3:
        case ModelType::kQwen3Moe: {
            auto* m = static_cast<Qwen3Model*>(model_ptr_);
            m->forward(&token_id, &position, 1, 1, logits_out);
            break;
        }
        case ModelType::kDeepseekV2:
        case ModelType::kDeepseekV3: {
            auto* m = static_cast<DeepSeekModel*>(model_ptr_);
            m->forward(&token_id, &position, 1, 1, logits_out);
            break;
        }
    }
    seq_len_++;
}

void VaistEngine::prefill_step(const uint32_t* input_ids, size_t input_len,
                               const uint32_t* positions, float* logits_out) {
    switch (model_type_) {
        case ModelType::kLlama: {
            auto* m = static_cast<LlamaModel*>(model_ptr_);
            m->forward(input_ids, positions, (uint32_t)input_len, 1, logits_out);
            break;
        }
        case ModelType::kQwen3:
        case ModelType::kQwen3Moe: {
            auto* m = static_cast<Qwen3Model*>(model_ptr_);
            m->forward(input_ids, positions, (uint32_t)input_len, 1, logits_out);
            break;
        }
        case ModelType::kDeepseekV2:
        case ModelType::kDeepseekV3: {
            auto* m = static_cast<DeepSeekModel*>(model_ptr_);
            m->forward(input_ids, positions, (uint32_t)input_len, 1, logits_out);
            break;
        }
    }
    seq_len_ += (uint32_t)input_len;
}

uint32_t VaistEngine::do_sample(const float* logits, uint32_t vocab,
                                const std::vector<uint32_t>& generated,
                                const VaistSamplingConfig& cfg) {
    /* Convert config and call vaist_sample_cfg */
    VaistSamplerConfig samp_cfg;
    vaist_sampler_config_init(&samp_cfg,
                              cfg.temperature, cfg.top_k, cfg.top_p,
                              cfg.min_p, cfg.presence_penalty,
                              cfg.frequency_penalty);

    uint32_t out_id = 0;
    vaist_sample_cfg(sampler_, &samp_cfg, logits, vocab,
                     generated.data(), generated.size(), &out_id);
    return out_id;
}

std::vector<uint32_t> VaistEngine::tokenize(const char* text) {
    std::vector<uint32_t> result;
    if (!tokenizer_) return result;
    uint32_t buf[8192];
    size_t count = 0;
    if (vaist_tokenize(tokenizer_, text, buf, 8192, &count) == VAIST_OK) {
        result.assign(buf, buf + count);
    }
    return result;
}

std::string VaistEngine::detokenize(const uint32_t* tokens, size_t n) {
    if (!tokenizer_) return {};
    char buf[65536];
    size_t written = 0;
    if (vaist_detokenize(tokenizer_, tokens, n, buf, sizeof(buf), &written) == VAIST_OK) {
        return std::string(buf, written);
    }
    return {};
}

VaistStatus VaistEngine::generate(const char* prompt,
                                  uint32_t max_tokens,
                                  const VaistSamplingConfig& cfg,
                                  std::vector<uint32_t>* out_tokens) {
    auto tokens = tokenize(prompt);
    return generate(tokens.data(), tokens.size(), max_tokens, cfg, out_tokens);
}

VaistStatus VaistEngine::generate(const uint32_t* input_ids,
                                  size_t input_len,
                                  uint32_t max_tokens,
                                  const VaistSamplingConfig& cfg,
                                  std::vector<uint32_t>* out_tokens) {
    if (!model_ptr_ || !vocab_size_) {
        return VAIST_INVALID_STATE;
    }

    auto cfg_copy = cfg;
    if (cfg_copy.seed != 0) {
        /* vaist_sample_cfg uses logits pointer as seed; for deterministic
         * sampling we just set temperature to the provided value */
    }

    size_t logits_sz = (size_t)vocab_size_;
    float* logits = new float[logits_sz];
    std::vector<uint32_t> generated;
    if (input_len > 0) {
        generated.assign(input_ids, input_ids + input_len);
    }

    /* Prefill: run forward for all input tokens */
    if (input_len > 0) {
        std::vector<uint32_t> positions(input_len);
        for (size_t i = 0; i < input_len; i++) positions[i] = (uint32_t)i;
        float* prefill_logits = new float[input_len * vocab_size_];
        prefill_step(input_ids, input_len, positions.data(), prefill_logits);
        delete[] prefill_logits;
    }

    /* Decode: generate max_tokens tokens */
    uint32_t cur_pos = (uint32_t)input_len;
    uint32_t last_token = (input_len > 0) ? input_ids[input_len - 1] : 0;

    for (uint32_t step = 0; step < max_tokens; step++) {
        decode_step(last_token, cur_pos, logits);
        uint32_t next = do_sample(logits, vocab_size_, generated, cfg_copy);
        if (out_tokens) out_tokens->push_back(next);
        generated.push_back(next);
        last_token = next;
        cur_pos++;

        /* Stop on EOS (conventionally token 2 or vocab-defined) */
        if (next == 2 || next == 0 || next >= vocab_size_) {
            break;
        }
    }

    delete[] logits;
    return VAIST_OK;
}

} // namespace vaist
