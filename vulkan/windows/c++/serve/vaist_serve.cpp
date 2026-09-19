/**
 * \file vaist_serve.cpp
 * \brief Main entry point for the VAiSt OpenAI-compatible HTTP server.
 *
 * Parses command-line arguments, loads a model via VaistEngine, and starts
 * the HTTP server listening for requests on the specified port.
 *
 * Usage:
 *   vaist_serve.exe --model <path> [--port 8080] [--address 0.0.0.0]
 *                   [--max-tokens 128] [--temperature 0.7]
 *                   [--top-p 0.9] [--top-k 40]
 *
 * Send SIGINT (Ctrl+C) or SIGTERM to shut down gracefully.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "vaist_http_server.hpp"
#include "vaist_engine.hpp"
#include "vaist_runtime.h"
#include "vaist_tokens.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <csignal>
#include <chrono>
#include <thread>

using namespace vaist;

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int sig) {
    (void)sig;
    g_shutdown.store(true);
}

/**
 * \brief Parse command-line arguments.
 * \param argc  Argument count.
 * \param argv  Argument vector.
 * \param[out] config  Output configuration populated from args.
 * \return true if parsing succeeded, false if --model was missing.
 */
struct ServeConfig {
    std::string weight_path;
    std::string config_path;
    std::string bind_addr = "0.0.0.0";
    uint16_t port = 8080;
    uint32_t max_tokens = 128;
    float temperature = 0.7f;
    float top_p = 0.9f;
    uint32_t top_k = 40;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    VaistEngine::ModelType model_type = VaistEngine::ModelType::kLlama;
};

static bool parse_args(int argc, char** argv, ServeConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string_view arg(argv[i]);
        if (arg == "--model" && i + 1 < argc) {
            config.weight_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config.config_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if (arg == "--address" && i + 1 < argc) {
            config.bind_addr = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_tokens = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--top-p" && i + 1 < argc) {
            config.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--top-k" && i + 1 < argc) {
            config.top_k = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--min-p" && i + 1 < argc) {
            config.min_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--presence-penalty" && i + 1 < argc) {
            config.presence_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--frequency-penalty" && i + 1 < argc) {
            config.frequency_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--model-type" && i + 1 < argc) {
            std::string_view mtype(argv[++i]);
            if (mtype == "llama") config.model_type = VaistEngine::ModelType::kLlama;
            else if (mtype == "qwen3") config.model_type = VaistEngine::ModelType::kQwen3;
            else if (mtype == "qwen3-moe") config.model_type = VaistEngine::ModelType::kQwen3Moe;
            else if (mtype == "deepseek-v2") config.model_type = VaistEngine::ModelType::kDeepseekV2;
            else if (mtype == "deepseek-v3") config.model_type = VaistEngine::ModelType::kDeepseekV3;
            else {
                std::fprintf(stderr, "Unknown model type: %.*s\n", static_cast<int>(mtype.size()), mtype.data());
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::printf(
                "Usage: vaist_serve.exe --model <path> [options]\n"
                "  --model <path>           Path to model weights (.safetensors or .gguf)\n"
                "  --model-type <type>      Model architecture: llama, qwen3, qwen3-moe, deepseek-v2, deepseek-v3\n"
                "  --config <path>          Path to config.json (optional)\n"
                "  --port <port>            Listen port (default: 8080)\n"
                "  --address <addr>         Bind address (default: 0.0.0.0)\n"
                "  --max-tokens <n>         Maximum new tokens (default: 128)\n"
                "  --temperature <t>        Sampling temperature (default: 0.7)\n"
                "  --top-p <p>              Nucleus sampling (default: 0.9)\n"
                "  --top-k <k>              Top-K sampling (default: 40)\n"
                "  --min-p <p>              Min-p filtering (default: 0.0)\n"
                "  --presence-penalty <p>   Presence penalty (default: 0.0)\n"
                "  --frequency-penalty <p>  Frequency penalty (default: 0.0)\n"
            );
            std::exit(0);
        }
    }

    if (config.weight_path.empty()) {
        std::fprintf(stderr, "Error: --model is required\n");
        std::fprintf(stderr, "Run with --help for usage.\n");
        return false;
    }

    return true;
}

/**
 * \brief Create and initialize a VaistEngine with a tokenizer.
 *
 * Attempts to locate a tokenizer directory alongside the model weights.
 * If the model path contains a directory, looks for tokenizer files there.
 * Falls back to the byte tokenizer if no tokenizer files are found.
 *
 * \param weight_path  Path to model weights.
 * \param model_type   Architecture type.
 * \param[out] engine  Engine to initialize.
 * \return VAIST_OK on success.
 */
static VaistStatus create_engine(const std::string& weight_path,
                                  VaistEngine::ModelType model_type,
                                  VaistEngine& engine) {
    // Try to create runtime (CPU backend for Windows)
    VaistRuntime* rt = nullptr;
    if (vaist_runtime_create(VAIST_BACKEND_CPU, &rt) != VAIST_OK) {
        std::fprintf(stderr, "Warning: could not create runtime, using nullptr\n");
    }

    // Load the model
    VaistStatus st = engine.load_model(model_type, weight_path);
    if (st != VAIST_OK) {
        std::fprintf(stderr, "Error: failed to load model from %s: %s\n",
                     weight_path.c_str(), vaist_status_string(st));
        if (rt) vaist_runtime_destroy(rt);
        return st;
    }

    // Try to create a tokenizer
    // Look for vocab.json/merges.txt or a tokenizer directory alongside the model
    std::string vocab_path;
    size_t slash = weight_path.find_last_of("/\\");
    if (slash != std::string::npos) {
        std::string dir = weight_path.substr(0, slash + 1);
        // Try directory with tokenizer files
        std::string test_path = dir + "tokenizer.json";
        FILE* f = std::fopen(test_path.c_str(), "rb");
        if (f) { std::fclose(f); vocab_path = test_path; }
        else {
            test_path = dir + "vocab.json";
            f = std::fopen(test_path.c_str(), "rb");
            if (f) { std::fclose(f); vocab_path = dir; }
        }
    }

    VaistTokenizer* tokenizer = nullptr;
    if (!vocab_path.empty()) {
        if (vaist_tokenizer_create(VAIST_TOKENIZER_BPE, vocab_path.c_str(), &tokenizer) != VAIST_OK) {
            std::fprintf(stderr, "Warning: failed to load BPE tokenizer, using bytes tokenizer\n");
            tokenizer = nullptr;
        }
    }

    // Fallback: byte tokenizer (always works, maps each byte to a token id)
    if (!tokenizer) {
        if (vaist_tokenizer_create(VAIST_TOKENIZER_BYTES, nullptr, &tokenizer) != VAIST_OK) {
            std::fprintf(stderr, "Warning: could not create byte tokenizer\n");
        }
    }

    engine.set_tokenizer(tokenizer);

    return VAIST_OK;
}

int main(int argc, char** argv) {
    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Parse arguments
    ServeConfig config;
    if (!parse_args(argc, argv, config)) {
        return 1;
    }

    std::printf("VAiSt Serve\n");
    std::printf("  Model:   %s\n", config.weight_path.c_str());
    std::printf("  Type:    %d\n", static_cast<int>(config.model_type));
    std::printf("  Port:    %u\n", config.port);
    std::printf("  Address: %s\n", config.bind_addr.c_str());
    std::printf("  Max tokens: %u\n", config.max_tokens);
    std::printf("  Temperature: %.2f\n", config.temperature);
    std::printf("  Top-p: %.2f\n", config.top_p);
    std::printf("  Top-k: %u\n", config.top_k);
    std::printf("\n");

    // Create and load the engine
    VaistEngine engine;
    VaistStatus st = create_engine(config.weight_path, config.model_type, engine);
    if (st != VAIST_OK) {
        std::fprintf(stderr, "Failed to initialize engine: %s\n", vaist_status_string(st));
        return 1;
    }

    std::printf("Model loaded successfully. Vocab size: %u\n", engine.vocab_size());
    std::fflush(stdout);

    // Determine model name for /v1/models
    std::string model_name = config.weight_path;
    size_t last_slash = model_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_name = model_name.substr(last_slash + 1);
    }
    // Strip extension
    size_t last_dot = model_name.find_last_of('.');
    if (last_dot != std::string::npos) {
        model_name = model_name.substr(0, last_dot);
    }
    if (model_name.empty()) {
        model_name = "vaist-model";
    }

    // Start the HTTP server
    VaistHttpServer server(&engine, config.port, config.bind_addr, model_name);
    if (!server.start()) {
        std::fprintf(stderr, "Failed to start HTTP server on port %u\n", config.port);
        return 1;
    }

    std::printf("Server listening on http://%s:%u\n", config.bind_addr.c_str(), config.port);
    std::printf("  POST /v1/chat/completions\n");
    std::printf("  POST /v1/completions\n");
    std::printf("  GET  /v1/models\n");
    std::printf("  GET  /health\n");
    std::fflush(stdout);

    // Main loop: wait for shutdown signal
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::printf("\nShutting down...\n");
    server.stop();
    server.join();
    std::printf("Server stopped.\n");

    return 0;
}
