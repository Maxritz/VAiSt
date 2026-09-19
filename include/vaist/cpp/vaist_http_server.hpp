/**
 * \file vaist_http_server.hpp
 * \brief OpenAI-compatible HTTP server for the VAiSt inference engine.
 *
 * Wraps a VaistEngine* behind a minimal Winsock2 HTTP/1.1 server that
 * implements the subset of the OpenAI REST API needed for serving:
 *
 *   - POST /v1/chat/completions  — ChatML multi-turn conversation
 *   - POST /v1/completions       — plain-text prompt completion
 *   - GET  /v1/models            — list loaded model(s)
 *   - GET  /health               — health-check endpoint
 *
 * The server uses a thread-per-connection model: each accepted socket
 * is handled by a dedicated std::thread that reads the HTTP request,
 * dispatches to the engine, and writes the response (or SSE stream).
 *
 * No external HTTP or JSON libraries are required — JSON construction
 * uses JsonWriter and JSON parsing uses JsonParser from vaist_json.hpp.
 */
#ifndef VAIST_CPP_HTTP_SERVER_HPP
#define VAIST_CPP_HTTP_SERVER_HPP

#include "vaist_engine.hpp"
#include "vaist_json.hpp"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdint>

namespace vaist {

/**
 * \brief Parsed HTTP request.
 */
struct HttpRequest {
    std::string method;       /**< GET, POST, etc. (uppercased).          */
    std::string path;         /**< Request path without query string.      */
    std::string query;        /**< Raw query string (may be empty).        */
    std::string body;         /**< Request body (may be empty).            */
    std::vector<std::pair<std::string, std::string>> headers;
};

/**
 * \brief Parsed OpenAI completion request parameters.
 */
struct CompletionRequest {
    std::string model;        /**< Model identifier (e.g. "llama-3").       */
    std::string prompt;       /**< Text prompt (for /v1/completions).      */
    std::vector<std::pair<std::string, std::string>> messages; /**< role,content pairs for chat */
    uint32_t max_tokens    = 128;       /**< Maximum new tokens to generate.  */
    float   temperature    = 0.7f;      /**< Sampling temperature (0 = greedy). */
    float   top_p          = 0.9f;      /**< Nucleus sampling threshold.        */
    uint32_t top_k          = 40;        /**< Top-K filter (0 = disabled).       */
    float   min_p          = 0.0f;       /**< Min-p filter (0 = disabled).       */
    float   presence_penalty = 0.0f;     /**< Presence penalty.                */
    float   frequency_penalty = 0.0f;    /**< Frequency penalty.               */
    bool    stream         = false;     /**< SSE streaming output.              */
    uint64_t seed          = 0;          /**< PRNG seed (0 = default).          */
};

/**
 * \brief Callback type for streaming token emission.
 *
 * Called once per generated token during streaming mode. The callback
 * receives the raw token text (not token ID) so the caller can write
 * SSE chunks.
 */
using StreamCallback = std::function<void(std::string_view token_text)>;

/**
 * \brief Result of a non-streaming completion.
 */
struct CompletionResult {
    uint32_t      prompt_tokens = 0;     /**< Number of input tokens.  */
    uint32_t      completion_tokens = 0; /**< Number of output tokens. */
    std::string   text;                  /**< Generated text.            */
    bool          stopped = false;       /**< True if stopped at EOS/max_tokens. */
};

/**
 * \brief OpenAI-compatible HTTP server.
 *
 * Wraps a VaistEngine and exposes the OpenAI REST API surface over HTTP/1.1
 * using Winsock2. Thread-per-connection: each accepted socket runs its own
 * thread until the request completes (or the server stops).
 *
 * Lifecycle:
 * \code
 *   VaistEngine engine;
 *   engine.load_model(...);
 *   VaistHttpServer server(&engine, 8080, "0.0.0.0");
 *   server.start();
 *   // ... run until Ctrl+C or SIGTERM ...
 *   server.stop();
 * \endcode
 */
class VaistHttpServer {
public:
    /**
     * \brief Construct the server.
     * \param engine   Pointer to a loaded VaistEngine (must outlive the server).
     * \param port     TCP port to listen on.
     * \param bind_addr  Bind address (e.g. "0.0.0.0" for all interfaces).
     * \param model_name  Human-readable model name for /v1/models.
     */
    VaistHttpServer(VaistEngine* engine,
                    uint16_t port,
                    std::string_view bind_addr,
                    std::string_view model_name = "vaist-model");

    /** \brief Destructor stops the server and cleans up Winsock. */
    ~VaistHttpServer();

    // Non-copyable, non-movable (owns a socket + thread pool).
    VaistHttpServer(const VaistHttpServer&) = delete;
    VaistHttpServer& operator=(const VaistHttpServer&) = delete;
    VaistHttpServer(VaistHttpServer&&) = delete;
    VaistHttpServer& operator=(VaistHttpServer&&) = delete;

    /**
     * \brief Start listening for connections.
     *
     * Spawns the listener thread. Returns immediately.
     * \return true if the listening socket was created and bound.
     */
    bool start();

    /**
     * \brief Stop the server and wait for the listener thread to exit.
     *
     * Closes the listening socket, causing accept() to fail and the
     * listener thread to exit. Does not forcibly kill connection threads.
     */
    void stop();

    /**
     * \brief Blocking wait for the listener thread to exit.
     * Call after stop().
     */
    void join();

    /** \brief True if start() has been called and the socket is open. */
    bool is_running() const noexcept { return running_; }

private:
    VaistEngine* engine_;
    uint16_t port_;
    std::string bind_addr_;
    std::string model_name_;

    SOCKET listen_sock_ = INVALID_SOCKET;
    std::atomic<bool> running_{false};
    std::thread listener_thread_;

    /** \brief Generate a unique request ID for the response. */
    static std::string generate_request_id();

    /** \brief Format an HTTP 200 response. */
    static std::string format_response(int status_code,
                                       std::string_view body,
                                       std::string_view content_type = "application/json");

    /** \brief Format an HTTP error response. */
    static std::string format_error(int status_code,
                                    std::string_view message);

    /** \brief Read an HTTP request from a socket. */
    static bool read_request(SOCKET sock, HttpRequest& out,
                             std::vector<uint8_t>& buffer, size_t& buf_pos,
                             size_t max_body_bytes = 64 * 1024 * 1024);

    /** \brief Parse the HTTP request line and headers. */
    static bool parse_http_request(const std::vector<uint8_t>& buf,
                                   size_t data_len,
                                   HttpRequest& out);

    /** \brief Write data to a socket (full send). */
    static bool send_all(SOCKET sock, const char* data, size_t len);

    /** \brief Handle a single HTTP connection. */
    void handle_connection(SOCKET sock);

    /** \brief Route the request to the appropriate handler. */
    void route(const HttpRequest& req, SOCKET sock);

    /** \brief Handle POST /v1/chat/completions. */
    void handle_chat_completions(const HttpRequest& req, SOCKET sock);

    /** \brief Handle POST /v1/completions. */
    void handle_completions(const HttpRequest& req, SOCKET sock);

    /** \brief Handle GET /v1/models. */
    void handle_models(const HttpRequest& req, SOCKET sock);

    /** \brief Handle GET /health. */
    void handle_health(const HttpRequest& req, SOCKET sock);

    /** \brief Parse OpenAI request JSON into CompletionRequest. */
    static bool parse_completion_request(const std::string& body,
                                           CompletionRequest& out);

    /** \brief Run non-streaming generation and return CompletionResult. */
    CompletionResult run_generation(CompletionRequest& req);

    /** \brief Run streaming generation, calling callback per token. */
    void run_generation_stream(CompletionRequest& req,
                               SOCKET sock,
                               StreamCallback callback);

    /** \brief Build SSE event chunk for a single token. */
    static std::string build_sse_chunk(const std::string& token_text);

    /** \brief Build the final SSE [DONE] message. */
    static std::string build_sse_done();

    /** \brief Build the final SSE usage chunk. */
    static std::string build_sse_usage(uint32_t prompt_tokens,
                                       uint32_t completion_tokens);

    /** \brief Listener thread main loop. */
    void listener_loop();

    /** \brief Winsock initialization (static, called once). */
    static bool init_winsock();

    /** \brief Winsock cleanup (static, called once). */
    static void cleanup_winsock();

    static std::atomic<int> wsa_refcount_;
};

} // namespace vaist

#endif // VAIST_CPP_HTTP_SERVER_HPP
