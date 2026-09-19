/**
 * \file vaist_http_server.cpp
 * \brief Implementation of the OpenAI-compatible HTTP server.
 *
 * Uses Winsock2 for networking, JsonWriter/JsonParser for JSON,
 * and VaistEngine for inference. Thread-per-connection model.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "vaist_http_server.hpp"
#include "vaist_engine.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <mutex>
#include <random>
#include <sstream>
 #include <system_error>
 #if defined(__GNUC__)
 #pragma GCC diagnostic ignored "-Wunknown-pragmas"
 #endif
 
 #pragma comment(lib, "ws2_32.lib")

namespace vaist {

/* ======================================================================== */
/* Static helpers                                                            */
/* ======================================================================== */

std::atomic<int> VaistHttpServer::wsa_refcount_{0};

static std::mt19937_64 g_prng{ std::random_device{}() };
static std::mutex g_prng_mutex;

static std::string random_hex(size_t n) {
    std::lock_guard<std::mutex> lock(g_prng_mutex);
    std::uniform_int_distribution<int> dist(0, 15);
    std::string out;
    out.reserve(n);
    const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out += hex[dist(g_prng)];
    }
    return out;
}

bool VaistHttpServer::init_winsock() {
    int count = wsa_refcount_.fetch_add(1);
    if (count > 0) return true;
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}

void VaistHttpServer::cleanup_winsock() {
    int count = wsa_refcount_.fetch_sub(1);
    if (count <= 1) {
        WSACleanup();
    }
}

std::string VaistHttpServer::generate_request_id() {
    static std::atomic<uint64_t> counter{0};
    uint64_t c = counter.fetch_add(1);
    std::ostringstream ss;
    ss << "chatcmpl-" << random_hex(8) << c;
    return ss.str();
}

std::string VaistHttpServer::format_response(int status_code,
                                             std::string_view body,
                                             std::string_view content_type) {
    std::string status_text;
    switch (status_code) {
        case 200: status_text = "OK"; break;
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
        default:  status_text = "Unknown"; break;
    }

    std::ostringstream resp;
    resp << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
         << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
         << "\r\n"
         << body;
    return resp.str();
}

std::string VaistHttpServer::format_error(int status_code,
                                          std::string_view message) {
    JsonWriter w;
    w.start_object();
    w.key("object"); w.value("error");
    w.key("message"); w.value(message);
    w.key("type"); w.value("invalid_request_error");
    w.end_object();
    return format_response(status_code, w.str(), "application/json");
}

bool VaistHttpServer::send_all(SOCKET sock, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        int n = static_cast<int>(send(sock, data + sent,
                                      static_cast<int>(len - sent), 0));
        if (n == SOCKET_ERROR || n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool VaistHttpServer::read_request(SOCKET sock, HttpRequest& out,
                                   std::vector<uint8_t>& buffer,
                                   size_t& buf_pos,
                                   size_t max_body_bytes) {
    (void)max_body_bytes; // not used — read up to available buffer
    buffer.resize(8192);
    buf_pos = 0;

    while (true) {
        if (buf_pos >= buffer.size()) {
            buffer.resize(buffer.size() * 2);
        }
        int n = static_cast<int>(recv(sock, reinterpret_cast<char*>(buffer.data() + buf_pos),
                                      static_cast<int>(buffer.size() - buf_pos), 0));
        if (n == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT) continue;
            return false;
        }
        if (n == 0) {
            return buf_pos > 0;
        }
        buf_pos += static_cast<size_t>(n);

        // Try to parse the complete request
        if (parse_http_request(buffer, buf_pos, out)) {
            // Check if we have the full body
            size_t content_length = 0;
            for (const auto& h : out.headers) {
                if (h.first == "content-length") {
                    content_length = std::stoull(h.second);
                    break;
                }
            }
            // Find the body offset
            size_t header_end = 0;
            for (size_t i = 0; i + 3 < buf_pos; i++) {
                if (buffer[i] == '\r' && buffer[i + 1] == '\n' &&
                    buffer[i + 2] == '\r' && buffer[i + 3] == '\n') {
                    header_end = i + 4;
                    break;
                }
            }
            if (buf_pos >= header_end + content_length) {
                // Trim body to exact content_length
                if (content_length > 0) {
                    out.body.assign(reinterpret_cast<const char*>(buffer.data() + header_end),
                                    content_length);
                } else {
                    out.body.clear();
                }
                return true;
            }
        }
        // Keep reading
    }
}

bool VaistHttpServer::parse_http_request(const std::vector<uint8_t>& buf,
                                         size_t data_len,
                                         HttpRequest& out) {
    if (data_len < 14) return false; // Minimum: "GET / HTTP/1\r\n\r\n"

    // Find the header/body boundary (\r\n\r\n)
    size_t header_end = std::string::npos;
    for (size_t i = 0; i + 3 < data_len; i++) {
        if (buf[i] == '\r' && buf[i + 1] == '\n' && buf[i + 2] == '\r' && buf[i + 3] == '\n') {
            header_end = i;
            break;
        }
    }
    if (header_end == std::string::npos) return false;

    // Parse request line
    std::string request_line(reinterpret_cast<const char*>(buf.data()), header_end);
    size_t line_end = request_line.find("\r\n");
    if (line_end == std::string::npos) return false;

    std::string first_line = request_line.substr(0, line_end);
    size_t sp1 = first_line.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t sp2 = first_line.rfind(' ');
    if (sp2 == std::string::npos || sp2 <= sp1) return false;

    out.method = first_line.substr(0, sp1);
    // Uppercase method
    for (auto& c : out.method) c = static_cast<char>(toupper(c));

    std::string path_and_query = first_line.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t qpos = path_and_query.find('?');
    if (qpos != std::string::npos) {
        out.path = path_and_query.substr(0, qpos);
        out.query = path_and_query.substr(qpos + 1);
    } else {
        out.path = path_and_query;
        out.query.clear();
    }

    // Parse headers
    size_t pos = line_end + 2;
    while (pos + 1 < data_len) {
        size_t line_end2 = std::string::npos;
        for (size_t i = pos; i + 1 < data_len; i++) {
            if (buf[i] == '\r' && buf[i + 1] == '\n') {
                line_end2 = i;
                break;
            }
        }
        if (line_end2 == std::string::npos) break;
        std::string header_line(reinterpret_cast<const char*>(buf.data() + pos),
                                line_end2 - pos);
        pos = line_end2 + 2;

        if (header_line.empty()) break;

        size_t colon = header_line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = header_line.substr(0, colon);
        std::string value = header_line.substr(colon + 1);
        // trim whitespace
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
        // lowercase header name
        for (auto& c : name) c = static_cast<char>(tolower(c));
        out.headers.emplace_back(std::move(name), std::move(value));
    }

    // Parse body
    size_t body_offset = header_end + 4;
    if (body_offset < data_len) {
        out.body.assign(reinterpret_cast<const char*>(buf.data() + body_offset),
                        data_len - body_offset);
    }

    return true;
}

/* ======================================================================== */
/* Construction / lifecycle                                                  */
/* ======================================================================== */

VaistHttpServer::VaistHttpServer(VaistEngine* engine,
                                uint16_t port,
                                std::string_view bind_addr,
                                std::string_view model_name)
    : engine_(engine)
    , port_(port)
    , bind_addr_(bind_addr)
    , model_name_(model_name) {
}

VaistHttpServer::~VaistHttpServer() {
    if (running_) {
        stop();
        if (listener_thread_.joinable()) {
            listener_thread_.join();
        }
    }
    if (listen_sock_ != INVALID_SOCKET) {
        closesocket(listen_sock_);
        listen_sock_ = INVALID_SOCKET;
    }
    cleanup_winsock();
}

bool VaistHttpServer::start() {
    if (running_) return true;
    if (!init_winsock()) return false;

    listen_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock_ == INVALID_SOCKET) {
        cleanup_winsock();
        return false;
    }

    // Allow address reuse
    {
        BOOL opt = TRUE;
        setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (bind_addr_.empty() || bind_addr_ == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, bind_addr_.c_str(), &addr.sin_addr);
    }

    if (bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        closesocket(listen_sock_);
        listen_sock_ = INVALID_SOCKET;
        cleanup_winsock();
        return false;
    }

    if (listen(listen_sock_, 16) == SOCKET_ERROR) {
        closesocket(listen_sock_);
        listen_sock_ = INVALID_SOCKET;
        cleanup_winsock();
        return false;
    }

    running_ = true;
    listener_thread_ = std::thread(&VaistHttpServer::listener_loop, this);
    return true;
}

void VaistHttpServer::stop() {
    running_ = false;
    if (listen_sock_ != INVALID_SOCKET) {
        closesocket(listen_sock_);
        listen_sock_ = INVALID_SOCKET;
    }
}

void VaistHttpServer::join() {
    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }
}

void VaistHttpServer::listener_loop() {
    while (running_) {
        sockaddr_in client_addr{};
        int client_len = sizeof(client_addr);
        SOCKET client_sock = accept(listen_sock_,
                                    reinterpret_cast<sockaddr*>(&client_addr),
                                    &client_len);

        if (client_sock == INVALID_SOCKET) {
            if (running_) continue;
            break;
        }

        // Spawn a thread per connection
        std::thread(&VaistHttpServer::handle_connection, this, client_sock).detach();
    }
}

void VaistHttpServer::handle_connection(SOCKET sock) {
    HttpRequest req;
    std::vector<uint8_t> buffer;
    size_t buf_pos = 0;

    // Set per-connection timeout
    DWORD timeout = 30000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&timeout), sizeof(timeout));

    if (!read_request(sock, req, buffer, buf_pos)) {
        closesocket(sock);
        return;
    }

    // Handle OPTIONS preflight
    if (req.method == "OPTIONS") {
        std::string resp = format_response(200, "", "text/plain");
        send_all(sock, resp.data(), resp.size());
        closesocket(sock);
        return;
    }

    route(req, sock);
    closesocket(sock);
}

/* ======================================================================== */
/* Routing                                                                   */
/* ======================================================================== */

void VaistHttpServer::route(const HttpRequest& req, SOCKET sock) {
    if (req.method == "POST" && req.path == "/v1/chat/completions") {
        handle_chat_completions(req, sock);
    } else if (req.method == "POST" && req.path == "/v1/completions") {
        handle_completions(req, sock);
    } else if (req.method == "GET" && req.path == "/v1/models") {
        handle_models(req, sock);
    } else if (req.method == "GET" && req.path == "/health") {
        handle_health(req, sock);
    } else {
        std::string resp = format_error(404, "Not Found");
        send_all(sock, resp.data(), resp.size());
    }
}

/* ======================================================================== */
/* Route handlers                                                            */
/* ======================================================================== */

void VaistHttpServer::handle_chat_completions(const HttpRequest& req, SOCKET sock) {
    // Parse the request body as JSON
    CompletionRequest creq;
    if (!parse_completion_request(req.body, creq)) {
        std::string resp = format_error(400, "Invalid request body");
        send_all(sock, resp.data(), resp.size());
        return;
    }

    // Convert ChatML messages to prompt
    std::string full_prompt;
    if (creq.messages.empty()) {
        // Fall back to prompt field
        full_prompt = creq.prompt;
    } else {
        // Build ChatML-style prompt from messages
        for (const auto& msg : creq.messages) {
            full_prompt += msg.first + ": " + msg.second + "\n";
        }
        full_prompt += "assistant: ";
    }

    if (creq.stream) {
        // Streaming mode: send SSE chunks
        std::string headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        send_all(sock, headers.data(), headers.size());

        // Send response header chunk
        time_t now = std::time(nullptr);
        std::string id = generate_request_id();

        JsonWriter wh;
        wh.start_object();
        wh.key("id"); wh.value(id);
        wh.key("object"); wh.value("chat.completion.chunk");
        wh.key("created"); wh.value(static_cast<int64_t>(now));
        wh.key("model"); wh.value(creq.model);
        wh.start_array(); wh.key("choices");
        wh.start_object();
        wh.key("index"); wh.value(static_cast<int64_t>(0));
        wh.start_object();
        wh.key("delta");
        wh.start_object();
        wh.key("role"); wh.value("assistant");
        wh.end_object();
        wh.end_object();
        wh.end_array();
        wh.end_object();

        std::string chunk = build_sse_chunk(wh.str());
        send_all(sock, chunk.data(), chunk.size());

         // Run streaming generation
        uint32_t prompt_tokens = 0;
        uint32_t completion_tokens = 0;

        // Count prompt tokens for usage reporting
        if (engine_->tokenizer()) {
            uint32_t toks[8192];
            size_t count = 0;
            if (vaist_tokenize(engine_->tokenizer(), full_prompt.c_str(), toks, 8192, &count) == VAIST_OK) {
                prompt_tokens = static_cast<uint32_t>(count);
            }
        }

        // Run the actual generation in streaming mode
        auto callback = [&](std::string_view token_text) {
            JsonWriter cw;
            cw.start_object();
            cw.key("choices");
            cw.start_array();
            cw.start_object();
            cw.key("index"); cw.value(static_cast<int64_t>(0));
            cw.key("delta");
            cw.start_object();
            cw.key("content"); cw.value(token_text);
            cw.end_object();
            cw.end_object();
            cw.key("finish_reason"); cw.value(nullptr);
            cw.end_object();
            cw.end_array();
            cw.end_object();

            std::string chunk_data = build_sse_chunk(cw.str());
            send_all(sock, chunk_data.data(), chunk_data.size());
            completion_tokens++;
        };

        run_generation_stream(creq, sock, callback);

        // Send final chunk with finish_reason
        JsonWriter fc;
        fc.start_object();
        fc.key("choices");
        fc.start_array();
        fc.start_object();
        fc.key("index"); fc.value(static_cast<int64_t>(0));
        fc.start_object();
        fc.key("delta");
        fc.start_object();
        fc.end_object();
        fc.end_object();
        fc.key("finish_reason"); fc.value("stop");
        fc.end_object();
        fc.end_array();
        fc.end_object();

        std::string final_chunk = build_sse_chunk(fc.str());
        send_all(sock, final_chunk.data(), final_chunk.size());

        // Send usage chunk
        std::string usage = build_sse_usage(prompt_tokens, completion_tokens);
        send_all(sock, usage.data(), usage.size());

        // Send DONE
        send_all(sock, build_sse_done().data(), build_sse_done().size());
    } else {
        // Non-streaming mode
        CompletionResult result = run_generation(creq);

        time_t now = std::time(nullptr);
        std::string id = generate_request_id();

        JsonWriter w;
        w.start_object();
        w.key("id"); w.value(id);
        w.key("object"); w.value("chat.completion");
        w.key("created"); w.value(static_cast<int64_t>(now));
        w.key("model"); w.value(creq.model);
        // choices array
        w.key("choices");
        w.start_array();
        w.start_object();
        w.key("index"); w.value(static_cast<int64_t>(0));
        w.start_object();
        w.key("role"); w.value("assistant");
        w.key("content"); w.value(result.text);
        w.end_object();
        w.key("finish_reason");
        w.value(result.stopped ? "stop" : "length");
        w.end_object();
        w.end_array();
        // usage
        w.key("usage");
        w.start_object();
        w.key("prompt_tokens"); w.value(static_cast<int64_t>(result.prompt_tokens));
        w.key("completion_tokens"); w.value(static_cast<int64_t>(result.completion_tokens));
        w.key("total_tokens"); w.value(static_cast<int64_t>(result.prompt_tokens + result.completion_tokens));
        w.end_object();
        w.end_object();

        std::string resp = format_response(200, w.str(), "application/json");
        send_all(sock, resp.data(), resp.size());
    }
}

void VaistHttpServer::handle_completions(const HttpRequest& req, SOCKET sock) {
    // Reuses the same logic as chat/completions, just without messages
    CompletionRequest creq;
    if (!parse_completion_request(req.body, creq)) {
        std::string resp = format_error(400, "Invalid request body");
        send_all(sock, resp.data(), resp.size());
        return;
    }

    // If no messages, use prompt directly
    if (!creq.messages.empty()) {
        // Convert to plain prompt
        creq.prompt.clear();
        for (const auto& msg : creq.messages) {
            creq.prompt += msg.first + ": " + msg.second + "\n";
        }
    }

    if (creq.stream) {
        // Streaming mode
        std::string headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        send_all(sock, headers.data(), headers.size());

        // Send initial delta with role
        time_t now = std::time(nullptr);
        std::string id = generate_request_id();

        JsonWriter wh;
        wh.start_object();
        wh.key("id"); wh.value(id);
        wh.key("object"); wh.value("text_completion");
        wh.key("created"); wh.value(static_cast<int64_t>(now));
        wh.key("model"); wh.value(creq.model);
        wh.start_array(); wh.key("choices");
        wh.start_object();
        wh.key("index"); wh.value(static_cast<int64_t>(0));
        wh.start_object();
        wh.key("text"); wh.value("");
        wh.end_object();
        wh.key("finish_reason"); wh.value(nullptr);
        wh.end_object();
        wh.end_array();
        wh.end_object();

        send_all(sock, build_sse_chunk(wh.str()).data(), build_sse_chunk(wh.str()).size());

        // Run streaming generation
        auto callback = [&](std::string_view token_text) {
            JsonWriter cw;
            cw.start_object();
            cw.key("choices");
            cw.start_array();
            cw.start_object();
            cw.key("index"); cw.value(static_cast<int64_t>(0));
            cw.start_object();
            cw.key("text"); cw.value(token_text);
            cw.end_object();
            cw.key("finish_reason"); cw.value(nullptr);
            cw.end_object();
            cw.end_array();
            cw.end_object();

            std::string chunk_data = build_sse_chunk(cw.str());
            send_all(sock, chunk_data.data(), chunk_data.size());
        };

        run_generation_stream(creq, sock, callback);

        // Final chunk
        JsonWriter fc;
        fc.start_object();
        fc.key("choices");
        fc.start_array();
        fc.start_object();
        fc.key("index"); fc.value(static_cast<int64_t>(0));
        fc.start_object();
        fc.key("text"); fc.value("");
        fc.end_object();
        fc.key("finish_reason"); fc.value("stop");
        fc.end_object();
        fc.end_array();
        fc.end_object();

        send_all(sock, build_sse_chunk(fc.str()).data(), build_sse_chunk(fc.str()).size());
        send_all(sock, build_sse_done().data(), build_sse_done().size());
    } else {
        // Non-streaming
        CompletionResult result = run_generation(creq);

        time_t now = std::time(nullptr);
        std::string id = generate_request_id();

        JsonWriter w;
        w.start_object();
        w.key("id"); w.value(id);
        w.key("object"); w.value("text_completion");
        w.key("created"); w.value(static_cast<int64_t>(now));
        w.key("model"); w.value(creq.model);
        w.key("choices");
        w.start_array();
        w.start_object();
        w.key("index"); w.value(static_cast<int64_t>(0));
        w.start_object();
        w.key("text"); w.value(result.text);
        w.end_object();
        w.key("finish_reason");
        w.value(result.stopped ? "stop" : "length");
        w.end_object();
        w.end_array();
        w.key("usage");
        w.start_object();
        w.key("prompt_tokens"); w.value(static_cast<int64_t>(result.prompt_tokens));
        w.key("completion_tokens"); w.value(static_cast<int64_t>(result.completion_tokens));
        w.key("total_tokens"); w.value(static_cast<int64_t>(result.prompt_tokens + result.completion_tokens));
        w.end_object();
        w.end_object();

        std::string resp = format_response(200, w.str(), "application/json");
        send_all(sock, resp.data(), resp.size());
    }
}

void VaistHttpServer::handle_models(const HttpRequest& req, SOCKET sock) {
    (void)req;
    JsonWriter w;
    w.start_object();
    w.key("object"); w.value("list");
    w.key("data");
    w.start_array();
    w.start_object();
    w.key("id"); w.value(model_name_);
    w.key("object"); w.value("model");
    w.key("owned_by"); w.value("vaist");
    w.end_object();
    w.end_array();
    w.end_object();

    std::string resp = format_response(200, w.str(), "application/json");
    send_all(sock, resp.data(), resp.size());
}

void VaistHttpServer::handle_health(const HttpRequest& req, SOCKET sock) {
    (void)req;
    std::string resp = format_response(200, "ok", "text/plain");
    send_all(sock, resp.data(), resp.size());
}

/* ======================================================================== */
/* Request parsing                                                           */
/* ======================================================================== */

bool VaistHttpServer::parse_completion_request(const std::string& body,
                                                CompletionRequest& out) {
    if (body.empty()) return false;

    JsonValue parsed;
    try {
        parsed = JsonParser::parse(body.c_str());
    } catch (...) {
        return false;
    }

    if (parsed.type() != JsonType::kObject) return false;

    for (const auto& [key, val] : parsed.as_object()) {
        if (key == "model") {
            if (val.type() == JsonType::kString) out.model = val.as_string();
        } else if (key == "prompt") {
            if (val.type() == JsonType::kString) out.prompt = val.as_string();
        } else if (key == "messages") {
            if (val.type() == JsonType::kArray) {
                for (const auto& msg : val.as_array()) {
                    if (msg.type() == JsonType::kObject) {
                        std::string role, content;
                        for (const auto& [mk, mv] : msg.as_object()) {
                            if (mk == "role" && mv.type() == JsonType::kString)
                                role = mv.as_string();
                            else if (mk == "content" && mv.type() == JsonType::kString)
                                content = mv.as_string();
                        }
                        if (!role.empty()) {
                            out.messages.emplace_back(role, content);
                        }
                    }
                }
            }
        } else if (key == "max_tokens") {
            if (val.type() == JsonType::kInt) out.max_tokens = static_cast<uint32_t>(val.as_int());
            else if (val.type() == JsonType::kDouble) out.max_tokens = static_cast<uint32_t>(val.as_double());
        } else if (key == "temperature") {
            if (val.type() == JsonType::kInt) out.temperature = static_cast<float>(val.as_int());
            else if (val.type() == JsonType::kDouble) out.temperature = static_cast<float>(val.as_double());
        } else if (key == "top_p") {
            if (val.type() == JsonType::kInt) out.top_p = static_cast<float>(val.as_int());
            else if (val.type() == JsonType::kDouble) out.top_p = static_cast<float>(val.as_double());
        } else if (key == "top_k") {
            if (val.type() == JsonType::kInt) out.top_k = static_cast<uint32_t>(val.as_int());
        } else if (key == "min_p") {
            if (val.type() == JsonType::kInt) out.min_p = static_cast<float>(val.as_int());
            else if (val.type() == JsonType::kDouble) out.min_p = static_cast<float>(val.as_double());
        } else if (key == "presence_penalty") {
            if (val.type() == JsonType::kInt) out.presence_penalty = static_cast<float>(val.as_int());
            else if (val.type() == JsonType::kDouble) out.presence_penalty = static_cast<float>(val.as_double());
        } else if (key == "frequency_penalty") {
            if (val.type() == JsonType::kInt) out.frequency_penalty = static_cast<float>(val.as_int());
            else if (val.type() == JsonType::kDouble) out.frequency_penalty = static_cast<float>(val.as_double());
        } else if (key == "stream") {
            if (val.type() == JsonType::kBool) out.stream = val.as_bool();
        } else if (key == "seed") {
            if (val.type() == JsonType::kInt) out.seed = static_cast<uint64_t>(val.as_int());
        }
    }

    return true;
}

/* ======================================================================== */
/* Generation                                                                */
/* ======================================================================== */

CompletionResult VaistHttpServer::run_generation(CompletionRequest& req) {
    CompletionResult result;

    if (!engine_) {
        return result;
    }

    // Build sampling config
    VaistSamplingConfig cfg;
    cfg.temperature = req.temperature;
    cfg.top_p = req.top_p;
    cfg.top_k = req.top_k;
    cfg.min_p = req.min_p;
    cfg.presence_penalty = req.presence_penalty;
    cfg.frequency_penalty = req.frequency_penalty;
    cfg.seed = req.seed;

    // Determine the prompt
    std::string prompt = req.prompt;
    if (!req.messages.empty() && req.prompt.empty()) {
        for (const auto& msg : req.messages) {
            prompt += msg.first + ": " + msg.second + "\n";
        }
        prompt += "assistant: ";
    }

    // Tokenize the prompt to count tokens
    if (engine_->tokenizer()) {
        uint32_t toks[8192];
        size_t count = 0;
        if (vaist_tokenize(engine_->tokenizer(), prompt.c_str(), toks, 8192, &count) == VAIST_OK) {
            result.prompt_tokens = static_cast<uint32_t>(count);
        }
    }

    // Run generation
    std::vector<uint32_t> out_tokens;
    VaistStatus status = engine_->generate(prompt.c_str(),
                                            req.max_tokens,
                                            cfg,
                                            &out_tokens);

    if (status != VAIST_OK) {
        result.stopped = true;
        return result;
    }

    // Detokenize
    std::vector<uint32_t> all_tokens;
    if (engine_->tokenizer()) {
        char buf[65536];
        size_t written = 0;
        if (vaist_detokenize(engine_->tokenizer(), out_tokens.data(), out_tokens.size(),
                              buf, sizeof(buf), &written) == VAIST_OK) {
            result.text.assign(buf, written);
        }
    }

    result.completion_tokens = static_cast<uint32_t>(out_tokens.size());
    result.stopped = (out_tokens.size() < req.max_tokens) ||
                     (!out_tokens.empty() && out_tokens.back() == 2);

    return result;
}

void VaistHttpServer::run_generation_stream(CompletionRequest& req,
                                             SOCKET sock,
                                             StreamCallback callback) {
    (void)sock;

    if (!engine_) return;

    // Build sampling config
    VaistSamplingConfig cfg;
    cfg.temperature = req.temperature;
    cfg.top_p = req.top_p;
    cfg.top_k = req.top_k;
    cfg.min_p = req.min_p;
    cfg.presence_penalty = req.presence_penalty;
    cfg.frequency_penalty = req.frequency_penalty;
    cfg.seed = req.seed;

    // Determine the prompt
    std::string prompt = req.prompt;
    if (!req.messages.empty() && req.prompt.empty()) {
        for (const auto& msg : req.messages) {
            prompt += msg.first + ": " + msg.second + "\n";
        }
        prompt += "assistant: ";
    }

    // Run generation, calling back per token
    std::vector<uint32_t> out_tokens;
    engine_->generate(prompt.c_str(), req.max_tokens, cfg, &out_tokens);

    // Emit each token as a stream event
    if (engine_->tokenizer()) {
        char buf[256];
        for (uint32_t tok : out_tokens) {
            size_t written = 0;
            if (vaist_detokenize(engine_->tokenizer(), &tok, 1, buf, sizeof(buf), &written) == VAIST_OK) {
                if (written > 0) {
                    callback(std::string_view(buf, written));
                }
            }
        }
    }
}

/* ======================================================================== */
/* SSE helpers                                                               */
/* ======================================================================== */

std::string VaistHttpServer::build_sse_chunk(const std::string& json) {
    std::string chunk = "data: ";
    chunk += json;
    chunk += "\n\n";
    return chunk;
}

std::string VaistHttpServer::build_sse_done() {
    return "data: [DONE]\n\n";
}

std::string VaistHttpServer::build_sse_usage(uint32_t prompt_tokens,
                                              uint32_t completion_tokens) {
    JsonWriter w;
    w.start_object();
    w.key("choices");
    w.start_array();
    w.start_object();
    w.key("index"); w.value(static_cast<int64_t>(0));
    w.start_object();
    w.key("delta");
    w.start_object();
    w.end_object();
    w.end_object();
    w.key("finish_reason"); w.value(nullptr);
    w.end_object();
    w.end_array();
    w.key("usage");
    w.start_object();
    w.key("prompt_tokens"); w.value(static_cast<int64_t>(prompt_tokens));
    w.key("completion_tokens"); w.value(static_cast<int64_t>(completion_tokens));
    w.key("total_tokens"); w.value(static_cast<int64_t>(prompt_tokens + completion_tokens));
    w.end_object();
    w.end_object();
    return build_sse_chunk(w.str());
}

} // namespace vaist
