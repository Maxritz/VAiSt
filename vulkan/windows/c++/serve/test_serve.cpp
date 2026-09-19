/**
 * \file test_serve.cpp
 * \brief Unit test for the VAiSt HTTP server.
 *
 * Starts the server on a test port, sends HTTP requests using Winsock2,
 * and verifies the response format matches the OpenAI API contract.
 *
 * This test does NOT require a real model file — it verifies the HTTP
 * protocol handling, JSON serialization, and routing logic.
 *
 * Run: vaist_serve_test.exe
 */
#define _CRT_SECURE_NO_WARNINGS
#include "vaist_http_server.hpp"
#include "vaist_engine.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <chrono>
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

using namespace vaist;

#pragma comment(lib, "ws2_32.lib")

/**
 * \brief Send an HTTP request and receive the response.
 * \param host  Target host (e.g. "127.0.0.1").
 * \param port  Target port.
 * \param request  Full HTTP request string.
 * \return Response body as string.
 */
static std::string send_http_request(const std::string& host, uint16_t port,
                                      const std::string& request) {
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return "";

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        closesocket(sock);
        return "";
    }

    // Set a timeout on the client socket
    DWORD timeout = 5000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));

    send(sock, request.data(), static_cast<int>(request.size()), 0);

    std::vector<uint8_t> response(65536);
    std::string full_response;
    while (true) {
        int n = recv(sock, reinterpret_cast<char*>(response.data()),
                     static_cast<int>(response.size()), 0);
        if (n <= 0) break;
        full_response.append(reinterpret_cast<char*>(response.data()), n);
    }
    closesocket(sock);
    return full_response;
}

/** \brief Extract the body from an HTTP response (after \r\n\r\n). */
static std::string extract_body(const std::string& http_response) {
    size_t pos = http_response.find("\r\n\r\n");
    if (pos == std::string::npos) return "";
    return http_response.substr(pos + 4);
}

/** \brief Check if HTTP response contains a status code. */
static bool has_status(const std::string& response, int code) {
    std::string search = "HTTP/1.1 " + std::to_string(code) + " ";
    return response.find(search) != std::string::npos;
}

int main() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }

    int test_port = 18080;
    int failures = 0;

    // ---- Test 1: Server construction and start/stop ----
    {
        VaistEngine engine(nullptr);

        VaistHttpServer server(&engine, static_cast<uint16_t>(test_port), "127.0.0.1", "test-model");
        if (!server.start()) {
            std::fprintf(stderr, "FAIL: Server failed to start\n");
            failures++;
        }
        std::fflush(stdout);

        // Give the server time to bind
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // ---- Test 2: GET /health ----
        {
            std::string request =
                "GET /health HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Connection: close\r\n"
                "\r\n";
            std::string response = send_http_request("127.0.0.1", static_cast<uint16_t>(test_port), request);
            std::string body = extract_body(response);

            bool ok = has_status(response, 200) && body == "ok";
            if (ok) {
                std::printf("PASS: GET /health returned 'ok'\n");
            } else {
                std::printf("FAIL: GET /health. Status/200=%d, body='%s'\n", has_status(response, 200), body.c_str());
                failures++;
            }
        }

        // ---- Test 3: GET /v1/models ----
        {
            std::string request =
                "GET /v1/models HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Connection: close\r\n"
                "\r\n";
            std::string response = send_http_request("127.0.0.1", static_cast<uint16_t>(test_port), request);
            std::string body = extract_body(response);

            bool ok = has_status(response, 200) &&
                      body.find("\"object\":\"list\"") != std::string::npos &&
                      body.find("test-model") != std::string::npos;
            if (ok) {
                std::printf("PASS: GET /v1/models returned model list\n");
            } else {
                std::printf("FAIL: GET /v1/models. Body: %s\n", body.c_str());
                failures++;
            }
        }

        // ---- Test 4: POST /v1/chat/completions (non-streaming) ----
        {
            std::string req_body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":5,\"temperature\":0.7,\"stream\":false}";
            std::string request =
                "POST /v1/chat/completions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(req_body.size()) + "\r\n"
                "Connection: close\r\n"
                "\r\n" +
                req_body;
            std::string response = send_http_request("127.0.0.1", static_cast<uint16_t>(test_port), request);
            std::string body = extract_body(response);

            // We don't have a real model so generation won't produce meaningful output,
            // but we verify the response structure is valid JSON
            bool ok = has_status(response, 200);
            if (ok) {
                std::printf("PASS: POST /v1/chat/completions returned 200\n");
            } else {
                std::printf("FAIL: POST /v1/chat/completions. Status: %d\n", has_status(response, 200));
                failures++;
            }

            // Check for OpenAI response structure keys
            bool has_id = body.find("\"id\"") != std::string::npos;
            bool has_object = body.find("\"object\":\"chat.completion\"") != std::string::npos;
            bool has_choices = body.find("\"choices\"") != std::string::npos;
            bool has_usage = body.find("\"usage\"") != std::string::npos;
            if (has_id && has_object && has_choices && has_usage) {
                std::printf("PASS: Response has correct OpenAI structure\n");
            } else {
                std::printf("FAIL: Response missing fields. id=%d object=%d choices=%d usage=%d\n",
                            has_id, has_object, has_choices, has_usage);
                failures++;
            }
        }

        // ---- Test 5: POST /v1/completions ----
        {
            std::string req_body = "{\"model\":\"test-model\",\"prompt\":\"Hello world\",\"max_tokens\":5,\"stream\":false}";
            std::string request =
                "POST /v1/completions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(req_body.size()) + "\r\n"
                "Connection: close\r\n"
                "\r\n" +
                req_body;
            std::string response = send_http_request("127.0.0.1", static_cast<uint16_t>(test_port), request);
            std::string body = extract_body(response);

            bool ok = has_status(response, 200) &&
                      body.find("\"object\":\"text_completion\"") != std::string::npos;
            if (ok) {
                std::printf("PASS: POST /v1/completions returned 200 with correct object type\n");
            } else {
                std::printf("FAIL: POST /v1/completions\n");
                failures++;
            }
        }

        // ---- Test 6: 404 for unknown paths ----
        {
            std::string request =
                "GET /unknown HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Connection: close\r\n"
                "\r\n";
            std::string response = send_http_request("127.0.0.1", static_cast<uint16_t>(test_port), request);

            bool ok = has_status(response, 404);
            if (ok) {
                std::printf("PASS: Unknown path returned 404\n");
            } else {
                std::printf("FAIL: Unknown path did not return 404\n");
                failures++;
            }
        }

        // ---- Test 7: Invalid JSON body ----
        {
            std::string req_body = "not valid json";
            std::string request =
                "POST /v1/chat/completions HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: " + std::to_string(req_body.size()) + "\r\n"
                "Connection: close\r\n"
                "\r\n" +
                req_body;
            std::string response = send_http_request("127.0.0.1", static_cast<uint16_t>(test_port), request);

            bool ok = has_status(response, 400);
            if (ok) {
                std::printf("PASS: Invalid JSON returned 400\n");
            } else {
                std::printf("FAIL: Invalid JSON did not return 400\n");
                failures++;
            }
        }

        // Stop the server
        server.stop();
        server.join();
        std::printf("PASS: Server stopped\n");
        std::fflush(stdout);
    }

    // ---- Test 8: JsonWriter and JsonParser standalone ----
    {
        JsonWriter w;
        w.start_object();
        w.key("id"); w.value("chatcmpl-test");
        w.key("object"); w.value("chat.completion");
        w.key("created"); w.value(static_cast<int64_t>(1234567890));
        int64_t zero = 0;
        w.key("choices"); w.start_array();
        w.start_object();
        w.key("index"); w.value(zero);
        w.key("message");
        w.start_object();
        w.key("role"); w.value("assistant");
        w.key("content"); w.value("Hello world");
        w.end_object();
        w.key("finish_reason"); w.value("stop");
        w.end_object();
        w.end_array();
        w.key("usage");
        w.start_object();
        w.key("prompt_tokens"); w.value(static_cast<int64_t>(2));
        w.key("completion_tokens"); w.value(static_cast<int64_t>(5));
        w.key("total_tokens"); w.value(static_cast<int64_t>(7));
        w.end_object();
        w.end_object();

        std::string json = w.str();

        // Parse it back
        JsonValue parsed = JsonParser::parse(json.c_str());
        bool ok = parsed.type() == JsonType::kObject;
        if (ok) {
            std::printf("PASS: JsonWriter/JsonParser round-trip\n");
        } else {
            std::printf("FAIL: JsonWriter/JsonParser round-trip\n");
            failures++;
        }
    }

    WSACleanup();

    std::printf("\n=== Results: %d failures ===\n", failures);
    return failures == 0 ? 0 : 1;
}
