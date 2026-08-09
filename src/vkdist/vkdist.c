/**
 * \file vkdist.c
 * \brief Distributed-compute remote-execution layer (Phase 0 loopback slice).
 *
 * Implements the vkdist client and server described in specs/VKDIST-DESIGN.md:
 *
 *  - Transport: Winsock2 on _WIN32, POSIX sockets otherwise. Blocking sockets
 *    with TCP_NODELAY; every read/write is a loop that transfers exactly the
 *    requested byte count so the stream can never desync on short transfers.
 *  - Framing: u32 LE length + u32 LE opcode + payload; all fields are packed
 *    little-endian explicitly (never by reinterpreting host structs).
 *  - Server: vkdist_server_run() creates a transient VkRuntime, a command
 *    pool/buffer/fence and a remote-buffer table, then serves HELLO /
 *    REGISTER_BUFFER / UPLOAD / DISPATCH_GEMM / READBACK until BYE. Every
 *    stage (upload/dispatch/readback) is separated by a full host-side wait,
 *    so no pipeline barriers are required (see spec section 5.3).
 *  - Client: fd-based request/reply helpers on top of the same framing.
 */

#include "vkdist.h"
#include "vkdist_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <errno.h>
#endif

/* ===========================================================================
 * Socket abstraction
 * ========================================================================== */

#ifdef _WIN32
typedef SOCKET vkdist_sock_t;
#define VKDIST_SOCK_INVALID INVALID_SOCKET
#define VKDIST_SOCK_ISERR(x) ((x) == VKDIST_SOCK_INVALID)
#else
typedef int vkdist_sock_t;
#define VKDIST_SOCK_INVALID (-1)
#define VKDIST_SOCK_ISERR(x) ((x) < 0)
#endif

static int vkdist_sock_startup(void)
{
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        return -1;
#endif
    return 0;
}

static void vkdist_sock_close(vkdist_sock_t s)
{
    if (VKDIST_SOCK_ISERR(s))
        return;
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

/* Send exactly n bytes. Returns 0 on success, -1 on error. */
static int vkdist_sock_send_all(vkdist_sock_t s, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    size_t sent = 0;
    while (sent < n) {
        int w = (int)send(s, p + sent, (int)(n - sent), 0);
        if (w < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEINTR || err == WSAEWOULDBLOCK)
                continue;
#else
            if (errno == EINTR)
                continue;
#endif
            return -1;
        }
        if (w == 0)
            return -1;
        sent += (size_t)w;
    }
    return 0;
}

/* Receive exactly n bytes. Returns 0 on success, 1 on clean peer close, -1 on
   error. */
static int vkdist_sock_recv_all(vkdist_sock_t s, void *buf, size_t n)
{
    char *p = (char *)buf;
    size_t got = 0;
    while (got < n) {
        int r = (int)recv(s, p + got, (int)(n - got), 0);
        if (r == 0)
            return 1; /* peer closed */
        if (r < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEINTR || err == WSAEWOULDBLOCK)
                continue;
#else
            if (errno == EINTR)
                continue;
#endif
            return -1;
        }
        got += (size_t)r;
    }
    return 0;
}

/* ===========================================================================
 * Little-endian pack / unpack
 * ========================================================================== */

static void vkdist_put_u32_le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void vkdist_put_u64_le(uint8_t *p, uint64_t v)
{
    vkdist_put_u32_le(p, (uint32_t)(v & 0xFFFFFFFFu));
    vkdist_put_u32_le(p + 4, (uint32_t)(v >> 32));
}

static void vkdist_put_i32_le(uint8_t *p, int32_t v)
{
    vkdist_put_u32_le(p, (uint32_t)v);
}

static void vkdist_put_f32_le(uint8_t *p, float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    vkdist_put_u32_le(p, u);
}

static uint32_t vkdist_get_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint64_t vkdist_get_u64_le(const uint8_t *p)
{
    uint64_t lo = vkdist_get_u32_le(p);
    uint64_t hi = vkdist_get_u32_le(p + 4);
    return lo | (hi << 32);
}

static int32_t vkdist_get_i32_le(const uint8_t *p)
{
    return (int32_t)vkdist_get_u32_le(p);
}

static float vkdist_get_f32_le(const uint8_t *p)
{
    uint32_t u = vkdist_get_u32_le(p);
    float v;
    memcpy(&v, &u, sizeof(v));
    return v;
}

/* ===========================================================================
 * Framing
 * ========================================================================== */

/* Send one frame: u32 LE length + u32 LE opcode + payload. */
static int vkdist_frame_send(vkdist_sock_t s, uint32_t opcode,
                             const uint8_t *payload, uint32_t len)
{
    uint8_t hdr[8];
    vkdist_put_u32_le(hdr, len);
    vkdist_put_u32_le(hdr + 4, opcode);
    if (vkdist_sock_send_all(s, hdr, sizeof(hdr)) != 0)
        return -1;
    if (len > 0 && payload != NULL && vkdist_sock_send_all(s, payload, len) != 0)
        return -1;
    return 0;
}

/* Receive one frame into buf (cap buf_cap). Returns 0 on success (out params
   set), 1 on clean peer close, -1 on transport error, -2 on oversize frame. */
static int vkdist_frame_recv(vkdist_sock_t s, uint8_t *buf, uint32_t buf_cap,
                             uint32_t *out_opcode, uint32_t *out_len)
{
    uint8_t hdr[8];
    int r = vkdist_sock_recv_all(s, hdr, sizeof(hdr));
    if (r == 1)
        return 1;
    if (r != 0)
        return -1;

    uint32_t len = vkdist_get_u32_le(hdr);
    uint32_t opcode = vkdist_get_u32_le(hdr + 4);
    if (len > buf_cap)
        return -2;

    if (len > 0) {
        r = vkdist_sock_recv_all(s, buf, len);
        if (r == 1)
            return 1;
        if (r != 0)
            return -1;
    }
    *out_opcode = opcode;
    *out_len = len;
    return 0;
}

/* ===========================================================================
 * Server helpers
 * ========================================================================== */

/* Reply payload = i32 result. */
static int vkdist_server_send_result(vkdist_sock_t s, uint32_t opcode,
                                     int32_t result)
{
    uint8_t p[4];
    vkdist_put_i32_le(p, result);
    return vkdist_frame_send(s, opcode, p, sizeof(p));
}

/* Reply payload = i32 result + u64 handle. */
static int vkdist_server_send_result_u64(vkdist_sock_t s, uint32_t opcode,
                                         int32_t result, uint64_t v)
{
    uint8_t p[12];
    vkdist_put_i32_le(p, result);
    vkdist_put_u64_le(p + 4, v);
    return vkdist_frame_send(s, opcode, p, sizeof(p));
}

/* Reply payload = i32 result + raw readback bytes. */
static int vkdist_server_send_readback(vkdist_sock_t s, int32_t result,
                                       const uint8_t *data, uint32_t data_len)
{
    size_t total = 4u + data_len;
    uint8_t *p = (uint8_t *)malloc(total > 0 ? total : 1);
    if (!p)
        return -1;
    vkdist_put_i32_le(p, result);
    if (data_len > 0)
        memcpy(p + 4, data, data_len);
    int rc = vkdist_frame_send(s, VKDIST_OP_READBACK, p, (uint32_t)total);
    free(p);
    return rc;
}

static vkdist_remote_buffer_t *vkdist_server_find_buffer(
    vkdist_server_state_t *st, uint64_t handle)
{
    for (uint32_t i = 0; i < st->buffer_count; i++) {
        if (st->buffers[i].handle == handle)
            return &st->buffers[i];
    }
    return NULL;
}

/* Dispatch handler: record vkblas_sgemm into the connection command buffer,
   submit with the fence, wait, and reset. C is used in place (read for beta,
   written as D); the GEMM shader reads data_C before writing data_D for the
   same element, so in-place is safe. Returns the final VkResult. */
static VkResult vkdist_server_dispatch(vkdist_server_state_t *st, int32_t m,
                                       int32_t n, int32_t k, float alpha,
                                       VkBuffer A, int32_t lda, VkBuffer B,
                                       int32_t ldb, VkBuffer C, int32_t ldc,
                                       float beta)
{
    VkResult vr;

    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vr = vkBeginCommandBuffer(st->cmd, &begin);
    if (vr == VK_SUCCESS)
        vr = vkblas_sgemm(st->blas, st->cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                          m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
                          C, ldc);
    if (vr == VK_SUCCESS)
        vr = vkEndCommandBuffer(st->cmd);

    if (vr == VK_SUCCESS) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &st->cmd;
        vr = vkQueueSubmit(st->queue, 1, &submit, st->fence);
        if (vr == VK_SUCCESS) {
            vr = vkWaitForFences(st->dev, 1, &st->fence, VK_TRUE, UINT64_MAX);
            vkResetFences(st->dev, 1, &st->fence);
        }
    }

    vkResetCommandBuffer(st->cmd, 0);
    return vr;
}

/* ===========================================================================
 * Server entry points
 * ========================================================================== */

int vkdist_server_start(uint16_t port, const char *ip)
{
    if (vkdist_sock_startup() != 0)
        return -1;

    vkdist_sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (VKDIST_SOCK_ISERR(s))
        return -1;

    int opt = 1;
#ifdef _WIN32
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&opt, sizeof(opt));
#else
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (ip != NULL && ip[0] != '\0') {
        if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
            vkdist_sock_close(s);
            return -1;
        }
    } else {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    }

    if (bind(s, (struct sockaddr *)&addr, (int)sizeof(addr)) != 0) {
        vkdist_sock_close(s);
        return -1;
    }
    if (listen(s, VKDIST_LISTEN_BACKLOG) != 0) {
        vkdist_sock_close(s);
        return -1;
    }
    return (int)s;
}

int vkdist_server_accept(int listen_fd)
{
    vkdist_sock_t s = (vkdist_sock_t)listen_fd;
    struct sockaddr_in addr;
#ifdef _WIN32
    int alen = (int)sizeof(addr);
#else
    socklen_t alen = (socklen_t)sizeof(addr);
#endif
    vkdist_sock_t c = accept(s, (struct sockaddr *)&addr, &alen);
    if (VKDIST_SOCK_ISERR(c))
        return -1;

    int opt = 1;
#ifdef _WIN32
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *)&opt, sizeof(opt));
#else
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif
    return (int)c;
}

VkResult vkdist_server_run(VkPhysicalDevice pd, VkDevice dev,
                           VkBLASContext *blas, int conn_fd)
{
    if (dev == VK_NULL_HANDLE || blas == NULL || conn_fd < 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    vkdist_sock_t s = (vkdist_sock_t)conn_fd;
    VkResult result = VK_ERROR_UNKNOWN;

    vkdist_server_state_t st;
    memset(&st, 0, sizeof(st));
    st.pd = pd;
    st.dev = dev;
    st.blas = blas;
    st.next_handle = 1;

    /* Setup: queue (family 0), transient runtime, command pool/buffer/fence. */
    vkGetDeviceQueue(dev, 0, 0, &st.queue);

    VkResult vr = vkr_create_runtime(pd, dev, st.queue, &st.runtime);
    if (vr != VK_SUCCESS)
        goto out_socket;

    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = 0;
    vr = vkCreateCommandPool(dev, &pool_info, NULL, &st.cmd_pool);
    if (vr != VK_SUCCESS)
        goto out_runtime;

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = st.cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    vr = vkAllocateCommandBuffers(dev, &alloc_info, &st.cmd);
    if (vr != VK_SUCCESS)
        goto out_pool;

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vr = vkCreateFence(dev, &fence_info, NULL, &st.fence);
    if (vr != VK_SUCCESS)
        goto out_cmd;

    /* One fixed-size receive buffer reused for the whole session. */
    uint8_t *frame = (uint8_t *)malloc(VKDIST_MAX_FRAME);
    if (!frame) {
        vr = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out_fence;
    }

    /* ── RPC loop ─────────────────────────────────────────────────────── */
    uint32_t opcode = 0;
    uint32_t len = 0;
    for (;;) {
        int fr = vkdist_frame_recv(s, frame, VKDIST_MAX_FRAME, &opcode, &len);
        if (fr == 1) {                       /* peer closed cleanly */
            result = VK_SUCCESS;
            break;
        }
        if (fr != 0) {                   /* transport or oversize error */
            result = VK_ERROR_UNKNOWN;
            break;
        }

        if (opcode == VKDIST_OP_HELLO) {
            if (len != 4 ||
                vkdist_get_u32_le(frame) != VKDIST_PROTOCOL_VERSION) {
                vkdist_server_send_result(s, opcode,
                                          (int32_t)VK_ERROR_INCOMPATIBLE_DRIVER);
                result = VK_ERROR_INCOMPATIBLE_DRIVER;
                break;
            }
            if (vkdist_server_send_result(s, opcode, (int32_t)VK_SUCCESS) != 0) {
                result = VK_ERROR_UNKNOWN;
                break;
            }
            continue;
        }

        if (opcode == VKDIST_OP_REGISTER_BUFFER) {
            if (len != 8) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                result = VK_ERROR_UNKNOWN;
                break;
            }
            if (st.buffer_count >= VKDIST_MAX_BUFFERS) {
                vkdist_server_send_result_u64(s, opcode,
                    (int32_t)VK_ERROR_OUT_OF_DEVICE_MEMORY, 0);
                continue;
            }
            uint64_t size = vkdist_get_u64_le(frame);
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            vr = vkr_malloc(st.runtime, (VkDeviceSize)size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            &buf, &mem);
            if (vr != VK_SUCCESS) {
                vkdist_server_send_result_u64(s, opcode, (int32_t)vr, 0);
                continue;
            }
            vkdist_remote_buffer_t *slot = &st.buffers[st.buffer_count++];
            slot->handle = st.next_handle++;
            slot->buffer = buf;
            slot->memory = mem;
            slot->size = (VkDeviceSize)size;
            vkdist_server_send_result_u64(s, opcode, (int32_t)VK_SUCCESS,
                                          slot->handle);
            continue;
        }

        if (opcode == VKDIST_OP_UPLOAD) {
            if (len < 24) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                result = VK_ERROR_UNKNOWN;
                break;
            }
            uint64_t handle = vkdist_get_u64_le(frame);
            uint64_t offset = vkdist_get_u64_le(frame + 8);
            uint64_t size = vkdist_get_u64_le(frame + 16);
            if (len != 24u + (uint32_t)size) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                result = VK_ERROR_UNKNOWN;
                break;
            }
            vkdist_remote_buffer_t *b = vkdist_server_find_buffer(&st, handle);
            if (b == NULL || offset + size > (uint64_t)b->size) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                continue;
            }
            vr = vkr_upload(st.runtime, st.cmd, st.queue, frame + 24,
                            b->buffer, (VkDeviceSize)offset, (VkDeviceSize)size);
            vkdist_server_send_result(s, opcode, (int32_t)vr);
            continue;
        }

        if (opcode == VKDIST_OP_DISPATCH_GEMM) {
            if (len != 56) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                result = VK_ERROR_UNKNOWN;
                break;
            }
            int32_t m = vkdist_get_i32_le(frame);
            int32_t n = vkdist_get_i32_le(frame + 4);
            int32_t k = vkdist_get_i32_le(frame + 8);
            float alpha = vkdist_get_f32_le(frame + 12);
            uint64_t hA = vkdist_get_u64_le(frame + 16);
            uint64_t hB = vkdist_get_u64_le(frame + 24);
            uint64_t hC = vkdist_get_u64_le(frame + 32);
            float beta = vkdist_get_f32_le(frame + 40);
            int32_t lda = vkdist_get_i32_le(frame + 44);
            int32_t ldb = vkdist_get_i32_le(frame + 48);
            int32_t ldc = vkdist_get_i32_le(frame + 52);

            vkdist_remote_buffer_t *A = vkdist_server_find_buffer(&st, hA);
            vkdist_remote_buffer_t *B = vkdist_server_find_buffer(&st, hB);
            vkdist_remote_buffer_t *C = vkdist_server_find_buffer(&st, hC);
            int valid = (m > 0 && n > 0 && k > 0)
                     && (lda >= m && ldb >= k && ldc >= m);
            if (!valid || A == NULL || B == NULL || C == NULL) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                continue;
            }
            uint64_t a_needed = (uint64_t)((int64_t)(k - 1) * lda + m) * 4u;
            uint64_t b_needed = (uint64_t)((int64_t)(n - 1) * ldb + k) * 4u;
            uint64_t c_needed = (uint64_t)((int64_t)(n - 1) * ldc + m) * 4u;
            if ((uint64_t)A->size < a_needed || (uint64_t)B->size < b_needed
                || (uint64_t)C->size < c_needed) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                continue;
            }

            vr = vkdist_server_dispatch(&st, m, n, k, alpha, A->buffer, lda,
                                        B->buffer, ldb, C->buffer, ldc, beta);
            vkdist_server_send_result(s, opcode, (int32_t)vr);
            continue;
        }

        if (opcode == VKDIST_OP_READBACK) {
            if (len != 24) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                result = VK_ERROR_UNKNOWN;
                break;
            }
            uint64_t handle = vkdist_get_u64_le(frame);
            uint64_t offset = vkdist_get_u64_le(frame + 8);
            uint64_t size = vkdist_get_u64_le(frame + 16);
            if (size > (uint64_t)(VKDIST_MAX_FRAME - 4u)) {
                vkdist_server_send_result(s, opcode,
                                          (int32_t)VK_ERROR_OUT_OF_DEVICE_MEMORY);
                continue;
            }
            vkdist_remote_buffer_t *b = vkdist_server_find_buffer(&st, handle);
            if (b == NULL || offset + size > (uint64_t)b->size) {
                vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
                continue;
            }
            void *host = malloc(size > 0 ? (size_t)size : 1);
            if (!host) {
                vkdist_server_send_result(s, opcode,
                                          (int32_t)VK_ERROR_OUT_OF_HOST_MEMORY);
                continue;
            }
            vr = vkr_download(st.runtime, st.cmd, st.queue, b->buffer,
                              (VkDeviceSize)offset, host, (VkDeviceSize)size);
            if (vr == VK_SUCCESS) {
                if (vkdist_server_send_readback(s, (int32_t)VK_SUCCESS,
                                                (const uint8_t *)host,
                                                (uint32_t)size) != 0)
                    vr = VK_ERROR_UNKNOWN;
            } else {
                vkdist_server_send_result(s, opcode, (int32_t)vr);
            }
            free(host);
            if (vr != VK_SUCCESS) {
                result = VK_ERROR_UNKNOWN;
                break;
            }
            continue;
        }

        if (opcode == VKDIST_OP_BYE) {
            result = VK_SUCCESS; /* graceful close, no reply */
            break;
        }

        /* Unknown opcode: desync — reply, then close. */
        vkdist_server_send_result(s, opcode, (int32_t)VK_ERROR_UNKNOWN);
        result = VK_ERROR_UNKNOWN;
        break;
    }

    free(frame);

    /* Release every remote buffer handle (vkr_destroy_runtime frees the block
       memory but does not destroy the VkBuffer objects). */
    for (uint32_t i = 0; i < st.buffer_count; i++)
        vkDestroyBuffer(dev, st.buffers[i].buffer, NULL);

out_fence:
    if (st.fence != VK_NULL_HANDLE)
        vkDestroyFence(dev, st.fence, NULL);
out_cmd:
    if (st.cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(dev, st.cmd_pool, 1, &st.cmd);
out_pool:
    if (st.cmd_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(dev, st.cmd_pool, NULL);
out_runtime:
    if (st.runtime != NULL)
        vkr_destroy_runtime(st.runtime);
out_socket:
    vkdist_sock_close(s);
    return result;
}

/* ===========================================================================
 * Client helpers
 * ========================================================================== */

/* Receive a reply frame, validate the opcode echo, return its payload length.
   Returns 0 on success, -1 on error. */
static int vkdist_client_recv_frame(vkdist_sock_t s, uint32_t expect_opcode,
                                    uint8_t *buf, uint32_t buf_cap,
                                    uint32_t *out_len)
{
    uint8_t hdr[8];
    int r = vkdist_sock_recv_all(s, hdr, sizeof(hdr));
    if (r != 0)
        return -1;
    uint32_t len = vkdist_get_u32_le(hdr);
    uint32_t opcode = vkdist_get_u32_le(hdr + 4);
    if (opcode != expect_opcode || len > buf_cap)
        return -1;
    if (len > 0 && vkdist_sock_recv_all(s, buf, len) != 0)
        return -1;
    *out_len = len;
    return 0;
}

static int vkdist_client_recv_result(vkdist_sock_t s, uint32_t expect_opcode,
                                     int32_t *out_result)
{
    uint8_t buf[4];
    uint32_t len = 0;
    if (vkdist_client_recv_frame(s, expect_opcode, buf, sizeof(buf), &len) != 0)
        return -1;
    if (len != 4)
        return -1;
    *out_result = vkdist_get_i32_le(buf);
    return 0;
}

/* ===========================================================================
 * Client entry points
 * ========================================================================== */

int vkdist_client_connect(const char *ip, uint16_t port)
{
    if (ip == NULL || ip[0] == '\0')
        return -1;
    if (vkdist_sock_startup() != 0)
        return -1;

    vkdist_sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (VKDIST_SOCK_ISERR(s))
        return -1;

    int opt = 1;
#ifdef _WIN32
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&opt, sizeof(opt));
#else
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        vkdist_sock_close(s);
        return -1;
    }
    if (connect(s, (struct sockaddr *)&addr, (int)sizeof(addr)) != 0) {
        vkdist_sock_close(s);
        return -1;
    }

    /* HELLO handshake before any op is accepted. */
    uint8_t payload[4];
    vkdist_put_u32_le(payload, VKDIST_PROTOCOL_VERSION);
    if (vkdist_frame_send(s, VKDIST_OP_HELLO, payload, sizeof(payload)) != 0) {
        vkdist_sock_close(s);
        return -1;
    }
    int32_t result = 0;
    if (vkdist_client_recv_result(s, VKDIST_OP_HELLO, &result) != 0 ||
        result != 0) {
        vkdist_sock_close(s);
        return -1;
    }
    return (int)s;
}

VkResult vkdist_register_buffer(int fd, VkDeviceSize size, uint64_t *handle)
{
    if (handle != NULL)
        *handle = 0;
    if (fd < 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint8_t payload[8];
    vkdist_put_u64_le(payload, (uint64_t)size);
    if (vkdist_frame_send((vkdist_sock_t)fd, VKDIST_OP_REGISTER_BUFFER,
                          payload, sizeof(payload)) != 0)
        return VK_ERROR_UNKNOWN;

    uint8_t reply[12];
    uint32_t len = 0;
    if (vkdist_client_recv_frame((vkdist_sock_t)fd, VKDIST_OP_REGISTER_BUFFER,
                                 reply, sizeof(reply), &len) != 0)
        return VK_ERROR_UNKNOWN;
    if (len != 12)
        return VK_ERROR_UNKNOWN;

    int32_t result = vkdist_get_i32_le(reply);
    if (result == 0 && handle != NULL)
        *handle = vkdist_get_u64_le(reply + 4);
    return (VkResult)result;
}

VkResult vkdist_upload(int fd, uint64_t handle, const void *host,
                       VkDeviceSize offset, VkDeviceSize size)
{
    if (fd < 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint64_t total = 24u + (uint64_t)size;
    if (total > VKDIST_MAX_FRAME)
        return VK_ERROR_UNKNOWN;

    uint8_t *p = (uint8_t *)malloc(total > 0 ? (size_t)total : 1);
    if (!p)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    vkdist_put_u64_le(p, handle);
    vkdist_put_u64_le(p + 8, (uint64_t)offset);
    vkdist_put_u64_le(p + 16, (uint64_t)size);
    if (size > 0 && host != NULL)
        memcpy(p + 24, host, (size_t)size);

    int rc = vkdist_frame_send((vkdist_sock_t)fd, VKDIST_OP_UPLOAD, p,
                               (uint32_t)total);
    free(p);
    if (rc != 0)
        return VK_ERROR_UNKNOWN;

    int32_t result = 0;
    if (vkdist_client_recv_result((vkdist_sock_t)fd, VKDIST_OP_UPLOAD,
                                  &result) != 0)
        return VK_ERROR_UNKNOWN;
    return (VkResult)result;
}

VkResult vkdist_sgemm(int fd, int32_t m, int32_t n, int32_t k,
                      const float *alpha, uint64_t A, uint64_t B, uint64_t C,
                      const float *beta, int32_t lda, int32_t ldb, int32_t ldc)
{
    if (fd < 0 || alpha == NULL || beta == NULL)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint8_t payload[56];
    vkdist_put_i32_le(payload, m);
    vkdist_put_i32_le(payload + 4, n);
    vkdist_put_i32_le(payload + 8, k);
    vkdist_put_f32_le(payload + 12, *alpha);
    vkdist_put_u64_le(payload + 16, A);
    vkdist_put_u64_le(payload + 24, B);
    vkdist_put_u64_le(payload + 32, C);
    vkdist_put_f32_le(payload + 40, *beta);
    vkdist_put_i32_le(payload + 44, lda);
    vkdist_put_i32_le(payload + 48, ldb);
    vkdist_put_i32_le(payload + 52, ldc);

    if (vkdist_frame_send((vkdist_sock_t)fd, VKDIST_OP_DISPATCH_GEMM, payload,
                          sizeof(payload)) != 0)
        return VK_ERROR_UNKNOWN;

    int32_t result = 0;
    if (vkdist_client_recv_result((vkdist_sock_t)fd, VKDIST_OP_DISPATCH_GEMM,
                                  &result) != 0)
        return VK_ERROR_UNKNOWN;
    return (VkResult)result;
}

VkResult vkdist_readback(int fd, uint64_t handle, VkDeviceSize offset,
                         VkDeviceSize size, void *host)
{
    if (fd < 0 || (size > 0 && host == NULL))
        return VK_ERROR_INITIALIZATION_FAILED;
    if ((uint64_t)size + 4u > VKDIST_MAX_FRAME)
        return VK_ERROR_UNKNOWN;

    uint8_t payload[24];
    vkdist_put_u64_le(payload, handle);
    vkdist_put_u64_le(payload + 8, (uint64_t)offset);
    vkdist_put_u64_le(payload + 16, (uint64_t)size);
    if (vkdist_frame_send((vkdist_sock_t)fd, VKDIST_OP_READBACK, payload,
                          sizeof(payload)) != 0)
        return VK_ERROR_UNKNOWN;

    uint32_t reply_cap = 4u + (uint32_t)size;
    uint8_t *reply = (uint8_t *)malloc((size_t)reply_cap);
    if (!reply)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    uint32_t len = 0;
    int rc = vkdist_client_recv_frame((vkdist_sock_t)fd, VKDIST_OP_READBACK,
                                      reply, reply_cap, &len);
    if (rc != 0) {
        free(reply);
        return VK_ERROR_UNKNOWN;
    }
    if (len < 4) {
        free(reply);
        return VK_ERROR_UNKNOWN;
    }
    int32_t result = vkdist_get_i32_le(reply);
    if (result == 0) {
        if (len != reply_cap) {
            free(reply);
            return VK_ERROR_UNKNOWN;
        }
        if (size > 0)
            memcpy(host, reply + 4, (size_t)size);
    }
    free(reply);
    return (VkResult)result;
}

void vkdist_close(int fd)
{
    if (fd < 0)
        return;
    vkdist_sock_t s = (vkdist_sock_t)fd;
    /* Best-effort BYE so the server tears down its session cleanly. */
    vkdist_frame_send(s, VKDIST_OP_BYE, NULL, 0);
    vkdist_sock_close(s);
}
