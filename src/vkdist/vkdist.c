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
 *  - Phase 1 multi-connection: vkdist_server_accept_many() accepts N clients
 *    and vkdist_server_serve_many() serves them concurrently, one POSIX
 *    pthread per connection, each running the same per-connection loop. The
 *    VkBLASContext is shared and not thread-safe (lazy pipeline cache +
 *    descriptor pool), so serve_many serializes the vkblas_sgemm dispatch with
 *    a mutex; everything else (per-connection runtime, command buffer, buffer
 *    table, queue submits) is already independent per connection.
 *  - Phase 1 client: vkdist_sgemm_partitioned() splits C's columns across
 *    worker connections (n_i = n/n_workers, last takes the remainder),
 *    registers/uploads A + the B column strip + the current C strip per
 *    worker, dispatches an sgemm per worker, and merges the read-back strips
 *    into the host C buffer.
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

/* POSIX threads for vkdist_server_serve_many. On Windows winsock2.h is
   included above (before anything that might pull in windows.h), and this
   MinGW-W64 build uses the "posix" threading model, so winpthreads' pthread.h
   is available and links with -lpthread. */
#include <pthread.h>

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
    if (vr == VK_SUCCESS) {
        /* The VkBLASContext is shared across connections in serve_many and is
           not thread-safe (lazy pipeline cache + descriptor pool). Take the
           caller-provided lock around the context-mutating call; each
           connection's command buffer and queue submits stay independent. */
        pthread_mutex_t *lock = (pthread_mutex_t *)st->blas_lock;
        if (lock != NULL)
            pthread_mutex_lock(lock);
        vr = vkblas_sgemm(st->blas, st->cmd, VKBLAS_OP_N, VKBLAS_OP_N,
                          m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
                          C, ldc);
        if (lock != NULL)
            pthread_mutex_unlock(lock);
    }
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

/* Single-connection serve loop. blas_lock may be NULL (no serialization) or a
   pthread_mutex_t* shared by serve_many worker threads. */
static VkResult vkdist_server_run_ex(VkPhysicalDevice pd, VkDevice dev,
                                     VkBLASContext *blas, int conn_fd,
                                     void *blas_lock)
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
    st.blas_lock = blas_lock;
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

        if (opcode == VKDIST_OP_CAPS) {
            /* Reply: i32 result + VKDIST_CAPS_PAYLOAD_SIZE bytes. */
            uint8_t p[4 + VKDIST_CAPS_PAYLOAD_SIZE];
            vkdist_put_i32_le(p, 0);

            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(st.pd, &props);
            VkRuntimeCaps rcaps;
            memset(&rcaps, 0, sizeof(rcaps));
            vkr_detect_capabilities(st.pd, st.dev, &rcaps);

            VkPhysicalDeviceMemoryProperties mem;
            vkGetPhysicalDeviceMemoryProperties(st.pd, &mem);
            uint64_t vram_total = 0, vram_free = 0;
            /* Sum distinct device-local heaps once each (multiple memory types
               can reference the same heap, so summing per-type double counts). */
            for (uint32_t h = 0; h < mem.memoryHeapCount; h++) {
                if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    vram_total += mem.memoryHeaps[h].size;
                    vram_free  += mem.memoryHeaps[h].size;
                }
            }

            uint8_t *q = p + 4;
            vkdist_put_u32_le(q, VKDIST_PROTOCOL_VERSION);
            vkdist_put_u64_le(q + 4, vram_total);
            vkdist_put_u64_le(q + 12, vram_free);
            vkdist_put_u32_le(q + 20, rcaps.arch_index);
            vkdist_put_u32_le(q + 24, rcaps.subgroup_size);
            vkdist_put_u32_le(q + 28, VKDIST_MAX_FRAME);
            memset(q + 32, 0, 128);
            strncpy((char *)q + 32, props.deviceName, 127);

            if (vkdist_frame_send(s, opcode, p, sizeof(p)) != 0) {
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

VkResult vkdist_server_run(VkPhysicalDevice pd, VkDevice dev,
                           VkBLASContext *blas, int conn_fd)
{
    return vkdist_server_run_ex(pd, dev, blas, conn_fd, NULL);
}

VkResult vkdist_server_accept_many(int listen_fd, uint32_t n, int *conn_fds)
{
    if (conn_fds == NULL || n == 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < n; i++) {
        int c = vkdist_server_accept(listen_fd);
        if (c < 0) {
            /* Roll back: close everything accepted so far. */
            for (uint32_t j = 0; j < i; j++)
                vkdist_sock_close((vkdist_sock_t)conn_fds[j]);
            return VK_ERROR_UNKNOWN;
        }
        conn_fds[i] = c;
    }
    return VK_SUCCESS;
}

/* ── serve_many: one pthread per connection, each running vkdist_server_run.
   Per-connection state (runtime, command pool/buffer/fence, buffer table) is
   created inside run_ex, so sessions are independent; only the shared
   VkBLASContext needs the mutex, which run_ex takes around vkblas_sgemm. ── */

typedef struct {
    VkPhysicalDevice pd;
    VkDevice         dev;
    VkBLASContext   *blas;
    int              conn_fd;
    pthread_mutex_t *blas_lock;
    VkResult         result;
} vkdist_serve_thread_arg_t;

static void *vkdist_serve_thread_fn(void *arg)
{
    vkdist_serve_thread_arg_t *a = (vkdist_serve_thread_arg_t *)arg;
    a->result = vkdist_server_run_ex(a->pd, a->dev, a->blas, a->conn_fd,
                                     a->blas_lock);
    return NULL;
}

VkResult vkdist_server_serve_many(VkPhysicalDevice pd, VkDevice dev,
                                  VkBLASContext *blas, const int *conn_fds,
                                  uint32_t n)
{
    if (conn_fds == NULL || blas == NULL || n == 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    pthread_mutex_t blas_lock;
    if (pthread_mutex_init(&blas_lock, NULL) != 0)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    vkdist_serve_thread_arg_t *args =
        (vkdist_serve_thread_arg_t *)malloc(n * sizeof(*args));
    pthread_t *threads = (pthread_t *)malloc(n * sizeof(*threads));
    if (args == NULL || threads == NULL) {
        free(args);
        free(threads);
        pthread_mutex_destroy(&blas_lock);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VkResult result = VK_SUCCESS;

    for (uint32_t i = 0; i < n; i++) {
        args[i].pd = pd;
        args[i].dev = dev;
        args[i].blas = blas;
        args[i].conn_fd = conn_fds[i];
        args[i].blas_lock = &blas_lock;
        args[i].result = VK_ERROR_UNKNOWN;
        if (pthread_create(&threads[i], NULL, vkdist_serve_thread_fn,
                           &args[i]) != 0) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            /* Close the sessions that were never started; join what did. */
            for (uint32_t j = i; j < n; j++)
                vkdist_sock_close((vkdist_sock_t)conn_fds[j]);
            for (uint32_t j = 0; j < i; j++)
                pthread_join(threads[j], NULL);
            free(args);
            free(threads);
            pthread_mutex_destroy(&blas_lock);
            return result;
        }
    }

    for (uint32_t i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
        if (args[i].result != VK_SUCCESS && result == VK_SUCCESS)
            result = args[i].result;
    }

    free(args);
    free(threads);
    pthread_mutex_destroy(&blas_lock);
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

VkResult vkdist_query_caps(int fd, VkDistCaps *caps)
{
    if (fd < 0 || caps == NULL)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (vkdist_frame_send((vkdist_sock_t)fd, VKDIST_OP_CAPS, NULL, 0) != 0)
        return VK_ERROR_UNKNOWN;

    uint8_t reply[4 + VKDIST_CAPS_PAYLOAD_SIZE];
    uint32_t len = 0;
    if (vkdist_client_recv_frame((vkdist_sock_t)fd, VKDIST_OP_CAPS,
                                 reply, sizeof(reply), &len) != 0)
        return VK_ERROR_UNKNOWN;
    if (len != sizeof(reply))
        return VK_ERROR_UNKNOWN;

    int32_t result = vkdist_get_i32_le(reply);
    if (result != 0)
        return (VkResult)result;

    const uint8_t *q = reply + 4;
    caps->protocol_version = vkdist_get_u32_le(q);
    caps->vram_total = vkdist_get_u64_le(q + 4);
    caps->vram_free  = vkdist_get_u64_le(q + 12);
    caps->arch_index = vkdist_get_u32_le(q + 20);
    caps->subgroup_size = vkdist_get_u32_le(q + 24);
    caps->max_frame = vkdist_get_u32_le(q + 28);
    memcpy(caps->gpu_name, q + 32, 128);
    caps->gpu_name[127] = '\0';
    return VK_SUCCESS;
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

VkResult vkdist_sgemm_partitioned(int n_workers, const int *fds,
                                  int32_t m, int32_t n, int32_t k,
                                  const float *alpha, const float *A,
                                  int32_t lda, const float *B, int32_t ldb,
                                  const float *beta, float *C, int32_t ldc)
{
    if (fds == NULL || n_workers <= 0 || m <= 0 || n <= 0 || k <= 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (alpha == NULL || beta == NULL || A == NULL || B == NULL || C == NULL)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = VK_SUCCESS;
    int32_t n_start = 0;

    for (int32_t w = 0; w < n_workers; w++) {
        /* Column strip: n_i = n/n_workers for every worker but the last, which
           takes the remainder so the split always covers all n columns. */
        int32_t n_i = (w == n_workers - 1) ? (n - n_start) : (n / n_workers);
        if (n_i <= 0)
            continue;
        int fd = fds[w];

        /* Per-strip byte spans. B and C are column-major with ldb/ldc strides,
           so a column strip is one contiguous block starting at n_start*ldb /
           n_start*ldc; the span extends to the last column's k/m-th element.
           A is the full m x k matrix (span to column k-1, row m-1). */
        const uint64_t szA =
            (uint64_t)(((int64_t)(k - 1) * lda + m) * (int64_t)sizeof(float));
        const uint64_t szB =
            (uint64_t)(((int64_t)(n_i - 1) * ldb + k) * (int64_t)sizeof(float));
        const uint64_t szC =
            (uint64_t)(((int64_t)(n_i - 1) * ldc + m) * (int64_t)sizeof(float));
        const char *stripB =
            (const char *)B + (size_t)n_start * (size_t)ldb * sizeof(float);
        char *stripC =
            (char *)C + (size_t)n_start * (size_t)ldc * sizeof(float);

        uint64_t hA = 0, hB = 0, hC = 0;
        VkResult vr = vkdist_register_buffer(fd, szA, &hA);
        if (vr == VK_SUCCESS)
            vr = vkdist_register_buffer(fd, szB, &hB);
        if (vr == VK_SUCCESS)
            vr = vkdist_register_buffer(fd, szC, &hC);
        if (vr == VK_SUCCESS)
            vr = vkdist_upload(fd, hA, A, 0, szA);
        if (vr == VK_SUCCESS)
            vr = vkdist_upload(fd, hB, stripB, 0, szB);
        if (vr == VK_SUCCESS)
            /* Upload the current C strip so the beta term reads the caller's
               data (needed for accumulating runs: C = alpha*A*B + beta*C). */
            vr = vkdist_upload(fd, hC, stripC, 0, szC);
        if (vr == VK_SUCCESS)
            vr = vkdist_sgemm(fd, m, n_i, k, alpha, hA, hB, hC, beta,
                              lda, ldb, ldc);
        if (vr == VK_SUCCESS)
            /* Column-major merge: the strip is contiguous in C, so copy the
               read-back bytes directly into C + n_start*ldc. */
            vr = vkdist_readback(fd, hC, 0, szC, stripC);

        if (vr != VK_SUCCESS && result == VK_SUCCESS)
            result = vr;

        n_start += n_i;
    }

    return result;
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

/* ===========================================================================
 * Master / worker coordinator
 * ========================================================================== */

#define VKDIST_MASTER_MAX_WORKERS 8u

/* Run `ssh -o BatchMode=yes <user>@<host> "exit 0"`; return 0 iff key auth OK.
   On Windows uses the OpenSSH client. _popen gives us the exit code portably
   (cmd.exe on Win, sh elsewhere); BatchMode guarantees no password prompt.
   The remote command is `exit 0` (no spaces) so it needs no shell quoting,
   which would otherwise be mangled by cmd.exe's argument re-parsing. */
static int vkdist_ssh_key_ok(const char *host, const char *user)
{
    /* Only safe characters allowed — host/user are interpolated into a shell
       command, so reject anything that could break out (cmd/powershell or sh). */
    for (const char *p = host; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')
              || (*p >= '0' && *p <= '9') || *p == '.' || *p == '-' || *p == '_'))
            return 0;
    }
    for (const char *p = user; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')
              || (*p >= '0' && *p <= '9') || *p == '_' || *p == '-' || *p == '.'))
            return 0;
    }

    char cmd[512];
    int n = snprintf(cmd, sizeof(cmd),
                     "ssh -o BatchMode=yes -o ConnectTimeout=5 "
                     "-o StrictHostKeyChecking=accept-new %s@%s exit 0",
                     user, host);
    if (n < 0 || (size_t)n >= sizeof(cmd))
        return 0;

    FILE *p = _popen(cmd, "r");
    if (p == NULL)
        return 0;
    int rc = _pclose(p);
    return rc == 0;
}

struct VkDistMaster {
    int          fds[VKDIST_MASTER_MAX_WORKERS];
    VkDistCaps   caps[VKDIST_MASTER_MAX_WORKERS];
    uint32_t     count;
};

VkResult vkdist_verify_ssh_key(const char *host, const char *user)
{
    if (host == NULL || host[0] == '\0' || user == NULL || user[0] == '\0')
        return VK_ERROR_INITIALIZATION_FAILED;
    return vkdist_ssh_key_ok(host, user) ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

VkResult vkdist_master_create(VkDistMaster **out)
{
    if (out == NULL)
        return VK_ERROR_INITIALIZATION_FAILED;
    VkDistMaster *m = (VkDistMaster *)calloc(1, sizeof(VkDistMaster));
    if (!m)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out = m;
    return VK_SUCCESS;
}

VkResult vkdist_master_add_worker(VkDistMaster *m, const char *ip,
                                  const char *ssh_user, uint16_t port,
                                  VkDistCaps *out_caps)
{
    if (m == NULL || ip == NULL)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (m->count >= VKDIST_MASTER_MAX_WORKERS)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    /* Security gate: refuse any remote worker the controlling host has no
       SSH key to. Loopback (127.0.0.1 / localhost) is inherently the local
       machine — no SSH auth needed to talk to yourself, and it lets a master
       add the local GPU as a worker beside remote ones. */
    int is_loopback = (strcmp(ip, "127.0.0.1") == 0 ||
                       strcmp(ip, "localhost") == 0 ||
                       strcmp(ip, "::1") == 0);
    if (!is_loopback && vkdist_ssh_key_ok(ip, ssh_user) == 0)
        return VK_ERROR_UNKNOWN;

    int fd = vkdist_client_connect(ip, port);
    if (fd < 0)
        return VK_ERROR_UNKNOWN;

    VkResult vr = vkdist_query_caps(fd, &m->caps[m->count]);
    if (vr != VK_SUCCESS) {
        vkdist_close(fd);
        return vr;
    }

    m->fds[m->count] = fd;
    if (out_caps != NULL)
        *out_caps = m->caps[m->count];
    m->count++;
    return VK_SUCCESS;
}

uint32_t vkdist_master_worker_count(VkDistMaster *m)
{
    return m ? m->count : 0;
}

const VkDistCaps *vkdist_master_worker_caps(VkDistMaster *m, uint32_t i)
{
    if (m == NULL || i >= m->count)
        return NULL;
    return &m->caps[i];
}

VkResult vkdist_master_sgemm(VkDistMaster *m,
                             int32_t m_, int32_t n_, int32_t k_,
                             const float *alpha, const float *A, int32_t lda,
                             const float *B, int32_t ldb,
                             const float *beta, float *C, int32_t ldc)
{
    if (m == NULL || m->count == 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    return vkdist_sgemm_partitioned((int)m->count, m->fds,
                                    m_, n_, k_, alpha, A, lda, B, ldb,
                                    beta, C, ldc);
}

void vkdist_master_destroy(VkDistMaster *m)
{
    if (m == NULL)
        return;
    for (uint32_t i = 0; i < m->count; i++)
        vkdist_close(m->fds[i]);
    free(m);
}
