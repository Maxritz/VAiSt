/**
 * \file vkdist.h
 * \brief Distributed-compute remote-execution layer (hipDist-style) for the
 *        Vulkan AI stack.
 *
 * vkdist lets one machine offload compute to another PC's Vulkan card over a
 * network. Phase 0 (this file) is a loopback-vertical slice: a TCP server
 * hosts one Vulkan device, and a client connects over the network, registers
 * buffers by opaque u64 handle, uploads host data, remotely executes
 * vkblas_sgemm() on the server device, and reads results back.
 *
 * All handles are file descriptors (ints) returned by the connect/accept
 * functions; all remote buffers are opaque uint64_t handles allocated on the
 * server. Every client call is synchronous and returns a VkResult (0 on
 * success). See specs/VKDIST-DESIGN.md for the wire protocol and roadmap.
 *
 * Threading: each fd is owned by one thread at a time. Different fds from
 * different threads are independent.
 */
#ifndef VKDIST_H
#define VKDIST_H

#include <vulkan/vulkan.h>
#include <stdint.h>

#include "vkblas.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Wire protocol version for this release.
 *
 * The client sends this in the HELLO handshake; the server rejects a
 * connection whose version does not match with VK_ERROR_INCOMPATIBLE_DRIVER.
 */
#define VKDIST_PROTOCOL_VERSION 1u

/* ===========================================================================
 * Server
 * ========================================================================== */

/**
 * \brief Create a listening TCP socket for a vkdist server.
 *
 * Binds to \p ip:\p port and starts listening. The returned descriptor is the
 * listen socket; hand it to vkdist_server_accept(). Pass port = 0 to let the
 * OS choose an ephemeral port (query it with getsockname()).
 *
 * \param port Port to bind, host byte order. 0 = ephemeral.
 * \param ip   Bind address as an IPv4 dotted-quad string (e.g. "127.0.0.1"),
 *             or NULL to bind INADDR_ANY.
 * \return Listening socket descriptor, or -1 on error.
 */
int vkdist_server_start(uint16_t port, const char *ip);

/**
 * \brief Block until a client connects to the listen socket.
 *
 * \param listen_fd Descriptor returned by vkdist_server_start().
 * \return Connected socket descriptor, or -1 on error.
 */
int vkdist_server_accept(int listen_fd);

/**
 * \brief Serve one client connection end to end on the server GPU.
 *
 * Runs the request/reply loop for the given connection: HELLO handshake,
 * REGISTER_BUFFER / UPLOAD / DISPATCH_GEMM / READBACK, until BYE or a
 * disconnect. The server owns the connection: it closes \p conn_fd before
 * returning and releases all per-connection Vulkan resources (a transient
 * VkRuntime, a command pool/buffer/fence, and every registered buffer).
 *
 * The device must expose a queue on queue family 0; the server fetches it with
 * vkGetDeviceQueue(dev, 0, 0, ...). The caller owns \p dev and \p blas and
 * destroys them after this returns.
 *
 * \param pd     Physical device backing \p dev (for runtime creation).
 * \param dev    Logical device the remote buffers and dispatches use.
 * \param blas   VkBLASContext bound to \p dev (not thread-safe; this call is
 *               the only concurrent user).
 * \param conn_fd Connected socket from vkdist_server_accept().
 * \retval VK_SUCCESS Session ended cleanly (BYE or peer close).
 * \retval VK_ERROR_INCOMPATIBLE_DRIVER HELLO version mismatch.
 * \retval VK_ERROR_UNKNOWN Protocol or transport error.
 * \retval other Driver/runtime failure during setup or an operation.
 */
VkResult vkdist_server_run(VkPhysicalDevice pd, VkDevice dev,
                           VkBLASContext *blas, int conn_fd);

/* ===========================================================================
 * Client
 * ========================================================================== */

/**
 * \brief Connect to a vkdist server and perform the HELLO handshake.
 *
 * Returns a connected, handshaked socket descriptor, or -1 on failure
 * (connect error, HELLO version mismatch, or transport error). On failure the
 * socket is closed internally.
 *
 * \param ip   Server IPv4 address, dotted quad (e.g. "127.0.0.1").
 * \param port Server listen port, host byte order.
 * \return Connection descriptor, or -1 on error.
 */
int vkdist_client_connect(const char *ip, uint16_t port);

/**
 * \brief Register a remote buffer of \p size bytes on the server device.
 *
 * The server allocates a STORAGE | TRANSFER buffer and returns an opaque
 * handle. The client passes this handle to vkdist_upload / vkdist_sgemm /
 * vkdist_readback. The buffer lives until the connection is closed.
 *
 * \param fd     Connected socket from vkdist_client_connect().
 * \param size   Requested byte size (> 0).
 * \param handle Receives the opaque server handle (0 on failure).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_DEVICE_MEMORY Server allocator exhausted.
 * \retval VK_ERROR_UNKNOWN Transport/protocol error.
 */
VkResult vkdist_register_buffer(int fd, VkDeviceSize size, uint64_t *handle);

/**
 * \brief Upload host bytes into a remote buffer region.
 *
 * Sends \p size bytes from \p host to the server, which stages them into the
 * remote buffer at \p offset via vkr_upload. Synchronous: returns after the
 * server's device copy has completed.
 *
 * \param fd     Connected socket.
 * \param handle Remote buffer handle from vkdist_register_buffer().
 * \param host   Host source pointer (readable for \p size bytes).
 * \param offset Byte offset into the remote buffer.
 * \param size   Byte count to copy. Must satisfy offset+size <= buffer size.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_UNKNOWN Bad handle, out-of-range, or transport error.
 */
VkResult vkdist_upload(int fd, uint64_t handle, const void *host,
                       VkDeviceSize offset, VkDeviceSize size);

/**
 * \brief Remotely execute C = alpha * A * B + beta * C (f32 GEMM, in place).
 *
 * Dispatches vkblas_sgemm on the server with VKBLAS_OP_N/VKBLAS_OP_N and C
 * used both as the beta-read source and the output (D), so C is overwritten
 * with the result. Matrices are column-major.
 *
 * \param fd    Connected socket.
 * \param m     Rows of op(A) and C.
 * \param n     Cols of op(B) and C.
 * \param k     Contraction dimension.
 * \param alpha Host pointer to the alpha scalar.
 * \param A     Remote handle of the A buffer (m x k, lda stride).
 * \param B     Remote handle of the B buffer (k x n, ldb stride).
 * \param C     Remote handle of the C buffer (m x n, ldc stride; read+write).
 * \param beta  Host pointer to the beta scalar.
 * \param lda   Leading dimension of A (>= m).
 * \param ldb   Leading dimension of B (>= k).
 * \param ldc   Leading dimension of C (>= m).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_UNKNOWN Bad handle, invalid dims, or transport error.
 */
VkResult vkdist_sgemm(int fd, int32_t m, int32_t n, int32_t k,
                      const float *alpha, uint64_t A, uint64_t B, uint64_t C,
                      const float *beta, int32_t lda, int32_t ldb,
                      int32_t ldc);

/**
 * \brief Read bytes back from a remote buffer region into host memory.
 *
 * \param fd     Connected socket.
 * \param handle Remote buffer handle.
 * \param offset Byte offset into the remote buffer.
 * \param size   Byte count to read. Must satisfy offset+size <= buffer size.
 * \param host   Host destination pointer (writable for \p size bytes).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_UNKNOWN Bad handle, out-of-range, or transport error.
 */
VkResult vkdist_readback(int fd, uint64_t handle, VkDeviceSize offset,
                         VkDeviceSize size, void *host);

/**
 * \brief Close a vkdist connection.
 *
 * Sends a best-effort BYE frame (so the server tears down cleanly) and closes
 * the socket. Safe to call on any fd, including -1. No further calls may use
 * \p fd.
 *
 * \param fd Connection or listen descriptor to close.
 */
void vkdist_close(int fd);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKDIST_H */
