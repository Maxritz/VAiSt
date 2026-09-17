/**
 * \file vkdist.h
 * \brief Distributed-compute remote-execution layer (hipDist-style) for the
 *        Vulkan AI stack.
 *
 * vkdist lets one machine offload compute to another PC's Vulkan card over a
 * network. Phase 0 is a loopback-vertical slice: a TCP server hosts one Vulkan
 * device, and a client connects over the network, registers buffers by opaque
 * u64 handle, uploads host data, remotely executes vkblas_sgemm() on the
 * server device, and reads results back. Phase 1 adds multi-connection
 * serving (vkdist_server_accept_many / vkdist_server_serve_many) and a
 * client-side column-partitioned GEMM (vkdist_sgemm_partitioned) that splits
 * the output columns across workers and merges the partial results.
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

/**
 * \brief Capability advertisement for a connected vkdist server.
 *
 * Filled by vkdist_query_caps(). Lets a master/coordinator discover what a
 * worker GPU can do before routing work to it: GPU name, VRAM budget, the
 * stack's shader-tier arch index (0=baseline, 1=subgroup, 2=coopmatrix), and
 * the server's frame-size limit.
 */
typedef struct {
    uint32_t protocol_version;  /**< VKDIST_PROTOCOL_VERSION echoed by server. */
    uint64_t vram_total;        /**< Device-local heap size in bytes.          */
    uint64_t vram_free;         /**< Device-local heap currently free (bytes). */
    uint32_t arch_index;        /**< 0=baseline, 1=subgroup, 2=coopmatrix.     */
    uint32_t subgroup_size;     /**< Device subgroup size (e.g. 64 RDNA2).      */
    uint32_t max_frame;         /**< Max payload bytes/frame on this server.    */
    char     gpu_name[128];     /**< VkPhysicalDeviceProperties::deviceName.    */
} VkDistCaps;

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

/**
 * \brief Accept exactly \p n connections on a listen socket (blocking).
 *
 * Reuses vkdist_server_accept() \p n times and stores each connected socket
 * descriptor in \p conn_fds[0..n). On error, any descriptors already accepted
 * are closed internally and VK_ERROR_UNKNOWN is returned.
 *
 * \param listen_fd Descriptor returned by vkdist_server_start().
 * \param n         Number of connections to accept (must be > 0).
 * \param conn_fds  Array of at least \p n ints, filled with the accepted
 *                  descriptors.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 * \retval VK_ERROR_UNKNOWN accept() failed before all \p n connected.
 */
VkResult vkdist_server_accept_many(int listen_fd, uint32_t n, int *conn_fds);

/**
 * \brief Serve \p n established connections concurrently, one thread each.
 *
 * Convenience wrapper for a multi-worker server: spawns \p n POSIX threads
 * (pthread_create) that each run vkdist_server_run() on one of \p conn_fds
 * until that connection sends BYE or disconnects. Each connection gets its own
 * transient VkRuntime, command pool/buffer/fence, and remote-buffer table
 * (vkdist_server_run already owns all of these per call), so the sessions are
 * fully independent except for the caller-owned \p blas.
 *
 * The shared VkBLASContext is NOT thread-safe (lazy pipeline cache + descriptor
 * pool), so serve_many internally serializes the vkblas_sgemm dispatches with a
 * mutex. Queue submits and per-connection command buffers are safe to use
 * concurrently, so upload/download/dispatch of different connections overlap;
 * only the context-mutating part of a GEMM dispatch is serialized. Callers that
 * also use vkdist_server_run() directly on the same \p blas must serialize
 * those calls themselves.
 *
 * This function requires POSIX threads: on this MinGW-W64 "posix" build it is
 * winpthreads, linked with -lpthread (the same source compiles unchanged on
 * Linux). The caller must have already accepted the connections (e.g. via
 * vkdist_server_accept_many()).
 *
 * \param pd        Physical device backing \p dev.
 * \param dev       Logical device the remote buffers and dispatches use.
 * \param blas      VkBLASContext bound to \p dev (shared; serialized here).
 * \param conn_fds  Array of \p n connected sockets to serve.
 * \param n         Number of connections/threads.
 * \retval VK_SUCCESS Every session ended cleanly (BYE or peer close).
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument or pthread failure.
 * \retval other    First VkResult error returned by any worker session.
 */
VkResult vkdist_server_serve_many(VkPhysicalDevice pd, VkDevice dev,
                                  VkBLASContext *blas, const int *conn_fds,
                                  uint32_t n);

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
 * \brief Query a connected server's capabilities (GPU name, VRAM, arch tier).
 *
 * Sends a CAPS request and fills \p caps from the server's reply. Lets a
 * master decide which worker to route work to based on VRAM and shader tier.
 *
 * \param fd   Connected socket from vkdist_client_connect().
 * \param caps Receives the capability struct.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_UNKNOWN Transport/protocol error or truncated reply.
 */
VkResult vkdist_query_caps(int fd, VkDistCaps *caps);

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
 * \brief Column-partitioned f32 GEMM across \p n_workers connections.
 *
 * Splits the C matrix (m x n) into contiguous column strips across
 * \p n_workers workers: worker i computes C_i (m x n_i) = alpha * A * B_i +
 * beta * C_i where B_i is the k x n_i column strip B[:, n_start:n_start+n_i)
 * and the strips are merged back into the host C buffer. The split is
 * n_i = n / n_workers for every worker except the last, which receives the
 * remainder (n - n_start), so uneven splits (e.g. n=16, 3 workers -> 5/5/6)
 * are supported.
 *
 * Per worker the function: registers A_i (m x k), B_i (k x n_i, sized to the
 * strip's ldb span), C_i (m x n_i, sized to the ldc span); uploads A in full,
 * the B column strip (B is column-major so the strip is one contiguous block
 * starting at B + n_start*ldb), and the current C strip (so the beta term
 * reads the caller's data — required for accumulating runs); dispatches
 * vkdist_sgemm with the strip's n = n_i and the original ldb/ldc strides; and
 * reads C_i back into C + n_start*ldc. All matrices are column-major. On
 * failure the first error is returned (buffers registered earlier on that
 * connection stay until the connection closes).
 *
 * \param n_workers Number of worker connections (must be > 0).
 * \param fds       Array of \p n_workers connected sockets.
 * \param m         Rows of op(A) and C.
 * \param n         Cols of op(B) and C.
 * \param k         Contraction dimension.
 * \param alpha     Host pointer to the alpha scalar.
 * \param A         Host A buffer (m x k, lda stride).
 * \param lda       Leading dimension of A (>= m).
 * \param B         Host B buffer (k x n, ldb stride).
 * \param ldb       Leading dimension of B (>= k).
 * \param beta      Host pointer to the beta scalar.
 * \param C         Host C buffer (m x n, ldc stride); read for beta, then
 *                  overwritten with the merged result.
 * \param ldc       Leading dimension of C (>= m).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED Invalid argument.
 * \retval VK_ERROR_UNKNOWN Transport/protocol error on any worker.
 */
VkResult vkdist_sgemm_partitioned(int n_workers, const int *fds,
                                  int32_t m, int32_t n, int32_t k,
                                  const float *alpha, const float *A,
                                  int32_t lda, const float *B, int32_t ldb,
                                  const float *beta, float *C, int32_t ldc);

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

/* ===========================================================================
 * Master / worker coordinator
 * ========================================================================== */

/** \brief Opaque master context holding N connected worker connections. */
typedef struct VkDistMaster VkDistMaster;

/**
 * \brief Verify SSH key-based authentication to a host (security gate).
 *
 * Runs `ssh -o BatchMode=yes <user>@<host> "exit 0"` so no password prompt is
 * possible: BatchMode makes ssh fail instead of prompting for a password.
 * Returns VK_SUCCESS only if key-based auth succeeds and the host answers.
 *
 * This is the security gate the master enforces before accepting any worker:
 * a worker is only usable when the controlling host has an established SSH
 * key to it. On Windows the OpenSSH client (`ssh.exe`) must be on PATH.
 *
 * \param host IPv4 address or resolvable hostname of the target.
 * \param user SSH login user on the target.
 * \retval VK_SUCCESS SSH key auth to host works.
 * \retval VK_ERROR_INITIALIZATION_FAILED NULL argument.
 * \retval VK_ERROR_UNKNOWN ssh not found, auth failed, or host unreachable.
 */
VkResult vkdist_verify_ssh_key(const char *host, const char *user);

/**
 * \brief Create an empty master coordinator.
 *
 * \param out Receives the new master (call vkdist_master_destroy() when done).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_OUT_OF_HOST_MEMORY
 */
VkResult vkdist_master_create(VkDistMaster **out);

/**
 * \brief Connect a worker and record its capabilities.
 *
 * Security: verifies SSH key-based auth to \p ip (as \p ssh_user) FIRST via
 * vkdist_verify_ssh_key(). Only if that succeeds does it connect (HELLO
 * handshake inside vkdist_client_connect), query capabilities, and add the
 * worker to the master. A worker the controlling host has no SSH key to is
 * rejected with VK_ERROR_UNKNOWN — the distributed system will not operate
 * between hosts that do not trust each other via SSH.
 *
 * Note on transport encryption: vkdist does not tunnel its own traffic.
 * The SSH key gate authenticates the worker; to encrypt the data path, run
 * the vkdist client through an SSH local port-forward established outside
 * the library, e.g. `ssh -N -L 7001:127.0.0.1:7000 user@worker` and point
 * vkdist_client_connect() at 127.0.0.1:7001.
 *
 * The connection is owned by the master and closed by vkdist_master_destroy().
 *
 * \param m        Master context.
 * \param ip       Worker IPv4 address.
 * \param ssh_user SSH login user on the worker (verified before connect).
 * \param port     Worker vkdist listen port.
 * \param out_caps Optional; filled with the worker's capabilities.
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED NULL master.
 * \retval VK_ERROR_UNKNOWN SSH-key gate failed, connect/handshake/caps failure.
 */
VkResult vkdist_master_add_worker(VkDistMaster *m, const char *ip,
                                  const char *ssh_user, uint16_t port,
                                  VkDistCaps *out_caps);

/**
 * \brief Number of connected workers.
 */
uint32_t vkdist_master_worker_count(VkDistMaster *m);

/**
 * \brief Capabilities of worker \p i.
 * \retval NULL on invalid index or NULL master.
 */
const VkDistCaps *vkdist_master_worker_caps(VkDistMaster *m, uint32_t i);

/**
 * \brief Column-partitioned f32 GEMM across all connected workers.
 *
 * Reuses vkdist_sgemm_partitioned() with the master's worker fds, so C's
 * columns are split across every worker (n_i = n/count, last takes the
 * remainder) and the strips are merged back into the host C buffer.
 *
 * \param m     Master with >= 1 worker.
 * \param m_    Rows of op(A) and C.
 * \param n_    Cols of op(B) and C.
 * \param k_    Contraction dimension.
 * \param alpha Host alpha scalar.
 * \param A     Host A buffer (m x k, lda stride).
 * \param lda   Leading dimension of A (>= m).
 * \param B     Host B buffer (k x n, ldb stride).
 * \param ldb   Leading dimension of B (>= k).
 * \param beta  Host beta scalar.
 * \param C     Host C buffer (m x n, ldc stride); read for beta, overwritten.
 * \param ldc   Leading dimension of C (>= m).
 * \retval VK_SUCCESS
 * \retval VK_ERROR_INITIALIZATION_FAILED No workers or bad args.
 * \retval VK_ERROR_UNKNOWN Any worker transport failure.
 */
VkResult vkdist_master_sgemm(VkDistMaster *m,
                             int32_t m_, int32_t n_, int32_t k_,
                             const float *alpha, const float *A, int32_t lda,
                             const float *B, int32_t ldb,
                             const float *beta, float *C, int32_t ldc);

/**
 * \brief Close every worker connection and free the master.
 */
void vkdist_master_destroy(VkDistMaster *m);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKDIST_H */
