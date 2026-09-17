# VKDIST Include — Local Contract

Child of root `AGENTS.md`. Governs the public API surface under
`include/vkdist/` and the design spec `specs/VKDIST-DESIGN.md`.

## Scope

`vkdist` is the distributed-compute remote-execution layer for the Vulkan AI
stack: a client offloads compute (currently `vkblas_sgemm`) to a remote PC's
Vulkan GPU over TCP. Phase 0 shipped a loopback-vertical slice (transport +
RPC + remote dispatch). Phase 1 (this subtree) adds multi-connection serving
(`vkdist_server_accept_many` / `vkdist_server_serve_many`, one pthread per
worker connection) and a client-side column-partitioned GEMM
(`vkdist_sgemm_partitioned`) that splits C's columns across workers and merges
the partial results. Multi-PC tensor-parallel orchestration beyond column
partitioning (attention/KV sharding, rendezvous + TLS) remains future work in
`specs/VKDIST-DESIGN.md`.

## Public API (exact signatures)

```
#define VKDIST_PROTOCOL_VERSION 1u

int      vkdist_server_start(uint16_t port, const char* ip);      /* listen fd, -1 err */
int      vkdist_server_accept(int listen_fd);                     /* conn fd,  -1 err */
VkResult vkdist_server_run(VkPhysicalDevice pd, VkDevice dev,
                           VkBLASContext* blas, int conn_fd);     /* serve until BYE */
VkResult vkdist_server_accept_many(int listen_fd, uint32_t n, int* conn_fds);
VkResult vkdist_server_serve_many(VkPhysicalDevice pd, VkDevice dev,
                                  VkBLASContext* blas, const int* conn_fds,
                                  uint32_t n);                    /* 1 pthread per conn */
int      vkdist_client_connect(const char* ip, uint16_t port);    /* conn fd,  -1 err */
VkResult vkdist_register_buffer(int fd, VkDeviceSize size, uint64_t* handle);
VkResult vkdist_upload(int fd, uint64_t handle, const void* host,
                       VkDeviceSize offset, VkDeviceSize size);
VkResult vkdist_sgemm(int fd, int32_t m, int32_t n, int32_t k,
                      const float* alpha, uint64_t A, uint64_t B, uint64_t C,
                      const float* beta, int32_t lda, int32_t ldb, int32_t ldc);
VkResult vkdist_sgemm_partitioned(int n_workers, const int* fds,
                                  int32_t m, int32_t n, int32_t k,
                                  const float* alpha, const float* A, int32_t lda,
                                  const float* B, int32_t ldb,
                                  const float* beta, float* C, int32_t ldc);
VkResult vkdist_readback(int fd, uint64_t handle, VkDeviceSize offset,
                         VkDeviceSize size, void* host);
void     vkdist_close(int fd);
```

## Contract rules

- **fd-based client**: every socket is an `int` fd. `vkdist_client_connect`
  performs the HELLO handshake before returning; all later calls are
  synchronous request/reply and return a `VkResult` (`0` = success).
- **Opaque remote handles**: buffers are `uint64_t` handles owned by the
  server; the client never sees a Vulkan handle across the wire.
- **One server = one GPU endpoint**: `vkdist_server_run` serves one connection
  serially on the caller's `(pd, dev, blas)`. It creates/destroys a transient
  `VkRuntime` and command pool/buffer/fence per connection and closes the
  connection on exit.
- **Multi-worker serving (Phase 1)**: `vkdist_server_accept_many` blocks until
  `n` connections arrive and fills `conn_fds`; `vkdist_server_serve_many`
  spawns one pthread per connection, each running the same per-connection loop.
  Per-connection state (runtime, command buffer, buffer table) stays owned by
  `vkdist_server_run`; only the shared `VkBLASContext` is shared, and it is
  serialized internally with a mutex around the context-mutating part of the
  GEMM dispatch (pipeline cache + descriptor pool). Requires POSIX threads
  (`pthread_create`; `-lpthread` on this MinGW-W64 posix build).
- **Partitioned GEMM (Phase 1)**: `vkdist_sgemm_partitioned` splits C's columns
  as `n_i = n/n_workers` for every worker but the last, which takes the
  remainder `n - n_start` (uneven splits supported). Per worker it registers
  A_i/B_i/C_i, uploads A in full + the B column strip (column-major ⇒ one
  contiguous block at `B + n_start*ldb`) + the current C strip (so the beta
  term reads real data and repeated accumulating calls are exact), dispatches
  an sgemm with the strip's `n = n_i` and the original ldb/ldc, and reads C_i
  back into `C + n_start*ldc`. Host pointers for A/B/C are owned by the caller.
- **In-place sgemm**: `vkdist_sgemm` computes `C = alpha*A*B + beta*C` with C
  as both the beta-read source and the output (the GEMM shader reads an element
  before writing it, so in-place is safe).
- **Synchronous semantics**: upload/readback include a full device wait on the
  server, so a returned `VK_SUCCESS` means the device work is complete.
- **Versioned wire protocol**: clients and servers must agree on
  `VKDIST_PROTOCOL_VERSION` or the HELLO handshake fails.
- **No stubs**: every declared function has a real implementation and a passing
  harness (`tests/test_vkdist.c`).
- **C99 + DOX**: DOX `\brief`/`\param`/`\retval` on every function.

## Threading

- A socket fd is owned by one thread at a time. Different fds are independent.
- `vkdist_server_run` is single-threaded per connection (matches
  `VkBLASContext` not being thread-safe); `vkdist_server_serve_many` runs one
  such loop per connection in its own pthread and serializes the shared
  context's dispatches internally. Callers using the same `VkBLASContext`
  outside serve_many must serialize those calls themselves.

## Files

| File | Purpose |
|------|---------|
| `include/vkdist/vkdist.h` | Public API (this file) |
| `specs/VKDIST-DESIGN.md` | Architecture, wire protocol, roadmap |
| `src/vkdist/vkdist_internal.h` | Opcodes, frame limits, server session state |
| `src/vkdist/vkdist.c` | Socket wrappers, framing, server RPC loop, client API |
| `tests/test_vkdist.c` | Loopback harness (server thread + remote sgemm) |
