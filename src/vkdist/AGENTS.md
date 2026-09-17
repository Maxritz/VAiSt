# VKDIST Source — Local Contract

Child of root `AGENTS.md`, `include/vkdist/AGENTS.md`, and
`specs/VKDIST-DESIGN.md`.

## Implementation Rules

### Transport
- Winsock2 on `_WIN32` (`<winsock2.h>`, `ws2_32`), POSIX sockets otherwise.
  Both paths are implemented in `vkdist.c` behind `vkdist_sock_t` and
  send/recv/close wrappers; the public API stays OS-agnostic.
- Blocking sockets with `TCP_NODELAY`; send/recv loop until the exact byte
  count is transferred so the stream cannot desync on short reads/writes.
- `VKDIST_MAX_FRAME` (1 MiB) bounds every payload; a larger frame is a
  protocol error that closes the connection. Oversize protection keeps the
  per-connection receive buffer fixed-size and reusable.

### Framing and endianness
- Every message: `u32 LE length` + `u32 LE opcode` + payload. Length counts
  payload bytes only. All fields are packed/unpacked little-endian explicitly
  (never by reinterpreting host structs), so the wire format is endian-portable.
- Every reply echoes the request opcode and begins with an `i32 VkResult`
  (0 = `VK_SUCCESS`). Reply payload layouts per opcode:
  - HELLO reply: `i32 result`
  - REGISTER_BUFFER reply: `i32 result`, `u64 handle`
  - UPLOAD reply: `i32 result`
  - DISPATCH_GEMM reply: `i32 result`
  - READBACK reply: `i32 result`, raw bytes (only on success)
  - BYE: no reply

### Server session (`vkdist_server_run`)
- Assumes queue family 0 (`vkGetDeviceQueue(dev, 0, 0, ...)`); documented in
  the spec as a Phase-0 assumption.
- Creates a transient `VkRuntime` + command pool/buffer/fence per connection,
  all destroyed on exit. The caller owns `dev` and `blas`.
- Remote buffers are `vkr_malloc`'d with `STORAGE_BUFFER | TRANSFER_SRC |
  TRANSFER_DST` (device-local memory class) and recorded in a per-connection
  table of up to `VKDIST_MAX_BUFFERS`; handles start at 1 and increment.
- Synchronization: upload (`vkr_upload`), dispatch, and readback
  (`vkr_download`) are each separated by a full host-side wait on the same
  queue, so no pipeline barriers are required (spec §5.3). The shared command
  buffer is always reset before the next op.
- `DISPATCH_GEMM` validates m/n/k > 0, `lda ≥ m`, `ldb ≥ k`, `ldc ≥ m`, that
  A/B/C handles exist, and that each buffer is large enough for the matrix it
  backs; then records one `vkblas_sgemm` with C as both source and output
  (in-place is safe: the GEMM shader reads `data_C` before writing `data_D`).
- Cleanup: destroy every remote `VkBuffer` handle explicitly (vkruntime's
  `vkr_destroy_runtime` frees block memory but not buffer objects), then
  destroy the fence / command buffer / pool / runtime, then close the socket.

### Multi-connection serving (Phase 1)
- `vkdist_server_accept_many(listen_fd, n, conn_fds)` accepts exactly `n`
  connections (reusing `vkdist_server_accept`); on failure it closes the
  descriptors already accepted and returns an error.
- `vkdist_server_serve_many(pd, dev, blas, conn_fds, n)` spawns `n` POSIX
  threads (`pthread_create`, link `-lpthread`; winpthreads on this MinGW-W64
  posix build), each running the per-connection RPC loop on one fd. Per-run
  state (transient runtime, command pool/buffer/fence, buffer table) is owned
  inside `vkdist_server_run_ex`, so the sessions are independent except for the
  shared `VkBLASContext`.
- The shared context is **not thread-safe** (lazy pipeline cache + descriptor
  pool), so `serve_many` creates one `pthread_mutex_t` and passes it into the
  per-connection loop as `st.blas_lock`; `vkdist_server_dispatch` takes it
  around the `vkblas_sgemm` call only. Queue submits and per-connection
  command buffers are safe concurrently, so upload/download/dispatch of
  different connections overlap.
- `vkdist_server_run` (single connection) calls the same internal loop with a
  NULL lock.

### Client
- `vkdist_client_connect` sends HELLO and requires `VK_SUCCESS` before
  returning a usable fd.
- Client frames are built in little-endian byte arrays; upload/readback
  payloads are heap-allocated and freed before returning.
- `vkdist_close` sends a best-effort BYE (no reply expected) then closes.

### Partitioned GEMM client (Phase 1)
- `vkdist_sgemm_partitioned` splits C's columns as `n_i = n/n_workers` (all but
  the last) and `n_i = n - n_start` (last), so uneven splits are covered
  exactly and every column belongs to exactly one worker.
- Per worker `w`: register `A_i` (span `((k-1)*lda + m)*4` bytes), `B_i` (span
  `((n_i-1)*ldb + k)*4`), `C_i` (span `((n_i-1)*ldc + m)*4`); upload A in full,
  the B column strip (`B + n_start*ldb`, contiguous because B is column-major),
  and the current C strip (`C + n_start*ldc`, so the beta term reads the
  caller's data and repeated accumulating calls are exact); dispatch sgemm with
  `n = n_i` and the original `lda/ldb/ldc`; read C_i back into `C + n_start*ldc`
  (column-major merge is one contiguous memcpy per worker). Buffers registered
  earlier that fail remain on the connection until it closes; the first error
  is returned.

## Truth tables before code

Per root AGENTS.md. Phase-0 trace for the loopback vertical slice:

| Condition | Expected | Actual |
|-----------|----------|--------|
| HELLO version == 1 | reply 0, session continues | ✅ |
| HELLO version mismatch | reply VK_ERROR_INCOMPATIBLE_DRIVER, close | ✅ |
| REGISTER 3 buffers | handles 1, 2, 3 returned | ✅ |
| table full (64) | reply VK_ERROR_OUT_OF_DEVICE_MEMORY + handle 0 | ✅ |
| UPLOAD len == 24+size, valid handle/bounds | vkr_upload, reply 0 | ✅ |
| UPLOAD bad handle / OOB / len mismatch | reply VK_ERROR_UNKNOWN | ✅ |
| DISPATCH valid dims + handles + sizes | begin→sgemm→end→submit→wait→reset, reply 0 | ✅ |
| DISPATCH bad handle / dims / undersized buffer | reply VK_ERROR_UNKNOWN | ✅ |
| READBACK valid, size ≤ 1 MiB−4 | vkr_download, reply 0 + bytes | ✅ |
| READBACK OOB / oversized | reply VK_ERROR_UNKNOWN / OUT_OF_DEVICE_MEMORY | ✅ |
| BYE | server closes, VK_SUCCESS, no reply | ✅ |
| unknown opcode | reply VK_ERROR_UNKNOWN, close | ✅ |
| frame len > 1 MiB | close (desync protection) | ✅ |
| peer closes mid-session | recv==0 → clean break, VK_SUCCESS | ✅ |

Phase-1 trace for multi-connection + partitioned GEMM:

| Condition | Expected | Actual |
|-----------|----------|--------|
| accept_many with n clients connected | n fds returned, VK_SUCCESS | ✅ |
| accept_many with n-1 clients (timeout/close) | VK_ERROR_UNKNOWN, earlier fds closed | ✅ |
| serve_many n=2, both sessions BYE cleanly | VK_SUCCESS | ✅ |
| shared blas dispatched from 2 threads | mutex serializes vkblas_sgemm; no race | ✅ |
| partition n=16, workers=2 | strips [8,8] | ✅ |
| partition n=16, workers=3 (uneven) | strips [5,5,6], all columns covered | ✅ |
| per-worker B strip offset = n_start*ldb | column-major contiguous copy is exact | ✅ |
| per-worker C strip uploaded before dispatch | beta term reads caller's C (accumulation exact) | ✅ |
| merge C_i → C + n_start*ldc | contiguous memcpy, full C matches CPU ref ≤ 1e-3 | ✅ |
| partitioned beta=0.5 run twice into same C | C = A*B + 0.5*(A*B + 0.5*C0), matches CPU ref | ✅ |
| client connect sequential (HELLO-inside-connect) + blocking accept_many | would deadlock; test connects clients concurrently | ✅ |

## Files

| File | Purpose |
|------|---------|
| `vkdist_internal.h` | Opcode enum, frame constants, server session struct |
| `vkdist.c` | Socket wrappers, framing, server RPC loop, client API |
