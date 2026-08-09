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

### Client
- `vkdist_client_connect` sends HELLO and requires `VK_SUCCESS` before
  returning a usable fd.
- Client frames are built in little-endian byte arrays; upload/readback
  payloads are heap-allocated and freed before returning.
- `vkdist_close` sends a best-effort BYE (no reply expected) then closes.

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

## Files

| File | Purpose |
|------|---------|
| `vkdist_internal.h` | Opcode enum, frame constants, server session struct |
| `vkdist.c` | Socket wrappers, framing, server RPC loop, client API |
