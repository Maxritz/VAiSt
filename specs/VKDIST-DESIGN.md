# VKDIST — Distributed Compute for the Vulkan AI Stack

Design specification for `vkdist`, a distributed-compute module that lets one
machine offload compute to another PC's Vulkan GPU over a network. The end goal
of the module is to run a single model split across multiple PCs' Vulkan GPUs
(tensor-parallel inference). This document covers the architecture, transport,
wire protocol, data movement, concurrency, fault handling, and security posture,
plus the phased roadmap from a single-machine loopback slice (Phase 0, shipped
in this subtree) to multi-PC tensor-parallel operation (P1–P4).

## 1. Scope and Goal

- **Phase 0 (this subtree):** a real, testable loopback vertical slice. One
  TCP server hosts a Vulkan device; a client connects over `127.0.0.1`,
  registers buffers by opaque `u64` handle, uploads host data, remotely
  executes `vkblas_sgemm` on the server, and reads results back. This proves
  transport + RPC + remote-dispatch plumbing end to end on localhost.
- **Later phases:** multiple PCs over IP (P1), distributed GEMM where m/n is
  partitioned across workers (P2), attention/KV sharding (P3), and
  rendezvous/discovery + TLS (P4). These are design targets documented below,
  not implemented here.

## 2. Architecture

### 2.1 Client/Server model

`vkdist` is a request/reply remote-execution layer. One server process hosts
exactly one GPU endpoint (one `VkPhysicalDevice` + `VkDevice` + one
`VkBLASContext`). A server can accept multiple connections (P1+), but in
Phase 0 each connection is served by one serialized RPC loop.

```
                     ┌────────────────────────────────────────────────┐
                     │                     CLIENT PC                  │
                     │                                                │
                     │   Model / app layer (future tensor-parallel)   │
                     │                     │                           │
                     │               vkdist client API                │
                     │  (register / upload / sgemm / readback on fd)  │
                     │                     │                           │
                     │            TCP socket -> ip:port               │
                     └─────────────────────┼──────────────────────────┘
                                           │  TCP (framed: len + opcode + payload)
                     ┌─────────────────────┼──────────────────────────┐
                     │                     ▼                          │
                     │           SERVER  (1 GPU endpoint)             │
                     │    vkdist_server_run (per-connection loop)     │
                     │     ┌─────────────────────────────┐            │
                     │     │  RPC loop: recv frame,      │            │
                     │     │  parse, dispatch to op      │            │
                     │     │  handler                    │            │
                     │     └──────┬───────┬───────┬──────┘            │
                     │            │       │       │                    │
                     │   ┌────────▼──┐ ┌──▼───────▼──┐ ┌──────────┐   │
                     │   │ vkruntime │ │   vkblas    │ │  buffer  │   │
                     │   │ alloc/    │ │   sgemm     │ │  table   │   │
                     │   │ upload/   │ │  dispatch   │ │ handle-> │   │
                     │   │ download  │ │  +fence wait│ │ VkBuffer │   │
                     │   └───────────┘ └─────────────┘ └──────────┘   │
                     └────────────────────────────────────────────────┘
```

### 2.2 One server = one GPU endpoint

A server advertises exactly one `(VkPhysicalDevice, VkDevice, VkQueue,
VkBLASContext)` tuple. Every remote buffer a client registers is allocated on
that device; every dispatched op runs on that device. This keeps the buffer
table and command-buffer ownership trivially serialized. Multi-GPU-per-machine
is a future extension (a second server endpoint on another port).

### 2.3 Handles

Remote buffers are **opaque `uint64_t` handles**. The server owns the mapping
handle → `(VkBuffer, VkDeviceMemory, VkDeviceSize)`. The client never sees
Vulkan handles across the wire; it only passes handles back. Handles are
assigned monotonically per connection starting at 1.

## 3. Transport

- **TCP/IP** over loopback in Phase 0; the same code runs over any routable
  interface in P1.
- Implementation: Winsock2 on `_WIN32` (`<winsock2.h>`, link `ws2_32`), POSIX
  sockets otherwise (`<sys/socket.h>` etc.). Both paths are implemented in
  `src/vkdist/vkdist.c` behind small send/recv/close wrappers.
- Sockets are blocking; `TCP_NODELAY` is set on both ends to keep
  request/reply latency low.
- All I/O is explicit length-bounded: a helper reads/writes *exactly* N bytes
  in a loop, so short reads/writes from the kernel cannot desync the stream.

### 3.1 Message framing

Every message on the wire is:

```
+------------------+------------------+------------------------------+
| u32 LE length    | u32 LE opcode    | payload (length bytes)       |
| (payload bytes)  |                  |                              |
+------------------+------------------+------------------------------+
```

- `length` is the byte count of the payload **only** (the 8-byte header is not
  counted).
- All integers/floats are little-endian, encoded explicitly (never by
  reinterpreting host structs), so the protocol is endian-portable.
- Phase 0 caps a single frame at **1 MiB** (`VKDIST_MAX_FRAME`). A frame
  larger than the cap is a protocol error and the connection is closed. This
  bound makes server-side receive buffers fixed-size and reusable per
  connection. Large weight/activation transfers are streamed/chunked in P1.
- Every **reply** echoes the request opcode and begins with an `i32 VkResult`
  (`0` = `VK_SUCCESS`).

## 4. Protocol

### 4.1 Opcodes

| Opcode | Name | Direction | Request payload | Reply payload |
|--------|------|-----------|-----------------|---------------|
| 1 | `HELLO` | C→S | `u32 version` | `i32 result` |
| 2 | `REGISTER_BUFFER` | C→S | `u64 size` | `i32 result`, `u64 handle` |
| 3 | `UPLOAD` | C→S | `u64 handle`, `u64 offset`, `u64 size`, raw bytes | `i32 result` |
| 4 | `DISPATCH_GEMM` | C→S | `i32 m, i32 n, i32 k`, `f32 alpha`, `u64 A, u64 B, u64 C`, `f32 beta`, `i32 lda, i32 ldb, i32 ldc` | `i32 result` |
| 5 | `READBACK` | C→S | `u64 handle`, `u64 offset`, `u64 size` | `i32 result`, raw bytes (only on success) |
| 6 | `BYE` | C→S | *(empty)* | *(none — server closes)* |

### 4.2 Session lifecycle

1. Client connects and sends `HELLO` with the protocol version
   (`VKDIST_PROTOCOL_VERSION = 1`). The server replies `VK_SUCCESS` on a match
   or a negative error (and closes) on mismatch.
2. The client registers buffers, uploads, dispatches, and reads back using the
   request/reply sequence above. The server is **strictly serial** per
   connection: it processes one request, sends its reply, then reads the next.
3. The client sends `BYE` (or just closes). The server tears down its
   per-connection runtime and returns.

### 4.3 Wire layout detail

- `REGISTER_BUFFER` — server `vkr_malloc`s a
  `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST` buffer (device-local memory
  class per vkruntime), stores it in the connection's table, replies with the
  handle.
- `UPLOAD` — payload length must equal `24 + size`; the raw bytes follow the
  three `u64` fields. Server calls `vkr_upload` (host→device staging copy,
  submitted + waited), replies with the `VkResult`.
- `DISPATCH_GEMM` — payload is exactly 56 bytes. Server validates m/n/k > 0,
  `lda ≥ m`, `ldb ≥ k`, `ldc ≥ m`, that A/B/C handles exist, and that the
  buffers are large enough for `(k×lda)`, `(n×ldb)`, `(n×ldc)` f32 elements
  respectively. It then records one `vkblas_sgemm` (in place: C is both the
  beta-read source and the output D, which the GEMM shader reads before it
  writes, so in-place is safe) into a per-connection command buffer, submits
  with a fence, waits for the fence, resets the command buffer, and replies.
- `READBACK` — server calls `vkr_download` (device→host staging copy,
  submitted + waited), replies `i32 result` followed by the raw bytes when
  successful. Reply payload is capped at `4 + VKDIST_MAX_FRAME - 4` so it
  fits the 1 MiB frame budget.

### 4.4 Server error mapping

| Condition | Reply `VkResult` |
|-----------|-------------------|
| `HELLO` version mismatch | `VK_ERROR_INCOMPATIBLE_DRIVER` (then close) |
| Unknown opcode / malformed payload / bad handle / out-of-range offset | `VK_ERROR_UNKNOWN` |
| `vkr_malloc` / `vkr_upload` / `vkr_download` failure | the returned `VkResult` |
| Oversized frame / protocol desync | close (no reply possible) |

## 5. Data movement

### 5.1 Host → Server GPU (upload)

```
client host buffer ──TCP──► server host frame ──vkr_upload──► server device buffer
```

The bytes cross TCP once and are staged into device memory once. `vkr_upload`
creates a transient host-visible staging buffer, memcpy's the received bytes
in, records a single `vkCmdCopyBuffer` into the connection command buffer,
submits to the server's compute/transfer queue, waits for idle, and frees the
staging buffer.

### 5.2 Server GPU → Host (readback)

```
server device buffer ──vkr_download──► server host staging ──TCP──► client host buffer
```

`vkr_download` performs the mirror copy (device → host-visible staging, submit,
wait), then the server frames the bytes back to the client.

### 5.3 Synchronization

Phase 0 keeps a single queue per connection and fully synchronizes between
every stage (submit + wait). Upload → `vkQueueWaitIdle` makes transfer writes
visible to the subsequent compute dispatch; dispatch → `vkWaitForFences` makes
compute writes visible to the subsequent readback copy. No explicit pipeline
barriers are required because every stage is separated by a host-side wait on
the same queue. This trades throughput for correctness and is the documented
Phase-0 choice; batching and multi-op command buffers with explicit barriers
are a P1 optimization.

## 6. Concurrency

- **Per-connection serialization:** `vkdist_server_run` is a single-threaded
  request/reply loop. One command buffer and one runtime serve one connection,
  so there is no concurrent Vulkan access. This matches the contract that
  `VkBLASContext` is not thread-safe.
- **One server, many connections (P1+):** the listen socket is multi-accept.
  Each accepted connection gets its own thread (or polled socket) and — to keep
  Vulkan serialization trivial — its own command buffer; the `VkBLASContext`
  and device are shared and must be guarded (mutex around `vkblas_*` calls, or
  per-connection contexts). The runtime stays per-device in P1.
- **The client API** is safe to call from one thread per socket. Different
  sockets from different threads are independent.

## 7. Fault handling

- **Peer close:** `recv == 0` at any frame boundary ends the session cleanly
  (`VK_SUCCESS`). A truncated frame mid-payload is detected by the
  read-exactly-N helper and treated as an error.
- **Socket errors:** any send/recv failure aborts the session; the server
  releases Vulkan resources and returns the last error. The client's
  subsequent calls fail fast with a negative `VkResult`.
- **Bad requests** (unknown handle, out-of-range offset, oversized frame,
  malformed length) are rejected with a negative `VkResult` reply; the server
  keeps the connection alive except for desync conditions (oversized/malformed
  header), where continuing would be unsafe.
- **Device errors:** a failed `vkblas_sgemm`, submit, or wait returns the
  driver `VkResult` to the client; the server stays up for the next request.
- **Resource leaks:** every remote buffer is `vkDestroyBuffer`-ed and every
  block freed by `vkr_destroy_runtime` when the session ends; staging buffers
  are transient. (vkruntime's `vkr_destroy_runtime` does not destroy
  outstanding `VkBuffer` handles, so `vkdist` explicitly destroys its table
  entries first.)

## 8. Security notes

- **No authentication in Phase 0.** The wire carries no credentials and no
  integrity/confidentiality protection. The protocol trusts the LAN.
- Bind the listener to `127.0.0.1` (loopback only) for local testing; binding
  to `0.0.0.0`/LAN IP exposes the GPU to any host that can reach the port.
- There is no size-authentication: a client may request arbitrarily many
  buffers up to the frame cap and device memory. A phase-0 server is intended
  for trusted networks.
- P4 adds TLS (mutual auth) and an optional capability/bandwidth token so a
  peer cannot silently exhaust another machine's VRAM.

## 9. Phased roadmap

| Phase | Scope | Deliverables |
|-------|-------|--------------|
| **P0 (this subtree)** | Loopback vertical slice | Design spec, `vkdist` lib, `tests/test_vkdist.c` (local sgemm via TCP, passes) |
| **P1** | Multi-PC over IP | Bind to LAN IP, streaming/chunked large payloads, multi-connection accept loop + per-connection command buffers, keep-alive/ping, bigger frame budget, per-device (not per-connection) runtime |
| **P2** | Distributed GEMM | Partition m/n across workers; client-side split of A/B/C; server-side partial GEMM into per-worker C slices; host reduce/sum of beta terms; grid split per worker |
| **P3** | Attention / KV sharding | Remote KV-cache buffers with incremental upload (cache-miss only), sharded attention (head or sequence partition), weight sharding (row/column split of Q/K/V/O and FFN), fused remote dequant GEMM (`qgemm_q4k`) |
| **P4** | Rendezvous / discovery + TLS | mDNS/JSON rendezvous to find worker GPUs, health + capability broadcast (VRAM, arch tier), mutual TLS, payload encryption, per-connection auth tokens, quotas |

### 9.1 Example multi-worker topology (P2 target)

```
        ┌──────────────┐  TCP   ┌──────────────┐
        │  Coordinator │◄──────►│ Worker A GPU │  C[:, 0..n/2)
        │  (rank 0)    │  TCP   ├──────────────┤
        │  splits A,B  │◄──────►│ Worker B GPU │  C[:, n/2..n)
        └──────────────┘  TCP   └──────────────┘
```

Each worker runs a `vkdist_server_run` session; the coordinator dispatches
`DISPATCH_GEMM` with per-worker n-slices and reduces the partial results. The
existing buffer/upload/readback primitives carry directly into this design.

## 10. Phase-0 limitations (explicit)

1. Single connection served serially; no concurrent sessions.
2. 1 MiB frame cap (no streaming of multi-GB weights yet).
3. Descriptor sets are allocated per dispatch from the `VkBLASContext` pool
   (256 max) and never freed; a long-running connection eventually exhausts
   the pool. P1 frees/reuses sets or moves GEMM dispatch to push descriptors.
4. Full GPU-idle wait after every upload, dispatch, and readback; no pipelining.
5. Runtime + command pool are created/destroyed per connection rather than
   shared per device.
6. IPv4 only (`inet_pton` AF_INET); IPv6 and hostname resolution deferred to
   P1.
7. Server assumes queue family 0 and uses `vkGetDeviceQueue(dev, 0, 0, ...)`.

## 11. Files

| File | Purpose |
|------|---------|
| `include/vkdist/vkdist.h` | Public API (client + server entry points) |
| `include/vkdist/AGENTS.md` | Public API contract |
| `src/vkdist/vkdist_internal.h` | Opcodes, frame constants, server state structs |
| `src/vkdist/vkdist.c` | Socket wrappers, framing, server RPC loop, client API |
| `src/vkdist/AGENTS.md` | Implementation contract |
| `tests/test_vkdist.c` | Loopback harness: server thread + remote sgemm vs CPU ref |
| `specs/VKDIST-DESIGN.md` | This document |
