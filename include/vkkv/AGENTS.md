# VKKV Include — Local Contract

Child of root `AGENTS.md`. Governs the public API surface under `include/vkkv/`.

## Scope

Cross-Model KV Cache Transfer per arXiv:2608.03893 (see the abstract at
https://arxiv.org/abs/2608.03893). The module implements the LINEAR-ALGEBRA
layer only: a per-head closed-form ridge mapper that maps a source model's K/V
cache to a target model's K/V cache so prefill can be skipped when swapping
between same-family models. No LLM runtime is built; the module consumes and
produces plain float buffers and is validated numerically against CPU linear
algebra.

## Mapping of the paper's three-step design

| Paper step | Representation in vkkv |
|------------|------------------------|
| 1. Strip RoPE from keys | **Caller's responsibility** — callers must present RoPE-free K/V to `vkkv_fit_cpu`/`vkkv_apply`. |
| 2. Per head, select top-k source layers | One fitted mapper **per head** (one Wh). Which calibration data feeds each head = the caller's selection. |
| 3. Per-head ridge fit + apply | `vkkv_fit_cpu` (host solve) + `vkkv_apply` (GPU matmul). |

## The mapper math (authoritative)

For a matched calibration set per head h with n samples:
- X_h is [n x src_dim] row-major (SOURCE keys/values, RoPE-stripped).
- Y_h is [n x tgt_dim] row-major (TARGET keys/values).
- Fit: `W_h = (X_h^T X_h + lambda*I)^-1 * X_h^T Y_h` (ridge regression).
- Apply: `TARGET[r][j] = sum_k SOURCE[r][k] * W_h[k][j]`.

All matrix buffers are **row-major float32**: element [r][c] of an [m x k]
buffer lives at `r*k + c`. There is no transpose in apply — `W_h` is stored
exactly as [src_dim x tgt_dim] row-major and multiplied on the right of the
source rows.

## Design decisions (documented, binding)

- **Fit is host-side** (C99 double precision inside `vkkv_fit_cpu`): the
  calibration set is small and the ridge solve is an offline step; a double
  Gauss-Jordan solve on the host is simpler and more accurate than a GPU
  GEMM+inverse chain. Fitted W is uploaded once to a device buffer.
- **Apply is GPU-side** (`vkkv_apply`): one compute dispatch of the
  `shaders/vkkv/baseline/apply.comp` shader — thread per output element,
  dot over src_dim, `local_size 256`, guarded. This is the production path.
- **W storage**: one device buffer [n_heads][src_dim x tgt_dim] floats,
  row-major; `W_h` block at element offset `h * src_dim * tgt_dim`.
- **Dependencies**: vkkv links **vkruntime only** (pooled allocator, staging
  upload/download, pipeline helpers). It does NOT link vkmath or vkblas: the
  fit is a host solve and apply is one self-contained shader.
- **No push-dependency on vkmath/vkblas shaders**: `apply.comp` is a plain
  Vulkan 1.4 core compute shader (no shaderInt64, no subgroup, no coopmatrix).

## Vulkan-native

- `VkKVTransfer` is opaque; layout lives only in `src/vkkv/vkkv_internal.h`.
- Every handle is a Vulkan object (`VkBuffer` args to `vkkv_apply`, device
  buffer for W, VkPipeline cached in the context).
- Command recording happens into a caller-supplied `VkCommandBuffer`
  (`vkkv_apply`); the transfer's internal command buffer is used only by
  vkruntime staging uploads/downloads.

## Thread safety

- `VkKVTransfer` is **not thread-safe**. Callers serialize concurrent
  `vkkv_*()` calls on the same transfer.
- `vkkv_apply` records into any caller command buffer; different transfers
  may be used concurrently on distinct buffers.

## Files

| File | Purpose |
|------|---------|
| `vkkv.h` | Public API: create/destroy, fit (CPU), apply (GPU) |
