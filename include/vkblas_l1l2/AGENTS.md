# VKBLAS-L1L2 Include — Local Contract

Child of root `AGENTS.md`, `include/vkblas/AGENTS.md` and `src/vkblas/AGENTS.md`.
Governs the public API surface under `include/vkblas_l1l2/`.

## Role

`vkblas_l1l2` is a COMPANION library to VKBLAS (GEMM). It implements BLAS
Level-1/Level-2 ops (axpy, scal, dot, nrm2, asum, amax, gemv) on the SAME
opaque `VkBLASContext`. It lives in its own subtree and never edits the shared
VKBLAS files (`include/vkblas/vkblas.h`, `src/vkblas/vkblas.c`,
`src/vkblas/vkblas_internal.h`), so it cannot conflict with parallel work.

## API Design Principles

### Reuse VkBLASContext (design option (a) — REQUIRED)
- Public functions take `VkBLASContext* ctx, VkCommandBuffer cmd` exactly like
  `vkblas_sgemm`. There is no vkblas_l1l2-owned context.
- `vkblas_l1l2` links against libvkblas and drives the shared context's
  descriptor pool, set layout, pipeline layout and pipeline cache through the
  internal helpers in `src/vkblas/vkblas_internal.h`.
- L1/L2 pipelines are inserted into the context's shared pipeline cache under
  hash keys carrying marker bit 63, so they can never collide with GEMM keys.

### Vulkan-native handles everywhere
- Vectors/matrices are `VkBuffer`. Data is COLUMN-MAJOR like rocBLAS.
- `incx`/`incy` are element strides (positive). Zero/negative `n` is a no-op
  returning `VK_SUCCESS`, mirroring BLAS.
- alpha/beta are f32 host pointers, dereferenced at call time (host pointer
  mode). The context pointer-mode field only affects the GEMM family.
- No heap allocation in hot paths; commands only recorded into the caller's
  command buffer.

### rocBLAS mirror naming
- Prefix `vkblas_l1_*` / `vkblas_l2_*`, `_f16` suffix for half variants.
- Signatures mirror `rocblas_sscal/saxpy/sdot/snrm2/sasum/isamax/sgemv`
  parameter order, adapted to Vulkan handles.

### Documented deviations
- `amax` returns a 0-based index (rocBLAS returns 1-based).
- f16 data is stored one-half-per-uint32 (low 16 bits) and converted with
  integer bit-packing in the shaders, so NO `shaderFloat16` device feature is
  required. alpha/beta remain f32 scalars.
- f16 `dot` accumulates and stores the result as f32.

## Files

| File | Purpose |
|------|---------|
| `vkblas_l1l2.h` | Public API: L1 (axpy/scal/dot/nrm2/asum/amax) + L2 (gemv) |

## Thread Safety

Same as VKBLAS: `VkBLASContext` is not thread-safe. Callers serialize
concurrent `vkblas_l1l2_*()` calls that touch the same context.
