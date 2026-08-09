# VKBLAS-L1L2 Source — Local Contract

Child of root `AGENTS.md`, `src/vkblas/AGENTS.md` and
`include/vkblas_l1l2/AGENTS.md`. Governs `src/vkblas_l1l2/`.

## Role

Companion to VKBLAS. Implements BLAS Level-1/Level-2 ops in its OWN subtree so
it never conflicts with parallel work on `src/vkblas/vkblas.c` /
`include/vkblas/vkblas.h`. It REUSES the opaque `VkBLASContext` (design option
(a)) and links against libvkblas.

## Implementation Rules

### Context reuse (design option (a) — REQUIRED)
- Do NOT edit `src/vkblas/vkblas_internal.h` (shared, nobody edits it).
- Do NOT edit `src/vkblas/vkblas.c` or `src/vkblas/shaders_spv.h` (parallel
  work owns them). Only READ.
- Reuse helpers: `vkblas_alloc_descriptor_set`, `vkblas_write_descriptor_set`
  (writes all 4 bindings 0/1/2/3 = read/read/read/write),
  `vkblas_push_pc`, `vkblas_load_shader_module`, `vkblas_get_pipeline_layout`,
  `vkblas_hash_key` (GEMM only — see below).
- Push constants reuse `vkblas_push_constants_t` (84 bytes) exactly; the GLSL
  push-constant blocks byte-match that struct.

### Pipeline caching (SHARED context cache)
- L1/L2 pipelines are created lazily on first dispatch and inserted into the
  context's shared `pipelines[]` cache (open addressing, linear probing).
- Keys come from `vkblas_l1l2_hash_key(kernel, data_type)` with **marker bit
  63** set — GEMM keys (`vkblas_hash_key`) are always < 2^32, so the two key
  spaces are disjoint.
- Insertion only ever fills empty slots (never leaves holes), preserving the
  linear-probing invariant for GEMM lookups.
- `vkblas_destroy_context` in vkblas.c destroys every valid cached pipeline,
  including L1/L2 ones — no extra cleanup needed.

### Dispatch pattern
- `vkblas_l1l2_dispatch` records pipeline bind + push constants + ONE
  descriptor set (all four bindings written with VK_WHOLE_SIZE ranges;
  unused bindings get a valid dummy VkBuffer) + `vkCmdDispatch`.
- Workgroup size 256 for every kernel.
- Reductions (dot/nrm2/asum/amax) are TWO dispatches with a
  `vkblas_l1l2_cmd_barrier()` (SHADER_WRITE -> SHADER_READ, compute->compute)
  between them. Stage 1 writes per-workgroup partials into a scratch region of
  the caller's result buffer; stage 2 (single workgroup) reduces into result[0].
- Result buffer sizing contract (documented in the public header): dot/nrm2/
  asum need `ceil(n/256)` floats, amax needs `2*ceil(n/256)` uint32s.

### Zero-tolerance
- No stubs, TODOs, placeholders or mock code paths in `vkblas_l1l2.c`.
- No heap allocation in hot paths; all params are caller pointers.
- All f16 shaders use integer bit-packing (no `float16_t`, no shaderFloat16).

## Files

| File | Purpose |
|------|---------|
| `vkblas_l1l2_internal.h` | Kernel codes, blob table type, helper decls |
| `vkblas_l1l2.c` | L1/L2 dispatch, pipeline cache, all public ops |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays (embed_l1l2.ps1) |

## Verification

- Build: `test_vkblas_l1l2` (see tests/test_vkblas_l1l2.c) — must PASS on the
  real GPU before changes merge.
- Harness-first: any change to an op requires running the harness.
