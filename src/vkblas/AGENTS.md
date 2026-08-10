# VKBLAS Source — Local Contract

Child of root `AGENTS.md` and `include/vkblas/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first GEMM call
- Cache: open-addressing hashmap, power-of-two, linear probing
- Hash key: (data_type, transA, transB, is_strided, tier_index, tile_m, tile_n, tile_k)
- No malloc per lookup; fixed-size cache array
- Fused quantized GEMMs use dedicated `data_type` codes
  (`VKBLAS_DTYPE_QGEMM_Q8_0` / `VKBLAS_DTYPE_QGEMM_Q4K` / `Q4_0` / `Q5K` /
  `Q6K` / `Q3K` / `IQ4XS` in `vkblas_internal.h`)
  so their cache keys never collide with the plain f32/f16/bf16/i8/f64 GEMMs.
  All seven qgemm formats ship **subgroup-tier** blobs
  (`subgroup/qgemm_<fmt>.comp` + `subgroup/qgemm_<fmt>_f16.comp`, fp16 output
  storage) in addition to baseline; the `ensure_pipeline` fallback walk
  resolves baseline for devices without subgroups.

### Capability detection at context creation
Delegated to VKRuntime's `vkr_detect_capabilities()` (see `src/vkruntime/`):
vkblas links vkruntime and calls it once in `vkblas_create_context` to fill the
context's capability fields. The driver-guarded `use_coopmat` (env
`VAIT_COOPMATRIX`) and `active_tier` selection stay in the lib. The public
`vkblas_init_capabilities()` re-detects via the same helper.

### SPIR-V embedding
- All shader SPIR-V files are compiled to C arrays (`shaders_spv.h`)
- Embedded as `static const uint32_t` arrays — zero dynamic allocation
- `embed_spv.ps1` runs during CMake configure to generate the header
- If shader files are missing, context creation returns `VK_ERROR_FEATURE_NOT_PRESENT`

### Push constant layout (std140)
Must match GLSL push_constant block exactly:
```
offset 0:   uint m, n, k (12 bytes)
offset 12:  float alpha, beta (8 bytes)
offset 20:  uint lda, ldb, ldc, ldd (16 bytes)
offset 36:  int transA, transB, beta_is_zero, pad (16 bytes)
offset 52:  [4 bytes padding for 8-byte alignment]
offset 56:  uint64_t strideA, strideB, strideC, strideD (32 bytes)
offset 88:  uint64_t batchCount (8 bytes)
offset 96:  uint64_t _pad1 (8 bytes)
Total: 104 bytes (fits in minPushConstantsSize = 128)
```

### Descriptor set layout (set=0)
| binding | type       | access   | stage |
|---------|------------|----------|-------|
| 0       | SSBO       | read     | compute |
| 1       | SSBO       | read     | compute |
| 2       | SSBO       | read     | compute |
| 3       | SSBO       | write    | compute |
| 4       | Uniform    | read     | compute |
*(binding 4 reserved for future UBO; currently all push constants)*

### Fused quantized GEMM (dequant-in-matmul)
`vkblas_qgemm_q8_0_f32` / `vkblas_qgemm_q4k_f32` / `vkblas_qgemm_q4_0_f32` /
`vkblas_qgemm_q5k_f32` / `vkblas_qgemm_q6k_f32` / `vkblas_qgemm_q3k_f32` /
`vkblas_qgemm_iq4xs_f32` dequantize the weight matrix *inside* the matmul
kernel (see `shaders/vkblas/AGENTS.md` for the shaders).
Dispatch mirrors `vkblas_gemm_common` (lazy pipeline, descriptor set,
push constants, tiled grid) with these differences:

- **Weight buffer (binding 0)** is raw quantized bytes read via
  `uint8_t`/`int8_t` scalar-layout SSBO (same style as the vkquant dequant
  shaders). The set layout is unchanged (4 SSBO bindings).
- **Binding map:** Wq -> 0, x -> 1, y -> 2 (beta read), y -> 3 (write). y is
  both C and D because the API computes in place.
- **Push constants** reuse `vkblas_push_constants_t`. The host swaps m/n:
  `pc.m` = weight rows (= API n), `pc.n` = activation cols (= API m),
  `pc.k` = depth. `pc.lda` = weight row byte stride (ldw), `pc.ldb` = ldx,
  `pc.ldc` = `pc.ldd` = ldy. `transA = transB = 0`.
- **Grid:** tier-dependent. Baseline: ceil(n/16) x ceil(m/16) (one workgroup
  per 16x16 output block). Subgroup: ceil(n/32) x ceil(m/8) (one 32-lane
  subgroup per 32x8 block, all seven formats). The resolved tier is read from
  the pipeline-cache entry via `vkblas_qgemm_resolved_tier`;
  `vkblas_qgemm_tile_dims` maps (kernel, tier) -> tile and the baseline grid
  is bit-for-bit unchanged.
- **fp16 output storage:** `vkblas_qgemm_<fmt>_f16` (private dtype codes
  32..38, see `vkblas.c`) store the f32 accumulator as `float16_t` into
  y/z (beta read + write in place), requiring only
  `storageBuffer16BitAccess` + `scalarBlockLayout` on the device.
- `vkblas_qgemm_get_tier(ctx, format, &tier)` (public, `vkblas.h`) reports the
  resolved execution tier (0/1/2) per quantized weight format for harness
  verification of the subgroup dispatch.

#### Weight layout (authoritative)
- W is n rows x k columns, stored row-major in blocks. Row r starts at byte
  offset `r * ldw`; `ldw` must be a multiple of 4.
- **Q8_0** (`qgemm_q8_0_f32`): 36 B/block of 32 elems — f32 `d` (bytes 0..3)
  + 32 x int8 `qs` (bytes 4..35). `dequant(i) = d * qs[i]`. Row needs
  `ceil(k/32)` blocks.
- **Q4_0** (`qgemm_q4_0_f32`): 20 B/block of 32 elems — f32 `d` (bytes 0..3)
  + 16 x uint8 packed nibbles (bytes 4..19).
  `dequant(i) = d * ((nib & 0xF) - 8)`. Row needs `ceil(k/32)` blocks.
- **Q4_K** (`qgemm_q4k_f32`): 144 B/block of 256 elems, canonical ggml
  `block_q4_K` — f16 `d` (0..1), f16 `dmin` (2..3), `scales[12]` (4..15),
  `qs[128]` nibbles (16..143). Per-32-group `get_scale_min_k4` decode:
  `out = d*sc*nib - dmin*mn` (see shader/header for the bit math). Row needs
  `ceil(k/256)` blocks.
- **Q5_K** (`qgemm_q5k_f32`): 176 B/block of 256 elems, canonical ggml
  `block_q5_K` — f16 `d` (0..1), f16 `dmin` (2..3), `scales[12]` (4..15),
  `qh[32]` (16..47), `qs[128]` (48..175). Per-32-group `get_scale_min_k4`
  scale/min pair plus a 5th bit from `qh`:
  `out = d*sc*level - dmin*mn`. Row needs `ceil(k/256)` blocks.
- **Q6_K** (`qgemm_q6k_f32`): 210 B/block of 256 elems, canonical ggml
  `block_q6_K` — `ql[128]` (0..127), `qh[64]` (128..191), int8 `scales[16]`
  (192..207), f16 `d` (208..209). `out = d*sc*(level-32)`. Row needs
  `ceil(k/256)` blocks.
- **Q3_K** (`qgemm_q3k_f32`): 110 B/block of 256 elems, canonical ggml
  `block_q3_K` — `hmask[32]` (0..31), `qs[64]` (32..95), `scales[12]`
  (96..107), f16 `d` (108..109). 16 x int8 scales recovered from the 12 packed
  bytes; `out = d*(sc-32)*level`. Row needs `ceil(k/256)` blocks.
- **IQ4_XS** (`qgemm_iq4xs_f32`): 136 B/block of 256 elems, canonical ggml
  `block_iq4_xs` — f16 `d` (0..1), `scales_h` (2..3), `scales_l[4]` (4..7),
  `qs[128]` (8..135). `out = d*(ls-32)*iq4nl[nib]` (kvalues_iq4nl LUT).
  Row needs `ceil(k/256)` blocks.

## Files

| File | Purpose |
|------|---------|
| `vkblas_internal.h` | Internal structs, push constants, pipeline cache |
| `vkblas.c` | Context lifecycle, GEMM dispatch, pipeline creation, shader selection |
