# VKBLAS Source — Local Contract

Child of root `AGENTS.md` and `include/vkblas/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first GEMM call
- Cache: open-addressing hashmap, power-of-two, linear probing
- Hash key: (data_type, transA, transB, is_strided, tier_index, tile_m, tile_n, tile_k)
- No malloc per lookup; fixed-size cache array

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

## Files

| File | Purpose |
|------|---------|
| `vkblas_internal.h` | Internal structs, push constants, pipeline cache |
| `vkblas.c` | Context lifecycle, GEMM dispatch, pipeline creation, shader selection |
