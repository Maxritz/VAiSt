# VKMath Source — Local Contract

Child of root `AGENTS.md` and `include/vkmath/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first dispatch call
- Cache: open-addressing hashmap, power-of-two (256), linear probing
- Hash key: (kernel, data_type, tier_index) — see `vkmath_hash_key()`
- No malloc per lookup; fixed-size cache array in context struct

### Capability detection at context creation
Delegated to VKRuntime's `vkr_detect_capabilities()` (see `src/vkruntime/`):
vkmath links vkruntime and calls it once in `vkmath_create_context` to fill the
context's capability fields and `active_tier`. The public
`vkmath_init_capabilities()` re-detects via the same helper.

### SPIR-V embedding
- All shader SPIR-V files are compiled to C arrays (`shaders_spv.h`)
- Embedded as `static const uint32_t` arrays — zero dynamic allocation
- `compile_shaders.ps1` runs during CMake configure to generate the header
- `vkmath_select_spirv()` returns the correct SPIR-V blob for a given
  (kernel, data_type, tier) combination at runtime

### Push constant layout (std140, 72 bytes)
Must match GLSL push_constant block exactly:
```
offset  0: uint  num_elements, num_rows, num_cols, _pad0   (16 bytes)
offset 16: float alpha, beta, _pad1, _pad2                  (16 bytes)
offset 32: uint64_t stride_a, stride_b, stride_out,
            batch_count, _pad3                              (40 bytes)
Total: 72 bytes (fits in minPushConstantsSize = 128)
```

### Descriptor set layout (set=0)
| binding | type  | access | stage     |
|---------|-------|--------|-----------|
| 0       | SSBO  | read   | compute   |
| 1       | SSBO  | read   | compute   |
| 2       | SSBO  | write  | compute   |

Binding 0 = input A, binding 1 = input B (optional), binding 2 = output.
For unary ops, only bindings 0 and 2 are bound. For ternary ops, all three.

## Files

| File | Purpose |
|------|---------|
| `vkmath_internal.h` | Internal structs, push constants, pipeline cache |
| `vkmath.c` | Context lifecycle, op dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |
