# VKQuant Source — Local Contract

Child of root `AGENTS.md` and `include/vkquant/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first dispatch call
- Cache: open-addressing hashmap, power-of-two (256), linear probing
- Hash key: (kernel, tier_index) — see `vkquant_hash_key()`
- No malloc per lookup; fixed-size cache array in context struct

### Capability detection at context creation
- Mirrors `vkmath_init_capabilities`: queries `shaderInt64`, subgroup
  properties, and cooperative-matrix features via pNext chains
- Only baseline shaders exist; `ensure_pipeline` falls back to baseline

### SPIR-V embedding
- Shader SPIR-V compiled to C arrays (`shaders_spv.h`) by `compile_shaders.ps1`
- Embedded as `static const uint32_t` arrays — zero dynamic allocation
- `vkquant_select_spirv()` returns the blob for a (kernel, tier) combination

### Push constant layout (std140, 16 bytes)
```
offset  0: uint num_blocks, _pad0, _pad1, _pad2   (16 bytes)
```
Must match the GLSL push_constant block exactly. uint32/float only — no
`shaderInt64` requirement.

### Descriptor set layout (set=0)
| binding | type | access | stage     |
|---------|------|--------|-----------|
| 0       | SSBO | read   | compute   |
| 2       | SSBO | write  | compute   |

Binding 0 = quantized input (bytes), binding 2 = dequantized output (f32).

## Quantization block formats

- **Q8_0**: 36 bytes/block — f32 scale `d` (bytes 0..3) + 32 int8 `qs`
  (bytes 4..35). `out[i] = d * qs[i]`.
- **Q4_0**: 20 bytes/block — f32 scale `d` (bytes 0..3) + 16 packed-nibble
  uint8 (bytes 4..19). `xi = qs[i>>1]`; `nibble = (i&1) ? xi>>4 : xi&0xF`;
  `out[i] = d * ((int)nibble - 8)`.

## Files

| File | Purpose |
|------|---------|
| `vkquant_internal.h` | Internal structs, push constants, pipeline cache |
| `vkquant.c` | Context lifecycle, op dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |
