# VKQuant Source — Local Contract

Child of root `AGENTS.md` and `include/vkquant/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first dispatch call
- Cache: open-addressing hashmap, power-of-two (256), linear probing
- Hash key: (kernel, tier_index) — see `vkquant_hash_key()`
- No malloc per lookup; fixed-size cache array in context struct

### Capability detection at context creation
Delegated to VKRuntime's `vkr_detect_capabilities()` (see `src/vkruntime/`):
vkquant links vkruntime and calls it once in `vkquant_create_context` to fill
the context's capability fields and `active_tier`. Only baseline shaders exist;
`ensure_pipeline` falls back to baseline. The public
`vkquant_init_capabilities()` re-detects via the same helper.

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
- **Q4_K** (dequant, 144 B/256 elems): ggml `block_q4_K` — f16 `d`,`dmin`,
  `scales[12]`, `qs[128]`. Per-32-group scale/min decode via ggml
  `get_scale_min_k4`; `out = d*sc*nib - dmin*mn`.
- **Q6_K** (dequant, 210 B/256 elems): ggml `block_q6_K` — `ql[128]`,
  `qh[64]`, int8 `scales[16]`, f16 `d`. `out = d*sc*((ql4|qh2<<4)-32)`.
- **IQ4_XS** (dequant, 136 B/256 elems): ggml `block_iq4_xs` — f16 `d`,
  u16 `scales_h`, `scales_l[4]`, `qs[128]`, plus `kvalues_iq4nl` LUT.
- **Quantize** (Q8_0/Q4_0 forward): f32 source -> our f32-scale formats.

## Kernel ids

| Kernel | id | Dispatch |
|--------|----|----------|
| dequant_q8_0 / q4_0 | 0 / 1 | `ceil(32*blocks/256)` |
| dequant_q4k / q6k / iq4xs | 2 / 3 / 4 | `num_blocks` (256 elems/block) |
| quantize_q8_0 / q4_0 | 5 / 6 | `ceil(num_blocks/8)` (8 blocks/wg) |

## Files

| File | Purpose |
|------|---------|
| `vkquant_internal.h` | Internal structs, push constants, pipeline cache |
| `vkquant.c` | Context lifecycle, op dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |
