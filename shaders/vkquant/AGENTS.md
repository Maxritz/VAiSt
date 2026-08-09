# VKQuant Shaders — Local Contract

Child of root `AGENTS.md`. Governs `shaders/vkquant/` and its tier subdirs.

## Shader Variant Strategy — Capability Tiers

Shaders are organized by Vulkan capability tier, matching the VKMath/VKBLAS
pattern. VKQuant currently ships **baseline only**:

```
shaders/vkquant/
└── baseline/       Tier 0: portable Vulkan 1.4 core (Q8_0/Q4_0 dequant)
```

### What runs where

| Kernel          | Baseline |
|-----------------|----------|
| dequant_q8_0    | yes      |
| dequant_q4_0    | yes      |
| dequant_q4k     | yes      |
| dequant_q6k     | yes      |
| dequant_iq4xs   | yes      |
| quantize_q8_0   | yes      |
| quantize_q4_0   | yes      |

### Fallback chain
- `vkquant_ensure_pipeline()` walks the full tier chain (coopmatrix ->
  subgroup -> baseline) exactly like `vkmath_ensure_pipeline()`, but because
  no subgroup/coopmatrix blobs exist for dequant kernels, every dispatch
  resolves to the baseline blob. The runtime active_tier is detected but never
  used to gate kernel selection.

### Push constant layout (std140, 16 bytes)
```
offset  0: uint  num_blocks, _pad0, _pad1, _pad2   (16 bytes)
Total: 16 bytes — no uint64_t, no shaderInt64 requirement
```
Must byte-match `vkquant_push_constants_t` in `src/vkquant/vkquant_internal.h`.

### Descriptor set layout (set=0)
| binding | type  | access |
|---------|-------|--------|
| 0       | SSBO  | read   |
| 2       | SSBO  | write  |

Binding 0 = raw quantized input bytes, binding 2 = f32 output. Only these two
bindings exist (no binding 1).

### Byte formats (authoritative)
- **Q8_0**: block at `block*36`. Bytes 0..3 = f32 `d`; bytes 4..35 = 32 x int8
  `qs`. `out[i] = d * qs[i]`.
- **Q4_0**: block at `block*20`. Bytes 0..3 = f32 `d`; bytes 4..19 = 16 x uint8
  packed nibbles. `v = (nibble - 8)`; `out[i] = d * v`.
- **Q4_K** (`dequant_q4k_f32`): 144 B/block, 256 elems. f16 `d` + f16 `dmin` +
  12 scale bytes + 128 nibble bytes (ggml `block_q4_K`). Per-32-group
  `get_scale_min_k4` decode: `out = d*sc*nib - dmin*mn`.
- **Q6_K** (`dequant_q6k_f32`): 210 B/block, 256 elems. `ql[128]` + `qh[64]` +
  16 int8 scales + f16 `d` (ggml `block_q6_K`).
  `out = d * scale * ((ql4 | qh2<<4) - 32)`.
- **IQ4_XS** (`dequant_iq4xs_f32`): 136 B/block, 256 elems. f16 `d` + u16
  `scales_h` + `scales_l[4]` + 128 nibble bytes (ggml `block_iq4_xs`) +
  `kvalues_iq4nl` LUT. `out = d*(ls-32) * iq4nl[nib]`.
- **Quantize** (`quantize_q8_0_f32` / `quantize_q4_0_f32`): f32 source,
  32 elems/block, output is our f32-scale Q8_0/Q4_0 (36/20 B). Round-trips
  through the corresponding dequant shader.

### Kernel strategy
- Dequant kernels: 256 threads/workgroup; 256 elems/block so
  `block = idx >> 8`, `lane = idx & 255`; `groups_x = num_blocks`.
- Quantize kernels: 256 threads/workgroup covering 8 blocks of 32 elements
  (`block = wgID*8 + (tid>>5)`); block-local max via shared-memory tree
  reduction + `barrier()`; `groups_x = ceil(num_blocks/8)`.
- Raw byte I/O via `uint8_t`/`int8_t` requires
  `GL_EXT_shader_explicit_arithmetic_types` and `GL_EXT_scalar_block_layout`
  for tight packing. No `shaderInt64` use. f16 scales are read as 2 bytes LE
  and converted with `unpackHalf2x16`.
