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

### Kernel strategy
- 256 threads/workgroup; `block = idx >> 5`, `lane = idx & 31`.
- `groups_x = ceil(32 * num_blocks / 256)` (at least 1).
- Raw byte I/O via `uint8_t`/`int8_t` requires
  `GL_EXT_shader_explicit_arithmetic_types` and `GL_EXT_scalar_block_layout`
  for tight packing. No `shaderInt64` use.
