# VKRAND Source — Local Contract

Child of root `AGENTS.md` and `include/vkrand/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first dispatch call
- Cache: open-addressing hashmap, power-of-two (256), linear probing
- Hash key: (kernel, tier) — see `vkrand_hash_key()`
- No malloc per lookup; fixed-size cache array in context struct

### Capability detection at context creation
Delegated to VKRuntime's `vkr_detect_capabilities()` (see `src/vkruntime/`):
vkrand links vkruntime and calls it once in `vkrand_create_context` to fill the
context's capability fields. Only a baseline shader exists, so the active tier
is clamped to baseline regardless of detected capabilities.

### SPIR-V embedding
- Shader SPIR-V compiled to C arrays (`shaders_spv.h`) by `compile_shaders.ps1`
- Embedded as `static const uint32_t` arrays — zero dynamic allocation
- `vkrand_select_spirv()` returns the blob for a (kernel, tier) combination

### Push constant layout (std140, 16 bytes)
```
offset  0: uint count, seed, _pad0, _pad1   (16 bytes)
```
Must match the GLSL push_constant block exactly. uint32/float only — no
`shaderInt64` requirement.

### Descriptor set layout (set=0)
| binding | type | access | stage     |
|---------|------|--------|-----------|
| 2       | SSBO | write  | compute   |

Binding 2 = output uniform floats.

## Algorithms

| Kernel | Generator | Mapping |
|--------|-----------|---------|
| `VKRAND_KERNEL_UNIFORM_F32` | Philox4x32-10 (stateless) | `float(c0 & 0xFFFFFF) / 16777216.0` |
| `VKRAND_KERNEL_THREEFRY_F32` | ThreeFry2x32-20 (stateless) | `float(X0 & 0xFFFFFF) / 16777216.0` |
| `VKRAND_KERNEL_NORMAL_F32` | Philox4x32-10 Box-Muller | N(0,1): `sqrt(-2 ln u0) * cos(2*pi*u1)` |
| `VKRAND_KERNEL_UNIFORM_UINT32` | Philox4x32-10 (stateless) | raw `c0` word (uint32) |

All stateless: thread i derives the counter from `i` and `seed`. Philox runs 10
rounds (M0=0xD2511F53, M1=0xCD9E8D57, W0=0x9E3779B9, W1=0xBB67AE85) and is
verified against the Random123 philox4x32-10 known-answer vectors. ThreeFry2x32
runs 20 rounds with the {13,15,26,6,17,29,16,24} rotation schedule, parity
0x1BD11BDA, and key injection after every 4-round group INCLUDING the last
(so the output includes the r=5 injection); verified against the Random123
threefry2x32x20 known-answer vectors. Normal draws two consecutive uniform
counters (2*i, 2*i+1) and applies Box-Muller.

## Files

| File | Purpose |
|------|---------|
| `vkrand_internal.h` | Internal structs, push constants, pipeline cache |
| `vkrand.c` | Context lifecycle, op dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |
