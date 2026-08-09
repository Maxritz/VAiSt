# VKRAND Source — Local Contract

Child of root `AGENTS.md` and `include/vkrand/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first dispatch call
- Cache: open-addressing hashmap, power-of-two (256), linear probing
- Hash key: (tier_index) — RNG has a single kernel, see `vkrand_hash_key()`
- No malloc per lookup; fixed-size cache array in context struct

### Capability detection at context creation
- Mirrors `vkmath_init_capabilities`: queries `shaderInt64`, subgroup
  properties, and cooperative-matrix features via pNext chains
- Only a baseline shader exists; `ensure_pipeline` falls back to baseline

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

## Algorithm

Philox4x32-10 counter-based PRNG. Stateless: thread i derives the counter
from `i` and `seed`, runs 10 rounds (M0=0xD2511F53, M1=0xCD9E8D57,
W0=0x9E3779B9, W1=0xBB67AE85), maps c0 to [0,1) via
`float(c0 & 0xFFFFFF) / 16777216.0`. Verified against the Random123
known-answer vectors.

## Files

| File | Purpose |
|------|---------|
| `vkrand_internal.h` | Internal structs, push constants, pipeline cache |
| `vkrand.c` | Context lifecycle, op dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |
