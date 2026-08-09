# VKRAND Shaders — Local Contract

Child of root `AGENTS.md` and `include/vkrand/AGENTS.md`.

## Shader Variant Strategy — Capability Tiers

Mirrors the VKMath pattern: shaders live under per-architecture tier
directories, and the runtime selects the highest supported tier that has a
blob for the requested kernel, falling back toward baseline.

```
shaders/vkrand/
└── baseline/       Tier 0: portable Vulkan 1.4 core
```

### What runs where

| Kernel           | Baseline |
|------------------|----------|
| uniform_f32      | yes      |

Only the baseline shader exists today; `active_tier` is clamped to baseline.

## Philox4x32-10 contract

- **Counter init**: `c0 = gl_GlobalInvocationID.x` (global thread index),
  `c1 = philox_seed_hash(seed)`, `c2 = 0x9E3779B9`, `c3 = 0xBB67AE85`.
- **Key init**: `k0 = seed`, `k1 = 0x9E3779B9 ^ seed`.
- **Round**: standard Philox4x32 multiply-xor step with
  `M0 = 0xD2511F53`, `M1 = 0xCD9E8D57`; hi-word product via `umulExtended`.
- **10 rounds**, key bumped by `(0x9E3779B9, 0xBB67AE85)` after each round.
- **Mapping**: `float(c0 & 0xFFFFFFu) / 16777216.0f` (24-bit mantissa -> [0,1)).

The CPU test reference must use the SAME algorithm and mapping, validated
against the Random123 known-answer vectors (see `include/vkrand/AGENTS.md`).

## Hard rules

- **No int64** — no `GL_ARB_gpu_shader_int64`, no `shaderInt64` feature.
  `umulExtended` is native 32-bit.
- **Push constants**: 16 bytes std140, `{uint count; uint seed; uint _pad0;
  uint _pad1;}`, byte-identical to `vkrand_push_constants_t` in the C runtime.
- **Descriptor set**: `set=0, binding=2` = output SSBO (write) only. The
  runtime layout declares exactly this binding; it must match the shader.
- **Dispatch**: `local_size_x = 256`, `groups_x = ceil(count/256)` (>= 1),
  shader guards `idx < count`.

## Adding a new kernel

Add a `*.comp` under a tier dir, add the blob to the runtime shader table
and CMake shader registry, then add a test that dispatches it.
