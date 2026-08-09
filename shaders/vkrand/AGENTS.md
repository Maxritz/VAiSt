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
| threefry_uniform_f32 | yes  |
| normal_f32       | yes      |
| uniform_uint32   | yes      |

Only the baseline tier shaders exist today; `active_tier` is clamped to
baseline.

## Philox4x32-10 contract (uniform_f32, normal_f32, uniform_uint32)

- **Counter init**: `c0 = gl_GlobalInvocationID.x` (global thread index),
  `c1 = philox_seed_hash(seed)`, `c2 = 0x9E3779B9`, `c3 = 0xBB67AE85`.
- **Key init**: `k0 = seed`, `k1 = 0x9E3779B9 ^ seed`.
- **Round**: standard Philox4x32 multiply-xor step with
  `M0 = 0xD2511F53`, `M1 = 0xCD9E8D57`; hi-word product via `umulExtended`.
- **10 rounds**, key bumped by `(0x9E3779B9, 0xBB67AE85)` after each round.
- **Mapping**: `float(c0 & 0xFFFFFFu) / 16777216.0f` (24-bit mantissa -> [0,1)).
  `uniform_uint32` writes the raw `c0` word instead. `normal_f32` draws
  counters `2*i` and `2*i+1` and applies Box-Muller (see below).

## ThreeFry2x32-20 contract

- **Counter init**: `X0 = gl_GlobalInvocationID.x` (global thread index),
  `X1 = threefry_seed_hash(seed)` (fmix32). Key: `k0 = seed`,
  `k1 = 0x9E3779B9 ^ seed`.
- **Round**: `X0 += X1; X1 = rotl(X1, rot[rr % 8]); X1 ^= X0;` with
  rotations `{13, 15, 26, 6, 17, 29, 16, 24}` (Random123 R_32x2_*_0).
- **Key schedule**: `ks = {k0, k1, 0x1BD11BDA ^ k0 ^ k1}`. Key injected after
  every 4-round group cycling (ks0,ks1)->(ks1,ks2)->(ks2,ks0)->... with
  `X1 += injection_index`. **The injection after the final (5th) group runs**
  (Random123 guards with `Nrounds>19`), so 20 rounds = 5 injections.
- **20 rounds**, mapping `float(X0 & 0xFFFFFFu) / 16777216.0f`.

## Normal N(0,1) contract

- Philox4x32-10 counter init at `2*i` and `2*i+1` (two consecutive uniform
  draws), Box-Muller `z = sqrt(-2 ln u0) * cos(2*pi*u1)`.
- `u0` clamped to `1e-30` minimum so `-2 ln(u0)` stays finite.
- Mapping inputs use the standard 24-bit mantissa uniform.

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
