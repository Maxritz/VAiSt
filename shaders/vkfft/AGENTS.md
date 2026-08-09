# VKFFT Shaders — Local Contract

Child of root `AGENTS.md`. Governs `shaders/vkfft/` and its tier subdirs.

## Shader Variant Strategy — Capability Tiers

Shaders are organized by Vulkan capability tier, matching the VKBLAS/VKMath
pattern:

```
shaders/vkfft/
├── baseline/       Tier 0: portable Vulkan 1.4 core
├── subgroup/       Tier 1: VK_KHR_shader_subgroup (not yet authored)
└── coopmatrix/     Tier 2: VK_KHR_cooperative_matrix (not yet authored)
```

Only the baseline tier exists today. `vkfft_ensure_pipeline` walks the try
order coop -> subgroup -> baseline and always lands on the baseline blob.

### File naming
- `fft_f32.comp` — forward/inverse radix-2 complex FFT, f32 I/O.
- `fft_f16.comp` — forward/inverse radix-2 complex FFT, f16 I/O (f32 internal
  compute; `GL_EXT_shader_explicit_arithmetic_types` + `layout(scalar)`).

## Kernel contract

- `layout(local_size_x = 256)`: one workgroup of 256 threads per FFT. Every
  thread strides over up to n/256 elements (`iters = (n + 255)/256`), so a
  single 256-thread workgroup covers the whole FFT for n up to 1024.
- **Barrier rule:** every thread in the workgroup must execute every
  `barrier()` the same number of times. The strided-loop trip count `iters` is
  derived from `pc.n`, which is uniform across the workgroup, so all threads
  run the same number of iterations. Memory ops are guarded by `(idx < pc.n)`;
  barriers stay outside any conditional.
- Shared-memory Cooley-Tukey DIT: bit-reversal permutation on load, then
  log2(n) radix-2 stages with a workgroup barrier between stages. Twiddles
  `exp(dir * 2*pi*i*j/m)` computed on the fly with `cos`/`sin`, where
  `dir = -1` for forward and `+1` for inverse (conjugate twiddles).
- Direction is selected at runtime via `pc.direction` (0 = forward,
  1 = inverse); both precisions share one shader each. The inverse is
  unnormalized (rocfft default), so forward-then-inverse recovers the original
  signal scaled by n.
- f16 shaders use `layout(scalar)` SSBOs of `float16_t` (interleaved Re/Im
  pairs, tightly packed) and convert to/from f32 at the buffer boundary;
  shared memory and all compute stay f32.

## Push constant layout (std140, 16 bytes)

Must match `vkfft_push_constants_t` exactly (no uint64_t):

```
offset  0: uint n, log2n, direction, _pad  (16 bytes)
Total: 16 bytes
```

`direction` = 0 forward, 1 inverse (VKFFT_DIR_FORWARD / VKFFT_DIR_INVERSE).

## Descriptor set layout (set=0)

| binding | type | access |
|---------|------|--------|
| 0       | SSBO | read   |
| 2       | SSBO | write  |

Binding 0 = input (interleaved Re/Im float pairs), binding 2 = output.
Binding 1 is unused.
