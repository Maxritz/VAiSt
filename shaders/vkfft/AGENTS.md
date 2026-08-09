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
- `fft_f32.comp` — forward radix-2 complex FFT, single precision.

## Kernel contract

- `layout(local_size_x = 256)`: one workgroup of 256 threads; only n threads
  (n <= 256) are active for memory ops.
- **Barrier rule:** every thread in the workgroup must execute every
  `barrier()` the same number of times. Memory ops are guarded by
  `if (idx < n)`; barriers stay outside any conditional.
- Shared-memory Cooley-Tukey DIT: bit-reversal permutation on load, then
  log2(n) radix-2 stages with a workgroup barrier between stages. Twiddles
  `exp(-2*pi*i*j/m)` computed on the fly with `cos`/`sin`.

## Push constant layout (std140, 16 bytes)

Must match `vkfft_push_constants_t` exactly (no uint64_t):

```
offset  0: uint n, log2n, _pad0, _pad1  (16 bytes)
Total: 16 bytes
```

## Descriptor set layout (set=0)

| binding | type | access |
|---------|------|--------|
| 0       | SSBO | read   |
| 2       | SSBO | write  |

Binding 0 = input (interleaved Re/Im float pairs), binding 2 = output.
Binding 1 is unused.
