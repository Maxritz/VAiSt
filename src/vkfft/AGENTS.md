# VKFFT Source — Local Contract

Child of root `AGENTS.md`, `include/vkfft/AGENTS.md`, and `shaders/vkfft/AGENTS.md`.

## Implementation Rules

### Plan = context
`VkFFTPlan` IS the context (mirrors rocfft plan semantics). It owns the
`VkDevice`, a driver-level pipeline cache, the descriptor set layout, the
pipeline layout, an optional descriptor pool (push-descriptor fallback), the
`vkCmdPushDescriptorSetKHR` function pointer, and a fixed-size open-addressing
pipeline cache, plus the FFT size `n`, `log2n`, and `is_2d` (1 = N x N
separable 2D plan). The plan is direction- and precision-agnostic: direction is
passed per-execute, and the f32/f16 pipelines are cached under different
(kernel, data_type) keys. 1D and 2D plans share the same FFT kernel and the
same 16-byte push-constant block (the `mode` field selects the buffer slice).

### Pipeline caching
- Pipeline created lazily on first `vkfft_execute_*`.
- Cache: open-addressing hashmap, power-of-two (256), linear probing.
- Hash key: (kernel, data_type, tier_index) — see `vkfft_hash_key()`.
  One kernel `VKFFT_KERNEL_FFT`; data_type distinguishes f32 (`VKFFT_DTYPE_F32`)
  from f16 (`VKFFT_DTYPE_F16`).
- No malloc per lookup; fixed-size cache array in the plan struct.

### Capability detection at plan creation
Delegated to VKRuntime's `vkr_detect_capabilities()` (see `src/vkruntime/`):
vkfft links vkruntime and calls it once in `vkfft_create_plan_internal` to fill
the plan's capability fields and `active_tier`. Only a baseline shader exists
today, so `vkfft_ensure_pipeline` walks coop -> subgroup -> baseline and always
lands on the baseline blob. `active_tier` is still detected and reported.

### Push constant layout (std140, 16 bytes)
Must match `fft_f32.comp` / `fft_f16.comp` exactly (no uint64_t):
```
offset  0: uint n, log2n, direction, mode  (16 bytes)
Total: 16 bytes
```
C99 static assert enforces `sizeof(vkfft_push_constants_t) == 16`. `mode` is
`VKFFT_MODE_ROW` (0, stride 1 — the 1D path and 2D row pass) or
`VKFFT_MODE_COL` (1, stride n — 2D column pass).

### Descriptor set layout (set=0)
| binding | type  | access | stage     |
|---------|-------|--------|-----------|
| 0       | SSBO  | read   | compute   |
| 2       | SSBO  | write  | compute   |

Binding 1 is intentionally unused. Push-descriptor path writes both bindings;
the fallback path sets `writes[i].dstSet = ds` BEFORE `vkUpdateDescriptorSets`.

### Dispatch
One workgroup (256 threads, n active via a strided loop) per FFT,
`vkCmdDispatch(1,1,1)` — batch 1. Works for n up to 1024.

### Execute functions
| Function | Kernel | dtype | direction |
|----------|--------|-------|-----------|
| `vkfft_execute_f32` | FFT | F32 | forward |
| `vkfft_execute_inverse_f32` | FFT | F32 | inverse |
| `vkfft_execute_f16` | FFT | F16 | forward |
| `vkfft_execute_inverse_f16` | FFT | F16 | inverse |
| `vkfft_execute_2d_f32` | FFT | F32 | forward (N x N separable) |
| `vkfft_execute_2d_inverse_f32` | FFT | F32 | inverse (N x N separable) |

The four 1D functions share the internal `vkfft_execute_dir` helper which
fills the push constants (mode = ROW, dispatch (1,1,1)) and dispatches one
workgroup. The two 2D functions share `vkfft_execute_dir_2d`: pass 1 dispatches
N workgroups in ROW mode (input -> output), a compute-to-compute memory barrier
(SHADER_WRITE -> SHADER_READ) follows, then pass 2 dispatches N workgroups in
COLUMN mode in place on `output`. The 2D round trip recovers the input scaled
by N*N (unnormalized inverse per axis). `vkfft_execute_dir_2d` rejects plans
with `is_2d == 0`.

## Files

| File | Purpose |
|------|---------|
| `vkfft_internal.h` | Push constants, plan struct, pipeline cache |
| `vkfft.c` | Plan lifecycle, dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |