# VKFFT Public API — Local Contract

Child of root `AGENTS.md`.

## Scope

Radix-2 complex FFT, f32 or f16 I/O, forward or inverse, n = power of two,
2 <= n <= 1024. One 256-thread workgroup per FFT (strided loop over n); batch
= 1 for this scope. The inverse is unnormalized (rocfft default), so
forward-then-inverse returns the original signal scaled by n.

## Buffer layout

Interleaved (Re, Im) pairs. For f32, element j lives at in[2j], in[2j+1] and
out[2j], out[2j+1]; each buffer holds 2*n floats. For f16, the same layout in
float16_t; each buffer holds 2*n float16_t values (n*4 bytes).

## Public API

| Function | Purpose |
|----------|---------|
| `vkfft_create_plan(pd, device, n, &plan)` | Create a plan; validates n is a power of two in [2,1024] (`VKFFT_ERROR_INVALID_ARGUMENT` otherwise). The plan owns the device, pipeline cache, set layout, pipeline layout, descriptor pool, and push-desc fn — the plan IS the context. It is direction- and precision-agnostic. |
| `vkfft_destroy_plan(plan)` | Release all cached pipelines / descriptors / handles. |
| `vkfft_execute_f32(plan, cmd, in, out)` | Forward f32 FFT dispatch (one 256-thread workgroup). |
| `vkfft_execute_inverse_f32(plan, cmd, in, out)` | Inverse f32 FFT dispatch (unnormalized). |
| `vkfft_execute_f16(plan, cmd, in, out)` | Forward f16 FFT dispatch (f32 internal compute). |
| `vkfft_execute_inverse_f16(plan, cmd, in, out)` | Inverse f16 FFT dispatch (unnormalized). |
| `vkfft_get_size(plan)` | Returns n. |
| `vkfft_get_arch_name(plan)` / `vkfft_get_arch_index(plan)` | Capability-tier queries. |

## Semantics

- Caller owns buffer memory and synchronization; VKFFT only records compute
  work into a caller-supplied `VkCommandBuffer`.
- Forward transform: X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n). Inverse:
  y[m] = sum_k X[k] * exp(+2*pi*i*k*m/n) (unnormalized; matches rocfft).
- The caller must order pipeline barriers so shader writes are visible before
  any readback — including between a forward dispatch and an inverse dispatch
  that reads its output.
- No heap allocation in hot paths; the plan owns fixed-size pipeline-cache
  storage and a descriptor pool.

## Descriptor set (set=0)

| binding | type | access | stage |
|---------|------|--------|-------|
| 0       | SSBO | read   | compute |
| 2       | SSBO | write  | compute |

Binding 1 is intentionally unused (input is a single buffer).
