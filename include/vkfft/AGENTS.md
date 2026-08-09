# VKFFT Public API — Local Contract

Child of root `AGENTS.md`.

## Scope

Forward radix-2 complex FFT, f32, n = power of two, 2 <= n <= 256. One
workgroup per FFT; batch = 1 for this scope.

## Buffer layout

Interleaved (Re, Im) float pairs. Element j lives at in[2j], in[2j+1] and
out[2j], out[2j+1]. Each buffer holds 2*n floats.

## Public API

| Function | Purpose |
|----------|---------|
| `vkfft_create_plan(pd, device, n, &plan)` | Create a plan; validates n is a power of two in [2,256] (`VKFFT_ERROR_INVALID_ARGUMENT` otherwise). The plan owns the device, pipeline cache, set layout, pipeline layout, descriptor pool, and push-desc fn — the plan IS the context. |
| `vkfft_destroy_plan(plan)` | Release all cached pipelines / descriptors / handles. |
| `vkfft_execute_f32(plan, cmd, input, output)` | Record a forward FFT dispatch into `cmd` (one 256-thread workgroup, n active). |
| `vkfft_get_size(plan)` | Returns n. |
| `vkfft_get_arch_name(plan)` / `vkfft_get_arch_index(plan)` | Capability-tier queries. |

## Semantics

- Caller owns buffer memory and synchronization; VKFFT only records compute
  work into a caller-supplied `VkCommandBuffer`.
- Forward transform: X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n).
- The caller must order pipeline barriers so shader writes are visible before
  any readback.
- No heap allocation in hot paths; the plan owns fixed-size pipeline-cache
  storage and a descriptor pool.

## Descriptor set (set=0)

| binding | type | access | stage |
|---------|------|--------|-------|
| 0       | SSBO | read   | compute |
| 2       | SSBO | write  | compute |

Binding 1 is intentionally unused (input is a single buffer).
