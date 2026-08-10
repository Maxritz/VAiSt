# VKMath Shaders — Local Contract

Child of root `AGANTS.md`. Governs `shaders/vkmath/` and its tier subdirs.

## Shader Variant Strategy — Capability Tiers

Shaders are organized by Vulkan capability tier, matching the VKBLAS pattern:

```
shaders/vkmath/
├── baseline/       Tier 0: portable Vulkan 1.4 core
├── subgroup/       Tier 1: VK_KHR_shader_subgroup
└── coopmatrix/     Tier 2: not used for elementwise (see README)
```

### What runs where

| Kernel | Baseline | Subgroup |
|--------|----------|----------|
| relu_f32 | yes | reuses baseline (subgroup provides no benefit) |
| relu_f16 | yes | — |
| silu_f32 | yes | yes (elementwise; same result as baseline) |
| silu_f16 | yes | yes (f32 compute, f16 I/O) |
| gelu_f32 | yes | yes (elementwise; same result as baseline) |
| gelu_f16 | yes | — |
| tanh_f32 | yes | reuses baseline |
| tanh_f16 | yes | — |
| sigmoid_f32 | yes | yes (elementwise; same result as baseline) |
| sigmoid_f16 | yes | — |
| add_f32 | yes | — |
| mul_f32 | yes | — |
| add_mul_f32 | yes | — |
| scale_f32 | yes | — |
| max_reduce_dim_f32 | yes | yes (subgroup reduction) |
| sum_reduce_dim_f32 | yes | yes (subgroup reduction) |

### Fallback chain
- Elementwise unary ops with subgroup shaders: runtime prefers subgroup tier,
  falls back to baseline tier for the same kernel.
- Reductions: same fallback. The subgroup reduction is mathematically
  identical to the baseline shared-memory tree reduction.

### Push constant layout (std140, 72 bytes)
```
offset  0: uint  num_elements, num_rows, num_cols, alpha  (16 bytes)
offset 16: float beta, _pad0, _pad1, _pad2                 (16 bytes)
offset 32: uint64_t stride_a, stride_b,
            stride_out, batch_count, _pad3                (40 bytes)
Total: 72 bytes
```

### Descriptor set layout (set=0)
| binding | type  | access |
|---------|-------|--------|
| 0       | SSBO  | read   |
| 1       | SSBO  | read   |
| 2       | SSBO  | write  |

Binding 0 = input A, binding 1 = input B, binding 2 = output.
For unary ops, only bindings 0 and 2 are bound.

### f16 shaders
Use `GL_EXT_shader_explicit_arithmetic_types` and `GL_EXT_scalar_block_layout`.
SSBOs use `layout(scalar)` for tight float16_t packing.

### bf16 cast shaders (`cast_f32_to_bf16.comp`, `cast_bf16_to_f32.comp`)
bf16 is stored as `uint16_t` in `layout(scalar)` SSBOs (low 16 bits = truncated
f32 top bits), bit-compatible with `gemm_bf16.comp`. Math:
- `f32 -> bf16`: `uint16_t(floatBitsToUint(f) >> 16)` (truncation, NOT RNE)
- `bf16 -> f32`: `uintBitsToFloat(uint(b) << 16)` (exact)

These shaders declare `StorageBuffer16BitAccess` (no `Int16` — they only
load/store `uint16_t`, no 16-bit arithmetic), so the device must enable
`storageBuffer16BitAccess` + `scalarBlockLayout`. Baseline-only tier; no
subgroup variant (elementwise, no benefit).

### bf16 elementwise shaders (`add_bf16.comp`, `mul_bf16.comp`,
`add_mul_bf16.comp`, `scale_bf16.comp`)
Same bf16 representation as the casts: `uint16_t` in `layout(scalar)` SSBOs,
`StorageBuffer16BitAccess`. Compute runs in f32, results truncate back to bf16:
```
float  a  = uintBitsToFloat(uint(data_a[idx]) << 16);
float  b  = uintBitsToFloat(uint(data_b[idx]) << 16);
float  r  = /* op in f32 (add / mul / (a+b)*alpha / alpha*a) */;
data_out[idx] = uint16_t(floatBitsToUint(r) >> 16);
```
Push-constant block identical to the f32/f16 shaders (`pc.num_elements`,
`pc.alpha` used by add_mul/scale). `add_mul_bf16.comp` and `scale_bf16.comp`
apply `pc.alpha` for the `(a+b)*alpha` / `alpha*a` forms. Baseline-only tier.
