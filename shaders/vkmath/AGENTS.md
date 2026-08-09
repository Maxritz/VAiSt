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
