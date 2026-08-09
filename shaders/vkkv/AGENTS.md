# VKKV Shaders — Local Contract

Child of root `AGENTS.md` and `src/vkkv/AGENTS.md`. Governs `shaders/vkkv/`.

## Variant strategy

Only a single baseline shader exists. The apply kernel is a per-output-element
dot product over the source dimension — a memory-bound, trivially parallel
workload where subgroup/coopmatrix tiers add no value:

```
shaders/vkkv/
└── baseline/       Tier 0: portable Vulkan 1.4 core (only tier)
    └── apply.comp
```

## Kernel

| Shader   | Description |
|----------|-------------|
| apply    | `dst[r][j] = sum_k src[r][k] * W_h[k][j]`, `local_size 256`, bounds guarded |

## Push constant layout (std140, 16 bytes)

```
offset  0: uint num_elements   (n * tgt_dim, total output elements)
offset  4: uint tgt_dim
offset  8: uint src_dim
offset 12: uint w_offset       (element offset of Wh in the stacked W buffer)
Total: 16 bytes
```

## Descriptor set layout (set=0)

| binding | type | access  | content      |
|---------|------|---------|--------------|
| 0       | SSBO | readonly | src [n x src_dim] |
| 1       | SSBO | readonly | W  [n_heads][src_dim*tgt_dim] stacked |
| 2       | SSBO | writeonly| dst [n x tgt_dim] |

## Requirements

- Vulkan 1.4 core compute only. No `shaderInt64`, no subgroup, no coopmatrix.
- `#version 450` GLSL, `-e main`, embedded via `glslangValidator -V --target-env
  vulkan1.4` into `src/vkkv/shaders_spv.h`.
- The GLSL push-constant block and the C `vkkv_push_constants_t` struct must
  stay byte-identical (both are 16 bytes, four `uint32_t`).
