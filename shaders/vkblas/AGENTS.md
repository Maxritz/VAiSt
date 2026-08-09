# VKBLAS Shaders — Local Contract

Child of root `AGENTS.md`. Governs `shaders/vkblas/` and its capability-tier subdirs.

## Shader Variant Strategy — Capability Tiers

Shaders are organized by **Vulkan capability tier**, not by GPU model. Each tier
is a self-contained SPIR-V module that produces numerically equivalent results.
The runtime selects the highest tier the physical device supports; each lower
tier provides a **portable Vulkan approximation** of features required by the
tier above.

```
shaders/vkblas/
├── baseline/       Tier 0: portable Vulkan 1.0 core, no extensions
├── subgroup/       Tier 1: requires VK_KHR_shader_subgroup
└── coopmatrix/     Tier 2: requires VK_KHR_cooperative_matrix
```

### Fallback chain (every feature must have a Vulkan approximation)
| Tier 2 feature (coopmatrix) | Tier 1 approximation (subgroup)        | Tier 0 approximation (baseline)          |
|-----------------------------|----------------------------------------|------------------------------------------|
| `OpCooperativeMatrixMul`    | Subgroup-shuffle tiled accumulation    | Scalar register accumulation           |
| Subgroup shuffle (32-wide)  | Same opcode, portable subgroup size    | Manual shared-mem broadcast               |
| `OpGroupFMulReduce`         | `subgroupAdd` + manual loop            | Sequential accumulation loop             |

Any device that lacks Tier 2 support falls back to Tier 1's subgroup
implementation, which itself falls back to Tier 0's portable code. New GPU
architectures need only declare which extensions they expose — no new shader
variants are required unless a genuinely new feature class is introduced.

### Specialization constants
All tile dimensions and unroll factors are specialization constants (`\c`
declared with `constant_id`), not `#define` macros. This allows one SPIR-V
binary to serve multiple tile configurations without recompilation.

### Push constants layout
```glsl
layout(push_constant) uniform PC {
    uint  m, n, k;           // matrix dimensions
    float alpha, beta;       // scaling scalars
    uint  lda, ldb, ldc, ldd;
    int   transA, transB;
    int   beta_is_zero;
    int   _pad0;
    int64_t strideA;         // element-stride (column stride) per batch
    int64_t strideB;
    int64_t strideC;
    int64_t strideD;
    int64_t batchCount;
    int64_t _pad1;
} pc;
```

Layout follows std140 packing rules. `int64_t` requires `shaderInt64` feature
(VK_KHR_vulkan_memory_model / VkPhysicalDeviceVulkan12Features::shaderInt64).

### Descriptor set convention (fixed across all tiers)
- `set=0, binding=0` → A buffer (SSBO, read)
- `set=0, binding=1` → B buffer (SSBO, read)
- `set=0, binding=2` → C buffer (SSBO, read)
- `set=0, binding=3` → D buffer (SSBO, write)

## File naming
- `gemm_f32.comp` — single-precision GEMM
- `gemm_f16.comp` — half-precision GEMM
- `gemm_bf16.comp` — bfloat16 GEMM
- `gemm_i8.comp`  — int8 GEMM
- `qgemm_q8_0.comp` — fused quantized GEMM, Q8_0 weights (dequant-in-matmul)
- `qgemm_q4k.comp`  — fused quantized GEMM, Q4_K weights (dequant-in-matmul)

Each variant exists in all three tier directories.

## Fused quantized GEMM shaders (baseline only)

`qgemm_q8_0.comp` / `qgemm_q4k.comp` reuse the gemm_f32 16x16 tiled structure
(shared `As`/`Bs`, k-loop with zero-fill, m/n write-guard) but the A-tile load
**dequantizes the quantized weight blocks into f32 shared memory** instead of
loading f32. The dequant math is ported verbatim from the vkquant shaders:

- **Q8_0**: 36 B/block (f32 `d` + 32 int8). `As = d * qs[i]`.
- **Q4_K**: 144 B/block (ggml `block_q4_K`). Per-32-group
  `get_scale_min_k4` decode; `out = d*sc*nib - dmin*mn`.

Byte I/O uses `uint8_t`/`int8_t` scalar-layout SSBOs (requires
`GL_EXT_shader_explicit_arithmetic_types` + `GL_EXT_scalar_block_layout`,
plus the `storageBuffer8BitAccess`/`shaderInt8` device features — enabled by
the test harness). The descriptor layout is unchanged from the plain GEMMs:
binding 0 = Wq (bytes, read), 1 = x (read), 2 = y (read, beta term),
3 = y (write).

Push constants reuse the plain-GEMM block; the host maps `pc.m` = weight rows,
`pc.n` = activation cols, `pc.lda` = weight row byte stride (ldw),
`pc.ldb` = ldx, `pc.ldd` = ldy. The grid is ceil(n/16) x ceil(m/16).
