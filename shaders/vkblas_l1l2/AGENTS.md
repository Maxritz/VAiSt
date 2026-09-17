# VKBLAS-L1L2 Shaders — Local Contract

Child of root `AGENTS.md` and `src/vkblas_l1l2/AGENTS.md`. Governs
`shaders/vkblas_l1l2/` and its tier subdirs.

## Layout

```
shaders/vkblas_l1l2/
└── baseline/       Tier 0: portable Vulkan 1.4 core, no extensions
```

Every op has exactly one baseline `.comp` shader. No subgroup/coopmatrix
variants exist for L1/L2 ops — elementwise/reduction/gemv workloads do not
benefit from those tiers.

## Kernels

| File                 | Op                       | Bindings used             |
|----------------------|--------------------------|---------------------------|
| axpy_f32.comp        | y = a*x + y              | 0=x(ro), 1=y(rw)          |
| axpy_f16.comp        | y = a*x + y (half)       | 0=x(ro,uint), 1=y(rw,uint)|
| scal_f32.comp        | x = a*x                  | 0=x(rw)                   |
| scal_f16.comp        | x = a*x (half)           | 0=x(rw,uint)              |
| dot_partial_f32.comp | partial[g] = sum x*y     | 0=x(ro), 1=y(ro), 3=partials(wo) |
| dot_partial_f16.comp | same, half inputs        | 0,1 uint ro, 3 partials f32 |
| dot_finalize_f32.comp| result[0] = sum partials | 3=rw                     |
| nrm2_partial_f32.comp| partial[g] = sum x^2     | 0=x(ro), 3=partials(wo)   |
| nrm2_finalize_f32.comp| result[0] = sqrt(sum)   | 3=rw                     |
| asum_partial_f32.comp| partial[g] = sum \|x\|    | 0=x(ro), 3=partials(wo)   |
| asum_finalize_f32.comp| result[0] = sum partials | 3=rw                     |
| amax_partial_f32.comp| partial[2g]=idx, [2g+1]=val | 0=x(ro), 3=pairs(wo,uint) |
| amax_finalize_f32.comp| result[0] = global idx    | 3=rw,uint                |
| gemv_f32.comp        | y = a*op(A)x + b*y       | 0=A(ro), 1=x(ro), 3=y(rw) |
| gemv_f16.comp        | same, half               | 0,1 uint ro, 3 uint rw    |

Binding convention follows VKBLAS: set=0, bindings 0/1/2 read, binding 3 write
(read-write for in-place ops). Shaders may declare a subset of the layout's
bindings; the C side always writes all four with valid handles.

## Push constants

Every shader declares the block below — it byte-matches
`vkblas_push_constants_t` in `src/vkblas/vkblas_internal.h` (84 bytes,
std140, all 4-byte members so no packing surprises):

```
m, n, k (uint), alpha, beta (float), lda, ldb, ldc, ldd (uint),
transA, transB, beta_is_zero, _pad0 (int),
strideA, strideB, strideC, strideD, batchCount, _pad1.._pad3 (uint)
```

Field packing used by the ops:
- axpy: n, alpha, lda=incx, ldb=incy
- scal: n, alpha, lda=incx
- dot/nrm2/asum partial: n, lda=incx, ldb=incy (dot only)
- amax partial: n, lda=incx
- finalize (all reductions): batchCount = number of partials G
- gemv: m, n, lda, alpha, beta, transA, ldb=incx, ldd=incy

## f16 storage

Halfs are stored ONE-PER-UINT32 (low 16 bits) and converted in-shader with the
`f16_pack`/`f16_unpack` bit helpers. This deliberately avoids `float16_t` /
`GL_EXT_shader_explicit_arithmetic_types`, so no `shaderFloat16` device feature
is required. alpha/beta remain f32. Reductions accumulate in f32.

## Reductions

Two-stage correctness model (no float atomics, no cross-group shared memory):
1. partial kernel: one workgroup per 256-element segment, shared-memory tree
   reduce, workgroup g writes `partial[g]` (amax writes an (index,value) pair).
2. finalize kernel: single workgroup, grid-stride loop over G partials, shared
   reduce, writes `result[0]` (nrm2 applies sqrt; amax returns the 0-based
   index, ties to the lowest index via strictly-greater comparison).

## Verification

Every `.comp` is compiled with
`glslangValidator -V --target-env vulkan1.4 -e main` and embedded by
`embed_l1l2.ps1` into `src/vkblas_l1l2/shaders_spv.h`. The generated header is
checked for the SPIR-V magic `0x07230203`. The full set must pass
`test_vkblas_l1l2` on the real GPU.
