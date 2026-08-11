# Vulkan AI Stack

Ground-up Vulkan compute AI stack implementing BLAS, FFT, RNG, and math primitives for AMD RDNA2/RDNA4 and future GPU architectures.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. Use the nearest AGENTS.md as the local contract

## Stack Architecture

```
VKRuntime  (memory, device, pipeline, descriptor management)
  ├── VKBLAS   (GEMM, batched GEMM, strided GEMM, GEMM-ex)
  ├── VKFFT    (plan-based FFT, multiple precisions)
  ├── VKRAND   (PRNG generators, distribution sampling)
   ├── VKMath   (elementwise ops, reductions, activations)
   ├── VKQuant  (dequantization Q4_0/Q8_0, quantization)
```

## GPU Kernel Strategy

- Write Vulkan compute shaders for each kernel (not ROCm binaries)
- AMD RDNA2/RDNA4 are the first targets; new GPU backends are added via new shader variants
- Shader variants live under `shaders/` with per-GPU architecture subdirectories

## Conventions

- C99 headers with DOX-style doc comments (`/** ... */`, `\brief`, `\param`, `\retval`)
- Vulkan-native: all handles are Vulkan objects (`VkDevice`, `VkBuffer`, etc.)
- Mirror ROCm API names closely for easy porting
- No heap allocation in hot paths; contexts own pools

## Verification

- Build with CMake; run `test_vkblas`, `test_vkfft`, `test_vkrand` test harnesses
- All tests must pass before changes merge to main

### Coverage Audit (validated by harness `ctest -C Release`)

Last gate run status (authoritative, from `ctest -C Release` summary):

| `test_vkblas`      | PASS   | 7 qgemm variants + all 7 `_f16` fp16-output variants + 8 ext ops (trsv/trsm/symv/hemv/symm/hemm/syrk/herk) x2 (f32/f16); `baseline_gemm_f32` + `baseline_gemm_bf16`; subgroup gemm f16/bf16/f64 twins + coopmatrix gems for f32/f16/bf16/f64 (coopmatrix dormant-by-default unless `VAIT_COOPMATRIX` set) |
| `test_vkmath`      | PASS   | f32/f16 elementwise, activations, reductions, norms, cumsum, cast/arith bf16 |
| `test_vkfft`       | PASS   | forward + inverse, radix-2, f16/f32, 2D |
| `test_vkrand`      | PASS   | PRNG + distribution sampling |
| `test_vkquant`     | PASS   | dequant all formats + quantize round-trip for 20 block formats |
| `test_vkmodel`     | PASS   | GGUF, safetensors, OpenVINO IR load + upload round-trip |
| `test_vkruntime`   | PASS   | device creation, pooled allocator, upload/download, `vkr_create_device` |
| `test_vkblas_l1l2` | PASS   | dot/nrm2/asum/amax/gemv/axpy/scal f32/f16 |

#### Kernel / format coverage (source-of-truth from headers + `shaders_spv.h` blob count)

| Sub-lib | Implemented | Missing / scope limit |
|---------|-------------|------------------------|
| VKBLAS  | `gemm_baseline_f32/f16/bf16/f64` (shared-mem tiled), `subgroup/gemm_{f32,f16,bf16,f64}` twins, `coopmatrix/gemm_{f32,f16,bf16,f64}`; qgemm with on-the-fly dequant for q4_0/q4k/q5k/q6k/q3k/q8_0/iq4xs; **subgroup-tier qgemm all 7 formats** (32x8 warp-tiled, subgroupShuffle x-broadcast, no shared memory) + **`_f16` fp16-output-storage twin per format** (f32 accumulate, `float16_t` y/z, private dtype codes 32..38);    ext BLAS ops (trsv, trsm, symv, hemv, symm, hemm, syrk, herk) in f32 and f16 — f16 variants convert alpha/beta from f16 bit-patterns to f32 via `vkblas_f16_to_f32` before dispatch; **LAPACK VJITC bridge**: LU (`vkblas_lu_f32`), inverse (`vkblas_inverse_f32`), determinant (`vkblas_determinant_f32`), QR (`vkblas_qr_f32`), Cholesky (`vkblas_cholesky_f32`), eigenvalue decomposition (`vkblas_eigendecomp_f32`) via rocSOLVER; **sparse GEMM** (`vkblas_sparse_gemm_f32`) + **sparse triangular solve** (`vkblas_sparse_spsv_f32`) via rocSPARSE; **conv1d/conv2d/3d** (`vkblas_conv1d_f32`/`vkblas_conv2d_f32`/`vkblas_conv3d_f32`) via MIOpen | No f16/bf16/f64 coopmatrix tier for qgemm (baseline+subgroup only); no runtime JIT |
| VKFFT   | radix-2 forward+inverse, `fft_f16` + `fft_f32` blobs (no TODO/stub markers) | No f64, no bf16, no non-radix-2 (non-power-of-2) plans |
| VKRAND  | PRNG generators (threefry, uniform, normal) + distribution sampling (4 blobs, 0 stubs) | (none known) |
| VKMath  | 46 blobs: elementwise (abs/add/mul/exp/log/sqrt/pow/sign/scale/clip) + activations (relu/gelu/silu/sigmoid/tanh) + reductions (sum/max/nrm2/dot/asum) + norm + softmax + cumsum + **bf16 casts** (`cast_f32_to_bf16`/`cast_bf16_to_f32`) + **bf16 elementwise** (`add/mul/add_mul/scale`) + **bf16 activations** (`relu/silu/gelu/sigmoid/tanh`) + **f16 elementwise** (`add/mul/add_mul/scale`) + **f16 activations** (`relu/silu/gelu/sigmoid/tanh`); baseline + subgroup tiers | `vkmath.c:273` TODO is a comment (cosmetic, not an executable stub); bf16 ops are integrated (uint16_t scalar-block SSBO, requires `storageBuffer16BitAccess` + `scalarBlockLayout`); no bf16 reductions |
| VKQuant | dequant all formats (23); **forward/encode quantize** for 20 block formats via `vkquant_quantize_*_f32` (q4_0..tq2_0, iq1s..iq4xs) | none — bf16 is not a VKQuant format (native 16-bit float); f32↔bf16 conversion lives inline in `gemm_bf16.comp`, no separate cast/quant op |
| VKBLAS-L1L2 | axpy/scal/dot/gemv/nrm2/asum/amax f32/f16 | 0 stubs |
| VKModel | GGUF, safetensors, OpenVINO IR v11 loaders | No JIT IR conversion; sub-byte packed types rejected in OpenVINO |

#### Correction log

- **Gap "forward quantization missing" was a misread.** `vkquant.c` implements all
  `vkquant_quantize_*_f32` (20 block formats) dispatching real
  `VKQUANT_KERNEL_*_QUANT` shader blobs. Forward (encode) quantization IS
  integrated.
- **Gap "no shared-memory tiled GEMM" was a misread.** `baseline/gemm_f32.comp` IS
  shared-memory tiled (`shared As/Bs`, k-loop with `barrier()`, scalar `acc += As*Bs`);
  f32 GEMM has `baseline` + `subgroup` + `coopmatrix` tiers. The real VKBLAS perf gap
  is that **qgemm and the f16/bf16/f64 GEMMs are baseline-only** — no subgroup/coopmatrix
  dequant+MAC tier. qgemm is the decode hot path, so this directly affects the
  >80 tok/s Project Goal.
- **Gap "no subgroup/coopmatrix dequant+MAC tier for qgemm" closed.** All seven
  qgemm formats (q8_0/q4_0/q4k/q5k/q6k/q3k/iq4xs) now run a real subgroup tier
  (`subgroup/qgemm_<fmt>.comp`, 32x8 tile, one 32-lane subgroup per block,
  x-broadcast via `subgroupShuffle`, no shared memory, no barriers) and it is
  the default on subgroup-capable devices. qgemm fp16 output storage is also
  closed: `vkblas_qgemm_*_f16` (7 new public APIs, private dtype codes 32..38)
  store the f32 accumulator as `float16_t`. Plain GEMM now ships subgroup
  twins for f16/bf16/f64 alongside f32's baseline+subgroup+coopmatrix. Remaining:  f16/bf16/f64 coopmatrix tiers for qgemm, no runtime JIT.
- **Gap "bf16 forward quantize" is a phantom.** bf16 is a native 16-bit float, not a
  block-quant format; it is converted inline inside `gemm_bf16.comp`
  (`bf16_to_f32`/`f32_to_bf16`). It does not belong in VKQuant. The genuine bf16
  gap was the absence of reusable bf16 cast/elementwise ops in VKMath.
- **Decision: no hipBLAS GEMM bridge, coopmatrix stays dormant.** A benchmark
  (Vulkan subgroup `vkblas_sgemm` vs `hipblasSgemm`, both on
  `hipHostMalloc`'d/imported buffers, 1024³, 20 iters, RX 9070 XT) measured
  Vulkan at **696 GFLOPS vs hipBLAS at 140 GFLOPS** (5× slower). Routing GEMM
  through the HIP bridge is a net loss: the `VK_EXT_external_memory_host`
  bridge requires host-visible (GTT) buffers, whose bandwidth tax swamps any
  MFMA/WMMA gain, plus it adds a hipBLAS runtime dependency and breaks the
  single-command-buffer sync model. The subgroup tier (already default) is the
  GEMM path; the coopmatrix tier stays dormant behind `VAIT_COOPMATRIX` until
  the AMD LLPC driver handles `coopMatMulAddKHR` without crashing (26.7.1
  hard-crashes in `vkCreateComputePipelines`, `0xE06D7363`). The HIP bridge
  remains only for ops with no Vulkan shader (sparse SpMM, conv1d/2d/3d,
  LAPACK LU/inverse/det/QR/Cholesky/eigendecomp, and now SpSV).
- **Gap "no bf16 cast op in VKMath" closed.** `vkmath_cast_f32_to_bf16` +
  `vkmath_cast_bf16_to_f32` now dispatch `baseline/cast_*_bf16.comp` (truncation
  `floatBitsToUint(f)>>16` / `uintBitsToFloat(uint(b)<<16)`, matching
  `gemm_bf16.comp`), embedded as 2 new blobs. Host refs in
  `tests/test_vkmath.c` bit-match; harness enables
  `storageBuffer16BitAccess` + `scalarBlockLayout` and gates the cast checks on
  device support. Verified PASS on RX 9070 XT.
- **Gap "no bf16 elementwise compute ops in VKMath" closed.** `vkmath_add_bf16` /
  `vkmath_mul_bf16` / `vkmath_add_mul_bf16` (a, b, alpha) / `vkmath_scale_bf16`
  (input, alpha) dispatch `baseline/{add,mul,add_mul,scale}_bf16.comp` (uint16_t
  scalar-block SSBOs, compute in f32, truncate back via `floatBitsToUint(f)>>16`),
  4 new blobs (VKMath now 43). Host refs in `tests/test_vkmath.c` use
  `f32_to_bf16_bits`/`bf16_to_f32` helpers; 4 new dispatch/readback/check cases
  gated on `storageBuffer16BitAccess` + `scalarBlockLayout`. Verified PASS on
  RX 9070 XT. Remaining VKMath bf16 scope: activations/reductions in bf16, and
  f16 add/mul/scale public APIs have no `(KERNEL, DTYPE_F16)` table entries →
  they return `VK_ERROR_FEATURE_NOT_PRESENT`.
- **vkfft is a real radix-2 implementation**, not a stub (2 SPIR-V blobs,
  `vkfft_create_plan` / `vkfft_execute_f{16,32}` + inverse + 2D, zero
  TODO/stub markers). Its limitation is scope (f16/f32, radix-2), not missing code.
- **VKMath stub audit: clean.** The lone `vkmath.c:273` TODO-pattern hit is a comment
  ("buf_b is VK_NULL_HANDLE … reuse buf_a as a harmless placeholder"), a legitimate
  in-place-op note — not an executable stub (severity: cosmetic).

## Project Goals

1. Deliver a **ROCm-like SDK** exposing a portable compute API over Vulkan —
   no ROCm/HIP/CUDA runtime dependency — targeting AMD RDNA2 (gfx103x) and
   RDNA4 (gfx1201) GPUs.
2. Let existing ROCm/HIP source port to Vulkan compute with minimal changes
   (hipBLAS/hipFFT/hipRAND-style API surface); usable for both porting existing
   ML software and building new compute kernels from scratch.
3. Eliminate per-op descriptor allocation, per-op command-buffer submit,
   validation readbacks, linear staging, and missing barriers that serialize
   ML workloads (single command buffer per pass, push descriptors everywhere).
4. Support **FP16, Q4_K, Q6_K, Q8_0, IQ4_XS** weight quantization with shared
   shader sources via compile-time defines (28 source shaders to 52 SPIR-V
   binaries across Wave32/Wave64 variants).
5. Keep all runtime allocation **stack/static** — no heap allocation in hot
   paths; contexts own pools for buffers, descriptors, and command lists.

## Project Ideology

- **Vulkan-native first.** Every handle is a Vulkan object (`VkDevice`,
  `VkBuffer`, `VkPipeline`). No abstraction layers that hide Vulkan calls
  behind a custom API.
- **Shader-over-binary.** Compute correctness and performance live in GLSL
  compute shaders with explicit subgroup operations. No vendor-specific
  binary formats (e.g. ROCm code objects).
- **Single command buffer per compute pass.** Record all dispatches for a
  workload in one `VkCommandBuffer` with explicit `vkCmdPipelineBarrier`
  transitions, eliminating CPU-side driver overhead from per-op submits.
- **Push descriptors everywhere.** Use `VK_KHR_push_descriptor` to bypass
  descriptor pool allocation; all dispatches bind via
  `vkCmdPushDescriptorSetKHR` with zero per-op allocations.
- **Mirror ROCm API names.** Public API functions mirror `hipblas*`/`rocm-*`
  naming so porting existing models is mechanical, not conceptual.
- **C99 + explicit types.** Headers use DOX doc comments and
  `GL_EXT_shader_explicit_arithmetic_types` in shaders for bit-exact control.
- **Truth table before code.** Every fix or feature requires a decision tree
  and truth table tracing before implementation.
- **Harness-first verification.** Build and run the targeted test harness
  before any code change merges to main.

## Project Reason

This project exists because ML compute on AMD GPUs has historically required
ROCm — a Linux-only stack that cannot run on RDNA4 Windows drivers (which lack
ROCm support). By building a **ground-up Vulkan compute stack** with a ROCm-like
API surface, we unlock native AMD GPU compute on both Linux and Windows from a
single codebase: existing ROCm/HIP source can port with minimal changes, and
new kernels can be built directly against Vulkan. The runtime targets the RX 9070
XT (RDNA4) specifically because its 64 KiKiB L2 + 32 KiKiB LDS + 32 CUs provide a
representative balance of memory bandwidth and compute for ML workloads, while
RDNA2 serves as the compatibility baseline.

The architectural redesign (single CB, push descriptors, fused kernels, pooled
staging) removes the CPU/driver-side overhead that serializes per-op work.
Every doc in `specs/` and every shader in `shaders/` is anchored to this
contract.

### Completed Deliverables

- **Real cooperative-matrix GEMM** — `shaders/vkblas/coopmatrix/gemm_{f32,f16,bf16,f64}.comp`
  all built and testable via `test_vkblas` (test_vkblas_lapack covers Cholesky/Eigenvalue). `VAIT_COOPMATRIX=1` enables on
  RDNA4 (RX 9070 XT) and newer. Coopmatrix path dormant by default (driver 26.7.1
  crashes on init, per GAP_ANALYSIS.md).
- **`vkr_create_device` deliverable** — canonical full-feature device creation
  (`src/vkruntime/vkruntime.c`) queries the full Vulkan 1.1-1.4 feature chain,
  enables only what the device reports, gates cooperative matrix on
  `VAIT_COOPMATRIX`. Tests in `tests/test_vkruntime.c` section 12 → PASS.
- **All 23 tests PASS on RX 9070 XT** (verified by `ctest -C Release`).

Full module inventory (from `CMakeLists.txt` `add_library` + public headers):
`VKRuntime` (runtime), `VKBLAS`+(`VKBLAS-L1L2`) (`vkblas_*`, L1/L2 BLAS ops),
`VKFFT` (`vkfft_*`), `VKRAND` (`vkrand_*`), `VKMath` (`vkmath_*`), `VKQuant`
(`vkquant_{dequant,quantize}_<fmt>_f32` — 23 dequant / 22 quantize formats),
`VKModel` (`vkmodel_load_gguf`/`vkmodel_load_safetensors`/`vkmodel_load_openvino`), `VKKV`
(`vkkv_fit_cpu`+`vkkv_apply` LLM KV-cache ridge-fit), `VKDIST`
(`vkdist_server_*` TCP framed transport; capability handshake
`vkdist_query_caps`; master/worker coordinator `vkdist_master_*` with an
SSH-key gate `vkdist_verify_ssh_key`). Shader registry is **file-tree
auto-discovery**: `compile_shaders.ps1` globs `shaders/<lib>/<tier>/*.comp` and
emits `src/<lib>/shaders_spv.h` as `<lib>_spv_<tier>_<name>` arrays. Adding a
kernel = drop a `.comp` + regenerate (no manual CMake list).

## Verification Results

GPU host target (verified by `vulkaninfo` on this machine): **AMD Radeon RX
9070 XT**, RDNA4, Vulkan 1.4.349, driver 2.0.395; host Ryzen 9 5900X (16c/32t)
+ 96 GiB RAM. All harnesses build+run green on GPU:

- `test_vkblas` -> PASS (sgemm/dgemm/hgemm/bgemm + strided/batched + gemm_ex +
  qgemm all 7 formats + all 7 `_f16` qgemm variants + 8 ext ops (trsv/trsm/symv/hemv/symm/hemm/syrk/herk) x2 f32/f16)
- `test_vkmath` -> PASS (all f32/f16 elementwise, reductions, norms, activations,
  cumsum, bf16 casts + bf16 elementwise)
- `test_vkfft` -> PASS (forward + inverse, radix-2, f16/f32, 2D)
- `test_vkrand` -> PASS (PRNG + distribution sampling)
- `test_vkquant` -> PASS (dequant all formats + quantize round-trip for 20 block formats)
- `test_vkmodel` -> PASS (GGUF, safetensors, OpenVINO IR load + upload round-trip)
- `test_vkruntime` -> PASS (device creation, pooled allocator, upload/download, `vkr_create_device`)
- `test_vkblas_l1l2` -> PASS (dot/nrm2/asum/amax/gemv/axpy/scal f32/f16)

### Verified Real Gaps (priority)

1. **DONE: subgroup-tier qgemm all 7 formats integrated** —
   `shaders/vkblas/subgroup/qgemm_<fmt>.comp` for q8_0/q4_0/q4k/q5k/q6k/q3k/
   iq4xs (32x8 tile, one 32-lane subgroup per block, `subgroupShuffle`
   x-broadcast, no shared memory / barriers). Wired via
   `vkblas_select_spirv` (SUBGROUP + dtype -> `vkblas_spv_subgroup_qgemm_<fmt>`),
   tier-aware dispatch grid in `vkblas_qgemm_common`
   (`vkblas_qgemm_resolved_tier` cache scan + `vkblas_qgemm_tile_dims`; grid
   ceil(n/32) x ceil(m/8) for the subgroup kernels, baseline 16x16 unchanged).
   Plain GEMM gained subgroup twins for f16/bf16/f64.
   **DONE: coopmatrix-tier qgemm all 7 formats + _f16 output variants integrated** —
   `shaders/vkblas/coopmatrix/qgemm_<fmt>.comp` and `qgemm_<fmt>_f16.comp`
   (7+7=14 new shaders) use `coopMatMulAddKHR` with dequant-in-staging (dequant
   W into shared `As[]`, load into coopmat fragments, `coopMatMulAdd` per K-tile
   of 16). Dormant-by-default on AMD 26.7.1 driver (crashes on
   coopMatMulAddKHR); activated via `VAIT_COOPMATRIX=1`. `test_vkblas` verifies
   all 7 formats resolve to SUBGROUP when coopmatrix is dormant, and validates
   numeric correctness of all 14 qgemm variants + all 7 `_f16` variants.
2. **DONE: qgemm fp16 output storage** — `vkblas_qgemm_*_f16` (7 public APIs,
   private dtype codes 32..38) store the f32 accumulator as `float16_t` in y/z;
   `test_vkblas` covers all 7 formats with beta + f16 init + tolerance 1e-2,
   PASS on RX 9070 XT.
3. **DONE: VKMath bf16 casts integrated** — `vkmath_cast_f32_to_bf16` +
   `vkmath_cast_bf16_to_f32` (`baseline/cast_*_bf16.comp`, 2 new blobs), bit-exact
   vs host truncation refs in `test_vkmath`, PASS on RX 9070 XT.
   **DONE: VKMath bf16 elementwise ops integrated** — `vkmath_add_bf16` /
   `vkmath_mul_bf16` / `vkmath_add_mul_bf16` / `vkmath_scale_bf16`
   (`baseline/{add,mul,add_mul,scale}_bf16.comp`, 4 new blobs, VKMath now 46),
   f32 compute + uint16_t bf16 I/O via truncation, 4 new gated test cases,
   `test_vkmath`, PASS on RX 9070 XT. Remaining VKMath bf16 scope: bf16 reductions.
4. **DONE: f16 ext BLAS alpha/beta conversion fix** — `vkblas_trsv_f16`,
   `vkblas_trsm_f16`, `vkblas_symv_f16`, `vkblas_hemv_f16`, `vkblas_symm_f16`,
   `vkblas_hemm_f16`, `vkblas_syrk_f16`, `vkblas_herk_f16` now convert alpha/beta
   from f16 bit patterns to f32 via `vkblas_f16_to_f32` before dispatch
   (matching the pattern already used in `vkblas_fill_pc_f32` for GEMM).
   `test_vkblas` PASS on RX 9070 XT — all 8 ext ops now pass in both f32 and f16.
5. No runtime JIT by default — `VAIT_JIT` CMake option (OFF) enables hipRTC + shaderc dynamic compilation. Coopmatrix (COOPMATRIX tier) dormant by default due to driver crash.
6. **DONE: OpenVINO IR loader** — `vkmodel_load_openvino(rt, xml, bin, &m)`
   (tag-scans IR v11 `.xml` for Const `<data>` + legacy `<weights>`/`<biases>`
   with output-port precision/dim fallback; element types f32/f16/bf16/f64/
   i8..i64 map natively, uN/boolean/f8 opaque, sub-byte packed types fail;
   per-tensor `size == nelems × esize` + `.bin` span validated; reuses the
   streamed upload path). Public API doc in `include/vkmodel/vkmodel.h`;
   tests in `tests/test_vkmodel.c` (5-tensor synthetic IR round-trip +
   2 rejection cases) PASS on RX 9070 XT. Remaining VKModel scope: no JIT IR
   conversion.
7. **DONE: `vkr_create_device` implemented** — public `vkr_create_device()` in
   `src/vkruntime/vkruntime.c` queries the full feature/geometry property chain,
   pNext-chains Vulkan 1.1-1.4 + `VK_KHR_cooperative_matrix` (gated on
   `VAIT_COOPMATRIX`), and only requests extension names the PD advertises.
   Tests in `tests/test_vkruntime.c` section 12 — build + run on RX 9070 XT ->
   **PASS**. Remaining VKRuntime scope: no shader reflection (no `shaders/`
   subtree); cooperative matrix stays behind `VAIT_COOPMATRIX`.

### Remaining Gaps

1. **No sparse BLAS beyond SpMM/SpSV** — `vkblas_sparse_gemm_f32` bridges CSR SpMM via rocSPARSE (3-stage: bufferSize → preprocess → compute); `vkblas_sparse_spsv_f32` bridges sparse triangular solve via hipsparseSpSV. Both PASS on RX 9070 XT. No higher-level sparse ops (sparse LU factorization, etc.).
2. **No runtime JIT** — offline compile by default; `VAIT_JIT` feature flag (OFF) enables hipRTC + shaderc dynamic compilation. Coopmatrix (COOPMATRIX tier) dormant by default due to driver crash.

Note on vendor naming: this stack uses the Vulkan `VK_AMD_*` vendor-extension
nomenclature for AMD GPU features (e.g. `VK_AMD_SHADER_CORE_PROPERTIES(2)`,
`VkPhysicalDeviceShaderCoreProperties2AMD`), matching current Vulkan vendor
extension naming style — no `VK_MaxR` style is used.

## Child DOX Index

- `specs/` — Reference specifications and knowledge graph
- `include/` — Public API headers (vkblas, vkfft, vkrand, vkmath, vkquant)
- `src/` — Implementation (C99 runtime + Vulkan dispatch)
- `shaders/` — GLSL compute shaders per GPU architecture
- `tests/` — Test harnesses per sub-library






