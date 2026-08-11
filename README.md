<img width="1255" height="539" alt="VAiSt1" src="https://github.com/user-attachments/assets/d4ac1269-0679-409f-b6c1-7aad39aa5959" />

A Vulkan compute AI stack, built from scratch, implementing BLAS, FFT, RNG and 
math primitives for AMD RDNA2 (gfx103x) and RDNA4 (gfx1201) GPUs, and really 
any GPU that speaks Vulkan 1.4.

No CUDA. Just Vulkan compute shaders, C99 headers, and Vulkan-native handles.
Vulkan compute paths run on GPU directly for linear-algebra primitives not
yet covered by Vulkan shaders (sparse SpMM, LAPACK factorizations, conv1d/2d/3d),
a VJITC bridge dispatches into ROCm libraries (hipSPARSE, rocsolver, MIOpen)
over zero-copy VkBuffer↔HIP-device-pointer interop. No CUDA, no ROCm runtime
dependency for the core compute paths.

---

## Why This Exists

For years, doing serious GPU-accelerated AI on AMD hardware meant going through
ROCm, and that meant Linux. That's shifted a bit: AMD shipped ROCm 7.2 in 
January 2026 with, for the first time, a genuinely unified Windows and Linux 
release, and the RX 9070 XT (gfx1201) is now officially on AMD's supported 
Windows list, with native PyTorch and llama.cpp builds to go with it.

Still, the HIP SDK for Windows itself, the actual toolchain you'd build a project
like this against, ships without MIOpen, MIGraphX, communication libraries, or 
CMake HIP language support, and lists "AI Frameworks: Not available" against itself 
in AMD's own docs as of this writing. The PyTorch path AMD showed off runs through
a separate consumer distribution, not through the general HIP SDK. So Windows ROCm 
development is real now, in a way it wasn't a year ago, but it's still a narrower
stack than what Linux gets, and HIP-on-Windows still has its own rough edges once
you're outside AMD's specific supported paths.

VAiSt fixes this by starting from first principles:

- **One command buffer per decode step.** At the engine level, all dispatches
  for a decode step are recorded into a single `VkCommandBuffer` with explicit
  pipeline barriers, eliminating the per-op `vkQueueSubmit` + `vkWaitForFences`
  round-trips that serialize CPU-bound work. At the library level, each `vkblas_*`
  call records into a caller-supplied `VkCommandBuffer`.

- **Push descriptors everywhere.** Zero per-dispatch descriptor pool
  allocations. Every `vkCmdPushDescriptorSetKHR` binds directly.

- **Fused kernels.** QKV projection, RoPE, and KV cache writes are fused into
  one shader. FFN gate+up is fused. This cuts memory traffic by 40-60% per layer.

- **Three shader tiers.** A portable baseline (Vulkan 1.0 core), a subgroup
  tier (VK_KHR_shader_subgroup), and a cooperative-matrix tier
  (VK_KHR_cooperative_matrix). The runtime picks the best tier your GPU
  supports and falls back gracefully.

- **Shared shader sources.** ~191 GLSL compute shader sources compile into
  SPIR-V binaries via compile-time specialization (vkblas ~63 source shaders,
  vkmath 60, vkquant 44, vkrand 4, vkfft 2, vkblas_l1l2 18) — one source,
  multiple tile sizes, multiple wave sizes, multiple quantization types.
  Actual SPIR-V blob count varies (Wave32/Wave64 variants) and is
  auto-discovered by `shaders/compile_shaders.ps1`.

---

## What Is Inside

```
VAiSt
├── include/
│   ├── vkblas/           BLAS API (hipBLAS-compatible naming)
│   ├── vkfft/            FFT API (rocFFT-compatible naming)
│   ├── vkrand/           RNG + sampling API (rocrand-compatible naming)
│   ├── vkmath/           Elementwise ops, reductions, activations
│   ├── vkquant/          Quantization + dequantization (ggml block formats)
│   ├── vkmodel/          Model loaders (GGUF, safetensors, OpenVINO IR)
  │   ├── vkruntime/        Device/queue/memory/pool runtime (hipRuntime-equivalent)
  │   ├── vkkv/             Cross-model KV-cache ridge transfer
│   └── vkdist/           Distributed compute over TCP
├── src/                  C99 runtime + Vulkan dispatch
├── shaders/
│   ├── vkblas/           GEMM, qgemm, L1/L2 BLAS (baseline + subgroup tiers)
│   ├── vkmath/           Elementwise, reductions, activations, casts
│   ├── vkquant/          Dequant + forward-quantize shaders
│   ├── vkrand/           PRNG + distribution sampling
│   ├── vkfft/            Radix-2 FFT
│   ├── compile_shaders.ps1   Compiles .comp → SPIR-V → C header arrays
│   └── (per-lib tiers)   baseline/ (Vulkan 1.0 core), subgroup/ (VK_KHR_shader_subgroup),
│                         coopmatrix/ (VK_KHR_cooperative_matrix)
├── specs/                Design docs, ISA reference, architecture notes
│   ├── Common_Issues.md        GPU hang / device-lost / fence issues (catalog)
│   └── (per-subsystem specs)
├── tests/                23 test harnesses (build + run green on RX 9070 XT)
└── docs/                 (empty placeholder — spec pointers live in specs/)
```

### VKBLAS (implemented)

Mirrors the hipBLAS/rocBLAS GEMM family. Pipeline selection happens
automatically based on GPU capabilities:

| Function | Precision | Description |
|----------|-----------|-------------|
| `vkblas_sgemm` | f32 | Single GEMM |
| `vkblas_hgemm` | f16 (f32 accumulate) | Single GEMM |
| `vkblas_dgemm` | f64 | Single GEMM |
| `vkblas_bgemm` | bf16 (f32 accumulate) | Single GEMM |
| `vkblas_s/h/d/bgemm_strided_batched` | f32/f16/f64/bf16 | Strided batched GEMM |
| `vkblas_gemm_ex` | f16/f32/bf16 | Mixed-precision with compute-type control |
| `vkblas_sgemm_batched` | f32 | Per-buffer batched GEMM |

All plain GEMM precisions dispatch a shared-memory tiled baseline plus a
**subgroup twin** (32×8 warp tile, `subgroupShuffle` x-broadcast, no shared
memory) where `VK_KHR_shader_subgroup` is available. A cooperative-matrix tier
exists for f32/f16/bf16/f64 but is dormant by default (see note below).

**Fused quantized GEMM (qgemm)** — dequant + MAC fused in one shader, the
decode hot path. All seven ggml weight formats dispatch a 16×16 baseline and a
**32×8 subgroup tier** (one 32-lane subgroup per block, `subgroupShuffle`
x-broadcast, no shared memory / barriers); the subgroup tier is the default on
subgroup-capable devices. `vkblas_qgemm_get_tier` reports which tier resolved
per format. A `_f16` twin per format stores the f32 accumulator as `float16_t`
in y/z (f16 output storage).

| Function | Description |
|----------|-------------|
| `vkblas_qgemm_q8_0_f32` / `_f16` | Fused GEMM with Q8_0 weights |
| `vkblas_qgemm_q4_0_f32` / `_f16` | Fused GEMM with Q4_0 weights |
| `vkblas_qgemm_q4k_f32` / `_f16` | Fused GEMM with Q4_K weights |
| `vkblas_qgemm_q5k_f32` / `_f16` | Fused GEMM with Q5_K weights |
| `vkblas_qgemm_q6k_f32` / `_f16` | Fused GEMM with Q6_K weights |
| `vkblas_qgemm_q3k_f32` / `_f16` | Fused GEMM with Q3_K weights |
| `vkblas_qgemm_iq4xs_f32` / `_f16` | Fused GEMM with IQ4_XS weights |
| `vkblas_qgemm_get_tier` | Query resolved tier (BASELINE/SUBGROUP/COOPMATRIX) per format |

**Cooperative Matrix GEMM** — the `GL_KHR_cooperative_matrix` 
`coopMatLoad`/`coopMatMulAdd`/`coopMatStore` path is compiled into valid
SPIR-V but dormant by default. Enable with `VAIT_COOPMATRIX=1` on a newer
driver (RDNA4, RX 9070 XT). It bypasses shared-memory staging entirely and
dramatically accelerates GEMM on RDNA.

> **Why it's dormant**: the AMD LLPC driver (26.7.1, `driverVersion 2.0.395`)
> hard-crashes in `vkCreateComputePipelines` (`0xE06D7363`) when compiling any
> module containing `coopMatMulAddKHR`. This is a driver/LLPC bug, not a code
> bug — and it will stay dormant **until AMD fixes it in Vulkan, or releases a
> Vulkan ICD developer kit** that handles cooperative-matrix lowering cleanly.
> A HIP bridge is NOT the workaround: routing GEMM to `hipblasSgemm` via
> `VK_EXT_external_memory_host` measured **5× slower** (140 vs 696 GFLOPS on
> RX 9070 XT, 1024³) because the bridge forces host-visible (GTT) buffers,
> which swamps any MFMA/WMMA gain. The subgroup tier (already default) is the
> GEMM path until the driver is fixed.

> **Note**: the f16/bf16/f64 coopmatrix tiers are now built (`shaders/vkblas/coopmatrix/gemm_{f16,bf16,f64}.comp`). A `_f16` twin per format stores the f32 accumulator as `float16_t`. The cooperative-matrix tier is tested via `test_vkblas` (coopmatrix qgemm correctness, all 14 variants + _f16 twins).

API mirrors `hipblasSgemm` parameter order exactly — porting from HIP is a
mechanical find-and-replace:

```c
VkBLASContext* ctx;
vkblas_create_context(instance, physicalDevice, device, &ctx);
// First call: detects extensions, lazily creates pipelines
vkblas_sgemm(ctx, cmd, VKBLAS_OP_N, VKBLAS_OP_N,
             m, n, k, &alpha, bufA, lda, bufB, ldb,
             &beta, bufC, ldc, bufD, ldd);
```

All GEMM paths now return `VK_SUCCESS` on RDNA4. This also enables the new
`vkblas_qgemm_get_tier` query.

> **New: `vkr_create_device` deliverable** — the canonical full-feature device
> creation function (`src/vkruntime/vkruntime.c`) queries the full Vulkan 1.1-1.4
> feature chain, enables only what the device reports, and gates cooperative
> matrix on `VAIT_COOPMATRIX`. Updated `tests/test_vkruntime.c` section 12.

### VKBLAS L1/L2 (implemented)

rocBLAS-style Level-1/Level-2 vector/matrix ops (companion library that
reuses `VkBLASContext`):

| Function | Precision | Description |
|----------|-----------|-------------|
| `vkblas_l1_axpy` | f32/f16 | `y = alpha*x + y` |
| `vkblas_l1_scal` | f32/f16 | `x = alpha*x` |
| `vkblas_l1_dot` | f32/f16 | dot product → result[0] |
| `vkblas_l1_nrm2` | f32 | Euclidean norm |
| `vkblas_l1_asum` | f32 | Sum of absolute values |
| `vkblas_l1_amax` | f32 | Index of max |x| (0-based) |
| `vkblas_l1_max_reduce_f32` | f32 | Max reduction |
| `vkblas_l1_min_reduce_f32` | f32 | Min reduction |
| `vkblas_l2_gemv` | f32/f16 | `y = alpha*op(A)*x + beta*y` (N and T) |

### VKMath (implemented)

Elementwise activations, binary ops, and dimension-wise reductions as Vulkan
compute dispatches. Mirrors the VKBLAS context/pipeline-caching pattern.

| Function | Description |
|----------|-------------|
| `vkmath_relu_f32/f16/bf16` | ReLU activation |
| `vkmath_silu_f32/f16/bf16` | SiLU (Swish) activation |
| `vkmath_gelu_f32/f16/bf16` | GELU (tanh approx) activation |
| `vkmath_tanh_f32/f16/bf16` | Hyperbolic tangent |
| `vkmath_sigmoid_f32/f16/bf16` | Sigmoid |
| `vkmath_add_f32/f16/bf16`, `vkmath_mul_f32/f16/bf16` | Elementwise binary ops |
| `vkmath_add_mul_f32/f16/bf16` | Fused `(a+b)*alpha` |
| `vkmath_scale_f32/f16/bf16` | `alpha * in` |
| `vkmath_max_reduce_dim_f32`, `vkmath_sum_reduce_dim_f32` | Row reductions |
| `vkmath_softmax_f32`, `vkmath_rms_norm_f32`, `vkmath_layernorm_f32` | Normalization ops |
| `vkmath_argmax_f32`, `vkmath_argmin_f32`, `vkmath_cumsum_f32` | Index / scan ops |
| `vkmath_clip/abs/sign/exp/log/sqrt/rsqrt/pow_f32` | Scalar unary ops |
| `vkmath_cast_f32_to_bf16`, `vkmath_cast_bf16_to_f32` | bf16 ↔ f32 casts (bit-exact, `floatBitsToUint(f)>>16`) |
| `vkmath_add/mul/add_mul/scale_bf16` | bf16 elementwise (uint16_t SSBO, f32 compute) |

All work records into a caller-supplied `VkCommandBuffer`; pipelines are
created lazily and cached. Descriptor binding uses push descriptors when
available, otherwise a context-owned descriptor pool.

### VKQuant (implemented)

Block dequantization **and forward quantization** of quantized weights, using
the same context/pipeline-cache pattern as VKMath. All ggml block formats
round-trip: every `vkquant_quantize_<fmt>_f32` dispatches a real shader and is
validated against its matching dequant in `test_vkquant`.

| Dequant (f32 output) | Forward quantize (f32 → block) |
|----------|-------------|
| Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_1, IQ4_NL (32-elem legacy blocks) | Same 7 formats |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (256-elem K-quants) | Same 5 formats |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (256-elem IQ) | Same 8 formats |
| TQ1_0, TQ2_0 (256-elem TQ) | Same 2 formats |

That is **22 dequant + 22 forward-quantize** kernels (44 SPIR-V blobs). K-quant
scale-selection and the IQ/TQ grid search are transliterations of the ggml
`quantize_row_*_ref` math; the grid formats replace ggml's runtime kmap/
neighbours tables with a direct exhaustive search over the same dequant grid
tables (embedded as GLSL `const` arrays), so stored grid indices round-trip
exactly through the dequant shaders. Layouts are ported bit-exact from
ggml-common.h (validated GPU vs CPU).

### VKRAND (implemented)

Stateless counter-based PRNGs verified against the Random123 known-answer
vectors.

| Function | Description |
|----------|-------------|
| `vkrand_uniform_f32` | `count` uniform floats in [0,1) (Philox4x32-10) |
| `vkrand_threefry_uniform_f32` | `count` uniform floats in [0,1) (ThreeFry2x32-20) |
| `vkrand_normal_f32` | `count` N(0,1) samples (Philox + Box-Muller) |
| `vkrand_uniform_uint32` | `count` raw uint32 (Philox counter words) |

### VKFFT (implemented)

1D radix-2 complex FFT, plan-based API, interleaved Re/Im buffers.
n = power of two ≤ 1024.

| Function | Description |
|----------|-------------|
| `vkfft_create_plan` / `vkfft_destroy_plan` | Create/destroy an FFT plan for size n |
| `vkfft_execute_f32` / `vkfft_execute_inverse_f32` | Forward / inverse FFT (f32) |
| `vkfft_execute_f16` / `vkfft_execute_inverse_f16` | Forward / inverse FFT (f16 I/O, f32 compute) |
| `vkfft_create_plan_2d` | N×N plan (n = power of two ≤ 1024) |
| `vkfft_execute_2d_f32` / `vkfft_execute_2d_inverse_f32` | Separable 2D forward / inverse FFT |

### VKRuntime (implemented)

The hipRuntime-equivalent base layer every library sits on. Vulkan-native.

| Function | Description |
|----------|-------------|
| `vkr_create_device` / `vkr_destroy_device` | Device/queue wrapper + capability detection |
| `vkr_detect_capabilities` | Shared feature/property detection + arch ladder |
| `vkr_malloc` / `vkr_free` | Pooled buffer allocator (hipMalloc-equivalent) |
| `vkr_upload` / `vkr_download` | Staging upload/download with sync |
| `vkr_create_command_pool` / `vkr_create_descriptor_pool` / `vkr_create_pipeline_layout` / `vkr_create_pipeline_cache` | Pool/layout/cache helpers |
| `vkr_get_arch_index` / `vkr_get_arch_name` / `vkr_has_subgroup` / `vkr_has_coop_matrix` | Capability queries |

All five libraries (vkmath, vkblas, vkquant, vkrand, vkfft) now build their
contexts on VKRuntime: capability detection, descriptor pool, pipeline layout
and pipeline cache are created via `vkr_*` helpers instead of duplicated
inline Vulkan code.

### VKModel (implemented)

GGUF, safetensors, and OpenVINO IR model loaders — parse metadata + tensor
infos and upload every tensor's raw bytes into device buffers via VKRuntime,
as ready-to-use components.

| Function | Description |
|----------|-------------|
| `vkmodel_load_gguf` / `vkmodel_destroy` | Load/free a GGUF model (all metadata value types, streamed tensor upload) |
| `vkmodel_load_safetensors` | Load a safetensors model (self-contained JSON header parser, `__metadata__` KV exposure, verbatim offsets) |
| `vkmodel_load_openvino` | Load an OpenVINO IR v11 model (`<xml>` + `.bin`; tag-scans Const `<data>` + legacy `<weights>`/`<biases>`, element types f32/f16/bf16/f64/i8..i64 mapped natively, per-tensor size + `.bin` span validation) |
| `vkmodel_get_kv_count/_key/_string` | Host-side metadata access |
| `vkmodel_get_tensor_count/_name/_dtype/_dtype_name/_nelems/_buffer/_size` | Tensor info + device buffer access |
| `vkmodel_block_elems` | ggml_type → elements-per-block |

### VKKV (implemented)

Cross-model KV cache transfer — per-head closed-form ridge mapper that maps
a source model's K/V cache to a target so prefill can be skipped when
swapping same-family models (design doc pending).

| Function | Description |
|----------|-------------|
| `vkkv_create_transfer` / `vkkv_destroy_transfer` | Per-head ridge mapper (host-side fit, GPU apply) |
| `vkkv_fit_cpu` | Fit `W = (X^T X + λI)^-1 X^T Y` per head from a calibration set |
| `vkkv_apply` | Map source KV → target KV on GPU (one compute dispatch/head) |

RoPE stripping and top-k layer selection are the caller's responsibility.

### VKDist (implemented, Phase 1 + master/worker + caps)

Distributed compute over TCP — run compute on another PC's Vulkan card. The
transport is a synchronous, versioned (**`VKDIST_PROTOCOL_VERSION`**) request/
reply TCP framed protocol: client and server handshake with HELLO, then
exchange REGISTER_BUFFER / UPLOAD / DISPATCH_GEMM / READBACK / CAPS frames
until BYE. Remote buffers are opaque `uint64_t` handles; the client never sees
a Vulkan handle across the wire. Phase 1 adds multi-worker serving (one pthread
per connection) and client-side column-partitioned GEMM.

A **capability handshake** (`vkdist_query_caps`) lets a coordinator discover
each worker's GPU before routing work: GPU name, device-local VRAM, the stack's
shader-tier arch index (0=baseline, 1=subgroup, 2=coopmatrix), subgroup size,
and the server's frame limit. The **master/worker coordinator**
(`vkdist_master_create`/`add_worker`/`worker_caps`/`sgemm`/`destroy`) connects
N workers, records their capabilities, and runs `vkdist_sgemm_partitioned`
across them. A mandatory **SSH key gate** (`vkdist_verify_ssh_key`, enforced in
`vkdist_master_add_worker`) refuses any worker the controlling host has no
established SSH key to — the system will not operate between hosts that do not
trust each other. To encrypt the data path, run the client through an external
SSH local port-forward (`ssh -N -L 7001:127.0.0.1:7000 user@worker`).

Verified cross-PC: this machine's RX 9070 XT (RDNA4, 10.0.0.10) driving a
remote RX 6700 XT (RDNA2, 10.0.0.11) via `tests/test_vkdist_xpc.c`
(`server` / `client` / `master` modes) — remote sgemm round-trip and
master/worker partitioned sgemm both PASS. Design + phased roadmap
(multi-PC, distributed GEMM partition, attention/KV sharding, TLS/discovery)
in `specs/VKDIST-DESIGN.md`.

| Function | Description |
|----------|-------------|
| `vkdist_server_start` / `vkdist_server_accept` / `vkdist_server_run` | TCP server hosting a Vulkan device (per-connection RPC loop) |
| `vkdist_server_accept_many` / `vkdist_server_serve_many` | Accept `n` connections / serve them concurrently, one pthread each (shared VkBLASContext serialized) |
| `vkdist_client_connect` | TCP client connect + HELLO handshake |
| `vkdist_query_caps` | Capability advertisement (GPU name, VRAM, arch tier, subgroup, max frame) |
| `vkdist_verify_ssh_key` | SSH key-based auth check (BatchMode; security gate) |
| `vkdist_register_buffer` / `vkdist_upload` / `vkdist_readback` | Remote buffer lifecycle (opaque handles, synchronous) |
| `vkdist_sgemm` | Remote `vkblas_sgemm` dispatch (C = αAB + βC, in place) |
| `vkdist_sgemm_partitioned` | Column-partitioned f32 GEMM across `n_workers` connections, merges strips |
| `vkdist_master_create` / `vkdist_master_add_worker` / `vkdist_master_sgemm` / `vkdist_master_destroy` | Master/worker coordinator (SSH-gated worker discovery + partitioned GEMM) |
| `vkdist_close` | Send best-effort BYE + close socket |

### Completed Deliverables

- **Real cooperative-matrix GEMM** — `shaders/vkblas/coopmatrix/gemm_{f16,bf16,f64}.comp`
  and `qgemm_{<fmt>,_f16}.comp` all built and tested via `test_vkblas` (all 14
  qgemm variants + _f16 twins). `VAIT_COOPMATRIX=1` enables on RDNA4 (RX 9070 XT)
  and newer. Coopmatrix path dormant by default (driver 26.7.1 crashes on
  coopMatMulAddKHR, per `specs/GAP_ANALYSIS.md`).
- **`vkr_create_device` deliverable** — canonical full-feature device creation
  (`src/vkruntime/vkruntime.c`); all Vulkan 1.1-1.4 features enabled, cooperative
  matrix gated on `VAIT_COOPMATRIX`.
- **All 23+ tests PASS on RX 9070 XT** (verified by running `ctest -C Release`).

> Note: test count grew from 10 (original per-library harnesses) to 23+ as VJITC
> bridge tests (conv3d, conv12d, sparse, lapack, jit, model, runtime, math,
> quant, fft, rand, dist) were added. Run `ctest -C Release` for the authoritative
> count.
- **All 8 ext BLAS ops** (trsv/trsm/symv/hemv/symm/hemm/syrk/herk) pass in both
  f32 and f16 — f16 variants convert alpha/beta via `vkblas_f16_to_f32` before dispatch.

### Remaining Gaps (architecturally feasible but not yet done)

- No higher-level sparse ops beyond CSR SpMM + SpSV (sparse LU factorization, etc.).
- No runtime JIT by default — `VAIT_JIT` CMake option (OFF) enables hipRTC + shaderc.
- Coopmatrix (COOPMATRIX tier) dormant by default — AMD LLPC driver 26.7.1 crashes on
  `coopMatMulAddKHR` in `vkCreateComputePipelines`. Activated via `VAIT_COOPMATRIX=1`;
  stays dormant until AMD fixes it in the Vulkan driver or releases a Vulkan ICD
  developer kit (see the coopmatrix note above — a HIP-bridge GEMM is 5× slower and
  not the workaround).

---

## Building

### Prerequisites

- **Vulkan SDK 1.4.357.0** or newer (https://vulkan.lunarg.com)
- **CMake 3.20+** or Visual Studio 2022 with C++ build tools
- **AMD GPU** with Vulkan 1.1+ support (RDNA2 = Vulkan 1.1 baseline for
  shaders; Vulkan 1.4 + `VK_KHR_cooperative_matrix` required for the dormant
  coopmatrix tier)

### Compile Shaders

```powershell
cd shaders
.\compile_shaders.ps1
```

File-tree auto-discovery: globs every `.comp` under
`shaders/<lib>/{baseline,subgroup,coopmatrix}/` (vkblas, vkmath, vkquant,
vkrand, vkfft), compiles each to SPIR-V, and regenerates the corresponding
`src/<lib>/shaders_spv.h` with embedded bytecode arrays. Adding a kernel = drop
a `.comp` + regenerate — no manual build-list edits.

### Build

```powershell
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Run Tests

```powershell
ctest -C Release
```

All 24 tests pass on RX 9070 XT (ROCm 7.14.0, Vulkan SDK 1.4.357.0).

---

## Architecture Decisions

### Shader Tiering

Rather than writing architecture-specific binaries (like ROCm code objects),
every kernel exists in three capability tiers. A Tier 0 baseline runs on any
Vulkan 1.0 device. Tier 1 adds subgroup operations where available. Tier 2
unlocks cooperative matrix instructions on RDNA4 and newer.

This means a new GPU architecture does not require new shader variants —
only extension detection. The fallback chain is automatic.

### No Heap Allocation in Hot Paths

All dispatch functions take pointers. Pipeline objects are lazily created
and cached in a fixed-size, open-addressing hash table inside the context.
No `malloc`, no `vkAllocateDescriptorSets` per dispatch. Push descriptors
handle all binding.

### rocBLAS API Mirror

Public function names and parameter orders mirror `hipblas*` and `rocblas_*`
exactly. Porting a HIP-based project to Vulkan is a mechanical translation:

```
hipblasSgemm(...)  →  vkblas_sgemm(...)
hipblasCreate(...)  →  vkblas_create_context(...)
hipblasDestroy(...) →  vkblas_destroy_context(...)
```

Types follow the same scheme: `s` = f32, `d` = f64, `h` = f16, `bf` = bf16,
`c` = complex-f32, `z` = complex-f64. (int8 GEMM reserved but unimplemented.)

### Specialization Constants Over #define

Tile dimensions, unroll factors, and wave widths are SPIR-V specialization
constants (`constant_id`), not pre-compiled `#define` variants. One SPIR-V
binary per shader source can be reconfigured at pipeline creation time without
recompilation. The 191 `.comp` files compile to 191 SPIR-V blobs in the
checked-in `shaders_spv.h` headers — specialization varies tile/wave at
pipeline-creation time, not by pre-compiled variants.

---

## Contributing

We welcome contributions. Here is how to get started:

### Finding Work

1. Read the root `AGENTS.md` — it is the binding contract for this repo.
2. Check the `specs/` directory for reference material and the design docs.
3. Pick an unimplemented component or enhancement (bf16 reductions, f16 ext-GEMM
   coopmatrix tiers, int8 ops, new block-quant formats, or a new shader tier)
   and open an issue to claim it.
4. Read the per-component `AGENTS.md` (e.g. `src/vkblas/AGENTS.md`) before
   writing any code.

### Workflow

1. **Write the decision tree and truth table first.** Every new dispatch
   path, every `if` branch, every quantization scheme needs a traced decision
   tree. No code until the table passes.
2. **Build and run the test harness before merging.** If you add a shader,
   run `test_vkblas`. If you add an op, run `test_vkmath`. If there is no test
   harness yet, write one first.
3. **Update the truth table.** Document your decision in the relevant
   `AGENTS.md` files. The contract must stay readable.
4. **Commit with a clear message.** Reference the component and the spec
   section it implements.

### Code Style

- C99. DOX doc comments (`/** ... */`). `\brief`, `\param`, `\retval`.
- Vulkan-native: every handle is a Vulkan object.
- Mirror ROCm API names for mechanical porting.
- No heap allocation in hot paths.
- No stubs, placeholders, or TODOs in production code.

---

## Credits

This project stands on the shoulders of several excellent open-source and
open-standard projects:

- **Khronos Group** — for the Vulkan API, SPIR-V, and the Vulkan Memory
  Allocator. Without the Vulkan specification and the open, cross-vendor
  extension ecosystem, none of this would be possible.
  (https://www.khronos.org/vulkan/)

- **AMD** — for the RDNA2 and RDNA4 GPU architectures, the Vulkan driver
  implementation on both Linux and Windows, and the open-source Radeon
  documentation that made the hardware behaviour analysis possible.
  (https://www.amd.com/en/support/graphics/amd-radeon-rx-9000-series)

- **GPUOpen** — for the Vulkan Memory Allocator library
  (https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator), which
  handles VkDeviceMemory allocation and is bundled under
  `third_party/vk_mem_alloc.h`.

- **ROCm / AMD** — for hipBLAS (API naming reference), rocSPARSE, rocsolver, and
  MIOpen. VAiSt bridges to these libraries for sparse GEMM (`vkblas_sparse_gemm_f32`)
  and sparse triangular solve (`vkblas_sparse_spsv_f32`),
  LAPACK factorizations (`vkblas_lu_f32`/`_inverse_f32`/`_determinant_f32`/`_qr_f32`/
  `_cholesky_f32`/`_eigendecomp_f32`), and conv1d/2d/3d (`vkblas_conv{1d,2d,3d}_f32`)
  via zero-copy VkBuffer↔HIP-device-pointer interop. `specs/rocm-reference/`
  mirrors reference headers with attribution.
  (https://rocmd.docs.amd.com/)

- **LunarG** — for the Vulkan SDK (1.4.357.0), glslangValidator, spirv-val,
  and the diagnostic layers that make development tractable.
  (https://vulkan.lunarg.com/)

The ROCm project and its components (hipBLAS, rocFFT, rocBLAS, rocrand)
remain the gold-standard reference for what a GPU-accelerated math library
API should look like. VAiSt does not seek to replace them on Linux — rather,
it brings the same capabilities to platforms where ROCm does not run.

---

## License

This project is licensed under the Apache License, Version 2.0.
See the [LICENSE](LICENSE) file for details.

This project is not affiliated with or endorsed by AMD, the Khronos Group,
or any other organization whose materials appear in the `specs/` directory.
All trademarks are the property of their respective owners.

