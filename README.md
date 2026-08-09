# VAiSt — Vulkan AI Stack

A ground-up Vulkan compute AI stack that implements BLAS, FFT, RNG, and math
primitives for AMD RDNA2 (gfx103x) and RDNA4 (gfx1201) GPUs — and any GPU
that speaks Vulkan 1.4.

**No ROCm. No HIP. No CUDA.** Just Vulkan compute shaders, C99 headers, and
Vulkan-native handles.

---

## Why This Exists

For years, doing serious GPU-accelerated AI on AMD hardware meant going
through ROCm. That is fine if you are on Linux with a professional GPU, but
it leaves RDNA2 and RDNA4 users on Windows completely out of the game —
Windows drivers do not ship with ROCm support, and HIP-on-Windows is a stub
at best.

I wanted to run LLM inference on my RX 9070 XT and found that the existing
HIP backends either did not compile on Windows or delivered less than 1 tok/s
due to a cascade of architectural problems: one command buffer per operation,
a descriptor-set allocation per kernel dispatch, validation readbacks stalling
the GPU mid-inference, and per-head attention launches that barely register
any occupancy. The whole stack was spending more time on CPU driver overhead
than on actual compute.

VAiSt fixes this by starting from first principles:

- **One command buffer per decode step.** All 225 dispatches for a 32-layer
  model are recorded into a single `VkCommandBuffer` with explicit pipeline
  barriers, eliminating the 640 `vkQueueSubmit` + `vkWaitForFences` round-trips
  that burned 64ms per token on the legacy path.

- **Push descriptors everywhere.** Zero per-dispatch descriptor pool
  allocations. Every `vkCmdPushDescriptorSetKHR` binds directly.

- **Fused kernels.** QKV projection, RoPE, and KV cache writes are fused into
  one shader. FFN gate+up is fused. This cuts memory traffic by 40-60% per layer.

- **Three shader tiers.** A portable baseline (Vulkan 1.0 core), a subgroup
  tier (VK_KHR_shader_subgroup), and a cooperative-matrix tier
  (VK_KHR_cooperative_matrix). The runtime picks the best tier your GPU
  supports and falls back gracefully.

- **Shared shader sources.** 28 GLSL compute shader sources compile into 52
  SPIR-V binaries via compile-time specialization — one source, multiple
  tile sizes, multiple wave sizes, multiple quantization types.

---

## What Is Inside

```
VAiSt
├── include/
│   └── vkblas/           BLAS API (hipBLAS-compatible naming)
│   ├── vkfft/            FFT API (rocFFT-compatible naming)
│   ├── vkrand/           RNG + sampling API (rocrand-compatible naming)
│   ├── vkmath/           Elementwise ops, reductions, activations
│   └── vkquant/          Quantized weight dequantization (Q4_K, Q8_0, etc.)
├── src/                  C99 runtime + Vulkan dispatch
├── shaders/
│   ├── baseline/         Tier 0: portable Vulkan 1.0 (works everywhere)
│   ├── subgroup/         Tier 1: subgroup shuffle optimizations
│   ├── coopmatrix/       Tier 2: VK_KHR_cooperative_matrix (RDNA4+)
│   └── compile_shaders.ps1   Compiles .comp → SPIR-V → C header arrays
├── specs/                Reference data, ISA docs, ROCm headers
├── tests/                Per-library test harnesses (coming)
└── docs/                 Architecture specifications (specs/)
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
| `vkblas_gemm_ex` | f16/f32 | Mixed-precision with compute-type control |
| `vkblas_sgemm_batched` | f32 | Per-buffer batched GEMM |

API mirrors `hipblasSgemm` parameter order exactly — porting from HIP is a
mechanical find-and-replace:

```c
VkBLASContext* ctx;
vkblas_create_context(physicalDevice, device, &ctx);
// First call: detects extensions, lazily creates pipelines
vkblas_sgemm(ctx, cmd, VKBLAS_OP_N, VKBLAS_OP_N,
             m, n, k, &alpha, bufA, lda, bufB, ldb,
             &beta, bufC, ldc, bufD, ldd);
```

> **Note (cooperative matrix):** a real `GL_KHR_cooperative_matrix`
> (`coopMatMulAddKHR`) GEMM path is implemented and compiles to valid SPIR-V,
> but it is **dormant by default** because the AMD 26.7.1 driver hard-crashes
> inside `vkCreateComputePipelines` on any coopmat pipeline (RDNA2). Set the
> env var `VAIT_COOPMATRIX=1` to enable it on a fixed/newer driver. The
> correct shared-memory GEMM is the default path.

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
| `vkblas_l2_gemv` | f32/f16 | `y = alpha*op(A)*x + beta*y` (N and T) |

### VKMath (implemented)

Elementwise activations, binary ops, and dimension-wise reductions as Vulkan
compute dispatches. Mirrors the VKBLAS context/pipeline-caching pattern.

| Function | Description |
|----------|-------------|
| `vkmath_relu_f32/f16` | ReLU activation |
| `vkmath_silu_f32/f16` | SiLU (Swish) activation |
| `vkmath_gelu_f32/f16` | GELU (tanh approx) activation |
| `vkmath_tanh_f32/f16` | Hyperbolic tangent |
| `vkmath_sigmoid_f32/f16` | Sigmoid |
| `vkmath_add_f32/f16`, `vkmath_mul_f32/f16` | Elementwise binary ops |
| `vkmath_add_mul_f32/f16` | Fused `(a+b)*alpha` |
| `vkmath_scale_f32/f16` | `alpha * in` |
| `vkmath_max_reduce_dim_f32`, `vkmath_sum_reduce_dim_f32` | Row reductions |

All work records into a caller-supplied `VkCommandBuffer`; pipelines are
created lazily and cached. Descriptor binding uses push descriptors when
available, otherwise a context-owned descriptor pool.

### VKQuant (implemented)

Block dequantization of quantized weights to f32, using the same
context/pipeline-cache pattern as VKMath.

| Function | Description |
|----------|-------------|
| `vkquant_dequant_q8_0_f32` | Q8_0 blocks (f32 scale + 32×int8) → 32 f32 each |
| `vkquant_dequant_q4_0_f32` | Q4_0 blocks (f32 scale + 16 packed nibbles) → 32 f32 each |
| `vkquant_dequant_q4k_f32` | Q4_K blocks (ggml, 256 elems) → 256 f32 each |
| `vkquant_dequant_q6k_f32` | Q6_K blocks (ggml, 256 elems) → 256 f32 each |
| `vkquant_dequant_iq4xs_f32` | IQ4_XS blocks (ggml + iq4nl LUT) → 256 f32 each |
| `vkquant_quantize_q8_0_f32` | Forward quantize f32 → Q8_0 blocks |
| `vkquant_quantize_q4_0_f32` | Forward quantize f32 → Q4_0 blocks |

Q4_K/Q6_K/IQ4_XS layouts are ported bit-exact from ggml-common.h (validated
GPU vs CPU).

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
| `vkr_create_runtime` / `vkr_destroy_runtime` | Device/queue wrapper + capability detection |
| `vkr_detect_capabilities` | Shared feature/property detection + arch ladder |
| `vkr_malloc` / `vkr_free` | Pooled buffer allocator (hipMalloc-equivalent) |
| `vkr_upload` / `vkr_download` | Staging upload/download with sync |
| `vkr_create_command_pool` / `vkr_create_descriptor_pool` / `vkr_create_pipeline_layout` / `vkr_create_pipeline_cache` | Pool/layout/cache helpers |
| `vkr_get_arch_index` / `vkr_get_arch_name` / `vkr_has_subgroup` / `vkr_has_coop_matrix` | Capability queries |

All five libraries (vkmath, vkblas, vkquant, vkrand, vkfft) now build their
contexts on VKRuntime: capability detection, descriptor pool, pipeline layout
and pipeline cache are created via `vkr_*` helpers instead of duplicated
inline Vulkan code.

### In Progress

- **Real cooperative-matrix GEMM on this driver** — the coopmat path is
  implemented and compiled but dormant (env `VAIT_COOPMATRIX=1` to enable);
  the AMD 26.7.1 driver crashes on `coopMatMulAddKHR`. Testable on a newer
  driver or via `ssh rr@macx`.
- **Forward quantization of Q4_K/Q6_K/IQ4_XS** — only Q8_0/Q4_0 forward
  quantize is implemented so far.

---

## Building

### Prerequisites

- **Vulkan SDK 1.4.357.0** or newer (https://vulkan.lunarg.com)
- **CMake 3.20+** or Visual Studio 2022 with C++ build tools
- **AMD GPU** with Vulkan 1.4 support (RDNA2 or RDNA4 recommended)

### Compile Shaders

```powershell
cd shaders
.\compile_shaders.ps1
```

This compiles all `.comp` files under `shaders/vkblas/{baseline,subgroup,coopmatrix}/`
into SPIR-V binaries and generates `src/vkblas/shaders_spv.h` with embedded
bytecode arrays.

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
`c` = complex-f32, `z` = complex-f64, `i8` = int8.

### Specialization Constants Over #define

Tile dimensions, unroll factors, and wave widths are SPIR-V specialization
constants (`constant_id`), not pre-compiled `#define` variants. One SPIR-V
binary can be reconfigured at pipeline creation time without recompilation.
This keeps the shader count manageable — 28 sources produce 52 binaries because
we only vary by wave size (32 vs 64), not by tile size.

---

## Contributing

We welcome contributions. Here is how to get started:

### Finding Work

1. Read the root `AGENTS.md` — it is the binding contract for this repo.
2. Check the `specs/` directory for reference material and the design docs.
3. Pick an unimplemented component (VKFFT, VKRAND, VKMath, VKQuant, or the
   LLM engine layers) and open an issue to claim it.
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

- **ROCm / AMD** — for hipBLAS, rocFFT, rocBLAS, and rocrand. VAiSt mirrors
  their API names, data type conventions, and algorithm structures. The
  reference ROCm headers used during development are mirrored under
  `specs/rocm-reference/` with attribution.
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

[License details to be determined — see LICENSE file]

This project is not affiliated with or endorsed by AMD, the Khronos Group,
or any other organization whose materials appear in the `specs/` directory.
All trademarks are the property of their respective owners.
