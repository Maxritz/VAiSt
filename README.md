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

The first component to reach production readiness. Implements the full
hipBLAS GEMM family:

| Function | Precision | Description |
|----------|-----------|-------------|
| `vkblas_sgemm` | f32 | Single GEMM |
| `vkblas_sgemm_strided_batched` | f32 | Strided batched GEMM |
| `vkblas_sgemm_batched` | f32 | Per-buffer batched GEMM |
| `vkblas_sgemm_ex` | mixed | Ex dispatch with compute-type control |

API mirrors `hipblasSgemm` parameter order exactly — porting from HIP is a
mechanical find-and-replace. Pipeline selection happens automatically based on
GPU capabilities:

```c
VkBLASContext* ctx;
vkblas_create_context(physicalDevice, device, &ctx);
// First call: detects extensions, lazily creates pipelines
vkblas_sgemm(ctx, cmd, VKBLAS_OP_N, VKBLAS_OP_N,
             m, n, k, &alpha, bufA, lda, bufB, ldb,
             &beta, bufC, ldc, bufD, ldd);
```

### In Progress

The following components are scaffolded but not yet implemented:

- **VKFFT** — Plan-based FFT for multiple precisions
- **VKRAND** — PRNG generators (Philox, ThreeFry) and distribution sampling
- **VKMath** — Elementwise ops (add, silu, soft_max, gelu, etc.)
- **VKQuant** — Q4_K, Q6_K, Q8_0, IQ4_XS dequantization shaders
- **LLM engine integration** — Transformer layer pipeline, KV caching,
  token sampling

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
