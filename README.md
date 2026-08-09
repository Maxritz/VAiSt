<img width="1255" height="539" alt="VAiSt1" src="https://github.com/user-attachments/assets/1562b926-a3db-46e2-a679-115e4942a383" />

# VAiSt - Vulkan AI Stack

A Vulkan compute AI stack, built from scratch, implementing BLAS, FFT, RNG and math primitives for AMD RDNA2 (gfx103x) and RDNA4 (gfx1201) GPUs, and really any GPU that speaks Vulkan 1.4.

No ROCm. No HIP. No CUDA. Just Vulkan compute shaders, C99 headers, and Vulkan-native handles.

## Why This Exists

For years, doing serious GPU-accelerated AI on AMD hardware meant going through ROCm, and that meant Linux. That's shifted a bit: AMD shipped ROCm 7.2 in January 2026 with, for the first time, a genuinely unified Windows and Linux release, and the RX 9070 XT (gfx1201) is now officially on AMD's supported Windows list, with native PyTorch and llama.cpp builds to go with it.

Still, the HIP SDK for Windows itself, the actual toolchain you'd build a project like this against, ships without MIOpen, MIGraphX, communication libraries, or CMake HIP language support, and lists "AI Frameworks: Not available" against itself in AMD's own docs as of this writing. The PyTorch path AMD showed off runs through a separate consumer distribution, not through the general HIP SDK. So Windows ROCm development is real now, in a way it wasn't a year ago, but it's still a narrower stack than what Linux gets, and HIP-on-Windows still has its own rough edges once you're outside AMD's specific supported paths.

I wanted to run LLM inference on my RX 9070 XT. The existing HIP backends either wouldn't compile on Windows, or when they did, gave less than 1 tok/s thanks to a pile of architectural problems: one command buffer per operation, a descriptor-set allocation per kernel dispatch, validation readbacks stalling the GPU mid-inference, per-head attention launches that barely register any occupancy at all. The stack was spending more time on CPU driver overhead than on actual compute. That's not a GPU problem, that's a plumbing problem.

VAiSt starts over, from first principles:

**One command buffer per decode step.** All 225 dispatches for a 32-layer model get recorded into a single VkCommandBuffer with explicit pipeline barriers. That kills the 640 vkQueueSubmit + vkWaitForFences round-trips that were burning 64ms per token on the old path.

**Push descriptors everywhere.** No per-dispatch descriptor pool allocations. Every vkCmdPushDescriptorSetKHR binds directly.

**Fused kernels.** QKV projection, RoPE and KV cache writes are fused into a single shader. FFN gate+up is fused too. Cuts memory traffic by 40-60% per layer, which matters more than it sounds like it should.

**Three shader tiers.** A portable baseline (Vulkan 1.0 core), a subgroup tier (VK_KHR_shader_subgroup), and a cooperative-matrix tier (VK_KHR_cooperative_matrix). The runtime picks whichever tier your GPU actually supports and falls back gracefully if it doesn't.

**Shared shader sources.** 28 GLSL compute shader sources compile down into 52 SPIR-V binaries through compile-time specialisation, one source, multiple tile sizes, multiple wave sizes, multiple quantisation types.

## What's Inside

```
VAiSt
├── include/
│   └── vkblas/           BLAS API (hipBLAS-compatible naming)
│   ├── vkfft/            FFT API (rocFFT-compatible naming)
│   ├── vkrand/            RNG + sampling API (rocrand-compatible naming)
│   ├── vkmath/            Elementwise ops, reductions, activations
│   └── vkquant/           Quantised weight dequantisation (Q4_K, Q8_0, etc.)
├── src/                  C99 runtime + Vulkan dispatch
├── shaders/
│   ├── baseline/         Tier 0: portable Vulkan 1.0 (works everywhere)
│   ├── subgroup/         Tier 1: subgroup shuffle optimisations
│   ├── coopmatrix/       Tier 2: VK_KHR_cooperative_matrix (RDNA4+)
│   └── compile_shaders.ps1   Compiles .comp -> SPIR-V -> C header arrays
├── specs/                Reference data, ISA docs, ROCm headers
├── tests/                Per-library test harnesses (coming)
└── docs/                 Architecture specifications (specs/)
```

## VKBLAS (implemented)

The first piece to actually reach a working state. Implements the full hipBLAS GEMM family:

| Function | Precision | Description |
|---|---|---|
| `vkblas_sgemm` | f32 | Single GEMM |
| `vkblas_sgemm_strided_batched` | f32 | Strided batched GEMM |
| `vkblas_sgemm_batched` | f32 | Per-buffer batched GEMM |
| `vkblas_sgemm_ex` | mixed | Ex dispatch with compute-type control |

The API mirrors `hipblasSgemm` parameter order exactly, so porting from HIP is mostly a mechanical find-and-replace. Pipeline selection happens automatically based on what the GPU can do:

```c
VkBLASContext* ctx;
vkblas_create_context(physicalDevice, device, &ctx);
// First call: detects extensions, lazily creates pipelines
vkblas_sgemm(ctx, cmd, VKBLAS_OP_N, VKBLAS_OP_N,
             m, n, k, &alpha, bufA, lda, bufB, ldb,
             &beta, bufC, ldc, bufD, ldd);
```

## In Progress

Scaffolded, not yet built:

- **VKFFT** - Plan-based FFT for multiple precisions
- **VKRAND** - PRNG generators (Philox, ThreeFry) and distribution sampling
- **VKMath** - Elementwise ops (add, silu, soft_max, gelu, etc.)
- **VKQuant** - Q4_K, Q6_K, Q8_0, IQ4_XS dequantisation shaders
- **LLM engine integration** - Transformer layer pipeline, KV caching, token sampling

## Building

### Prerequisites

- Vulkan SDK 1.4.357.0 or newer (https://vulkan.lunarg.com)
- CMake 3.20+, or Visual Studio 2022 with C++ build tools
- An AMD GPU with Vulkan 1.4 support (RDNA2 or RDNA4 recommended)

### Compile Shaders

```
cd shaders
.\compile_shaders.ps1
```

This compiles every `.comp` file under `shaders/vkblas/{baseline,subgroup,coopmatrix}/` into SPIR-V binaries, and generates `src/vkblas/shaders_spv.h` with the bytecode embedded.

### Build

```
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Run Tests

```
ctest -C Release
```

## Architecture Decisions

### Shader Tiering

Instead of writing architecture-specific binaries the way ROCm code objects do, every kernel exists in three capability tiers. Tier 0 is a baseline that runs on any Vulkan 1.0 device. Tier 1 adds subgroup operations where they're available. Tier 2 unlocks cooperative matrix instructions on RDNA4 and newer.

Which means a new GPU architecture doesn't need new shader variants, just extension detection. The fallback chain handles the rest on its own.

### No Heap Allocation in Hot Paths

Every dispatch function takes pointers. Pipeline objects are created lazily and cached in a fixed-size, open-addressing hash table inside the context. No malloc, no `vkAllocateDescriptorSets` per dispatch. Push descriptors handle all the binding.

### rocBLAS API Mirror

Public function names and parameter order mirror `hipblas*` and `rocblas_*` exactly on purpose. Porting a HIP project over to Vulkan becomes a mechanical translation:

```
hipblasSgemm(...)   ->  vkblas_sgemm(...)
hipblasCreate(...)  ->  vkblas_create_context(...)
hipblasDestroy(...) ->  vkblas_destroy_context(...)
```

Types follow the same scheme too: `s` = f32, `d` = f64, `h` = f16, `bf` = bf16, `c` = complex-f32, `z` = complex-f64, `i8` = int8.

### Specialisation Constants Over #define

Tile dimensions, unroll factors and wave widths are SPIR-V specialisation constants (`constant_id`), not pre-compiled `#define` variants. One SPIR-V binary can be reconfigured at pipeline creation time, no recompile needed. That's how 28 sources produce 52 binaries, not more, we only vary by wave size (32 vs 64), not by tile size.

## Contributing

Contributions welcome. Here's how to get started:

### Finding Work

1. Read the root `AGENTS.md` first, it's the binding contract for this repo.
2. Check the `specs/` directory for reference material and design docs.
3. Pick an unimplemented component (VKFFT, VKRAND, VKMath, VKQuant, or the LLM engine layers) and open an issue to claim it.
4. Read the component-level `AGENTS.md` (e.g. `src/vkblas/AGENTS.md`) before writing any code.

### Workflow

1. Write the decision tree and truth table first. Every new dispatch path, every `if` branch, every quantisation scheme needs a traced decision tree. No code until the table passes.
2. Build and run the test harness before merging. Added a shader? Run `test_vkblas`. Added an op? Run `test_vkmath`. No test harness yet for that piece? Write one first.
3. Update the truth table. Document the decision in the relevant `AGENTS.md` files. The contract has to stay readable, not just correct.
4. Commit with a clear message, referencing the component and the spec section it implements.

### Code Style

- C99. DOX doc comments (`/** ... */`). `\brief`, `\param`, `\retval`.
- Vulkan-native: every handle is a Vulkan object.
- Mirror ROCm API names for mechanical porting.
- No heap allocation in hot paths.
- No stubs, placeholders or TODOs in production code.

## Credits

This project stands on work done by others:

**Khronos Group** - for the Vulkan API, SPIR-V and the Vulkan Memory Allocator. Without the Vulkan spec and the open, cross-vendor extension ecosystem, none of this gets built. (https://www.khronos.org/vulkan/)

**AMD** - for the RDNA2 and RDNA4 architectures, the Vulkan driver on both Linux and Windows, and the open Radeon documentation that made the hardware behaviour analysis possible in the first place. (https://www.amd.com/en/support/graphics/amd-radeon-rx-9000-series)

**GPUOpen** - for the Vulkan Memory Allocator library (https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator), which handles `VkDeviceMemory` allocation and is bundled under `third_party/vk_mem_alloc.h`.

**ROCm / AMD** - for hipBLAS, rocFFT, rocBLAS and rocrand. VAiSt mirrors their API names, data type conventions and algorithm structures. Reference ROCm headers used during development are mirrored under `specs/rocm-reference/` with attribution. (https://rocm.docs.amd.com/)

**LunarG** - for the Vulkan SDK (1.4.357.0), glslangValidator, spirv-val, and the diagnostic layers that make any of this development tractable. (https://vulkan.lunarg.com/)

ROCm and its components (hipBLAS, rocFFT, rocBLAS, rocrand) remain the gold standard for what a GPU-accelerated math library API should look like. VAiSt isn't trying to replace them on Linux, it's just bringing the same capabilities to platforms where ROCm doesn't run.

## License

Apache License 2.0, see the [LICENSE](LICENSE) file for the full text. Permissive, includes a patent grant, free for commercial and non-commercial use.

Participation in this project is also governed by our [Code of Conduct](CODE_OF_CONDUCT.md), the short version: build and use this however you like, just not to develop or promote hatred, harassment, or violence against any person or group.

This project is not affiliated with or endorsed by AMD, the Khronos Group, or any other organisation whose materials appear in the `specs/` directory. All trademarks are the property of their respective owners.
