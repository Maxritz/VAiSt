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
  └── VKQuant  (dequantization Q4_0/Q8_0, quantization)
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

## Project Goals

1. Deliver a **pure Vulkan compute** AI inference stack — no ROCm, HIP, or
   CUDA dependencies — targeting AMD RDNA2 (gfx103x) and RDNA4 (gfx1201) GPUs.
2. Achieve **>80 tok/s** decode throughput on RX 9070 XT for Qwen3.5-9B-class
   models at Q4_K quantization (baseline: ~103 tok/s observed on HIP backend).
3. Eliminate the seven architectural flaws documented in the LLM Engine
   Architecture doc (per-op descriptor allocation, single-command-buffer CB
   per op, validation readbacks, linear staging, missing barriers,
   per-head dispatches, 640 vkQueueSubmit+wait per decode step).
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
- **Single command buffer per decode step.** Record all 225 dispatches in one
  `VkCommandBuffer` with explicit `vkCmdPipelineBarrier` transitions,
  eliminating CPU-side driver overhead from per-op submits.
- **Push descriptors everywhere.** Use `VK_KHR_push_descriptor` to bypass
  descriptor pool allocation; all 225 dispatches bind via
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

This project exists because high-performance LLM inference on AMD GPUs has
historically required ROCm — a Linux-only stack that cannot run on RDNA4
Windows drivers (which lack ROCm support). By building a **ground-up Vulkan
compute stack**, we unlock native AMD GPU inference on both Linux and Windows
from a single codebase. The engine targets the RX 9070 XT (RDNA4) specifically
because its 64 KiKiB L2 + 32 KiKiB LDS + 32 CUs provide a representative
balance of memory bandwidth and compute for sub-100B parameter models, while
RDNA2 serves as the compatibility baseline.

The architectural redesign (single CB, push descriptors, fused kernels,
pooled staging) directly addresses the 0.19 tok/s failure mode of the legacy
approach. Every doc in `docs/specs/` and every shader in `shaders/` is
anchored to this contract.

## Child DOX Index

- `specs/` — Reference specifications and knowledge graph
- `include/` — Public API headers (vkblas, vkfft, vkrand, vkmath, vkquant)
- `src/` — Implementation (C99 runtime + Vulkan dispatch)
- `shaders/` — GLSL compute shaders per GPU architecture
- `tests/` — Test harnesses per sub-library
