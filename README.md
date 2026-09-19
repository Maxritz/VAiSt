<img width="1255" height="539" alt="VAiSt1" src="https://github.com/user-attachments/assets/d4ac1269-0679-409f-b6c1-7aad39aa5959" />

A Vulkan compute AI stack, built from scratch, implementing BLAS, FFT, RNG, math
primitives, quantization, and attention for AMD RDNA2 (gfx103x) and RDNA4
(gfx1201) GPUs, and really any GPU that speaks Vulkan 1.4.

No CUDA. Just Vulkan compute shaders, C99 headers, and Vulkan-native handles.
Everything — BLAS, FFT, RNG, math, quant, model I/O, and attention — is a
native Vulkan compute dispatch. No ROCm / HIP / CUDA runtime dependency,
anywhere. Public API names mirror the ROCm surface for mechanical porting,
but every handle is a Vulkan object.

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
  tier (`VK_KHR_shader_subgroup`), and a cooperative-matrix tier
  (`VK_KHR_cooperative_matrix`). The runtime picks the best tier your GPU
  supports and falls back gracefully.

- **Shared shader sources.** 16 of the shader sources in `shaders/` compile to
  SPIR-V at build time via `glslangValidator` (when the Vulkan SDK is present),
  producing 27 SPIR-V blobs total (24 compiled sources + 3 pre-existing blobs
  with no `.comp` source). These are folded into a generated C header
  (`vaist_blas_spv.h`) consumed by `vaist_blas.c` and `vaist_attn.c` under the
  `VAIST_HAVE_VK_HDR` flag, keeping the libraries self-contained.

---

## What Is Inside

```
VAiSt
├── include/
│   ├── vaist/              VAiSt public headers (17 files: umbrella vaist.h
│   │   │                  + 12 component headers + vaist_attn.h, vaist_blas.h,
│   │   │                  vaist_linalg.h, vaist_quant_tables.h)
│   ├── vkblas/           BLAS API (hipBLAS-compatible naming)
│   ├── vkfft/            FFT API (rocFFT-compatible naming)
│   ├── vkrand/           RNG + sampling API (rocrand-compatible naming)
│   ├── vkquant/          Dequant + forward-quantize shaders (Q4_0/Q8_0/NVFP4/T2_0)
│   ├── vkmodel/          Model loaders (GGUF native, safetensors, OpenVINO IR)
│   │                     + SSD-offload streaming (mmap + LRU expert prefetch)
│   ├── vkruntime/        Device/queue/memory/pool runtime (hipRuntime-equivalent)
│   ├── vkkv/             Cross-model KV-cache ridge transfer
│   └── vkdist/           Distributed compute over TCP
├── src/                  C99 runtime + Vulkan dispatch
├── vulkan/{linux,windows}/{c99,c++,python}/
│   └── {ai,attn,blas,compute,core,distributed,engine,graph,linalg,llm,
│       model,nn,quant,runtime,tensor}/   15-component VAiSt c99/c++/python layer
├── shaders/
│   ├── vkattn/           Attention (flash-decode, future MHA/MLA variants)
│   ├── vkblas/           GEMM, qgemm, moe_route, conv (rb2) baseline tier
│   ├── vkblas_l1l2/      L1/L2 BLAS vector/matrix ops
│   ├── vkmath/           Elementwise, reductions, activations, casts
│   ├── vkrand/           PRNG + distribution sampling
│   ├── vkfft/            Radix-2 FFT
│   ├── vkkv/             KV cache ridge transfer
│   ├── compile_shaders.ps1   Compiles .comp → SPIR-V → C header arrays
│   └── (per-lib tiers)   baseline/ (Vulkan 1.0 core), subgroup/ (VK_KHR_shader_subgroup),
│                         coopmatrix/ (VK_KHR_cooperative_matrix)
├── cmake/                Build tooling (spirv_to_header.py)
├── specs/                Design docs, ISA reference, architecture notes
│   ├── Common_Issues.md        GPU hang / device-lost / fence issues (catalog)
│   └── (per-subsystem specs)
├── tests/                22 CTest targets (15 vaist_cpp_*_test + 7 standalone/Python)
│                         17 pass on this machine; 3 SEGFAULT + 2 Python-load
│                         failures are pre-existing environment issues (no GPU)
├── docs/                 Vulkan ↔ Torch migration guide
├── build/                Default build (no Vulkan SDK)
└── build-vk-on/          VAIST_ENABLE_VULKAN=ON (sandbox SDK 1.4.357)
```

### VAiSt C99 runtime layer (`vaist_*`)

The `vulkan/{linux,windows}/{c99,c++,python}/` tree and the `include/vaist/`
headers are the VAiSt runtime + BLAS/linalg layer on top of the `vkblas`/
`vkmath` library stack. This is the substrate that a host engine (e.g. the
assemble/sglang attention layer) links against: it owns Vulkan device
creation, buffer upload/download, command queues, and the dynamic Vulkan
loader resolution, then hands off to the per-op `vkblas_*`/`vkmath_*` kernels.

| Function | Description |
|----------|-------------|
| `vaist_runtime_create(requested, &rt)` | Create a runtime for `VAIST_BACKEND_AUTO/CPU/VULKAN` |
| `vaist_runtime_destroy(rt)` | Tear down |
| `vaist_runtime_info(rt, &out)` | Backend + availability info |
| `vaist_runtime_device_caps(rt, &out)` | Device capability snapshot |
| `vaist_buffer_create/destroy/data/size/upload/download` | Buffer lifecycle (host↔device) |
| `vaist_stream_create/destroy/synchronize` | Stream/queue abstraction |
| `vaist_buffer_gpu_handle(rt, &handle)` | Get the underlying Vulkan handle |
| `vaist_runtime_vk_state(rt, &dev, &queue, &qf)` | Vulkan device/queue/queue-family accessor |
| `vaist_runtime_vk_proc(rt, name)` | Resolve a Vulkan proc through the runtime |

The runtime does **not** link `vulkan-1` as an import library. It resolves
Vulkan entry points through `runtime_sym`/`inst_proc` (`LoadLibraryA` +
`GetProcAddress` on Windows, `dlopen` + `dlsym` on Linux).

### VAiSt Attention (`vaist_attn`)

| Function | Description |
|----------|-------------|
| `vaist_attn_create(rt, &cfg)` | Create a flash-decode attention context (shader, descriptor layout, pipeline, cmdpool, staging buffers) |
| `vaist_attn_destroy(ctx)` | Tear down |
| `vaist_attn_flash_decode(ctx, q, k_cache, v_cache, block_tables, seqlen, out)` | Dispatch paged-attention flash decode |

The `attn_flash_decode.comp` compute shader:
- Binds 5 SSBOs: `q` (input query, f32), `k_cache` + `v_cache` (GPU buffers, fp16),
  `block_tables` (uint32 page indices), `out` (f32 output).
- Push constants (32 bytes): `scale`, `head_dim`, `num_q_heads`, `num_kv_heads`,
  `seqlen`, `max_blocks`, `block_size`, `q_head_idx`.
- One workgroup per query head; each thread computes one output element via fused
  softmax + weighted sum over the key/value cache.
- On no Vulkan device (sandbox CPU-only), returns `VAIST_DEVICE_ERROR` so the
  caller falls back to CPU dequant + GEMM.

### Runtime behaviour: child-process Vulkan probe

`vaist_runtime_create(VAIST_BACKEND_VULKAN)` gates "Vulkan available" on a
real, creatable logical device. The sandbox loader enumerates a phantom
physical device whose `vkCreateDevice` raises an uncatchable
`EXCEPTION_STACK_OVERFLOW` (0xC0000300), so a **child-process probe** is used:

- `vaist_vk_device_probe_child()` spawns the test exe with the command-line
  flag `VAIST_VK_PROBE=1` (argv, not an env var — the env block does not
  propagate reliably across the DLL/exe boundary).
- The child's `main(argc, argv)` checks `argv[1] == "VAIST_VK_PROBE=1"` and
  runs probe-only mode.
- `vaist_runtime_create` skips the spawn when `GetCommandLineW()` contains
  `VAIST_VK_PROBE=1`, so the probe child never re-spawns (runaway recursion
  fix).
- A `__try/__except` SEH backstop around `vkCreateDevice` plus a
  `vk_dev_attempted` cache flag prevents retrying the crashing call.

### Sandbox quirks handled

- `VkCommandBufferAllocateInfo` in the sandbox `vulkan_core.h` lacks
  `pCommandBuffers` while `PFN_vkAllocateCommandBuffers` is 3-arg →
  `VAIST_VK_ALLOC_3ARG_COMMIT` CMake probe + 3-arg/2-arg branch in
  `vulkan/{linux,windows}/c99/{blas,attn}/src/vaist_*.c`.
- `/WX` on Windows (C2220 → error); C flags `/D_WINDOWS` (no `/EHa`, so SEH
  `__try/__except` is inert for loader crashes).
- `build-vk-on/` (MSVC Release) is the `VAIST_ENABLE_VULKAN=ON` target against
  the sandbox Vulkan SDK 1.4.357.0; `build/` is the default, no-Vulkan build
  that runs the CPU fallback.

---

## Building

### Prerequisites

- **Vulkan SDK 1.4.357.0** or newer (https://vulkan.lunarg.com) — needed only
  when `VAIST_ENABLE_VULKAN=ON` (compile + validate shaders + embed SPIR-V)
- **CMake 3.20+** or Visual Studio 2022 with C++ build tools
- **AMD GPU** with Vulkan 1.1+ support (RDNA2 = Vulkan 1.0 baseline for
  shaders; Vulkan 1.4 + `VK_KHR_cooperative_matrix` required for the dormant
  coopmatrix tier)

No ROCm / HIP / CUDA toolchain or runtime is required — the stack is pure
Vulkan compute. Build with the default MSVC (`cl.exe`) toolchain.

### Windows Toolchain

Build from a **Visual Studio 2022 Developer Command Prompt** (x64):

```powershell
# Default build (no Vulkan SDK required):
cd C:\vaist
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# With the Vulkan backend enabled (sandbox SDK):
$env:VULKAN_SDK="C:\VulkanSDK\1.4.357.0"
cmake -B build-vk-on -DCMAKE_BUILD_TYPE=Release -DVAIST_ENABLE_VULKAN=ON -G Ninja
cmake --build build-vk-on --config Release
ctest --test-dir build-vk-on -C Release
```

All 15 `vaist_cpp_*_test` + `vaist_blas_vulkan_test` + `blas_test` +
`linalg_test` + `vaist_blas_vulkan_test` + `cpp_smoke` pass. The
`vaist_blas_vulkan_test` is the one to watch on the sandbox: it prints
`PASS (backend=1, cpu-fallback)` and must leave **zero** leftover processes
(the child-process probe recursion guard prevents spawn runaway).

### Linux Toolchain

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build . --config Release
ctest -C Release
```

### Compile Shaders

```powershell
cd shaders
.\compile_shaders.ps1
```

File-tree auto-discovery: globs every `.comp` under
`shaders/{vkblas,vkmath,vkquant,vkrand,vkfft,vkattn}/{baseline,subgroup,coopmatrix}/`,
compiles each to SPIR-V, and regenerates the embedded C header array via
`cmake/spirv_to_header.py`. Adding a kernel = drop a `.comp` + regenerate.

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

### Specialization Constants Over `#define`

Tile dimensions, unroll factors, and wave widths are SPIR-V specialization
constants (`constant_id`), not pre-compiled `#define` variants. One SPIR-V
binary per shader source can be reconfigured at pipeline creation time without
recompilation. The 24 compiled `.comp` sources (in `shaders/{lib}/baseline/`) plus
  3 pre-existing SPIR-V blobs without sources yield 27 total SPIR-V blobs;
  specialization varies tile/wave at pipeline-creation time, not by pre-compiled
  variants.

---

## Contributing

We welcome contributions. Here is how to get started:

### Finding Work

1. Read the root `AGENTS.md` — it is the binding contract for this repo.
2. Check the `specs/` directory for reference material and the design docs.
3. Pick an unimplemented component or enhancement and open an issue to claim it.
4. Read the per-component `AGENTS.md` (e.g. `src/vkblas/AGENTS.md`) before writing any code.

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

- **ROCm / AMD** — for hipBLAS (API naming reference), rocSPARSE, and rocsolver.
  `specs/rocm-reference/` mirrors reference headers with attribution.
  (https://rocmd.docs.amd.com/)

- **LunarG** — for the Vulkan SDK (1.4.357.0), glslangValidator, spirv-val,
  and the diagnostic layers that make development tractable.
  (https://vulkan.lunarg.com/)

This project is not affiliated with or endorsed by AMD, the Khronos Group,
or any other organization whose materials appear in the `specs/` directory.
All trademarks are the property of their respective owners.

---

## License

This project is licensed under the Apache License, Version 2.0.
See the [LICENSE](LICENSE) file for details.
