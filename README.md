# VAiSt

Full multi-platform C99/C++/Python source baseline following the locked VAiSt
architecture.

## Current build target

- **Vulkan target:** 1.4.357 (sandbox SDK at `C:\VulkanSDK\1.4.357.0`)
- **Host:** Windows 10/11, MSVC (VS 17 2022 generator), CMake
- **C99 public ABI** (`vulkan/{linux,windows}/c99/`), C++17 convenience/RAII
  layer (`c++/`), Python ctypes layer (`python/`)
- CPU fallback preserved

## Runtime behaviour

The runtime does **not** link `vulkan-1` as an import library. It resolves
Vulkan entry points through `runtime_sym` (`LoadLibraryA` + `GetProcAddress`)
and `inst_proc` (`vkGetInstanceProcAddr(instance, name)`).

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
  guard).
- A `__try/__except` SEH backstop around `vkCreateDevice` plus a
  `vk_dev_attempted` cache flag prevents retrying the crashing call.

## Sandbox quirks handled

- `VkCommandBufferAllocateInfo` in the sandbox `vulkan_core.h` lacks
  `pCommandBuffers` while `PFN_vkAllocateCommandBuffers` is 3-arg →
  `VAIST_VK_ALLOC_3ARG` CMake probe + 3-arg/2-arg branch in `vaist_blas.c`.
- `VAIST_ENABLE_VULKAN=ON` build: `build-vk-on/` (MSVC Release).
- Default build (no Vulkan): `build/`.

## Layout

- `include/vaist/` — VAiSt public headers (ai, blas, compute, core,
  distributed, engine, graph, linalg, llm, model, nn, quant, runtime, tensor)
- `include/vk*/` + `src/vk*/` — the Vulkan BLAS/math/quant/fft/model/runtime
  libraries (vkblas, vkblas_l1l2, vkdist, vkfft, vkkv, vkmath, vkmodel,
  vkquant, vkrand, vkruntime, vkstream) with their SPIR-V shaders under
  `shaders/vk*/`
- `vulkan/{linux,windows}/{c99,c++,python}/{ai,blas,compute,core,distributed,
  engine,graph,linalg,llm,model,nn,quant,runtime,tensor}/` — the VAiSt
  implementation tree (cooperative matrix isolated under
  `vulkan/DO_NOT_USE/cooperative_matrix/`)
- `cmake/spirv_to_header.py` — SPIR-V → C header tool
- `tests/` — `vaist_blas_vulkan_test.c` (probe-only child entry),
  `blas_test.c`, `linalg_test.c`, `stack_smoke.c`, plus `test_vk*.c` for each
  library and C++/Python smoke tests
- `docs/vulkan_torch_migration_guide.md` — migration notes
- `specs/`, `specs-large/` — Vulkan spec/reference material
- `build/` (default, no Vulkan) and `build-vk-on/` (`VAIST_ENABLE_VULKAN=ON`)

The code is intentionally buildable without requiring the Vulkan SDK for the
baseline CPU path. The runtime detects the platform Vulkan loader
dynamically and records API availability; Vulkan kernel implementations
belong behind the runtime/compute boundaries and must be capability-selected.

The supplied source is a complete baseline implementation for the exposed
APIs. The user's full Windows/Vulkan hardware build is the next integration
validation step.