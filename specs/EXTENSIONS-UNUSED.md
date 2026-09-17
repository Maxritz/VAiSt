# Unused Vulkan Extensions → Achievable Features (RDNA2 vs RDNA4)

> **Document type**: durable research / audit spec
> **Scope**: every compute-relevant Vulkan extension (and core 1.1–1.4 feature)
> available on the RDNA2 (RX 6700 XT) and RDNA4 (RX 9070 XT) target devices that
> the VAiT stack does **not** enable or consume today, and what enabling each one
> unlocks.
> **Status**: analysis only — no library code depends on this table yet. The
> companion deliverable `specs/VK-MAXR-EXTENSIONS.md` defines the extensions the
> stack proposes on its own (`VK_MaxR_*`) for the gaps below that Vulkan cannot
> reach.

---

## 1. Enumeration ground truth

| Device | Architecture | Device ext count | Source |
|--------|--------------|------------------|--------|
| AMD Radeon RX 6700 XT | RDNA2 (gfx1031) | 198 | `specs/GPU_CAPABILITIES.md` §8 (`vulkaninfo` JSON, SDK 1.4.357.0) |
| AMD Radeon RX 9070 XT | RDNA4 (gfx1201) | 240 | `vulkaninfo` on this machine, SDK 1.4.357.0 |

Driver branch is identical (Adrenalin 26.7.1) on both devices, so every
compute-relevant extension below is advertised on **both** architectures unless
flagged `[RDNA4 only]`. All availability rows are therefore valid for both
unless marked.

### What the codebase enables today (the "used" set)

- Device extensions listed at `vkCreateDevice`: **only** `VK_KHR_cooperative_matrix`
  (and that gated behind `VAIT_COOPMATRIX`; see `tests/test_vkblas.c:406`).
- Core features enabled in the pNext chain (partial, duplicated per test
  harness): `shaderInt64`, `shaderFloat64`, `shaderInt16`,
  `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`,
  `shaderFloat16`, `shaderInt8`, `scalarBlockLayout`, `storageBuffer8BitAccess`.
- Detected but not formally enabled at device create: push descriptors
  (`vkCmdPushDescriptorSetKHR` resolved via `vkGetDeviceProcAddr`, vkruntime.c:120-123).
- Effectively **zero** named extensions from the list below are consumed.

Every other compute-relevant extension/feature below is **unused**.

---

## 2. Unused core 1.1–1.4 features (enable via pNext — no extension name needed)

These are promoted core features. They must be enabled in the
`VkPhysicalDeviceFeatures2` pNext chain at `vkCreateDevice`; they do **not**
appear in `ppEnabledExtensionNames`.

| # | Feature | Core | What it unlocks | Component | Priority |
|---|---------|------|-----------------|-----------|----------|
| 1 | `shaderIntegerDotProduct` | 1.3 | dp4a packed int8/int16 dot → a real `gemm_i8` qgemm tier. Both archs accelerate 8/16-bit + 4×8 packed, **not** mixed-sign or 32-bit (GPU_CAPABILITIES §6) | vkblas qgemm | **HIGH** |
| 2 | `subgroupSizeControl` + `computeFullSubgroups` | 1.3 | Force `requiredSubgroupSize=32` on subgroup-tier stages. RDNA4: 2,048 Wave32 slots vs 1,024 Wave64 (GPU_CAPABILITIES §2) | vkblas/vkmath pipeline factory | **HIGH** |
| 3 | `synchronization2` | 1.3 | `vkCmdPipelineBarrier2` for the fused single-CB barrier graph | vkruntime | **HIGH** |
| 4 | `pipelineCreationCacheControl` | 1.3 | Immediate/async pipeline compile — kills the lazy first-GEMM stall | vkblas/vkquant/vkmath | MED |
| 5 | `maintenance4` | 1.3 | `maxBufferSize`/`maxMemoryAllocationSize` = 2 GiB — caps pooled-allocator block growth | vkruntime | MED |
| 6 | `bufferDeviceAddress` | 1.2 | GPU-side pointers: fused kernels, one "weights table" buffer, no per-op descriptor rebinds (complements push descriptors) | vkblas/vkmath fused decode | **HIGH** |
| 7 | `shaderSubgroupExtendedTypes` | 1.2 | fp16/int8 subgroup reduce/shuffle for subgroup-tier reductions (fp16 softmax, norms) | vkmath | MED |
| 8 | `vulkanMemoryModel` | 1.2 | Device-scope atomics ordering for fused reductions | vkmath | LOW |
| 9 | `shaderBufferInt64Atomics` + `shaderSharedInt64Atomics` | 1.0 | 64-bit atomic accumulators for fused reductions (softmax stats) | vkmath | LOW |
| 10 | `timelineSemaphore` | 1.2 | Multi-queue (QF0 graphics / QF1 compute / QF2 transfer) without per-step wait-for-idle | vkruntime | MED |
| 11 | `shaderExpectAssume` (`VK_KHR_shader_expect_assume`, core 1.4) | 1.4 | `OpExpectKHR`/`OpAssumeKHR` branch-prediction hints in the GEMM k-loop | vkblas qgemm | LOW |
| 12 | `shaderBfloat16` (`VK_KHR_shader_bfloat16`) | ext | Native bf16 convert ops in GLSL — replaces the `>>16` truncation trick with real `OpConvertFToBF16` (round-to-nearest) | vkmath bf16 casts/gemm_bf16 | MED |
| 13 | `shaderFloat8` (`VK_EXT_shader_float8`) | ext | fp8 (E4M3/E5M2) scalar types in shaders — prerequisite for the RDNA4 WMMA-FP8 matrix path | vkquant fp8 KV/weights | LOW (research) |

---

## 3. Unused named device extensions (opt in by name)

| # | Extension | Class | What it unlocks | Component | Priority |
|---|-----------|-------|-----------------|-----------|----------|
| 1 | `VK_KHR_push_descriptor` | EXPLOIT | Formally enables the fn the runtime already resolves (vkruntime.c:120). Should be listed at device create | vkruntime | **HIGH** |
| 2 | `VK_KHR_pipeline_binary` | EXPLOIT | Bake 120 SPIR-V pipelines to disk once; near-zero cold-start compile. `VkPhysicalDevicePipelineBinaryFeaturesKHR.pipelineBinaries` | vkblas/vkmath/vkquant caches | **HIGH** |
| 3 | `VK_EXT_pipeline_creation_cache_control` | EXPLOIT | `pipelineCreationCacheControl` (redundant with core 1.3; keep for pre-1.3) | vkblas | LOW |
| 4 | `VK_AMD_shader_core_properties` | EXPLOIT | `wavefrontSize` (64 RDNA2 / 32 RDNA4), `simdPerComputeUnit`, `vgprsPerSimd`, `maxVgprAllocation` → Wave32-vs-Wave64 + occupancy math (GPU_CAPABILITIES §2) | vkruntime detect | **HIGH** |
| 5 | `VK_AMD_shader_core_properties2` | EXPLOIT | `activeComputeUnitCount` (40 RDNA2 / 32 RDNA4) → wave-slot budget (1,280 RDNA2 Wave64 / 2,048 RDNA4 Wave32) | vkruntime detect | **HIGH** |
| 6 | `VK_EXT_shader_atomic_float` (+`_float2`) | OPTIONAL | fp32 (and fp16 via float2) `atomicAdd/min/max` for fused softmax/attention — genuine win vs lock-free int tricks | vkmath | MED |
| 7 | `VK_AMD_device_coherent_memory` | OPTIONAL/RISKY | Coherent `DEVICE_LOCAL` types (this GPU exposes type 2 = DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT on heap 1 / ReBAR) → zero-copy CPU-write-to-VRAM path, DirectStorage-style, no staging copy | vkruntime | **HIGH** (ties to memory-type probe) |
| 8 | `VK_EXT_pageable_device_local_memory` | OPTIONAL | Pin/unpin device-local pages — pairs with #7 for the ReBAR aperture | vkruntime | MED |
| 9 | `VK_AMD_buffer_marker` | OPTIONAL | `vkCmdWriteBufferMarker2AMD` GPU timestamps inside the single-CB decode path | dev profiler | LOW |
| 10 | `VK_AMD_shader_info` | OPTIONAL | `numUsedVgprs`/`numUsedSgprs` per pipeline — the only sanctioned occupancy *measurement* | dev profiler | LOW |
| 11 | `VK_EXT_memory_budget` / `VK_EXT_memory_priority` | OPTIONAL | Heap budget hints for vkruntime pool sizing; priority for pinning weights | vkruntime | LOW |
| 12 | `VK_EXT_descriptor_indexing` | OPTIONAL | Variable-count descriptor arrays — one weight binding across all decode layers | vkblas fused decode | MED |
| 13 | `VK_EXT_inline_uniform_block` | OPTIONAL | Small scalar constants (scales, tile params) in the descriptor set, no push-constant budget | vkblas/vkmath | LOW |
| 14 | `VK_EXT_mutable_descriptor_type` | OPTIONAL | One descriptor layout that is f32/f16/bf16/int8 depending on binding — fewer pipeline variants | vkblas/vkmath | LOW |
| 15 | `VK_KHR_shader_clock` | OPTIONAL | `clock64()` for GPU-side timing/debug counters | dev profiler | LOW |
| 16 | `VK_KHR_shader_subgroup_rotate` | OPTIONAL | `subgroupRotate` — rotation-based scan/FFT/attention patterns | vkfft/vkmath | LOW |
| 17 | `VK_KHR_shader_subgroup_uniform_control_flow` | OPTIONAL | Compiler optimisation for kernels whose branches are wave-uniform | vkblas qgemm | LOW |
| 18 | `VK_KHR_shader_quad_control` | OPTIONAL | Explicit quad subgroup ops | vkfft | LOW |
| 19 | `VK_KHR_workgroup_memory_explicit_layout` | OPTIONAL | Explicit LDS block layout control for tiled GEMM staging | vkblas | LOW |
| 20 | `VK_KHR_zero_initialize_workgroup_memory` | OPTIONAL | `zeroInitializeWorkgroupMemory` — avoids explicit LDS zeroing in reductions | vkmath | LOW |
| 21 | `VK_KHR_shader_long_vector` | OPTIONAL | Longer vectors in shaders → wider loads/stores, fewer ALU ops | all | LOW |
| 22 | `VK_EXT_shader_module_identifier` | OPTIONAL | Module-identifier cache keys for pipeline-cache hits across runs | vkblas | LOW |
| 23 | `VK_KHR_shader_maximal_reconvergence` | OPTIONAL | Post-divergence reconvergence semantics | vkblas | LOW |
| 24 | `VK_EXT_calibrated_timestamps` | OPTIONAL | CPU↔GPU time correlation for the profiler | dev profiler | LOW |
| 25 | `VK_KHR_map_memory2` / `VK_EXT_memory_priority` | OPTIONAL | `vkMapMemory2` + flush semantics for the coherent path | vkruntime | LOW |
| 26 | `VK_EXT_descriptor_buffer` / `VK_EXT_descriptor_heap` | DEFERRED | Zero-CPU-cost descriptor updates. Deliberately deferred (audit §2.1) — adds a second descriptor paradigm | — | DEFERRED |
| 27 | `VK_KHR_copy_memory_indirect` | LOW | Indirect/address-based copies for the descriptor-buffer path | — | DEFERRED |

---

## 4. What we can achieve, by pipeline stage

### 4.1 Upload path (DirectStorage-style zero-copy)
Enable **`VK_AMD_device_coherent_memory`** + **`VK_EXT_pageable_device_local_memory`** +
**`bufferDeviceAddress`**. This GPU (RDNA4, ReBAR/SAM enabled) exposes
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (memory type 2, heap 1 = the
15.92 GiB VRAM heap — verified via `vulkaninfo`). Result: host `memcpy` lands
directly in VRAM; GPU reads it at device-local bandwidth; no staging buffer, no
`vkCmdCopyBuffer`, no queue submit for the data path. RDNA2 without ReBAR only
maps a small BAR aperture — must fall back to the existing staged path (probe
`memoryTypeBits` per device).

### 4.2 Compute path (decode hot loop)
Enable **`shaderIntegerDotProduct`** → new `gemm_i8` dp4a qgemm tier (RDNA2/4
accelerate 8-bit + 4×8 packed). Enable **`subgroupSizeControl`** +
**`VK_AMD_shader_core_properties(2)`** → force Wave32 (2,048 slots on RDNA4),
tile to 4 waves/CU @ 64 VGPR. **`synchronization2`** + **`timelineSemaphore`**
→ single-CB barrier graph across 3 queues. **`pipelineCreationCacheControl`** +
**`VK_KHR_pipeline_binary`** → zero-latency pipeline bring-up.

### 4.3 Reduction / attention path
**`VK_EXT_shader_atomic_float`** → fp32/fp16 atomic softmax/attention stats;
**`shaderSubgroupExtendedTypes`** → fp16 subgroup reductions;
**`vulkanMemoryModel`** + int64 atomics → device-scope ordering.

### 4.4 Quality / bit-exactness
**`VK_KHR_shader_bfloat16`** → native round-to-nearest bf16 converts (replaces
truncation `>>16`); `VK_KHR_shader_float8` → fp8 scalar groundwork for the
RDNA4 WMMA-FP8 matrix tier.

---

## 5. Excluded (present but NOT useful to this stack)

Graphics/RT/video/swapchain: `VK_KHR_swapchain*`, `VK_KHR_present_*`,
`VK_KHR_surface*`, `VK_KHR_ray_tracing_*`, `VK_KHR_acceleration_structure`,
`VK_KHR_deferred_host_operations`, `VK_KHR_video_*` (decode/encode),
`VK_EXT_mesh_shader`, `VK_EXT_shader_object`, `VK_EXT_graphics_pipeline_library`,
`VK_KHR_dynamic_rendering*`, `VK_EXT_host_image_copy` (no images),
`VK_AMD_*` display/HDR/anti-lag/legacy GCN set (§2.3 of RDNA-EXTENSIONS.md),
`VK_AMD_gcn_shader`, `VK_AMD_gpu_shader_half_float/int16` (superseded by core),
`VK_AMD_shader_ballot` (superseded by core subgroup), `VK_AMD_shader_trinary_minmax`
(negligible ALU savings), `VK_AMD_memory_overallocation_behavior` (exact
allocation control), `VK_EXT_external_memory/semaphore/fence_win32` (headless),
`VK_KHR_external_*` (headless).

---

## 6. Recommended adoption order

1. **Canonical full feature enablement at `vkCreateDevice`** (core 1.1–1.4 +
   the §3 EXPLOIT set) in one vkruntime helper — the prerequisite for
   everything below. *(being implemented — see vkruntime `vkr_create_device`)*
2. **`VK_AMD_shader_core_properties(2)` + `subgroupSizeControl`** → Wave32 tier
   (measured, not assumed).
3. **`shaderIntegerDotProduct`** → `gemm_i8` dp4a tier (per-format
   `integerDotProduct*Accelerated` gate).
4. **`VK_AMD_device_coherent_memory` + ReBAR type-2 pool** → zero-copy upload.
5. **`VK_KHR_pipeline_binary`** → persisted pipeline store.
6. **`VK_EXT_shader_atomic_float`** → fused reductions.

Gaps Vulkan cannot reach (register budget, cache control, wave matrix, fp8
matrix) are proposed as the stack's own `VK_MaxR_*` extensions — see
`specs/VK-MAXR-EXTENSIONS.md`.

---

## 7. Honesty statement

- RDNA4 rows verified today via `vulkaninfo` on the RX 9070 XT (SDK 1.4.357.0).
- RDNA2 rows from `specs/GPU_CAPABILITIES.md` (RX 6700 XT, same driver branch).
- The "unused" classification means *not referenced by any file under `src/`,
  `tests/`, or `include/`* at the time of writing (verified by grep).
- Availability is not the same as correctness: `VK_KHR_cooperative_matrix` is
  advertised but driver-crashing on 26.7.1; probe every RISKY/OPTIONAL item
  before relying on it (harness-first rule).
