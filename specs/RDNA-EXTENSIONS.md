# RDNA/AMD Vulkan Extension Audit — VAiT Vulkan AI Stack

> **Document type**: durable research / audit spec
> **Scope**: all RDNA/AMD-relevant Vulkan device extensions available to the
> VAiT stack, how each is (or could be) used, plus candidate new extensions
> that would better expose AMD RDNA architecture features.
> **No library code or CMake was modified to produce this document.**

---

## 1. Methodology

| Item | Value |
|------|-------|
| Device enumerated | AMD Radeon RX 6700 XT (gfx1031 / Navi 22) |
| Target device (design) | AMD Radeon RX 9070 XT (gfx1201 / Navi 48, RDNA4) |
| Driver | AMD proprietary Adrenalin 26.7.1, driver version 2.0.353 (8388961) |
| SDK / loader | Vulkan SDK 1.4.357.0, instance/device `apiVersion` 1.4.357, conformance 1.4.0.0 |
| Extension inventory | 198 device extensions reported by `vulkaninfo` on the RX 6700 XT (see `specs/GPU_CAPABILITIES.md` §8) |
| Date | 2026-08-09 |

### Source material read

- `specs/GPU_CAPABILITIES.md` — capability matrix, §8 extension list, §2/§3 wave & SIMD facts, §10/§11 RDNA2↔RDNA4 decision drivers.
- `specs/vk_amd_extensions.txt` — raw AMD extension name dump (25 AMD extension macros; the §8 "19 total" count covers the *device-enabled* subset).
- `specs/vk_api_structs.txt` — API object/lifetime reference (used for synchronization and object-ownership notes).
- `specs/vk_coop_matrix.txt` — Vulkan spec excerpt (cooperative matrix is a ratified VK_KHR extension).
- `specs/vulkan_layers_whitepaper.txt` — layer configuration reference (used for validation-layer recommendations in §6; not a device extension).
- `specs/vulkan_core.h` — actual header (Vulkan 1.4.357); every struct/field name cited below was verified against it.
- Root `AGENTS.md`, `CMakeLists.txt`, and `src/*/AGENTS.md`, plus `shaders/compile_shaders.ps1` and `shaders/vkblas/coopmatrix/gemm_f32.comp` — to ground the "SDK use" column in what the codebase actually does today.

### Ground-truth caveats

- All *feature availability* and *revision* numbers are from the RX 6700 XT (RDNA2). RDNA4 (RX 9070 XT) numbers are the same driver branch (26.7.1) and are marked where the design assumes them (§2/§3/§11 of GPU_CAPABILITIES), but only the RX 6700 XT was actually enumerated.
- `GPU_CAPABILITIES.md` §8 lists `VK_EXT_push_descriptor` — **that name does not exist**. The header (vulkan_core.h:10684-10686) defines `VK_KHR_push_descriptor` (spec version 2). This document uses `VK_KHR_push_descriptor` everywhere.
- The cooperative-matrix tier is **driver-buggy on 26.7.1** (hard crash in `vkCreateComputePipelines` on `OpCooperativeMatrixMulAddKHR`; see `shaders/vkblas/coopmatrix/gemm_f32.comp:14-27`). It is classified RISKY and must remain env-gated.

---

## 2. Full extension inventory

Classification legend:

| Class | Meaning |
|-------|---------|
| **CORE** | Promoted into Vulkan 1.2/1.3/1.4 core; no opt-in needed. SDK should enable the feature through the `VkPhysicalDeviceVulkan*Features` pNext chain. |
| **EXPLOIT** | Still an extension (or a core feature the SDK must deliberately enable); the SDK should enable and actively use it. |
| **OPTIONAL** | Nice-to-have; enable if present, never required for correctness. |
| **NOT-NEEDED** | Graphics/raytracing/video/legacy — no use in a compute-only inference stack. |
| **RISKY** | Advertised but driver-buggy on 26.7.1; must be gated behind an env var or driver-version check. |

### 2.1 The ~30 most relevant extensions

| Extension | Rev | Class | SDK Use |
|-----------|-----|-------|---------|
| `VK_KHR_cooperative_matrix` | 2 | **RISKY** | Tier-2 GEMM path (`shaders/vkblas/coopmatrix/gemm_f32.comp`, 16×16×16 f32 fragments). **Driver hard-crashes on 26.7.1** in `vkCreateComputePipelines`; already gated behind `VAIT_COOPMATRIX` in `vkblas.c`. Keep gated; un-gate only on a driver that survives `coopMatMulAdd`. |
| `VK_KHR_shader_integer_dot_product` | 1 | CORE (1.3) + EXPLOIT | Quantized GEMM / Q4/Q6 dequant on RDNA2 via dp4a; `shaderIntegerDotProduct` is core 1.3. Verify the per-format `integerDotProduct*Accelerated` flags in `VkPhysicalDeviceVulkan13Properties` before emitting packed int8 dot (RDNA2 accelerates 8/16-bit and 4×8-bit packed; **not** mixed-signedness or 32-bit — GPU_CAPABILITIES §6). |
| `VK_KHR_shader_float16_int8` | 1 | CORE (1.2) | `shaderFloat16` + `shaderInt8` — foundation for all f16/int8 kernels (gemm_f16/bf16, silu_f16, relu_f16, dot_partial_f16, quantize). Enable via `VkPhysicalDeviceShaderFloat16Int8Features` (or `VkPhysicalDeviceVulkan12Features`). |
| `VK_EXT_subgroup_size_control` | 2 | CORE (1.3) + EXPLOIT | **Wave32 vs Wave64 specialization.** On RDNA4 the driver requires `subgroupSize=32` for compute (`requiredSubgroupSizeStages`). Set `requiredSubgroupSize=32` in `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (vulkan_core.h:7770-7774) chained to the stage for the `subgroup/` tier. |
| `VK_EXT_descriptor_indexing` | 2 | CORE (1.2) + OPTIONAL | Variable-count descriptor arrays for batched decode (one weight binding across layers). Not yet used by the SDK's fixed set-0 layouts; reserve for the fused single-CB decode design. |
| `VK_EXT_scalar_block_layout` | 1 | CORE (1.2) + EXPLOIT | std430/scalar layout for **tight quantized weight packing** (Q4/Q6 nibble layouts, f16 scale words). Already relied on by vkmath f16 SSBOs via `layout(scalar)`. |
| `VK_KHR_push_descriptor` | 2 | **EXPLOIT** | Zero-per-op descriptor allocation for the 225-dispatch single-CB decode path. `vkruntime` already resolves `vkCmdPushDescriptorSetKHR` via `vkGetDeviceProcAddr` (vkruntime.c:120-123). Must be listed as a device extension (not core). |
| `VK_KHR_buffer_device_address` | 1 | CORE (1.2) + EXPLOIT | GPU-side pointers for fused kernels (no descriptor re-binds per op), future BDA-based weight tables. Not used yet; enables LDS↔global pointer arithmetic in fused attention. |
| `VK_EXT_pipeline_creation_cache_control` | 3 | CORE (1.3) + EXPLOIT | `pipelineCreationCacheControl` — asynchronous/immediate pipeline compile for lazy first-GEMM creation in vkblas/vkquant. |
| `VK_KHR_pipeline_binary` | 1 | **EXPLOIT** | **Faster startup on RDNA4**: bake 52 SPIR-V pipelines to binaries once, reload on next launch. Feature struct `VkPhysicalDevicePipelineBinaryFeaturesKHR.pipelineBinaries` (vulkan_core.h:13243-13247); consult `VkPhysicalDevicePipelineBinaryPropertiesKHR` (13249-13257). Consumer: vkblas/vkquant/vkmath pipeline caches. |
| `VK_KHR_synchronization2` | 1 | CORE (1.3) | `synchronization2` — `vkCmdPipelineBarrier2` for the fused single-CB barrier graph. |
| `VK_KHR_timeline_semaphore` | 2 | CORE (1.2) + EXPLOIT | Multi-queue pipeline (QF0 graphics, QF1 compute, QF2 transfer) without per-step wait-for-idle. Future. |
| `VK_KHR_maintenance4` | 1 | CORE (1.3) | `maxBufferSize`/`maxMemoryAllocationSize` = 2 GiB caps the vkruntime pooled allocator's per-block growth; must be honored by `vkr_malloc`. |
| `VK_KHR_shader_subgroup_extended_types` | 1 | CORE (1.2) | fp16/int8/int64 subgroup ops (reduce/shuffle on f16) for subgroup-tier reductions. |
| `VK_KHR_16bit_storage` | 1 | CORE (1.1) | `storageBuffer16BitAccess`/`uniformAndStorageBuffer16BitAccess` — f16 SSBO access in gemm_f16/gemv_f16. |
| `VK_KHR_8bit_storage` | 1 | CORE (1.2) | `storageBuffer8BitAccess` — int8/quantized payloads in vkquant SSBOs. |
| `VK_KHR_vulkan_memory_model` | 1 | CORE (1.2) | Memory ordering for atomic-based fused reductions / device-scope atomics. |
| `VK_KHR_compute_shader_derivatives` | 1 | OPTIONAL | Compute `dFdx/dFdy` — only useful for non-LLM kernels; not planned. |
| `VK_EXT_descriptor_buffer` | 1 | OPTIONAL | Future descriptor-buffer path for zero-CPU-cost descriptor updates. Currently a "future optimization" in §8; do not add now (adds a second descriptor paradigm to the codebase). |
| `VK_AMD_shader_core_properties` | 2 | **EXPLOIT** | `wavefrontSize` (64 on RDNA2 / 32 on RDNA4), SIMD/CU counts, VGPR allocation limits (vulkan_core.h:18184-18201). **Add to `vkr_detect_capabilities()` pNext chain** to select the Wave32 variant automatically on RDNA4. |
| `VK_AMD_shader_core_properties2` | 1 | **EXPLOIT** | `activeComputeUnitCount` (vulkan_core.h:18901-18906) — feeds wave-slot math (2,048 Wave32 slots on RDNA4) and occupancy-driven workgroup sizing. |
| `VK_AMD_shader_info` | 1 | OPTIONAL | `vkGetShaderInfoAMD` returns `VkShaderStatisticsInfoAMD.numUsedVgprs`/`numUsedSgprs` (vulkan_core.h:15532-15548) — the only sanctioned way to *measure* occupancy per pipeline. Use in a dev-only profiler to validate Wave32 tile sizes. |
| `VK_AMD_pipeline_compiler_control` | 1 | OPTIONAL | `VkPipelineCompilerControlCreateInfoAMD.compilerControlFlags` (vulkan_core.h:18142-18146). **Today only a single undefined flag exists** — cannot express register limits. Candidate to lobby for a register/occupancy flag (see §5.2). |
| `VK_AMD_buffer_marker` | 1 | OPTIONAL | `vkCmdWriteBufferMarker2AMD` (vulkan_core.h:18109-18110) — GPU timestamps inside the single-CB decode path for per-dispatch profiling. Dev-only. |
| `VK_AMD_device_coherent_memory` | 1 | OPTIONAL/RISKY | `deviceCoherentMemory` (vulkan_core.h:18914-18918) exposes coherent DEVICE_LOCAL types (§5 types 4-7). Potential CPU-peek at GPU results without full readback; semantics are vendor-defined — evaluate before adopting. |
| `VK_AMD_shader_trinary_minmax` | 1 | OPTIONAL | `fmin3/fmax3/fmid3` — could shave a couple ALU ops in clamp-style activations (relu6, hardswish); negligible, not a priority. |
| `VK_AMD_gcn_shader` | 1 | **NOT-NEEDED** | Legacy GCN ISA injection. Conflicts with the "shader-over-binary" ideology (AGENTS.md) and targets pre-RDNA GCN. |
| `VK_AMD_gpu_shader_half_float` | 1 | **NOT-NEEDED** | Superseded by core `shaderFloat16`. |
| `VK_AMD_gpu_shader_int16` | 1 | **NOT-NEEDED** | Superseded by core `shaderInt16`. |
| `VK_AMD_shader_ballot` | 1 | **NOT-NEEDED** | Superseded by core subgroup ballot ops. |
| `VK_AMD_memory_overallocation_behavior` | 1 | **NOT-NEEDED** | We control allocation exactly (pooled allocator); over-allocation hint is irrelevant. |
| `VK_EXT_shader_atomic_float` | 1 | OPTIONAL | fp32 `atomicAdd/min/max` for fused reductions (softmax/attention) — genuine win vs. lock-free int trickery if AMD advertises it; absent from the §8 critical table, so probe it. |

### 2.2 Notable exclusions (present but not needed)

| Extension | Class | Why excluded |
|-----------|-------|--------------|
| `VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations` | NOT-NEEDED | RT hardware is not used by an LLM inference stack. |
| `VK_KHR_video_decode_queue`, `VK_KHR_video_decode_h264/h265`, `VK_KHR_video_maintenance` | NOT-NEEDED | Offline pipeline; no media encode/decode (queue families QF3/QF4, GPU_CAPABILITIES §9). |
| `VK_KHR_swapchain`, `VK_KHR_present_id`, `VK_KHR_present_wait` | NOT-NEEDED | Headless engine; no presentation. |
| `VK_KHR_external_memory_win32`, `VK_KHR_external_semaphore_win32`, `VK_KHR_external_fence_win32` | NOT-NEEDED | Headless, no cross-process/API interop. |
| `VK_EXT_shader_object`, `VK_EXT_graphics_pipeline_library` | NOT-NEEDED | Pipeline-object model is the design; shader objects don't remove the pipeline-cache benefit. |
| `VK_EXT_mesh_shader`, `VK_NV_mesh_shader` | NOT-NEEDED | Graphics. |
| `VK_KHR_dynamic_rendering` | NOT-NEEDED | Compute-only; no render passes. |
| `VK_EXT_memory_budget`, `VK_EXT_memory_priority` | OPTIONAL | Could inform vkruntime pool sizing and pin weights; low priority. |
| `VK_EXT_debug_utils`, `VK_EXT_validation_features`, `VK_EXT_layer_settings` | OPTIONAL | Dev/validation only (see §6 layers note). |

### 2.3 Full AMD set classification (all 25 from `vk_amd_extensions.txt`)

`VK_AMD_anti_lag`, `VK_AMD_display_native_hdr`, `VK_AMD_draw_indirect_count`,
`VK_AMD_mixed_attachment_samples`, `VK_AMD_negative_viewport_height`,
`VK_AMD_rasterization_order`, `VK_AMD_shader_early_and_late_fragment_tests`,
`VK_AMD_shader_explicit_vertex_parameter`, `VK_AMD_shader_fragment_mask`,
`VK_AMD_shader_image_load_store_lod`, `VK_AMD_texture_gather_bias_lod` →
**NOT-NEEDED** (graphics/display/legacy).

`VK_AMD_gpa_interface` → **NOT-NEEDED** (third-party profiler session hooks;
we prefer `VK_AMD_buffer_marker` + `VK_AMD_shader_info`).

**EXPLOIT/OPTIONAL** AMD set: `shader_core_properties` (2), `shader_core_properties2` (1),
`shader_info`, `buffer_marker`, `device_coherent_memory`, `pipeline_compiler_control`,
`shader_trinary_minmax` — see §2.1 rows.

**Exclusions in the header that are NOT in §8's critical table but matter**:
`VK_KHR_maintenance5/6/7/8` (maintenance8 advertised in §6) — `maintenance5`/`maintenance6`
add `VkPipelineCreateFlags2` and descriptor-set improvements; `maintenance8`
(`shaderExpectAssume` in §6) adds `OpExpectKHR`/`OpAssumeKHR` — **EXPLOIT candidate**
for branch-prediction hints in hot GEMM loops (core 1.4).

---

## 3. Feature enablement map

Checklist the SDK **should** apply at `vkCreateDevice`. All names verified in
`specs/vulkan_core.h`. Feature structs go in the `pNext` chain of
`VkDeviceCreateInfo` (which is the *only* way to enable core 1.1+ features —
`pEnabledFeatures` is the legacy 1.0 base struct).

### 3.1 Device extensions to list in `ppEnabledExtensionNames`

| Extension | When | Note |
|-----------|------|------|
| `VK_KHR_push_descriptor` | always (core pipeline) | NOT core in 1.4; required for the single-CB decode path. |
| `VK_KHR_cooperative_matrix` | **only** when `VAIT_COOPMATRIX` set | device extension + feature must both be enabled; see RISKY note. |
| `VK_KHR_pipeline_binary` | always on RDNA4 | feature `pipelineBinaries` must also be set. |
| `VK_AMD_shader_core_properties` | always | property query only, but must be enabled at device create to query via pNext. |
| `VK_AMD_shader_core_properties2` | always | same. |
| `VK_EXT_descriptor_buffer` | never (for now) | deliberate deferral. |

### 3.2 Feature pNext chain (base struct `VkPhysicalDeviceFeatures2`)

| Feature struct | Field(s) to set `VK_TRUE` | SDK component consuming it |
|----------------|----------------------------|----------------------------|
| `VkPhysicalDeviceFeatures` | `shaderInt64`, `shaderInt16`, `shaderFloat64`, `shaderBufferInt64Atomics`, `shaderSharedInt64Atomics`, `tessellationShader=0` (ensure off) | vkblas (gemm_f64, stride int64 push constants), vkmath, vkquant |
| `VkPhysicalDeviceVulkan11Features` | `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`, `storageInputOutput16` | vkblas gemm_f16/gemv_f16, vkmath f16 SSBOs, vkfft f16 |
| `VkPhysicalDeviceVulkan12Features` | `storageBuffer8BitAccess`, `uniformAndStorageBuffer8BitAccess`, `shaderFloat16`, `shaderInt8`, `scalarBlockLayout`, `descriptorIndexing`, `runtimeDescriptorArray`, `bufferDeviceAddress`, `shaderSubgroupExtendedTypes`, `timelineSemaphore`, `vulkanMemoryModel` | vkquant (int8 payloads), vkmath f16, weight packing, fused decode, future BDA, subgroup reductions, multi-queue sync |
| `VkPhysicalDeviceVulkan13Features` | `subgroupSizeControl`, `computeFullSubgroups`, `synchronization2`, `pipelineCreationCacheControl`, `maintenance4`, `shaderIntegerDotProduct`, `dynamicRendering=0` | Wave32 tier (subgroup shaders), single-CB barriers, lazy pipeline creation, 2 GiB caps, dp4a quantized GEMM |
| `VkPhysicalDeviceShaderFloat16Int8Features` | `shaderFloat16`, `shaderInt8` | redundant with Vulkan12 fields; set both for portability on pre-1.2 drivers |
| `VkPhysicalDeviceSubgroupSizeControlFeatures` | `subgroupSizeControl`, `computeFullSubgroups` | Wave32/64 tier selection |
| `VkPhysicalDeviceShaderIntegerDotProductFeatures` | `shaderIntegerDotProduct` | quantized GEMM dp4a |
| `VkPhysicalDeviceDescriptorIndexingFeatures` | `shaderStorageBufferArrayNonUniformIndexing`, `descriptorBindingVariableDescriptorCount`, `runtimeDescriptorArray` | fused decode weight arrays |
| `VkPhysicalDeviceBufferDeviceAddressFeatures` | `bufferDeviceAddress` | GPU-side pointer kernels (future) |
| `VkPhysicalDeviceSynchronization2Features` | `synchronization2` | `vkCmdPipelineBarrier2` in single-CB path |
| `VkPhysicalDeviceMaintenance4Features` | `maintenance4` | allocator cap query |
| `VkPhysicalDeviceTimelineSemaphoreFeatures` | `timelineSemaphore` | multi-queue pipeline |
| `VkPhysicalDevicePipelineBinaryFeaturesKHR` | `pipelineBinaries` | startup binary cache (vkblas/vkquant/vkmath) |
| `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` | `cooperativeMatrix`, `cooperativeMatrixRobustBufferAccess` | tier-2 GEMM — **gate on `VAIT_COOPMATRIX`** |

### 3.3 Per-pipeline enablement (stage-level pNext, not device-level)

| Where | Struct | Field | Use |
|-------|--------|-------|-----|
| Each compute stage, subgroup tier | `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (vulkan_core.h:7770) | `requiredSubgroupSize = 32` | Force Wave32 on RDNA4 for the subgroup tier; leave unset for baseline tier (let driver pick Wave64 on RDNA2). |
| Optional per pipeline | `VkPipelineCompilerControlCreateInfoAMD` | `compilerControlFlags` | future register/occupancy hints (§5.2). |

### 3.4 Property queries (read side, `VkPhysicalDeviceProperties2` pNext)

| Struct | Fields used | Consumer |
|--------|-------------|----------|
| `VkPhysicalDeviceSubgroupSizeControlProperties` | `minSubgroupSize`/`maxSubgroupSize`/`requiredSubgroupSizeStages` | tier selection in `vkr_detect_capabilities()` |
| `VkPhysicalDeviceVulkan13Properties` | `integerDotProduct4x8BitPackedUnsignedAccelerated`, `integerDotProduct8BitSignedAccelerated`, … | gate dp4a kernel variants |
| `VkPhysicalDevicePushDescriptorProperties` | `maxPushDescriptors` (vulkan_core.h:8643-8647) | size the fused decode descriptor set |
| `VkPhysicalDeviceShaderCorePropertiesAMD` | `wavefrontSize`, `simdPerComputeUnit`, `wavefrontsPerSimd`, `vgprsPerSimd`, `maxVgprAllocation` | Wave32-vs-Wave64 + occupancy math (GPU_CAPABILITIES §2) |
| `VkPhysicalDeviceShaderCoreProperties2AMD` | `activeComputeUnitCount` | wave-slot budget (1,280 RDNA2 Wave64 / 2,048 RDNA4 Wave32) |
| `VkPhysicalDeviceMaintenance4Properties` | `maxBufferSize` (= 2 GiB) | vkruntime block growth cap |
| `VkPhysicalDevicePipelineBinaryPropertiesKHR` | `pipelineBinaryInternalCache`, `pipelineBinaryPrecompiledInternalCache` | whether binary persistence is worth it |

---

## 4. Leverage analysis — RDNA capabilities through existing extensions

For each known RDNA feature, *which existing extension/feature unlocks it* and
*exactly how the SDK exploits it*.

### 4.1 Wave32 vs Wave64 (RDNA4: 2,048 Wave32 slots vs 1,024 Wave64; RDNA2: 1,280 Wave64)

- **Extension**: `VK_EXT_subgroup_size_control` (core 1.3 `subgroupSizeControl`).
- **How**: On RDNA4 the driver already mandates `requiredSubgroupSize=32` for
  compute (§3). The subgroup tier must emit `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`
  with `requiredSubgroupSize=32`, giving 2× wave slots vs Wave64 at equal VGPR
  cost (GPU_CAPABILITIES §11). Workgroup sizing should target 4 active waves/CU
  @64 VGPR (Wave32) instead of 2 (Wave64) — i.e. tile GEMM for 32-lane waves.
- **Owner**: vkblas + vkmath pipeline factory (tier selection), fed by
  `VkPhysicalDeviceShaderCorePropertiesAMD.wavefrontSize` + `wavefrontsPerSimd`.
- **RDNA2 caveat**: RDNA2 is Wave64-only (`wavefrontSize=64`); the baseline tier
  stays on Wave64 and the subgroup tier must *not* force 32 there.

### 4.2 int8/int16 dot products (dp4a / dlus) for quantized GEMM

- **Extension**: `VK_KHR_shader_integer_dot_product` (core 1.3).
- **How**: RDNA2/RDNA4 accelerate 8-bit signed/unsigned, 16-bit, and 4×8-bit
  packed dot products — but *not* mixed-signedness or 32-bit (GPU_CAPABILITIES
  §6). `vkquant`/`vkblas` quantized GEMM should use `dot4i8u8packed`/`dot8`
  only for **same-sign** operand pairs; if Q4/Q6 dequant mixes signs, dequantize
  to f16 first or keep the scalar path. Gate per-format on the
  `integerDotProduct*Accelerated` flags in `VkPhysicalDeviceVulkan13Properties`.
- **Owner**: vkblas (new `gemm_i8.comp` tier-2 kernel per `shaders/vkblas/AGENTS.md`),
  vkquant (dequant kernels).

### 4.3 Cooperative matrix 16×16×16 (RDNA4 only)

- **Extension**: `VK_KHR_cooperative_matrix` (rev 2). **RISKY on 26.7.1** —
  the f32 16×16×16 fragment is advertised but `OpCooperativeMatrixMulAddKHR`
  crashes `vkCreateComputePipelines` (uncatchable). 
- **How (when fixed)**: tier-2 GEMM with one Wave32 subgroup per workgroup
  (`coopmatrix/gemm_f32.comp`), A/B staged through LDS then `coopMatMulAdd`.
- **Owner**: vkblas. **Action**: keep `VAIT_COOPMATRIX` gate, add a driver
  version check (e.g. reject < 26.8.x until a known-good driver is found), and
  land the tier-2 test harness before un-gating (harness-first rule).

### 4.4 32 KiB LDS

- **Extension**: none needed (core limit `maxComputeSharedMemorySize` = 32,768).
- **How**: RDNA4 f32 16×16×16 fragment tiles fit comfortably; the fused decode
  design should stage KV-cache and weight tiles in LDS and reuse the pooled
  staging pattern. LDS is the primary bandwidth amplifier — tile GEMM
  dimensions (e.g. 128×128 f32 = 64 KiB → 2× split) must be chosen to fit LDS
  at the active wave count.

### 4.5 VGPR / occupancy

- **Extension**: `VK_AMD_shader_info` (measure), `VK_AMD_shader_core_properties`
  (limits). 
- **How**: after creating each pipeline, read `VkShaderStatisticsInfoAMD.numUsedVgprs`
  (dev-only) and compare against `vgprsPerSimd`/`maxVgprAllocation` to compute
  achieved wave occupancy. Use this to pick tile sizes and, later, register
  hints (§5.2). Wave slot math: `activeComputeUnitCount × simdPerComputeUnit ×
  wavefrontsPerSimd × (64/wavefrontSize)`.

### 4.6 Push descriptors → zero per-op allocation

- **Extension**: `VK_KHR_push_descriptor` (device extension).
- **How**: all 225 decode dispatches bind via `vkCmdPushDescriptorSetKHR` with no
  `vkAllocateDescriptorSets` in the loop. `vkruntime` already resolves the fn
  pointer (vkruntime.c:120-123); the fused decode set layout must fit
  `maxPushDescriptors`.

### 4.7 Pipeline binary → faster startup

- **Extension**: `VK_KHR_pipeline_binary`.
- **How**: on first run, create pipelines once and capture binaries
  (`vkCreatePipelineBinariesKHR`); persist to a cache file; on next launch pass
  `VkPipelineBinaryInfoKHR` in `pPipelineInfo.pNext`. Cuts the 52-SPIR-V
  shader-compile wall from every cold start to once per driver/model hash.
- **Owner**: vkblas/vkquant/vkmath pipeline caches (open-addressing caches in
  each context).

### 4.8 Buffer device address → GPU-side pointers / fused kernels

- **Extension**: `VK_KHR_buffer_device_address` (core 1.2).
- **How**: fused attention/RoPE kernels dereference pointer-holding payloads
  stored in a single "weights table" buffer, avoiding per-op descriptor
  re-binds entirely (complement to push descriptors). Requires the pooled
  allocator to bind buffers contiguously to a single `VkDeviceMemory` (already
  guaranteed: vkruntime binds each buffer to one region — `vkr_malloc`).

### 4.9 scalar_block_layout → tight quantized packing

- **Extension**: `VK_EXT_scalar_block_layout` (core 1.2 `scalarBlockLayout`).
- **How**: Q4_0/Q4_K/Q6_K/IQ4_XS blocks are byte- and nibble-packed with f16
  scale words. `layout(scalar)` removes std430 padding so blocks can be
  addressed by index without a copy. Already used by vkmath f16 SSBOs; extend to
  vkquant SSBOs.

---

## 5. Gap analysis + candidate new extensions

### 5.1 What RDNA exposes that Vulkan cannot reach today (honest list)

| RDNA feature | Reachable through current Vulkan? | Gap |
|--------------|-----------------------------------|-----|
| MFMA / wave matrix (WMMA-style) f32 | Nominally `VK_KHR_cooperative_matrix`, **but driver crashes on 26.7.1** | Usable path does not exist in practice |
| FP8 (E4M3/E5M2) matrix cores (RDNA4 `WMMA-FP8`) | Component types `FLOAT8_E4M3_EXT`/`FLOAT8_E5M2_EXT` exist in `VkComponentTypeKHR` (vulkan_core.h:13529) but no driver advertises an fp8 `VkCooperativeMatrixPropertiesKHR` | No fp8 matrix fragment reachable |
| L2/GL1 cache-control hints (discard, streaming) | **No** Vulkan surface for compute storage | KV-cache streaming writes pollute L2 |
| Explicit VGPR/register budget per pipeline | No — `VK_AMD_pipeline_compiler_control` has only an undefined flag (vulkan_core.h:18138-18141) | Cannot trade occupancy↔spills programmatically |
| Wave-slot occupancy at runtime | Derivable from `VK_AMD_shader_core_properties(2)` + `VK_AMD_shader_info`, but only after pipeline creation | No pre-flight occupancy estimate |
| FP32/FP16 atomics for fused reductions | `VK_EXT_shader_atomic_float` (+`VK_KHR_shader_float16_int8` for the fp16 case) — **exists**, absent from §8 critical table | Probe/adopt, no new extension needed |
| 64-bit atomics | Core `shaderBufferInt64Atomics`/`shaderSharedInt64Atomics` — exists | No gap |
| Fused KV-cache scatter | Plain strided SSBO stores | No hardware scatter; a "scatter instruction" proposal would be a no-op — **rejected** |

### 5.2 Candidate new extensions (3-5 concrete proposals)

#### P1 — `VK_AMD_wave_matrix` (working WMMA/MFMA path)

- **What it exposes**: explicit wave-matrix fragments in GLSL
  (`layout(amd_wave_matrix) mat<N> frag;`) with `amdWaveMatrixMul/Add/Load/Store`
  ops lowered **directly to MFMA**, all operands ordinary GLSL variables
  (no opaque cooperative-matrix values). Mirrors the proven HIP/D3D12-ML
  lowering, bypassing the buggy `GL_KHR_cooperative_matrix` path.
- **Consumes**: vkblas tier-2 GEMM, replacing the crash-prone coopmatrix blob.
- **Sketch**: new shader tier `shaders/vkblas/wavematrix/gemm_f32.comp`; GLSL
  extension `GL_AMD_wave_matrix`; SPIR-V capability `WaveMatrixAMD`;
  `vkblas.c` picks it when `amd_wave_matrix` device extension present and
  `VAIT_COOPMATRIX` set; fallback chain unchanged (subgroup → baseline).
- **Feasibility**: vendor-specific but *real* — it is exactly what AMD's
  closed WMMA backends emit. Cost: no public spec, AMD must ship the
  extension + `glslang` support.
- **Pros**: fixes the tier-2 dead end on RDNA4; keeps shader-over-binary (no ISA blobs).
  **Cons**: speculative (no proposal exists), vendor lock-in, glslang/SPIRV-Tools
  work needed.

#### P2 — `VK_AMD_wave_matrix_fp8` (fp8 matrix fragments)

- **What it exposes**: fp8 (E4M3/E5M2) matrix fragments with **f32 accumulator**,
  fp8→fp16 convert, and wider K per instruction (e.g. 16×16×32 @ fp8).
- **Consumes**: vkquant (fp8 KV cache, fp8 weight quant), vkblas GEMM — a
  direct line to the >80 tok/s decode target at Q8-ish precision.
- **Sketch**: reuse `VkComponentTypeKHR` FLOAT8 enums already in the header;
  add `VkCooperativeMatrixPropertiesKHR` rows for fp8; SPIR-V ops
  `OpCooperativeMatrixMulAddKHR` with fp8 A/B and f32 acc; runtime gate on a
  **new** property so it never runs on 26.7.1.
- **Feasibility**: the API scaffolding already exists (component types are
  defined); the true gap is *driver* support, not the spec. Realistic if AMD
  ships any fp8 matrix path on RDNA4.
- **Pros**: unlocks RDNA4's headline inference feature. **Cons**: depends on
  AMD's fp8 roadmap; same vendor-risk as P1; ideally P1 + P2 ship together.

#### P3 — `VK_AMD_cache_control` (L2/GL1 discard + streaming hints for compute)

- **What it exposes**: shader-level cache ops `amdCacheDiscard(level, addr)`,
  `amdCacheHint(streaming|persistent)` and/or a `VkCommandBuffer`-level
  `vkCmdDiscardRangeAMD` for compute buffers.
- **Consumes**: vkblas GEMM epilogue (discard weight lines after use —
  weights are read once per decode step), vkmath softmax/KV writes (streaming,
  don't pollute L2), vkruntime staging (discard after copy).
- **Sketch**: SPIR-V capability `CacheControlAMD`; GLSL built-ins;
  AMD already has the ISA (`s_dcache_discard`, cache-hint bits) — the work is
  a spec + driver wiring, not new silicon.
- **Feasibility**: medium; real ISA exists, no public Vulkan proposal today.
  **Pros**: direct memory-hierarchy win on the bandwidth-bound decode path.
  **Cons**: vendor-specific, L2 policy effects are hard to validate; keep behind
  a flag and benchmark.

#### P4 — `VK_AMD_shader_register_limits` (occupancy hint via register budget)

- **What it exposes**: pipeline-stage `pNext` `VkShaderRegisterLimitsAMD`
  (`maxVgprsPerWave`, optional `minWavesPerCu`) giving the compiler an explicit
  occupancy↔VGPR trade, like CUDA `__launch_bounds__` / HIP.
- **Consumes**: vkruntime pipeline factory — pass
  `maxVgprs = 64` on RDNA4 Wave32 to guarantee 4 active waves/CU for GEMM
  tiles, or relax for LDS-heavy kernels.
- **Sketch**: extend `VK_AMD_pipeline_compiler_control` (natural home — the
  struct exists, the flag is currently empty) or a sibling extension; read
  back achieved occupancy via `VK_AMD_shader_info`.
- **Feasibility**: highest of the five — reuses existing extension plumbing,
  no new shader model. **Pros**: makes the §4.5 occupancy lever deterministic;
  complements P3/P1. **Cons**: still vendor AMD-only; needs AMD to define the
  flag bits.

#### P5 — `VK_AMD_wave_slot_properties` (pre-flight occupancy query)

- **What it exposes**: a device-level query
  `vkGetPhysicalDeviceWaveSlotPropertiesAMD(pd, waveSize, maxVgprs, &slots)`
  returning the *achievable* active wave slots for a kernel profile before
  pipelines exist.
- **Consumes**: vkruntime arch tuning — decides Wave32 vs Wave64 and tile size
  at runtime-creation time, replacing the current hardcoded tier ladder.
- **Sketch**: driver computes slots from CU count, SIMD width, VGPR pool, LDS —
  the exact math in GPU_CAPABILITIES §2/§11.
- **Feasibility**: low-medium; the data is derivable but the driver has no such
  API. **Pros**: removes guesswork from occupancy tuning. **Cons**: speculative,
  would mostly replicate what `VK_AMD_shader_core_properties` + `shader_info`
  already provide; low ROI relative to P1/P3/P4. **Recommendation: do NOT
  lobby for this as a new extension; fold the math into the SDK instead.**

---

## 6. Recommendations (prioritized)

### Top 3 — do these first

1. **Enable the full Vulkan 1.1–1.3 feature set at `vkCreateDevice` in one
   canonical helper** (vkruntime owns it; every library shares it). Specifically
   `shaderFloat16`, `shaderInt8`, `scalarBlockLayout`, `bufferDeviceAddress`,
   `subgroupSizeControl`, `computeFullSubgroups`, `shaderIntegerDotProduct`,
   `synchronization2`, `timelineSemaphore`, `maintenance4`,
   `pipelineCreationCacheControl`, plus `storageBuffer{8,16}BitAccess` and
   `shaderSubgroupExtendedTypes`. This is the single highest-leverage change:
   every downstream tier/extension decision depends on it, and most of these
   features are already *detected* by `vkr_detect_capabilities()` but not all
   *enabled* in the device pNext chain (tests currently enable a partial subset —
   see `tests/test_vkblas.c:330-366`).
   **Owner**: vkruntime → all of vkblas/vkquant/vkmath/vkfft/vkrand.

2. **Wire Wave32 for the compute tier via subgroup_size_control + AMD core
   properties.** Query `VkPhysicalDeviceShaderCorePropertiesAMD.wavefrontSize`
   and `activeComputeUnitCount` in `vkr_detect_capabilities()`, and on RDNA4
   force `requiredSubgroupSize=32` on subgroup-tier stages. This is the
   prerequisite for the 2,048-wave-slot decode path that hits >80 tok/s.
   **Owner**: vkruntime (detect) + vkblas/vkmath (stage pNext).

3. **Adopt `VK_KHR_pipeline_binary` in the three pipeline caches**
   (vkblas/vkquant/vkmath) with a persisted, keyed-by-(driver,model,shader-hash)
   binary store. Second-order win: add `VK_AMD_buffer_marker` + `VK_AMD_shader_info`
   behind a dev flag to instrument the single-CB decode path. **Owner**:
   vkblas/vkquant/vkmath pipeline factories + a dev profiler.

### Immediate shortlist (in order)

| # | Action | Gate | Component |
|---|--------|------|-----------|
| 4 | Keep `VK_KHR_cooperative_matrix` enabled **only** under `VAIT_COOPMATRIX`, and add a driver-version guard rejecting 26.7.1; add the tier-2 harness before un-gating | `VAIT_COOPMATRIX` + driver ≥ fixed | vkblas |
| 5 | Build `gemm_i8` dp4a tier-2 kernel gated on `integerDotProduct*Accelerated` flags | feature flags (§3.4) | vkblas |
| 6 | Probe `VK_EXT_shader_atomic_float` for fused softmax/attention reductions | extension present | vkmath |
| 7 | Defer `VK_EXT_descriptor_buffer`, `VK_AMD_device_coherent_memory`, timeline semaphores until the single-CB decode path exists | — | vkruntime |
| 8 | Track the five candidate extensions (§5.2); lobby is only justified for **P4** (register limits) and **P3** (cache control) which are real ISA features with low spec cost | — | — |

### Validation-layer note

For dev builds only, enable `VK_LAYER_KHRONOS_validation` (+ optional
`VK_EXT_layer_settings` to toggle `validate_sync`) — per the layers whitepaper
the sync checks catch missing barriers in the fused single-CB path. Never in
release; the SDK's `synchronization2` + timeline-semaphore plan removes the
need for per-step submits.

### Honesty statement

- Everything in §3 is verified against `specs/vulkan_core.h` and the SDK's
  actual source/AGENTS.md contracts.
- Feature-availability rows are RX 6700 XT (RDNA2); RDNA4 rows are design
  assumptions from GPU_CAPABILITIES §2/§3/§11, not measured on this machine.
- §5.2 P1/P2/P3/P5 are **speculative proposals** — none exist in any Vulkan
  registry today; P4 is a natural extension of an existing AMD extension but
  still requires AMD. Only the §5.1 "reachable today" items and the §6 top-3
  are ready to act on now.
