# VK_MaxR_* Extension Proposals — VAiT Vulkan AI Stack

> **Document type**: durable research / design spec
> **Scope**: a family of *proposed* Vulkan device extensions under the stack's
> own vendor tag `MaxR`, following the Vulkan extension documentation format.
> These target the RDNA features that current Vulkan cannot reach (verified gap
> table in `specs/RDNA-EXTENSIONS.md` §5.1 and `specs/EXTENSIONS-UNUSED.md`).
> **Status**: proposals only. None are registered with the Vulkan registry;
> none are implemented in any driver. They are written to the exact
> specification-page format so the research is ready to hand to AMD/Khronos.
>
> Naming scheme follows the Vulkan convention `VK_<vendor-tag>_<ext-name>`;
> `MaxR` is this project's vendor tag (`github.com/Maxritz`).

---

## Vendor & numbering preamble (applies to all pages below)

| Item | Value |
|------|-------|
| Vendor tag | `MaxR` |
| Contact | Maxritz (github.com/Maxritz) |
| Extension number range | `400000–400099` (proposed — the registry vendor range for `MaxR` is unassigned; numbers below are placeholders to be allocated by the Vulkan registry if ever submitted) |
| IP status | No known IP claims; shader-level, no new silicon required beyond what RDNA2/RDNA4 already expose at the ISA level |
| Ratification | None — all **Provisional** |
| Common dependency | `VK_KHR_get_physical_device_properties2` / Vulkan 1.1+ (device extension feature/property queries) |

---

# VK_MaxR_register_limits

- **Name**: `VK_MaxR_register_limits`
- **Type**: Device
- **Registered extension number**: 400001 (proposed)
- **Revision**: 1
- **Dependencies**: Vulkan 1.1, `VK_EXT_pipeline_creation_cache_control`
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

RDNA wave-slot occupancy is a deterministic function of VGPR allocation
(`simdPerComputeUnit × wavefrontsPerSimd` waves per SIMD, at
`vgprsPerSimd / vgprsPerWave` waves). On both RDNA2 and RDNA4 the hardware
allocates VGPRs in fixed granularities (16 registers; `maxVgprAllocation` 256
— see `specs/GPU_CAPABILITIES.md` §2), but Vulkan gives the application **no
way to express a register budget or a minimum active-wave count** per pipeline.
The compiler picks the register count in a black box, so a kernel that *needs*
4 active waves/CU for latency hiding can silently compile to 2 (or spill).

This extension adds a stage-level pNext that tells the compiler the VGPR budget
and/or the minimum wave count to target, mirroring CUDA `__launch_bounds__` /
HIP `__launch_bounds__`, and — via `VK_AMD_shader_info` — lets the application
verify the achieved occupancy afterward.

## New API Interfaces

### New Structures

- `VkShaderRegisterLimitsMaxR`
  - `sType` — `VK_STRUCTURE_TYPE_SHADER_REGISTER_LIMITS_MAXR`
  - `pNext` — must be NULL
  - `maxVgprsPerWave` — upper bound on VGPRs per wave; must be a multiple of
    the device's `vgprAllocationGranularity` (16 on RDNA2/RDNA4) and ≤
    `maxVgprAllocation`.
  - `minWavesPerCu` — compiler must not exceed the register budget that would
    prevent this many active waves per CU. Mutually exclusive intent with
    `maxVgprsPerWave` (whichever is set binds; if both are zero the struct is
    ignored).

### New Constants

| Constant | Value |
|----------|-------|
| `VK_STRUCTURE_TYPE_SHADER_REGISTER_LIMITS_MAXR` | 1000401001 (proposed) |

### New Commands

None — consumed as `pNext` of `VkPipelineShaderStageCreateInfo`.

## Issues

1. **Which limit wins if both are set?** The compiler must satisfy the tighter
   of the two, i.e. `minWavesPerCu` implies `maxVgprsPerWave ≤
   vgprsPerSimd / (simdPerComputeUnit × minWavesPerCu)`. This matches the
   AMD compiler's existing occupancy-first model.
2. **Spills?** A budget below the kernel's floor forces spills; the application
   reads `VkShaderStatisticsInfoAMD.numUsedVgprs` to detect this and relax.
3. **Why not extend `VK_AMD_pipeline_compiler_control`?** That struct exists but
   its flag field is undefined (vulkan_core.h:18142-18146). A dedicated
   structure avoids depending on an undefined flag and is trivially extensible.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

# VK_MaxR_cache_control

- **Name**: `VK_MaxR_cache_control`
- **Type**: Device
- **Registered extension number**: 400002 (proposed)
- **Revision**: 1
- **Dependencies**: Vulkan 1.1
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

RDNA exposes L2/GL1 cache-discard and cache-hint instructions in the ISA
(`s_dcache_discard`, streaming/persistent hint bits), but Vulkan has **no
compute-surface** for them: storage-buffer writes to a streaming KV cache
pollute L2, and GEMM weight lines that are read exactly once per decode step
are retained pointlessly. This extension exposes three levers:

- **Discard**: explicitly drop cache lines after use (weights, staged copies).
- **Streaming hint**: mark a buffer region as read-once / write-once
  (KV cache, activations).
- **Persistent hint**: mark a region as hot across dispatches (weights,
  embedding tables).

## New API Interfaces

### New Commands

- `void vkCmdDiscardRangesMaxR(VkCommandBuffer commandBuffer, VkAccessFlags2 accessFlags, uint32_t rangeCount, const VkBufferRangeMaxR* pRanges)`
  - Hints the driver that the given byte ranges of each buffer are no longer
    needed and may be dropped from L2/GL1 without waiting for eviction.
  - Must be recorded **after** the last command that reads the range, and a
    memory dependency must exist that would have ordered those reads.
- `void vkCmdSetBufferCacheHintMaxR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkCacheHintMaxR hint)`
  - `VK_CACHE_HINT_STREAMING_MAXR` / `VK_CACHE_HINT_PERSISTENT_MAXR` /
    `VK_CACHE_HINT_DEFAULT_MAXR`, applied to a bound SSBO for subsequent
    dispatches.

### New Structures

- `VkBufferRangeMaxR` — `{ VkBuffer buffer; VkDeviceSize offset; VkDeviceSize size; }`

### New Enums

- `VkCacheHintMaxR` — `VK_CACHE_HINT_DEFAULT_MAXR = 0`,
  `VK_CACHE_HINT_STREAMING_MAXR = 1`, `VK_CACHE_HINT_PERSISTENT_MAXR = 2`.

### New Constants

| Constant | Value |
|----------|-------|
| `VK_STRUCTURE_TYPE_BUFFER_RANGE_MAXR` | 1000402001 (proposed) |

### New Shader Capabilities

None at rev 1 — host-only hints (keeps the shader-over-binary ideology; the
GLSL built-in form `maxrCacheHint()` is deferred to a possible rev 2).

## Issues

1. **Is this an optimization or a correctness contract?** An optimization.
   The driver may ignore every call; results must never change. This keeps the
   extension safe to ship behind a flag.
2. **Discard ordering** — the write-read hazard is the caller's responsibility,
   identical to every other cache-coherence primitive in Vulkan.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

# VK_MaxR_wave_matrix

- **Name**: `VK_MaxR_wave_matrix`
- **Type**: Device
- **Registered extension number**: 400003 (proposed)
- **Revision**: 1
- **Dependencies**: Vulkan 1.1, `VK_KHR_shader_float16_int8`
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

RDNA MFMA (matrix-fused-multiply-add) is the hardware the decode path needs for
>80 tok/s, but the only Vulkan route — `VK_KHR_cooperative_matrix` — is
**driver-broken on Adrenalin 26.7.1** (hard crash in `vkCreateComputePipelines`
on `OpCooperativeMatrixMulAddKHR`; see `shaders/vkblas/coopmatrix/gemm_f32.comp`
and `specs/RDNA-EXTENSIONS.md` §4.3). Rather than wait, this extension defines
an explicit, GLSL-level wave-matrix model (the same model AMD's own closed
WMMA backends lower to) where matrix fragments are **ordinary GLSL values**
held in register, not opaque opaque cooperative-matrix objects.

## New API Interfaces

### New GLSL Extension

`GL_MaxR_wave_matrix` — provides:

- `maxrWaveMatrix<N, M, K> mat;` — a matrix fragment type declared
  `layout(maxr_wave_matrix) uniform mat;`
- `maxrWaveMatrixMulAdd(dst, a, b, acc);` — `D = A×B + C` on fragments
- `maxrWaveMatrixLoad(dst, buffer, lda, /*reg*/);` and
  `maxrWaveMatrixStore(buf, src, lda);`

Fragments are lowered directly to `V_MFMA` instructions (f32/f16/bf16
combinations, same K-extension as the ISA).

### New SPIR-V Capability

- `WaveMatrixMaxR` (proposed capability number 6120).

### New Commands / Structures / Enums

None — pure shader-level, mirroring how `VK_KHR_cooperative_matrix` introduces
component types without host API.

## Issues

1. **Why not fix `VK_KHR_cooperative_matrix`?** We cannot; the bug is in the
   driver. This proposal is the *shader-over-binary* (AGENTS.md) compatible
   workaround and is a genuine requirement, not a speculative nicety.
2. **Vendor lock-in?** Yes — but the ISA it targets (MFMA) is RDNA-specific
   anyway; the extension is designed so a `VK_KHR_cooperative_matrix` port is a
   mechanical rewrite once that path is fixed.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

# VK_MaxR_wave_matrix_fp8

- **Name**: `VK_MaxR_wave_matrix_fp8`
- **Type**: Device
- **Registered extension number**: 400004 (proposed)
- **Revision**: 1
- **Dependencies**: `VK_MaxR_wave_matrix` (rev 1), `VK_EXT_shader_float8`
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

RDNA4 has WMMA-FP8 (E4M3/E5M2) matrix units; the component types already exist
in the Vulkan header (`VkComponentTypeKHR` `FLOAT8_E4M3_EXT`/`FLOAT8_E5M2_EXT`,
vulkan_core.h:13529) but no driver exposes an fp8 cooperative-matrix fragment.
This extension extends `VK_MaxR_wave_matrix` fragments to fp8 A/B operands with
an f32 accumulator (16×16×K with K ≥ 32 per instruction) and adds an fp8→fp16
convert for KV-cache read paths.

## New API Interfaces

### New Shader Types / Ops

- `maxrWaveMatrix` with element type `float8_t` (E4M3/E5M2) on A/B, f32
  accumulator.
- `maxrConvertF8toF16` — element-wise convert for KV-cache dequant.

### New SPIR-V Capability

- `WaveMatrixFp8MaxR` (proposed 6121), depends on `WaveMatrixMaxR`.

### New Structures / Enums

- `VkPhysicalDeviceWaveMatrixFp8FeaturesMaxR` — `{ sType, pNext,
  waveMatrixFp8 }` — queried via `vkGetPhysicalDeviceFeatures2`.

## Issues

1. **Precision contract** — the extension mandates f32 accumulation
   (matching the LLM decode accumulator) and documents fp8 operand rounding
   per the existing `VK_EXT_shader_float8` semantics.
2. **Requires silicon** — RDNA4-only (RDNA2 has no WMMA-FP8); RDNA2 devices
   report `waveMatrixFp8 = VK_FALSE`.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

# VK_MaxR_zero_copy_memory

- **Name**: `VK_MaxR_zero_copy_memory`
- **Type**: Device
- **Registered extension number**: 400005 (proposed)
- **Revision**: 1
- **Dependencies**: Vulkan 1.1, `VK_KHR_external_memory` (family-awareness),
  `VK_KHR_buffer_device_address`
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

On ReBAR-enabled discrete GPUs (and on APUs) a `DEVICE_LOCAL | HOST_VISIBLE |
HOST_COHERENT` memory type exists whose whole heap is CPU-writable at
device-local bandwidth — verified on this stack's RX 9070 XT (memory type 2 on
heap 1, the 15.92 GiB VRAM heap). `VK_AMD_device_coherent_memory` exposes the
*semantics* but with vendor-defined scope. This extension makes the zero-copy
path a first-class, portable contract: explicit query for the type(s), explicit
CPU-visibility scope, and an optional coherent range-flush primitive so
non-coherent ReBAR apertures can still be used.

## New API Interfaces

### New Commands

- `VkResult vkGetZeroCopyMemoryTypesMaxR(VkPhysicalDevice device, uint32_t* pMemoryTypeCount, uint32_t* pMemoryTypes)`
  - Returns indices of memory types that are `DEVICE_LOCAL | HOST_VISIBLE`
    **and** whose host mapping exposes the full heap (ReBAR). Fills like
    `vkGetPhysicalDeviceQueueFamilyProperties`.
- `void vkCmdFlushZeroCopyRangesMaxR(VkCommandBuffer commandBuffer, uint32_t rangeCount, const VkBufferRangeMaxR* pRanges)`
  - For non-coherent zero-copy types: orders prior host writes into the GPU's
    view before subsequent reads (a no-op for coherent types).

### New Structures

- `VkPhysicalDeviceZeroCopyMemoryFeaturesMaxR` — `{ sType, pNext,
  zeroCopyMemory }` — feature gate.
- `VkPhysicalDeviceZeroCopyMemoryPropertiesMaxR` — `{ sType, pNext,
  maxZeroCopyAllocationSize, hostCoherent }` — heap span + coherence.

### New Constants

| Constant | Value |
|----------|-------|
| `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_COPY_MEMORY_FEATURES_MAXR` | 1000405001 (proposed) |
| `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_COPY_MEMORY_PROPERTIES_MAXR` | 1000405002 (proposed) |
| `VK_STRUCTURE_TYPE_BUFFER_RANGE_MAXR` | reused from `VK_MaxR_cache_control` |

## Issues

1. **Relationship to `VK_AMD_device_coherent_memory`** — this is the portable,
   explicit-query subset; if that extension is present its coherent types can
   back the query result. The scope definition (full-heap visibility) is the
   value-add over the AMD extension.
2. **Driver support** — on RDNA2 without ReBAR the query returns zero types
   and the stack keeps its staged path (existing `vkr_upload`); no correctness
   risk.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

# VK_MaxR_l2_cache_reservation

- **Name**: `VK_MaxR_l2_cache_reservation`
- **Type**: Device
- **Registered extension number**: 400006 (proposed)
- **Revision**: 1
- **Dependencies**: `VK_MaxR_cache_control` (rev 1)
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

RDNA4 ships 64 MiB of L2 (RDNA2 64 MiB L2 for the 6900-series class; the 6700 XT
has 96 MiB infinity-cache variants — the *capacity* is queried per device, the
mechanism is identical). The decode path has two sharply different working sets:
weights (read once per step — want L2 *not* to cache them) and the KV cache
(streamed, want a guaranteed resident window). This extension lets the stack
reserve a hard L2 carve-out for a buffer heap so hot KV lines are never evicted
by weight traffic.

## New API Interfaces

### New Commands

- `VkResult vkReserveCacheRangeMaxR(VkDevice device, VkReserveCacheRangeInfoMaxR* pInfo, VkReservedCacheRangeMaxR* pReservation)`
- `void vkDestroyReservedCacheRangeMaxR(VkDevice device, VkReservedCacheRangeMaxR reservation)`

### New Structures

- `VkReserveCacheRangeInfoMaxR` — `{ sType, pNext, heapIndex, size,
  accessPattern (streaming|persistent), flags }`
- `VkReservedCacheRangeMaxR` — opaque reservation handle (64-bit).

### New Enums

- `VkCacheAccessPatternMaxR` — `VK_CACHE_ACCESS_PATTERN_STREAMING_MAXR`,
  `VK_CACHE_ACCESS_PATTERN_PERSISTENT_MAXR`.

### New Constants

| Constant | Value |
|----------|-------|
| `VK_STRUCTURE_TYPE_RESERVE_CACHE_RANGE_INFO_MAXR` | 1000406001 (proposed) |
| `VK_OBJECT_TYPE_RESERVED_CACHE_RANGE_MAXR` | 1000406002 (proposed) |

## Issues

1. **`VK_EXT_memory_priority` overlap** — that extension gives per-allocation
   priority hints; this one gives a *hard, deterministic carve-out*. They
   compose (priority = software intent, reservation = hardware commitment).
2. **Portability** — reservations are best-effort; the driver reports the
   granted size in `pReservation`.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

# VK_MaxR_occupancy_query

- **Name**: `VK_MaxR_occupancy_query`
- **Type**: Device
- **Registered extension number**: 400007 (proposed)
- **Revision**: 1
- **Dependencies**: `VK_MaxR_register_limits`, `VK_AMD_shader_core_properties`
- **Deprecation state**: (none)
- **Last modified date**: 2026-08-10

## Overview

Wave-slot occupancy is currently only *measurable after* a pipeline exists
(`VK_AMD_shader_info`). This extension adds a **pre-flight** estimate: given a
pipeline-create template (shader stage, expected VGPR budget from
`VK_MaxR_register_limits`, LDS size, and the device's CU/SIMD geometry from
`VK_AMD_shader_core_properties`), return the achievable active wave count and
the derived tile recommendation. This turns the tier-selection ladder
(Wave32 vs Wave64, tile 16×16 vs 32×8) into a runtime decision instead of a
hardcoded guess.

## New API Interfaces

### New Commands

- `VkResult vkGetPipelineOccupancyMaxR(VkDevice device, const VkPipelineShaderStageCreateInfo* pStage, const VkShaderOccupancyEstimateMaxR* pEstimateInput, VkShaderOccupancyEstimateMaxR* pEstimate)`

### New Structures

- `VkShaderOccupancyEstimateMaxR` — `{ sType, pNext, wavesPerCu, simdOccupancy,
  vgprsPerWave, suggestedTileX, suggestedTileY }`

### New Constants

| Constant | Value |
|----------|-------|
| `VK_STRUCTURE_TYPE_SHADER_OCCUPANCY_ESTIMATE_MAXR` | 1000407001 (proposed) |

## Issues

1. **Honest recommendation** — the audit (`RDNA-EXTENSIONS.md` §5.2 P5)
   recommends folding the occupancy math into the SDK rather than lobbying for
   this as a new extension. It is documented here for completeness; the
   *wave-slot math itself* (from `GPU_CAPABILITIES.md` §2) is being folded into
   `vkr_detect_capabilities()` regardless of whether this extension ever ships.

## Version History

- Rev 1 — 2026-08-10 — initial proposal.

---

## Summary table

| Extension | Gap it closes | Needs new silicon? | Primary consumer |
|-----------|---------------|--------------------|------------------|
| `VK_MaxR_register_limits` | no VGPR/occupancy control | no (ISA VGPR granularity exists) | vkruntime pipeline factory |
| `VK_MaxR_cache_control` | no L2/GL1 discard/hint for compute | no (`s_dcache_discard` exists) | vkblas GEMM epilogue, KV writes |
| `VK_MaxR_wave_matrix` | coopmatrix driver bug → MFMA dead end | no (MFMA exists) | vkblas tier-2 GEMM |
| `VK_MaxR_wave_matrix_fp8` | fp8 matrix fragments unreachable | RDNA4 WMMA-FP8 (exists) | vkquant fp8 KV/weights |
| `VK_MaxR_zero_copy_memory` | ReBAR zero-copy not a first-class contract | no | vkruntime upload path |
| `VK_MaxR_l2_cache_reservation` | KV cache evicted by weight traffic | no (L2 partitions exist) | KV cache heap |
| `VK_MaxR_occupancy_query` | pre-flight occupancy is guesswork | no | vkruntime arch tuning |

## Adoption roadmap (stack-side)

1. Immediately usable without any driver support: the **wave-slot math** behind
   `VK_MaxR_occupancy_query` (fold into `vkr_detect_capabilities`), the
   **zero-copy type query** (already enumerable via `vkGetPhysicalDeviceMemoryProperties`),
   and the **dp4a/bfloat16/atomics** items from `EXTENSIONS-UNUSED.md` §3.
2. Driver-gated: `register_limits`, `cache_control`, `wave_matrix` family —
   each behind a runtime probe + env gate, with a harness before un-gating
   (harness-first rule).
3. Only `VK_MaxR_*` that describe *already-exposed* behavior
   (`register_limits`, `zero_copy_memory`) are candidates for actual
   submission; the rest stay as documented research.

## Honesty statement

- Every RDNA instruction/fact referenced (VGPR granularity, MFMA, fp8 units,
  L2 capacity, `s_dcache_discard`, wave-slot math) is sourced from
  `specs/rdna2_isa.txt`, `specs/rdna4_isa.txt`, and `specs/GPU_CAPABILITIES.md`.
- The extension numbers, struct `sType` constants, and SPIR-V capability numbers
  are **proposed placeholders**; actual allocation requires the Vulkan registry.
- Nothing here is implemented in any driver or SDK as of 2026-08-10.
