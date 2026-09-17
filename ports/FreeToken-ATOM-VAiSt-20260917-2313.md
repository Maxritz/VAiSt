# Port Study: FreeToken-ROCm + ATOM → VAiSt (Vulkan 1.4.357)

Date: 2026-09-17  
Target: C++ Vulkan 1.4.377 / C99 / ASM, Windows 11, no Python/Triton  
Sources:  
- FreeToken-ROCm `python/freetoken/kernel/csrc/gguf/*.cuh` (13 CUDA kernel files, 7814 lines)  
- ATOM `atom/diffusion/attention.py`, `atom/attention/*`, `atom/distributed/*`, `atom/kv_transfer/*` (pure Python, sglang fork)

---

## 1. Repo Classification

### FreeToken-ROCm
- **Compute kernels**: 215 CUDA warp intrinsics (`__shfl`, `__dp4a`, `__ldg`, `__ballot`, `__popc`, …) scattered across `vecdotq.cuh` (45 hits), `mmq.cuh` (2), `moe.cuh` (2).
- The `hip_compat.cuh` shim (if present) aliases CUDA *runtime APIs* only — warp-level intrinsics remain unresolved under ROCm/HIP, let alone Vulkan.
- Kernel architecture:
  - `dequantize.cuh` (486 L) — per-block dequant templates. Byte-unpack + arithmetic. **No warp MMA**. Pure scalar thread loops.
  - `mmq.cuh` (803 L) — `mul_mat_q_*` kernels. **`dp4a` (packed-int4 dot-product) at the inner loop**. Uses `warp_reduce_sum`, `__ldg`, `__shfl_sync`.
  - `mmvq.cuh` (323 L) — `mul_mat_vec_q_*`. Same `dp4a` path.
  - `vecdotq.cuh` (1653 L) — vecdot_q templates + fused `mmq`-style matmul with `dp4a`. **45 intrinsic sites**.
  - `moe.cuh` (1299 L) — MoE dispatch + fused matmul+gate. `dp4a` + `__syncthreads` + `blockIdx` routing.
  - `moe_vec.cuh` (383 L) — MoE vecmat variants, same intrinsic pattern.
- **Summary**: dequant paths = scalar byte logic (adaptable to C99/Vulkan). Matmul/vecdot/moe paths = `dp4a`-dependent (rethink under Vulkan C99; requires `GL_KHR_cooperative_matrix` or delegation to `vaist_blas`).

### ATOM
- Pure Python sglang fork. `attention.py` implements attention dispatch policies (MLA, MLA-HA, chunked, sink). `atom/attention/{cache,metadata,flash_decoding}.py` — orchestration only. No `.cu`/`.cpp`/`.rs` kernel sources exist.
- `atom/kv_transfer/` — Python KV-transfer protocols (lookahead, lookahead-lite). 
- `atom/distributed/` — tensor-parallel sharding + NCCL wrapper.
- **Summary**: policy/routing/TP-dispatch layer, reference only. No compute kernels to port. Reimplement routing in Vulkan compute if needed; reuse vaist_blas for distributed matmul.

---

## 2. Function-to-Target Mapping

Legend:  
- **direct** = byte-identical algorithm; drop-in C99 replacement.  
- **adapt** = same algorithm, minor rewrite (thread→scalar loop, fp16 codec swap).  
- **rethink** = algorithm depends on CUDA warp intrinsic w/o Vulkan native equiv; delegate or redesign.  
- **green** = no blocker. **yellow** = needs design decision. **red** = requires new Vulkan extension or external dep.

### dequantize.cuh — 15 kernels → ADAPT (all green)

| FreeToken CUDA fn              | Block type    | Target                        | Status  | Notes |
|-------------------------------|---------------|-------------------------------|---------|-------|
| `dequantize_block_q4_0`        | block_q4_0    | `vaist_dequantize_f32` q4_0   | direct  | Already ported. |
| `dequantize_block_q4_1`        | block_q4_1    | `vaist_dequantize_f32` q4_1   | direct  | Already ported. |
| `dequantize_block_q5_0`        | block_q5_0    | `vaist_dequantize_f32` q5_0   | direct  | Already ported. |
| `dequantize_block_q5_1`        | block_q5_1    | `vaist_dequantize_f32` q5_1   | direct  | Already ported. |
| `dequantize_block_q8_0`        | block_q8_0    | `vaist_dequantize_f32` q8_0   | direct  | Already ported. |
| `dequantize_block_q8_1`        | block_q8_1    | `vaist_dequantize_f32` q8_1   | direct  | Already ported. |
| `dequantize_block_q2_K`        | block_q2_K    | `vaist_dequantize_f32` q2_K   | direct  | Struct 84 B (pack(1)); ported. |
| `dequantize_block_q3_K`        | block_q3_K    | `vaist_dequantize_f32` q3_K   | adapt   | Bit-scaled hmask decode; scalar loop. |
| `dequantize_block_q4_K`        | block_q4_K    | `vaist_dequantize_f32` q4_K   | adapt   | `get_scale_min_k4` nibbles; ported. |
| `dequantize_block_q5_K`        | block_q5_K    | `vaist_dequantize_f32` q5_K   | adapt   | `get_scale_min_k4` + qh bit-sel. |
| `dequantize_block_q6_K`        | block_q6_K    | `vaist_dequantize_f32` q6_K   | adapt   | ql/qh/scales decode. |
| `dequantize_block_iq2_xxs`     | iq2_xxs       | `vaist_dequantize_f32` iq2_xxs | direct  | Grid lookup; tables ported. |
| `dequantize_block_iq2_xs`      | iq2_xs        | `vaist_dequantize_f32` iq2_xs | direct  | Grid + scales; tables ported. |
| `dequantize_block_iq2_s`       | iq2_s         | `vaist_dequantize_f32` iq2_s  | **defer** | 4-bit packed + qh; return VAIST_UNSUPPORTED. |
| `dequantize_block_iq3_xxs`     | iq3_xxs       | `vaist_dequantize_f32` iq3_xxs | direct  | `iq3xxs_grid[256]`; tables ported. |
| `dequantize_block_iq3_s`       | iq3_s         | `vaist_dequantize_f32` iq3_s  | **defer** | Sign-bit-packed; VAIST_UNSUPPORTED. |
| `dequantize_block_iq1_s`       | iq1_s         | `vaist_dequantize_f32` iq1_s  | defer   | 16-to-8 expansion; defer. |
| `dequantize_block_iq1_m`       | iq1_m         | `vaist_dequantize_f32` iq1_m  | defer   | `iq1s_grid_gpu[2048]`; complex. |
| `dequantize_block_iq4_nl`      | iq4_nl        | `vaist_dequantize_f32` iq4_nl | direct  | `kvalues_iq4nl[16]`; tables ported. |
| `dequantize_block_iq4_xs`      | iq4_xs        | `vaist_dequantize_f32` iq4_xs | defer   | scales_h/scales_l split; deferred. |

**Dequant summary**: 15 implemented in `vaist_quant.c` (q4_0→q6_K + iq2_xxs/iq2_xs/iq3_xxs/iq4_nl). 5 iQuants deferred (iq2_s/iq3_s/iq1_s/iq1_m/iq4_xs) → return VAIST_UNSUPPORTED; add on demand.

### vecdotq.cuh — fused vecdot+matmul → RETHINK (red)

| FreeToken CUDA fn group        | Intrinsic deps         | Target                         | Status  |
|-------------------------------|------------------------|--------------------------------|---------|
| `vecdot_q_int8` family        | `__dp4a` (45 sites)    | vaist_blas `mul_mat_q`         | rethink |
| `vecdot_q4_0_4_1`             | `__dp4a` + `__shfl`    | vaist_blas `mul_mat_q`         | rethink |
| `vecdot_q3_K` / `q4_0_4_1`    | `__dp4a` + `__ldg`     | vaist_blas `mul_mat_q`         | rethink |
| `gemv_q4_0_4_1`               | `__shfl_sync` reduce   | vaist_blas vec path            | rethink |

These fuse dequant + dot into one kernel to keep partial sums in registers. Under Vulkan C99 there's no `dp4a`. Options:
1. **Delegation**: dequant to fp16/bf16 via `vaist_quant`, then dense matmul via `vaist_blas` (VULKAN_TILE path). 2× bandwidth, simpler.
2. **GL_KHR_cooperative_matrix**: requires Vulkan 1.4.377 + `shaderIntegerDotProduct` extension (not guaranteed vendor-wide).

**Decision**: delegate to (vaist_quant → vaist_blas) for now. Mark red-path `dp4a` sites as `vaist_quant_dequantize → vaist_blas_mul_mat` split. The fused kernel is a future optimization if `GL_KHR_cooperative_matrix` is available on target hardware.

### mmq.cuh / mmvq.cuh — quantized matmul → RETHINK (red)

| Kernel                        | Intrinsic           | Target                     | Status |
|------------------------------|---------------------|----------------------------|--------|
| `mul_mat_q4_0`, `mul_mat_q8_0` | `__dp4a`, `__shfl`  | `vaist_blas` (dense)       | rethink |
| `mul_mat_vec_q4_0`            | `__shfl_sync`       | `vaist_blas vec`           | rethink |

`vaist_blas` already covers: VULKAN_DOT→VULKAN_TILE→MATMUL_FREE→SIMD→SCALAR dispatch + ternary/binary MatMul-free paths. The dense matmul after dequant lands on the same ladder.

### moe.cuh / moe_vec.cuh — MoE dispatch → RETHINK (red)

| Kernel                        | Intrinsic           | Target                          | Status |
|------------------------------|---------------------|---------------------------------|--------|
| `moe` (256×128 tile)         | `dp4a`, `__syncthreads`, blockIdx routing | vaist_blas tiled + policy shim | rethink |
| `moe_vec_q4_0`               | `dp4a`              | vaist_blas (dense fallback)     | rethink |

**Routing policy** (top-k gate, expert parallelism) lives in Python (`freetoken/.../moe.py`). ATOM's attention routing (`atom/attention/{flash_decoding,cache}.py`) is the policy reference — reimplement gate dispatch in a Vulkan compute shader or reuse existing vaist dispatch. The compute body (matmul per expert) delegates to vaist_blas.

### ATOM attention dispatch — POLICY / direct

| ATOM component                | Target                          | Status |
|------------------------------|---------------------------------|--------|
| `attention.py` dispatch policies | Vulkan compute dispatcher     | direct (C++) |
| `kv_transfer/lookahead.py`   | Vulkan KV-cache copy/push       | adapt   |
| `distributed/tensor_parallel` | vaist_tp (not yet in scope)    | defer   |

No kernels — pure orchestration. Port as C++ Vulkan compute dispatcher calling `vaist_blas` + `vaist_quant`.

---

## 3. Debug-Hypothesis Map

| Target layer        | Risk                                | Hypothesis test                                    |
|---------------------|-------------------------------------|----------------------------------------------------|
| fp16 codec          | Denorm/normal branch mismath        | Self-test q4_0/q8_0/q8_1 round-trip (PASS ✅ 2026-09-17) |
| iQuant q4_0/nl tables | Grid-table endianness / sign flip   | Compare against llama.cpp `ggml-common.h` tables (bit-exact ✅) |
| K-quant q2_K/q3_K   | `get_scale_min_k4` nibble decode    | Add q3_K/q2_K round-trip to self-test (TODO)        |
| Matmul `dp4a` replace | Precision loss fp16 accum         | Compare vaist_blas fp16 output vs CUDA fp32 (TODO) |
| MoE routing         | Expert-lb imbalance                 | Synthetic 64-expert load test (TODO)                |

Current verified: q4_0, q4_1, q5_0, q8_0, q8_1 quantize+round-trip PASS (maxerr within nibble/byte rounding). Block sizes match GGUF pack(1) layout.

---

## 4. vaist_quant Status (verified)

`vaist_quant.c` + `vaist_quant.h` (enum VAIST_Q2_K…VAIST_Q8_1 + ternary/binary) + `vaist_quant_tables.h` (iQuant grids + kvalues_iq4nl + ksigns_iq2xs + kmask_iq2xs). MSVC BuildTools 14.44 self-test PASS. Linux copy synced (no gcc in env).

Dequant implemented: q8_0, q8_1, q4_0, q4_1, q5_0, q5_1, q6_0, q2_K, q3_K, q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq4_nl (+ q8_0/q8_1/q4_0/q4_1 quantize). Deferred: iq2_s, iq3_s, iq1_s, iq1_m, iq4_xs.

---

## 5. Action Items

1. **vaist_quant** — add q2_K/q3_K/q4_K/q5_K/q6_K round-trip to self-test; port iq2_s/iq3_s/iq4_xs when FreeToken MoE hits those weights.
2. **vaist_blas** — `mul_mat_q` entry point: dequant→dense via existing VULKAN_DOT/TILE ladder. Add `GL_KHR_cooperative_matrix` path as optional fast lane.
3. **MoE routing shim** — C++ Vulkan dispatcher mirroring ATOM's `atom/attention/flash_decoding.py` attention dispatch + FreeToken MoE top-k gate.
4. **Port study doc** — this file lives at `ports/FreeToken-ATOM-VAiSt-20260917-2313.md`.
