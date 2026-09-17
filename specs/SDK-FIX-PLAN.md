# SDK Fix Plan — Flowcharts, Truth Tables, Function Mappings

Covers the five remaining gaps, each traced with a directed flow chart and a
truth table before any code is written (per root AGENTS.md: truth table before
code, harness-first verification).

Execution order: Tasks 1–4 are disjoint per-library and run in parallel (one
agent each). Task 5 (VKRuntime refactor) touches every library and runs LAST,
after 1–4 merge, to avoid file conflicts.

---

## Task 1 — vkblas_bgemm (bf16): stub → real

### FLOW
```
vkblas_bgemm(ctx, cmd, transA, transB, m,n,k, alpha16, A, lda, B, ldb, beta16, C, ldc, D, ldd)
  │   alpha16/beta16 are const uint16_t* (bfloat16 bits)
  ▼
vkblas_gemm_common(..., dtype=VKBLAS_DTYPE_BF16, ...)
  │   bf16 alpha/beta → f32 via bits<<16 for push constants
  ▼
vkblas_ensure_pipeline(ctx, BF16, tA, tB, is_strided, &pipeline)
  ▼
vkblas_select_spirv(ctx, BF16, tier, &size)     [tier walk: coop → subgroup → baseline]
  ├── baseline: gemm_bf16.comp EXISTS? ──YES──► create pipeline, cache, dispatch  ✅
  └── NO ────────────────────────────────────► return VK_ERROR_FEATURE_NOT_PRESENT ✗ (current)
```

### TRUTH TABLE
| Condition | Expected | Current | After fix | PASS? |
|-----------|----------|---------|-----------|-------|
| gemm_bf16.comp present + baseline tier | blob found → real GEMM | FEATURE_NOT_PRESENT | real GEMM | ✅ |
| alpha/beta are bf16 (uint16) | converted f32 in push consts | n/a | bits<<16 → f32 | ✅ |
| f32 accumulation, bf16 output | result within 1e-2 | — | 1e-2 vs CPU | ✅ |

### Function mapping
- `vkblas_bgemm` / `vkblas_bgemm_strided_batched` → `vkblas_gemm_common(..., BF16, ...)`
- new `shaders/vkblas/baseline/gemm_bf16.comp` → `vkblas_spv_baseline_gemm_bf16` (uint16_t storage, `bits<<16`↔f32, f32 accumulate, scalar layout)

---

## Task 2 — Real cooperative-matrix GEMM (driver-blocked)

### FLOW
```
vkblas_create_context
  → vkblas_init_capabilities → has_coop_matrix (coop_features.cooperativeMatrix)
  → active_tier = COOPMATRIX, use_coopmat = env("VAIT_COOPMATRIX")? (default 0)
vkblas_sgemm(tier=COOPMATRIX)
  → select coopmatrix_gemm_f32 (REAL coopMatMulAddKHR version)
  → vkCreateComputePipelines
       ├── AMD 26.7.1 driver → C++ exception 0xE06D7363 (uncatchable HARD CRASH) ✗
       └── fixed/other driver → SUCCESS → real matrix-core GEMM ✅
  └── use_coopmat == 0 → shared-memory fallback blob (current, SAFE) ✅
```

### TRUTH TABLE
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| driver 26.7.1 + coopMatMulAddKHR pipeline | works | hard crash in vkCreateComputePipelines | ✗ (driver bug) |
| coopmat path disabled (default) | shared-mem GEMM, correct | correct (tested) | ✅ |
| `VAIT_COOPMATRIX=1` on a fixed driver | real coopmat GEMM | code-complete, SPIR-V-valid, untestable here | ⚠ guarded |

### Function mapping
- `vkblas_init_capabilities`: set `use_coopmat = (getenv("VAIT_COOPMATRIX")!=NULL && has_coop_matrix)`
- `vkblas_select_spirv`: coopmatrix blob returned only when `use_coopmat && tier==COOPMATRIX`
- `shaders/vkblas/coopmatrix/gemm_f32.comp`: REAL coopmat shader (coopMatLoad/Store/MulAdd); shared-mem path removed only when real path is active

**Note:** cannot be GPU-verified on this machine — the driver hard-crashes on any
`coopMatMulAddKHR` pipeline. Deliverable is a guarded, compiled, correct path + docs.

---

## Task 3 — VKFFT 2D (separable N×N)

### FLOW
```
vkfft_create_plan_2d(pd, dev, n, &plan)          // n power of two, n≤1024
  → internal context (reuse create path), store n, is_2d=1, log2n

vkfft_execute_2d_f32(plan, cmd, in, out)
  ├── pass 1 (ROWS): dispatch n workgroups; wg r FFTs row r: read in[r*n+k], write out[r*n+k]
  │       shader mode=0: base = wgID*n, stride 1
  ├── vkCmdPipelineBarrier(COMPUTE→COMPUTE, SHADER_WRITE→SHADER_READ)
  └── pass 2 (COLS): dispatch n workgroups; wg c FFTs col c: read out[r + c*n], write out[r + c*n]
          shader mode=1: base = wgID, stride n  (in-place on `out`)
```

### TRUTH TABLE
| Condition | Expected | PASS? |
|-----------|----------|-------|
| N×N forward = FFT rows then FFT cols | == CPU 2D DFT (1e-3) | ✅ |
| mode=0 load index base+k (stride 1) | row correct | ✅ |
| mode=1 load index base+k*n (stride n) | col correct | ✅ |
| barrier between passes | no stale reads | ✅ |

### Function mapping
- `vkfft_create_plan_2d` → internal create(n, is_2d=1)
- `vkfft_execute_2d_f32` → `vkfft_execute_dir_2d(plan, cmd, in, out, fwd, ROW)` + barrier + `...COL`
- push constant gains `mode` (0=row,1=col); shader `fft_f32.comp` gains mode + `gl_WorkGroupID.x` base; new `fft_2d_f32.comp` (or mode in fft_f32.comp)

---

## Task 4 — VKQuant Q4_K / Q6_K / IQ4_XS dequant + forward quantization

### FLOW
```
vkquant_dequant_q4k_f32(ctx, cmd, num_blocks, in, out)      // 256 elems/block
vkquant_dequant_q6k_f32(...)
vkquant_dequant_iq4xs_f32(...)
  → vkquant_ensure_pipeline(kernel, ...) → dispatch ceil(256*blocks/256)
  → each thread: block=idx>>8, lane=idx&255; decode via EXACT ggml layout; write out[idx]

vkquant_quantize_q8_0_f32(ctx, cmd, num_blocks, in_f32, out_q8_0)   // 32 elems/block
vkquant_quantize_q4_0_f32(...)
  → per block: scale=max(|x|)/qmax; q=round(x/scale); pack; write our f32-scale format
```

### TRUTH TABLE (formats — authoritative = ggml-common.h)
| Format | Block bytes | Scale | Dequant rule (bit-exact GPU==CPU required) |
|--------|------------|-------|-------------------------------------------|
| Q4_K | 144 | f16 d,dmin | 16 packed scales (8 low + 4 high nibbles), 128 nibble qs; ggml q4_K |
| Q6_K | 210 | f16 d | 128 4-bit ql + 64 2-bit qh + 16 int8 scales; ggml q6_K |
| IQ4_XS | 146 | f16 d | 16 scales + 128 nibble qs + iq4nl LUT; ggml iq4_xs |
| Q8_0 (fwd) | 36 | f32 | d=max/127; q=round(x/d)∈[-127,127] |
| Q4_0 (fwd) | 20 | f32 | d=max/8; q=round(x/d)+8 ∈[0,15]; pack nibbles |

### Function mapping
- dequant: `vkquant_dequant_q4k_f32`, `vkquant_dequant_q6k_f32`, `vkquant_dequant_iq4xs_f32`
- quant: `vkquant_quantize_q8_0_f32`, `vkquant_quantize_q4_0_f32`
- CPU references in the test implement the same ggml-common.h rules → bit-exact comparison

---

## Task 5 — Refactor five per-library contexts onto VKRuntime

### FLOW (current → target)
```
CURRENT (duplicated ×5: vkmath, vkblas, vkquant, vkrand, vkfft)
  vkL_create_context:
    [dup] capability detection (features2/properties2 pNext chains)
    [dup] push-desc fn load (vkGetDeviceProcAddr)
    [dup] descriptor pool create
    [dup] pipeline layout create (set layout + push-constant range)
    [dup] pipeline cache create
    [dup] pipeline cache array (open addressing)

TARGET
  vkL_create_context:
    → caps = vkr_detect_capabilities(pd, dev)        // single implementation
    → vkr_create_descriptor_pool(dev, ...)
    → vkr_create_pipeline_layout(dev, set_layout, pc_range, ...)
    → vkr_create_pipeline_cache(dev, ...)
    → [lib-specific] set layout + push constants stay in the lib (they differ)
    → [lib-specific] pipeline cache array stays in the lib
```

### TRUTH TABLE
| Aspect | Before | After | Regressions? |
|--------|--------|-------|--------------|
| Public create_context signature | per-lib (pd, dev) | unchanged | none |
| Capability detection | 5 copies | 1 (vkr_detect_capabilities) | behavior identical |
| Descriptor pool / layout / cache creation | 5 copies | vkr_* helpers | behavior identical |
| Push-desc load | 5 copies | vkr helper | identical |
| Tests | 7 passing | must stay 7 passing | harness-first |

### Function mapping
- NEW `vkr_detect_capabilities(VkPhysicalDevice, VkDevice, VkRuntimeCaps*)` → struct
  {has_shader_int64, has_subgroup, has_coop_matrix, has_push_descriptor, subgroup_size,
   max_workgroup_size[3], push_desc_fn, arch_index, arch_name}
- NEW `vkr_create_descriptor_pool(dev, max_sets, ssbo_count, *pool)`,
  `vkr_create_pipeline_layout(dev, set_layout, push_range_count, ranges, *layout)`,
  `vkr_create_pipeline_cache(dev, *cache)`
- Each lib's `vkL_init_capabilities` / context-creation body calls the vkr_* helpers
  instead of inline Vulkan; lib-specific set-layout + push-constant range stay local.
- Verify: all 7 ctest suites still pass unchanged.
