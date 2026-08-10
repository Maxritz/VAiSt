# HIP-Vulkan Bridge: Indirect Leverage Strategy

> **Document type**: durable research / design spec  
> **Scope**: Leveraging HIP as an intermediate layer to bypass Vulkan ICD gaps on AMD GPUs  
> **Status**: active integration — Phase 1 (Bridge Design) complete  

---

## Module Classification Table

All 16 RDNA4 ISA gaps classified by target module:

| Gap # | ISA Feature | GAP_ANALYSIS § | Module | HIP Backend | Status |
|-------|------------|----------------|--------|-------------|--------|
| 1 | `S_DCACHE_INV` | §1 Cache Control | **VKISA** | ❌ No | Stub |
| 2 | Temporal hints TH[2:0] | §1 Cache Control | **VKISA** | ❌ No | Stub |
| 3 | `S_GET_BARRIER_STATE` | §2 Barriers | **VKISA** | ❌ No | Stub |
| 4 | Trap barriers | §2 Barriers | **VKISA** | ❌ No | Stub |
| 5 | `V_WMMA_*` wave matrix | §3 WMMA/SWMMAC | **VKBLAS** | ✅ hipblasLtMatmul | Active |
| 6 | `V_SWMMAC_*` sparse | §3 WMMA/SWMMAC | **VKBLAS** | ✅ hipblasLtMatmul (sparse) | Active |
| 7 | 4:2 structured sparsity | §3 WMMA/SWMMAC | **VKBLAS** | ✅ hipblasLtMatmul | Active |
| 8 | `V_PERMLANE_*` | §6 DPP/PERMLANE | **VKISA** | ❌ No | Stub |
| 9 | `V_DOT4_*_FP8/BF8` | §5 Dot Prod | **VKBLAS** | ✅ hipblasLtMatmul (fp8/bf8) | Active |
| 10 | `BUFFER_ATOMIC_MIN/MAX_NUM_F32` | §5 Atomics | **VKISA** | ❌ No | Stub |
| 11 | `IMAGE_ATOMIC_MIN/MAX_F32` | §5 Atomics | **VKISA** | ❌ No | Stub |
| 12 | `PK_ADD_F16/_BF16` | §5 Atomics | **VKISA** | ❌ No | Stub |
| 13 | `DEALLOC_VGPRS` | §4 VGPR | **VKISA** | ✅ hipify auto | Active |
| 14 | Context switch state | §7 Context | **VKISA** | ❌ No | Stub |
| 15 | `V_CVT_*_FP8/BF8` | §7 FP8 Conv | **VKMath** | ✅ Custom HIP kernel | Active |
| 16 | `V_CVT_PK_*_FP8/BF8` | §7 FP8 Conv | **VKMath** | ✅ Custom HIP kernel | Active |

### Module Summary

| Module | Gaps Bridged | HIP Backend File | Compile Flag |
|--------|-------------|------------------|--------------|
| **VKBLAS** | 5, 6, 7, 9 | `vkblas_hip.c` | `--DNO_HIP` to disable |
| **VKMath** | 15, 16 | `vkmath_hip.c` | `--DNO_HIP` to disable |
| **VKStream** | BAR zero-copy | `vkstream_hip.c` | `--DNO_HIP` to disable |
| **VKISA** | 1, 2, 3, 4, 8, 10, 11, 12, 13, 14 | `vkisa_hip.c`, `vkisa_stub.c` | `--DFORCE_STUB` to disable |

---

## Bridge Architecture (Refined)

```
App Code
  │
  ├─ vkXXX_* API (stable interface, identical regardless of backend)
  │
  └─ vkXXX-backend-dispatch
      ├── vkXXX_vulkan.c  (primary: uses Vulkan extensions)
      ├── vkXXX_hip.c     (fallback: HIP APIs bridge the gap)
      └── vkXXX_stub.c    (fallback: no-op stubs when neither works)

Runtime selection per module:
├── Check Vulkan extension availability
├── If missing/unavailable → use HIP backend
├── If HIP unavailable → use stub
└── Compile flag --DNO_HIP forces Vulkan-only (stub on gap)
```

---

## HIP-Vulkan Bridge Details (per module)

### VKBLAS HIP Backend (`vkblas_hip.c`)
**Gaps covered**: #5 (WMMA), #6 (SWMMAC), #9 (FP8 dot)

```c
// When VK_KHR_cooperative_matrix crashes on driver 26.7.1:
vkblas_hip_gemm_f16(ctx, m, n, k, a, b, c, alpha, beta);
// Internally calls:
// hipblasLtMatmul(handle, HIPBLASLT_MATMUL_CONFIG_F16, ...)
```

- Loads `hipblasLtMatmul` via `hipGetProcAddress`
- Creates HIP buffer from existing VkBuffer device pointer
- Runs hipblasLt matmul, signals completion via Vulkan fence
- Returns control to Vulkan stream

### VKMath HIP Backend (`vkmath_hip.c`)
**Gaps covered**: #15 (FP8 CVT), #16 (BF8 CVT)

```c
// When SPIR-V has no OpConvert for FP8:
vkmath_hip_cvt_f32_to_fp8(ctx, input, output, n);
// Internally runs HIP kernel:
// hipModuleLoad + hipModuleGetFunction("cvt_f32_fp8")
```

### VKISA HIP Backend (`vkisa_hip.c`)
**Gaps covered**: #13 (DEALLOC_VGPRS, auto-managed by hipify)

### VKStream HIP Backend (`vkstream_hip.c`)
**Purpose**: Zero-copy BAR allocation when `VK_EXT_external_memory_host` missing

---

## Build System

### CMakeLists.txt (per-module options)
```cmake
# HIP backend toggles — each can be disabled independently
option(VKBLAS_USE_HIP "Enable HIP backend for VKBLAS (coop matrix fallback)" ON)
option(VKMath_USE_HIP "Enable HIP backend for VKMath (fp8/bf8 conversion)" ON)
option(VKSTREAM_USE_HIP "Enable HIP backend for VKStream (zero-copy BAR)" ON)
option(VKISA_USE_HIP "Enable HIP backend for VKISA" OFF)

# Master switch — disable all HIP backends
option(NO_HIP "Disable all HIP backends (-DNO_HIP=ON)" OFF)

if(NO_HIP)
    set(VKBLAS_USE_HIP OFF)
    set(VKMath_USE_HIP OFF)
    set(VKSTREAM_USE_HIP OFF)
    set(VKISA_USE_HIP OFF)
endif()

# Find HIP if any backend needs it
set(HIP_NEEDED OFF)
if(VKBLAS_USE_HIP OR VKMath_USE_HIP OR VKSTREAM_USE_HIP OR VKISA_USE_HIP)
    set(HIP_NEEDED ON)
endif()

if(HIP_NEEDED AND NOT NO_HIP)
    find_package(HIP QUIET PATHS "$ENV{ROCM_ROOT}/..")
    if(NOT HIP_FOUND)
        message(WARNING "HIP requested but not found — falling back to stubs")
    endif()
endif()
```

### Runtime Selection (in vkstream.c / vkblas.c)
```c
// Example: VKBLAS backend selection
VKBLAS_BACKEND select_gemm_backend(vkblas_context_t* ctx) {
    if (check_coop_matrix_crashes()) {
        #if defined(USE_HIP) && (VKBLAS_USE_HIP==1)
        if (hipblasLt_available()) {
            LOG_WARN("Coop matrix crashes — using HIP backend");
            return VKBLAS_BACKEND_HIP_BLASLT;
        }
        #endif
    }
    return VKBLAS_BACKEND_VULKAN_COOPMATRIX;  // or VKBLAS_BACKEND_VULKAN_BASELINE
}
```

---

## Usage

### Building with HIP bridges
```bash
# With HIP (recommended on AMD)
cmake -B build -DUSE_HIP=ON ..

# Without HIP (Vulkan-only — will hit gaps)
cmake -B build -DNO_HIP=ON ..

# Force HIP even if Vulkan ext available (for testing)
cmake -B build -DFORCE_HIP_VKBLAS=ON ..
```

### Runtime log output
```
vkblas: WARN: Vulkan VK_KHR_cooperative_matrix detected crash on init
vkblas: INFO: Falling back to HIP backend (hipblasLt)
vkmath: INFO: FP8 conversion not available in Vulkan, using HIP kernel
vkstream: INFO: VK_EXT_external_memory_host missing, using hipHostMalloc
```
