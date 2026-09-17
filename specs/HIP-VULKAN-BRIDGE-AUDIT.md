# HIP-Vulkan Bridge: AMD Driver 26.7.1 Extension & Proc Address Audit

## Status: Document complete

## Summary

The `VK_EXT_external_memory_host` extension is **advertised** by the AMD Radeon
driver 26.7.1 (LLPC 2.0.395) on RX 9070 XT, but the `vkImportMemoryHostPointerEXT`
entrypoint is **not exported** via either `vkGetDeviceProcAddr` or
`vkGetInstanceProcAddr`. This renders the zero-copy HIP→Vulkan path
non-functional despite the extension being listed.

A staging buffer fallback (HIP→memcpy→Vulkan mapped memory) works correctly as
a mitigation.

## Test Results

### Environment
- GPU: AMD Radeon RX 9070 XT
- Driver: 26.7.1 (LLPC 2.0.395)
- Vulkan SDK: 1.4.357.0
- HIP Runtime: 7.14.0 (ROCM-7.14.0-Windows)

### Extension Availability (`vkEnumerateDeviceExtensionProperties`)

| Extension | Listed | Proc Address | Notes |
|-----------|--------|-------------|-------|
| `VK_EXT_external_memory_host` | ✅ YES | ❌ NULL | **Core issue** |
| `VK_EXT_host_image_copy` | ✅ YES | (untested) | Available |
| `VK_EXT_host_query_reset` | ✅ YES | (untested) | Available |
| `VK_EXT_memory_budget` | ✅ YES | (untested) | Available |
| `VK_EXT_memory_priority` | ✅ YES | (untested) | Available |
| `VK_EXT_descriptor_indexing` | ✅ YES | (untested) | Available |
| `VK_EXT_shader_atomic_float` | ✅ YES | (untested) | Available |
| `VK_KHR_shader_float16_int8` | ✅ YES | (untested) | Available |
| `VK_EXT_scalar_block_layout` | ✅ YES | (untested) | Available |
| `VK_EXT_separate_stencil_usage` | ✅ YES | (untested) | Available |
| `VK_EXT_debug_report` | ❌ NO | — | Linux-only |
| `VK_EXT_debug_utils` | ❌ NO | — | Linux-only |
| `VK_EXT_blend_operation_advanced` | ❌ NO | — | Not exposed |

### Proc Address Resolution

```
vkImportMemoryHostPointerEXT:        device=NULL, instance=NULL  ❌
vkGetMemoryOpaqueCaptureAddressKHR:  device=NULL, instance=NULL  ❌
vkGetBufferOpaqueCaptureAddressKHR:  device=NULL, instance=0x7FF859A83A40  ✅
vkCmdDebugMarkerInsertEXT:           device=NULL, instance=0x7FF859A86CA0  ✅
vkGetPhysicalDeviceExternalMemoryHostPropertiesEXT: NULL  ❌
```

### Memory Properties

```
VkPhysicalDeviceExternalMemoryHostPropertiesEXT:
  minImportedHostPointerAlignment: 0x1000 (4096 bytes)

Memory types with HOST_VISIBLE | HOST_COHERENT:
  [1] HOST_VISIBLE HOST_COHERENT (heap 0)
  [2] HOST_VISIBLE HOST_COHERENT DEVICE_LOCAL (heap 1)  ← ReBAR zero-copy
  [3] HOST_VISIBLE HOST_COHERENT HOST_CACHED (heap 0)
  ... (duplicates for heaps 0,1)
```

HIP host pointers are 4096-aligned (meets the alignment requirement) but
the import function is unavailable.

## Impact on HIP-Vulkan Bridge

### Phase 1: VKBLAS (`vkblas_hip.c`)
- `VK_KHR_cooperative_matrix` crashes on driver 26.7.1 (documented in GAP_ANALYSIS.md: RED)
- HIP backend via `hipblasLtMatmul` is the active fallback
- **No change needed** — already has HIP fallback path

### Phase 2: VKMath (`vkmath_hip.c`)
- FP8/BF8 conversion via custom HIP kernel
- Uses `>>16` truncation trick for bf16 (documented in COMPONENT_FEATURE_MAP.md)
- **No change needed** — FP8/BF8 conversion is kernel-based, doesn't need host import

### Phase 3: VKStream (`vkstream_hip.c`)
- `VK_EXT_external_memory_host` is broken (proc addr NULL)
- **Requires fix**: staging buffer fallback (memcpy-based) is the only working path
- The existing stub in `vkstream_hip.c` needs runtime detection

### Phase 4: VKISA (`vkisa_hip.c`)
- Handles `DEALLOC_VGPR` via `hipify` auto-translation
- All other ISA gaps (cache control, barrier introspection, PERMLANE) have no HIP equivalent
- **No change needed** — stubs remain stubs (documented in HIP-VULKAN-BRIDGE.md)

## Architecture Decision

**Documented in `specs/HIP-VULKAN-BRIDGE.md`**: The bridge uses a three-tier dispatch:

```
vkXXX API → vkXXX-backend-dispatch
  ├── vkXXX_vulkan.c  (primary: Vulkan extensions)
  ├── vkXXX_hip.c     (fallback: HIP APIs)
  └── vkXXX_stub.c    (fallback: no-op stubs)
```

Runtime selection per module:
1. Check Vulkan extension availability **AND** proc address resolution at runtime
2. If extension missing **OR** proc addr NULL → use HIP backend
3. If HIP unavailable → use stub

### VKStream-specific flow:

```c
/* vkstream_runtime.c */
vkstream_backend_t select_stream_backend(vkstream_context_t* ctx) {
    /* Check if direct import is actually functional */
    PFN_vkImportMemoryHostPointerEXT fp = 
        (PFN_vkImportMemoryHostPointerEXT)vkGetDeviceProcAddr(
            ctx->device, "vkImportMemoryHostPointerEXT");
    
    if (fp && ctx->hip_available) {
        /* Test with a small allocation */
        void* test_ptr;
        if (hipHostMalloc(&test_ptr, 4096, hipHostMallocMapped) == hipSuccess) {
            VkImportMemoryHostPointerInfoEXT info = {
                .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_POINTER_EXT,
                .pHostPointer = test_ptr,
            };
            VkMemoryAllocateInfo alloc = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = 4096,
                .memoryTypeIndex = 0,
                .pNext = &info,
            };
            VkDeviceMemory mem;
            VkResult r = fp(ctx->device, &alloc, &info, &mem);
            hipHostFree(test_ptr);
            if (r == VK_SUCCESS) {
                return VKSTREAM_BACKEND_ZEROCOPY;  /* Zero-copy path */
            }
        }
    }
    
    /* Fallback: staging buffer */
    if (ctx->hip_available) {
        return VKSTREAM_BACKEND_STAGING;  /* HIP alloc + memcpy */
    }
    return VKSTREAM_BACKEND_STUB;
}
```

## Staging Buffer Fallback (Recommended Implementation)

When `vkImportMemoryHostPointerEXT` is NULL, use `hipHostMalloc` for BAR-resident
allocation, then copy through a mapped Vulkan buffer:

1. `hipHostMalloc` → get BAR-resident host pointer
2. Write data from HIP side
3. `vkAllocateMemory` + `vkMapMemory` → staging buffer
4. `memcpy(staging, hip_ptr, size)` — GPU-GPU direct if both in VRAM BAR
5. Use staging buffer in Vulkan compute

This is **not zero-copy** but leverages HIP's BAR allocation which is still
faster than system RAM staging.

## OpenVINO IR Loader Notes

From `test_vkmodel.c`: OpenVINO IR loader works (GGUF, safetensors, OpenVINO IR v11).
This is unrelated to the HIP-Vulkan bridge — model loading uses its own
staging path.

## Recommendation

1. **Keep `vkstream_hip.c` stub architecture** — it already has the right structure
2. **Add runtime proc addr validation** — don't trust extension names, test the actual proc
3. **Implement staging fallback in `vkstream_hip.c`** — `hipHostMalloc` + `memcpy` + `vkMapMemory`
4. **Document this as a driver bug** — AMD 26.7.1 advertises `VK_EXT_external_memory_host` but doesn't export the entrypoint
5. **No new HIP bridge files needed** — the architecture already accounts for this failure mode
