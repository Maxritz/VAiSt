# VKRuntime Source — Local Contract

Child of root `AGENTS.md` and `include/vkruntime/AGENTS.md`.

## Implementation Rules

### Capability detection at runtime creation
| Query | Struct | Detects |
|-------|--------|---------|
| `VkPhysicalDeviceFeatures2.features.shaderInt64` | core | 64-bit shader ints |
| `VkPhysicalDeviceSubgroupProperties.supportedStages` | pNext on `VkPhysicalDeviceProperties2` | compute-stage subgroup |
| `VkPhysicalDeviceCooperativeMatrixFeaturesKHR.cooperativeMatrix` | pNext on `VkPhysicalDeviceFeatures2` | cooperative matrix |
| `vkGetDeviceProcAddr("vkCmdPushDescriptorSetKHR")` | device fn ptr | push descriptors |

Same pNext-chain technique as `vkmath_init_capabilities()`. Arch index mirrors
the vkmath tier ladder: 2 = coopmatrix, 1 = subgroup, 0 = baseline.

### Pooled allocator (hipMalloc/hipFree equivalent)
- Two pools: device-local (index 0) and host-visible+coherent (index 1).
- Each pool = linked list of large `VkDeviceMemory` blocks (default growth
  16 MiB). Each block bump-allocates with a **coalescing free list**.
- `vkr_malloc`: create buffer → query `VkMemoryRequirements` → align size up →
  pick memory type → first-fit free list, else bump, else grow a block →
  `vkBindBufferMemory` → record (buffer → memory/offset/size) in an
  open-addressing table (power-of-two, linear probing, grows on load > 60%).
- `vkr_free`: look up record → return region to its block's free list
  (coalesce; tail frees roll the cursor back) → destroy buffer. No
  `vkFreeMemory`.
- `vkr_destroy_runtime`: unmap + `vkFreeMemory` every block, free bookkeeping.
- Host-visible blocks are mapped once at creation (`vkMapMemory`); staging
  upload/download memcpy straight into `block->mapped + offset`, no per-call
  map/unmap.

### Staging upload/download
- Transient host-visible staging buffer from the host pool; one
  `vkCmdCopyBuffer` (host→dev for upload, dev→host for download); single
  `vkQueueSubmit` + `vkQueueWaitIdle`; command buffer reset afterwards.
- Documented contract: caller supplies `cmd` + `queue`; `cmd` must be unbegun
  and reusable. This keeps synchronization explicit and matches the project
  rule of caller-owned command buffers.

### Memory type selection
- Device-local pool: prefer `DEVICE_LOCAL` without `HOST_VISIBLE`, else any
  `DEVICE_LOCAL`.
- Host pool: prefer `HOST_VISIBLE | HOST_COHERENT`, else any `HOST_VISIBLE`.
- Both restricted to the buffer's `memoryTypeBits`.

## Truth tables before code

Per root AGENTS.md. Reference trace for the allocator free-list carve path
(region reuse after `vkr_free`):

| Condition | Expected | Actual |
|-----------|----------|--------|
| free region exactly fits request | carved, region removed | ✅ |
| free region larger, split | leading + trailing remainders both kept | ✅ |
| tail free (`off+size == cursor`) | cursor rolls back, no region entry | ✅ |
| free adjacent to existing region | coalesced into one region | ✅ |
| bump hit and free list empty | cursor advanced | ✅ |
| no block room | new 16 MiB block grown | ✅ |

## Files

| File | Purpose |
|------|---------|
| `vkruntime_internal.h` | Block/pool/record/caps structs, pool index constants |
| `vkruntime.c` | Context lifecycle, caps, allocator, upload/download, pool helpers |
