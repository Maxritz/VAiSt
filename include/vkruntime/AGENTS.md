# VKRuntime Public API — Local Contract

Child of root `AGENTS.md`. VKRuntime is the base layer every other library
(vkmath, vkblas, vkquant, vkrand, vkfft) sits on; this contract governs the
public API only. Implementation rules live in `src/vkruntime/AGENTS.md`.

## Scope

The hipRuntime-equivalent foundation layer, Vulkan-native (no ROCm/HIP/CUDA):
device/context abstraction, capability detection, pooled buffer allocation,
staging upload/download, and command/descriptor pool helpers. VKRuntime needs
no shaders (no `shaders_spv.h`, no `shaders/` subtree).

## Public API

```
typedef struct VkRuntime VkRuntime;   /* opaque */

VkResult vkr_create_runtime(VkPhysicalDevice, VkDevice, VkQueue, VkRuntime**);
void     vkr_destroy_runtime(VkRuntime*);
uint32_t    vkr_get_arch_index(VkRuntime*);
const char* vkr_get_arch_name(VkRuntime*);
VkBool32    vkr_has_subgroup(VkRuntime*);
VkBool32    vkr_has_coop_matrix(VkRuntime*);
uint32_t    vkr_get_subgroup_size(VkRuntime*);
VkDevice    vkr_get_device(VkRuntime*);
VkResult vkr_malloc(VkRuntime*, VkDeviceSize, VkBufferUsageFlags,
                    VkBuffer*, VkDeviceMemory*);
void     vkr_free(VkRuntime*, VkBuffer, VkDeviceMemory);
VkResult vkr_upload(VkRuntime*, VkCommandBuffer, VkQueue, const void*,
                    VkBuffer, VkDeviceSize, VkDeviceSize);
VkResult vkr_download(VkRuntime*, VkCommandBuffer, VkQueue, VkBuffer,
                      VkDeviceSize, void*, VkDeviceSize);
VkResult vkr_create_command_pool(VkRuntime*, uint32_t queue_family, VkCommandPool*);
VkResult vkr_create_descriptor_pool(VkRuntime*, uint32_t max_sets,
                                    uint32_t ssbo_count, VkDescriptorPool*);
void     vkr_wait_idle(VkRuntime*);
```

## Contract rules

- **Opaque context**: `VkRuntime` layout lives only in
  `src/vkruntime/vkruntime_internal.h`.
- **Vulkan-native**: every returned handle is a Vulkan object. No custom
  allocator handles, no pointer-based device memory.
- **hipMalloc/hipFree equivalence**: `vkr_malloc`/`vkr_free` sub-allocate from
  a small set of large `VkDeviceMemory` blocks; `vkr_free` returns the region
  to the pool (no per-free syscall). Only `vkr_destroy_runtime` frees blocks.
- **Memory class selection from `usage`**: pure transfer-only usage
  (TRANSFER_SRC/DST only) → host-visible + host-coherent staging; anything
  with compute/graphics usage → device-local. Every pooled buffer also gets
  TRANSFER_SRC | TRANSFER_DST so `vkr_upload`/`vkr_download` work on any
  pooled buffer (mirrors hipMemcpy-on-any-hipMalloc-pointer).
- **Command buffer ownership in upload/download**: `vkr_upload`/`vkr_download`
  take the caller's command buffer over for exactly one submission
  (begin → one vkCmdCopyBuffer → end → submit → wait → reset). On return the
  command buffer is reusable; it must be unbegun on entry and come from a pool
  with RESET_COMMAND_BUFFER_BIT on the same queue family as the submitted queue.
- **Pool helpers return caller-owned objects**: command/descriptor pools are
  destroyed by the caller via `vkDestroy*`, not by the runtime.
- **No stubs**: every declared function has a real implementation and a
  passing harness test (tests/test_vkruntime.c).
- **C99 + DOX**: DOX `\brief`/`\param`/`\retval` doc comments on every function.

## Files

| File | Purpose |
|------|---------|
| `include/vkruntime/vkruntime.h` | Public API (this file) |
| `src/vkruntime/vkruntime_internal.h` | Pool/block/record/caps structs |
| `src/vkruntime/vkruntime.c` | Implementation |
| `tests/test_vkruntime.c` | Public-API harness (runs on the real GPU) |
