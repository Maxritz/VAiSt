# VKQuant Include — Local Contract

Child of root `AGENTS.md`. Governs the public API surface under `include/vkquant/`.

## API Design Principles

### Vulkan-native handles everywhere
- `VkQuantContext` wraps a `VkDevice`, `VkPipelineCache`, and a descriptor-set allocator
- All buffer arguments are `VkBuffer` handles (no CUDA/hip pointers)
- Users manage memory via Vulkan; VKQuant never allocates device memory
- Command recording happens into a user-supplied `VkCommandBuffer` (never a vkQueue)

### Naming mirror
- Prefix `vkquant_` (mirrors `ggml`/`llama.cpp` quant naming for easy porting)
- Type suffix `_f32` on dequant ops: input is a raw quantized byte buffer, output is f32

### Block formats (authoritative)
| Format | Block size | Layout |
|--------|-----------|--------|
| Q8_0   | 36 bytes  | f32 `d` + 32 x int8 `qs` — `out[i] = d * qs[i]` |
| Q4_0   | 20 bytes  | f32 `d` + 16 x uint8 packed nibbles — `v = nibble - 8`; `out[i] = d * v` |

Byte offsets are exact: Q8_0 block at `block*36`, Q4_0 block at `block*20`.
Output is always `num_blocks * 32` f32 values.

### No heap allocation in hot paths
- All dispatch functions take pointers, never allocate
- Pipeline objects are created lazily and cached inside `VkQuantContext`
- Descriptor sets are allocated from a pool owned by the context (fallback path)

## Thread Safety

- `VkQuantContext` is **not thread-safe**. Caller serializes concurrent
  `vkquant_*()` calls that touch the same context.
- Recording into different `VkCommandBuffer`s from different threads is fine,
  but each must use a distinct context or be serialized.

## Files

| File | Purpose |
|------|---------|
| `vkquant.h` | Public API: context mgmt, Q8_0/Q4_0 dequant ops, arch queries |
