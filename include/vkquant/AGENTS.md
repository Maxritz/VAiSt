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
| Q4_K   | 144 bytes | ggml `block_q4_K` — `out[i] = d*sc*nib - dmin*mn` (256 elems) |
| Q6_K   | 210 bytes | ggml `block_q6_K` — `out[i] = d*sc*((ql4\|qh2<<4)-32)` (256 elems) |
| IQ4_XS | 136 bytes | ggml `block_iq4_xs` + kvalues_iq4nl — `out[i] = d*(ls-32)*iq4nl[nib]` (256 elems) |

Byte offsets are exact: Q8_0 block at `block*36`, Q4_0 block at `block*20`,
Q4_K at `block*144`, Q6_K at `block*210`, IQ4_XS at `block*136`.
Legacy dequant output is `num_blocks * 32` f32 values; K-format dequant output
is `num_blocks * 256` f32 values.

### Forward quantization
`vkquant_quantize_q8_0_f32` / `vkquant_quantize_q4_0_f32` convert f32 data
(32 elems/block) into the f32-scale Q8_0/Q4_0 formats above; results
round-trip through the corresponding dequant ops.

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
