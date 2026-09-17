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

The remaining ggml formats (all f16-scale, byte-exact ggml layouts) are also
implemented with a matching `vkquant_dequant_<name>_f32` / (where applicable)
`vkquant_quantize_<name>_f32` op:

| Format | Bytes | elems/block | Dequant rule (ggml) |
|--------|-------|-------------|---------------------|
| Q4_1   | 20    | 32          | `out = d*nib + m` |
| Q5_0   | 22    | 32          | `out = d*((nib\|xh)-16)` |
| Q5_1   | 24    | 32          | `out = d*(nib\|xh)+m` |
| Q8_1   | 36    | 32          | `out = d*qs[i]` |
| IQ4_NL | 18    | 32          | `out = d*iq4nl[nib]` |
| Q2_K   | 84    | 256         | `out = d*(sc&0xF)*level - dmin*(sc>>4)*level` |
| Q3_K   | 110   | 256         | `out = d*(sc-32)*level` (12-byte 6-bit scale decode) |
| Q5_K   | 176   | 256         | `out = d*sc*(nib + qh_bit*16) - dmin*mn` |
| IQ1_S  | 50    | 256         | iq1s grid + IQ1S_DELTA |
| IQ1_M  | 56    | 256         | iq1s grid + IQ1S_DELTA (scale from 12 bits) |
| IQ2_XXS| 66    | 256         | iq2xxs grid + ksigns_iq2xs |
| IQ2_XS | 74    | 256         | iq2xs grid + ksigns_iq2xs |
| IQ2_S  | 82    | 256         | iq2s grid + ksigns_iq2xs |
| IQ3_XXS| 98    | 256         | iq3xxs grid + ksigns_iq2xs |
| IQ3_S  | 110   | 256         | iq3s grid + ksigns_iq2xs |
| TQ1_0  | 54    | 256         | ternary 5-in-1 byte groups |
| TQ2_0  | 66    | 256         | ternary 2 bits/element |

Byte offsets are exact ggml layouts: legacy block at `block*size`, K/IQ/TQ
super-block at `block*size`. Legacy dequant output is `num_blocks * 32` f32
values; K-format dequant output is `num_blocks * 256` f32 values.

### Forward quantization
`vkquant_quantize_q8_0_f32` / `vkquant_quantize_q4_0_f32` convert f32 data
(32 elems/block) into the f32-scale Q8_0/Q4_0 formats; results round-trip
through the corresponding dequant ops. The f16-scale legacy formats
(Q4_1/Q5_0/Q5_1/Q8_1) and the K-quants (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) have GPU
quantizers implementing the ggml reference scale-selection math; all quantized
bytes round-trip through the matching dequant with per-element error < 1e-1.
IQ and TQ formats are dequant-only — their quantizers are search/heuristic
based and are intentionally omitted.

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
| `vkquant.h` | Public API: context mgmt, all ggml dequant ops, forward quant ops, arch queries |
