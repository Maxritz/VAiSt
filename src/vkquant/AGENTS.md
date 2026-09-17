# VKQuant Source — Local Contract

Child of root `AGENTS.md` and `include/vkquant/AGENTS.md`.

## Implementation Rules

### Pipeline caching
- Pipelines created lazily on first dispatch call
- Cache: open-addressing hashmap, power-of-two (256), linear probing
- Hash key: (kernel, tier_index) — see `vkquant_hash_key()`
- No malloc per lookup; fixed-size cache array in context struct

### Capability detection at context creation
Delegated to VKRuntime's `vkr_detect_capabilities()` (see `src/vkruntime/`):
vkquant links vkruntime and calls it once in `vkquant_create_context` to fill
the context's capability fields and `active_tier`. Only baseline shaders exist;
`ensure_pipeline` falls back to baseline. The public
`vkquant_init_capabilities()` re-detects via the same helper.

### SPIR-V embedding
- Shader SPIR-V compiled to C arrays (`shaders_spv.h`) by `compile_shaders.ps1`
- Embedded as `static const uint32_t` arrays — zero dynamic allocation
- `vkquant_select_spirv()` returns the blob for a (kernel, tier) combination

### Push constant layout (std140, 16 bytes)
```
offset  0: uint num_blocks, _pad0, _pad1, _pad2   (16 bytes)
```
Must match the GLSL push_constant block exactly. uint32/float only — no
`shaderInt64` requirement.

### Descriptor set layout (set=0)
| binding | type | access | stage     |
|---------|------|--------|-----------|
| 0       | SSBO | read   | compute   |
| 2       | SSBO | write  | compute   |

Binding 0 = quantized input (bytes), binding 2 = dequantized output (f32).

### Block formats (dequant, f32 out = num_blocks * block_elems)
- **Legacy 32-elem blocks** (f16 ggml layouts): Q8_0 (36 B, f32 d),
  Q4_0 (20 B), Q4_1 (20 B, d+m), Q5_0 (22 B, d+qh), Q5_1 (24 B, d+m+qh),
  Q8_1 (36 B, d+s), IQ4_NL (18 B, d). Dispatch `ceil(32*blocks/256)`.
- **K-quants 256-elem blocks**: Q2_K (84 B), Q3_K (110 B), Q4_K (144 B),
  Q5_K (176 B), Q6_K (210 B). Dispatch `num_blocks`.
- **IQ 256-elem blocks**: IQ1_S (50 B), IQ1_M (56 B), IQ2_XXS (66 B),
  IQ2_XS (74 B), IQ2_S (82 B), IQ3_XXS (98 B), IQ3_S (110 B), IQ4_XS (136 B).
  Grid/ksigns/kvalues tables are embedded as GLSL `const` arrays generated from
  ggml-common.h (see tests/vkquant_tables.h for the C copies).
- **TQ 256-elem blocks**: TQ1_0 (54 B), TQ2_0 (66 B).

### Forward quantization (f32 -> block)
- Existing f32-scale: Q8_0 (36 B), Q4_0 (20 B) — 8 blocks/workgroup.
- Legacy f16-scale (ggml ref math): Q4_1, Q5_0, Q5_1, Q8_1 — one block per
  workgroup of 1 thread (thread 0 transliteration of quantize_row_*_ref).
- K-quants (ggml scale-selection math): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K — one
  256-elem block per workgroup of 1 thread. Round-trips through the matching
  dequant.
- IQ/TQ forward quantizers (all GPU, one 256-elem block per workgroup of 1
  thread; IQ4_NL uses 32-elem blocks): TQ1_0, TQ2_0 and IQ4_NL, IQ4_XS are
  direct transliterations of quantize_row_tq{1,2}_0_ref /
  quantize_row_iq4_{nl,xs}_ref (kvalues_iq4nl best-index search). The grid
  formats IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S transliterate
  the ggml quantize_row_*_ref scale-selection search but replace ggml's
  runtime-generated kmap/neighbours tables with a DIRECT exhaustive search
  over the same dequant grid tables (embedded as GLSL const arrays), searching
  in dequant units (q = grid_byte/DIV) so the chosen grid indices round-trip
  exactly through the existing dequant shaders. Each grid-search shader runs a
  scale search (is-loop) plus a final requantize of grid indices against the
  quantized block scale (db), matching ggml's structure. All round-trip
  through the matching dequant; the test harness asserts per-format max-abs
  error bounds (see tests/test_vkquant.c).

## Kernel ids

| Kernel | id | Dispatch |
|--------|----|----------|
| dequant_q8_0 / q4_0 | 0 / 1 | `ceil(32*blocks/256)` |
| dequant_q4k / q6k / iq4xs | 2 / 3 / 4 | `num_blocks` (256 elems/block) |
| quantize_q8_0 / q4_0 | 5 / 6 | `ceil(num_blocks/8)` (8 blocks/wg) |
| dequant q4_1/q5_0/q5_1/q8_1/iq4_nl | 7..10, 14 | `ceil(32*blocks/256)` |
| dequant q2k/q3k/q5k/iq1_s/iq1_m/iq2_xs/iq2_s/iq2_xxs/iq3_s/iq3_xxs/tq1_0/tq2_0 | 11..13, 15..23 | `num_blocks` |
| quantize q4_1/q5_0/q5_1/q8_1 | 24..27 | `num_blocks` (1 block/wg) |
| quantize q2k/q3k/q4k/q5k/q6k | 28..32 | `num_blocks` (1 block/wg) |
| quantize iq1_s/iq1_m/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq4_nl/iq4xs/tq1_0/tq2_0 | 33..43 | `num_blocks` (1 block/wg) |

## Files

| File | Purpose |
|------|---------|
| `vkquant_internal.h` | Internal structs, push constants, pipeline cache |
| `vkquant.c` | Context lifecycle, op dispatch, pipeline creation, shader selection |
| `shaders_spv.h` | Auto-generated SPIR-V blob arrays |
