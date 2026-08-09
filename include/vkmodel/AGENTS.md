# VKModel Include — Local Contract

Child of root `AGENTS.md`. Governs the public API surface under `include/vkmodel/`.

## API Design Principles

### Vulkan-native handles everywhere
- `VkModel` is an opaque handle wrapping host metadata storage + per-tensor
  device `VkBuffer`s allocated through the runtime (`vkr_malloc`).
- All tensor buffers are standard `VkBuffer` handles with
  `STORAGE | TRANSFER_SRC | TRANSFER_DST` usage so they feed directly into
  VKQuant dequant dispatch or further copies.

### Naming mirror
- Prefix `vkmodel_` (mirrors `gguf_*` / llama.cpp loader naming for easy
  porting). Getters mirror `llama_model_loader` accessor style.

### Loader scope (what it does / does not do)
- **Parses**: GGUF magic/version, metadata KVs (all 13 value types:
  uint8/16/32/64, int8/16/32/64, float16/32/64, bool, string, array of any of
  these incl. nesting), and tensor infos (name, n_dims, dims, ggml_type,
  file-relative data offset used verbatim).
- **Uploads**: each tensor's raw bytes into a model-owned device buffer via
  `vkr_malloc` + `vkr_upload`, streamed in bounded chunks (no whole-file RAM).
- **Does NOT dequantize**: the loader only reports `ggml_type` per tensor and
  exposes raw bytes. Dequantization is VKQuant's job; the reported dtype
  selects the kernel.
- **Does NOT quantize / graph-build**: no computation-graph construction.

### Queue/pool convention
- `vkmodel_load` assumes the runtime's queue lives on queue family 0 (the
  stack-wide single-queue convention used by every test harness). One
  command pool + one command buffer are created at load and reused across
  every tensor upload (`vkr_upload` resets it after each submit).

### ggml_type mapping (authoritative enum values)
```
0=F32 1=F16 2=Q4_0 3=Q4_1 (4,5 removed) 6=Q5_0 7=Q5_1 8=Q8_0 9=Q8_1
10=Q2_K 11=Q3_K 12=Q4_K 13=Q5_K 14=Q6_K 15=Q8_K
16=IQ2_XXS 17=IQ2_XS 18=IQ3_XXS 19=IQ1_S 20=IQ4_NL 21=IQ3_S 22=IQ2_S
23=IQ4_XS 24=I8 25=I16 26=I32 27=I64 28=F64 29=IQ1_M 30=BF16
34=TQ1_0 35=TQ2_0 39=MXFP4
```
- `vkmodel_block_elems` returns 32 (Q4_0..Q8_1, IQ4_NL), 256 (K super-blocks,
  IQ/TQ), the MXFP4/NVFP4/Q1_0/Q2_0 block sizes, 1 for non-blocked types
  (F32/F16/BF16/I8/I16/I32/I64/F64), 0 for removed/unknown.

### Byte sizes (block layout)
- Block byte sizes match the stack's VKQuant canonical layouts (the classic
  ggml f32-scale blocks): Q8_0=36, Q4_0=20, Q4_K=144, Q6_K=210, IQ4_XS=136.
  Other types use the classic ggml block sizes. These are the layouts the
  stack's dequant consumes; a loader must be re-checked against the writer's
  quantization version when the target model family changes.

## No heap allocation in hot paths
- Loading is host-side and heap-allocates metadata/tensor arrays by design
  (acceptable per root contract: "host-side loading"). Runtime dispatch
  after load is pure handle lookup, no allocation.

## Thread Safety
- `VkModel` is **not thread-safe**. Callers serialize concurrent access.

## Files
| File | Purpose |
|------|---------|
| `vkmodel.h` | Public API: load/destroy, metadata + tensor getters, block-size map |
