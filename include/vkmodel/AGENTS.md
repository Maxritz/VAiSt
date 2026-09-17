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
- **Parses GGUF**: magic/version, metadata KVs (all 13 value types:
  uint8/16/32/64, int8/16/32/64, float16/32/64, bool, string, array of any of
  these incl. nesting), and tensor infos (name, n_dims, dims, ggml_type,
  file-relative data offset used verbatim).
- **Parses safetensors**: 8-byte little-endian header length N + N-byte UTF-8
  JSON header (self-contained recursive-descent JSON parser in `vkmodel.c`),
  exposing `__metadata__` string entries as metadata KVs and each tensor's
  dtype/shape/data_offsets. Data is read verbatim at absolute offset
  `8 + N + data_offsets[0]`; tensor size is validated against dtype × nelems.
- **Parses OpenVINO IR v11** (`vkmodel_load_openvino`): tag-scans the `.xml`
  for `<layer type="Const">` weight nodes plus legacy `<weights>`/`<biases>`
  elements. Element type and shape come from `<data element_type shape
  offset size>`; when absent, they fall back to the output `<port>`
  precision/dims. Slices bytes at the declared `.bin` offset verbatim, size
  checked against element-type × nelems, and validates every tensor's span
  fits inside the `.bin`. Unsupported (sub-byte packed / non-static) element
  types fail the load; opaque U8/U16/U32/U64/boolean/f8 shapes are exposed
  like safetensors unknowns (`VKMODEL_DTYPE_UNKNOWN`, dtype_name = file
  element type), while bf16 maps natively to ggml 30.
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

### Safetensors dtype → ggml mapping
| safetensors dtype | ggml_type | notes |
|-------------------|-----------|-------|
| F32  | 0  | 1:1 |
| F16  | 1  | 1:1 |
| BF16 | 30 | 1:1 |
| F64  | 28 | 1:1 |
| I8   | 24 | 1:1 |
| I16  | 25 | 1:1 |
| I32  | 26 | 1:1 |
| I64  | 27 | 1:1 |
| U8/U16/U32/U64/BOOL/F8_E4M3/F8_E5M2 | `VKMODEL_DTYPE_UNKNOWN` (0xFFFFFFFF) | opaque bytes, no ggml 1:1; `vkmodel_get_tensor_dtype_name()` still returns the file name |
- Unknown safetensors dtypes fail the load (element size cannot be derived).
- Tensors with more than `GGML_MAX_DIMS` (4) dimensions are rejected.
- `vkmodel_get_tensor_dtype_name()` returns the safetensors dtype name for
  safetensors tensors and the canonical ggml name (from a static table) for
  GGUF tensors.

### OpenVINO element type → ggml mapping
| element_type | ggml_type | notes |
|--------------|-----------|-------|
| f32   | 0  | 1:1 |
| f16   | 1  | 1:1 |
| bf16  | 30 | 1:1 (native bf16, same as safetensors BF16) |
| f64   | 28 | 1:1 |
| i8/i16/i32/i64 | 24/25/26/27 | 1:1 |
| u8/u16/u32/u64/boolean/f8e4m3/f8e5m2/f8e8m0 | `VKMODEL_DTYPE_UNKNOWN` (0xFFFFFFFF) | opaque bytes, esize 1/2/4/8; `_dtype_name()` returns the file element type |
| i4/u1/u2/u3/u4/u6/nf4/f4e2m1/string/dynamic/undefined | load fails | sub-byte packed / non-static: element size underivable |
- Numeric `ov::element::Type_t` enum strings are accepted as aliases
  (dynamic=0, boolean=1, bf16=2, f16=3, f32=4, f64=5, i4=6, i8=7, i16=8,
  i32=9, i64=10, u1=11, u2=12, u3=13, u4=14, u6=15, u8=16, u16=17, u32=18,
  u64=19, nf4=20, f8e4m3=21, f8e5m2=22, string=23, f4e2m1=24, f8e8m0=25);
  lookup is case-insensitive ASCII.
- Shape order is OpenVINO's `<dim>` order (row-major, same as GGUF/safetensors).
  A scalar Const (empty shape) yields nelems 1 / size 1×element_size.
- Layer without `<data element_type>` (e.g. `FullyConnected` with `<weights>`)
  resolves dtype+dims from the output `<port id="0">` precision and `<dim>`
  list; port-precision strings ("FP32"/"FP16"/"BF16"/"I8"/"BOOL"/"FP8E4M3"/...)
  map to the IR element-type names via `vkmodel_ov_port_aliases`. If the port
  is also absent the load fails.
- Declared `<data size>` must exactly equal element_size × nelems (validated);
  the `.bin` span `offset+size` is checked against the file size. A `.bin`
  larger than needed (unused tail) is accepted.

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
| `vkmodel.h` | Public API: load/destroy (GGUF + safetensors + OpenVINO IR), metadata + tensor getters, dtype-name getter, block-size map |

### Public API surface
- `vkmodel_load` / `vkmodel_load_safetensors` / `vkmodel_load_openvino` —
  parse + upload a GGUF / safetensors / OpenVINO IR (.xml + .bin) file; all
  three produce the same `VkModel` opaque.
- `vkmodel_destroy` — frees host arrays + vkr_free()s every tensor buffer,
  for models loaded by any of the three loaders.
- Metadata: `vkmodel_get_kv_count` / `_get_kv_key` / `_get_kv_string`.
- Tensor: `vkmodel_get_tensor_count` / `_name` / `_dtype` / `_dtype_name` /
  `_nelems` / `_buffer` / `_size`.
- Mapping: `vkmodel_block_elems` (dtype → elements per block).
