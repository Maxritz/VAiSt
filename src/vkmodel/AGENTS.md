# VKModel Source — Local Contract

Child of root `AGENTS.md` and `include/vkmodel/AGENTS.md`.

## Implementation Rules

### Host parser (no Vulkan)
- Pure C99 `FILE*` reading through a small little-endian byte reader
  (`vkmodel_reader_t`). All multi-byte fields are assembled byte-by-byte, so
  the parser is endian-explicit and alignment-safe.
- GGUF header: magic `0x46554747` ("GGUF"), version 2..3 accepted (v2/v3 both
  use u64 counts; v1 used u32 counts and is rejected), tensor_count + kv_count.
- Metadata: for each KV — key (u64 len + bytes), u32 value type, typed value.
  All 13 value types are stored as `VkModelValue` (scalars in `u64`/`f64`,
  strings as NUL-terminated copies, arrays recursive). `general.alignment`
  (UINT32 or UINT64) overrides the default 32 alignment and must be a
  multiple of 8.
- Tensor infos: name, n_dims (1..4), dims[], u32 ggml_type, u64 offset.
  `data_offset = align_up(end_of_infos, alignment)`. Each tensor is located at
  `data_offset + offset` with the **offset field used verbatim** (never
  re-derived from sizes/alignment).
- Sanity caps (malformed-file guards): counts ≤ 2^20, array length ≤ 2^24,
  string length ≤ 64 MiB.

### Safetensors parser (same file)
- Header: 8-byte little-endian length N (1..64 MiB), then N bytes of UTF-8
  JSON parsed by a self-contained recursive-descent parser
  (`vkmodel_json_*`, ~250 lines, no external dependency). Supports objects,
  arrays, strings (incl. `\uXXXX` + surrogate pairs → UTF-8), integers,
  true/false/null; fractions/exponents consumed but truncated.
- Root object: each non-`__metadata__` member is a tensor
  {`dtype`, `shape`, `data_offsets`}; `__metadata__` string entries become
  `VkModelKV` (STRING), non-string metadata values are skipped.
- The loader sets `m->data_offset = 8 + N` and `t->offset = data_offsets[0]`
  so the **shared** GGUF streamed-upload path (`vkmodel_upload_tensor`)
  locates bytes verbatim with no changes. Tensor size = `end - start`, which
  must equal `nelems × element_size` (validated).
- Dtype map lives in `vkmodel_st_dtypes`; unknown dtypes fail the load. Shapes
  limited to 4 dims (GGML_MAX_DIMS); dims/offsets validated for overflow.
- JSON node children are stored as array slots inside each node's `members`
  array (never individually malloc'd) — the free routine releases keys/strings/
  child arrays recursively but only `free()`s the node itself (see
  `vkmodel_json_free` / `vkmodel_json_free_children`).

### Tensor byte size
- `size = ceil(nelems / blck_elems) * blck_bytes` from the two ggml_type
  tables in `vkmodel.c`. Block byte sizes follow the stack's VKQuant canonical
  layout (f32-scale Q4_0=20/Q8_0=36, Q4_K=144, Q6_K=210, IQ4_XS=136) and the
  classic ggml layout for the remaining types. Unknown/removed enum values
  yield size 0 → load fails.

### OpenVINO IR parser (same file)
- `vkmodel_load_openvino(rt, xml_path, bin_path, &m)`: opens the `.bin`
  (fseek/ftell size, capped at 64 MiB) then tag-scans the `.xml` with the
  shared `vkmodel_reader_t` + a small `vkmodel_ov_tag_t` scanner
  (`<name>`/`</name>`/`<name/>`/attributes). No external XML dependency.
- Const layer → tensor: `type="Const"` (any opset, incl. opset10 `Const` with
  embedded data and `version="opset1"`). Element type + shape + span from
  `<data element_type shape offset size>`; the legacy path resolves layers
  with `<weights offset size>` / `<biases offset size>` elements, whose
  dtype/dims fall back to the output `<port id="0">` `precision` + `<dim>`
  list. Port-precision strings ("FP32"/"FP16"/"BF16"/"I8"/"BOOL"/"FP8E4M3"/...)
  are resolved through `vkmodel_ov_port_aliases` to the matching IR element
  type name before the dtype lookup. Layers with neither Const nor
  weights/biases are
  skipped (e.g. `Parameter`). The IR `<dim>` order maps 1:1 to the GGUF dim
  order; a scalar (empty shape) has nelems 1.
- Element-type map lives in `vkmodel_ov_dtypes` (f32/f16→ggml 0/1,
  i8/i16/i32/i64→24/25/26/27, i4/u4/f8/bool→reject). bf16 and the unsigned
  widths map to `VKMODEL_DTYPE_UNKNOWN` with size 2 (bf16) / 1..8 (uN) so the
  loader still uploads the raw bytes; `_dtype_name()` reports the file element
  type. Sub-byte packed types fail (element size underivable).
- Validation (per tensor, in `vkmodel_ov_fill_tensor`): dtype resolvable and
  whole-byte (`i4`/`u1`..`u6`/`nf4`/`f4e2m1`/`string`/`dynamic`/`undefined`
  fail) → nelems from `<data shape>` (empty shape = scalar, nelems 1; no
  shape → output-port `<dim>`s) → declared `size` must EXACTLY equal
  `nelems × element_size` (line 2112) → span `offset + size` must fit inside
  the `.bin`. Any failure returns `VK_ERROR_INITIALIZATION_FAILED` and leaves
  `*pModel` NULL. The `.bin` may be larger than the total tensor span; there
  is no file-size vs declared-size check.
- Upload reuses the shared `vkmodel_upload_tensor` (streamed chunked `vkr_upload`
  into the same model-owned device buffers). Host structs (`VkModelOvLayer`)
  are freed by `vkmodel_destroy`.

### Streamed upload
- Per tensor: `vkr_malloc` a device buffer (STORAGE|TRANSFER_SRC/DST), then
  read the data region in `VKMODEL_STREAM_CHUNK` (64 MiB) slices via
  fseek/fread and `vkr_upload` each slice at the running buffer offset. Peak
  host RAM is bounded by the chunk size, not the model size.
- One command pool (family 0) + one command buffer allocated at load; reused
  across all tensors (each `vkr_upload` resets it). Queue resolved as
  family 0 / index 0 — the stack-wide convention for the runtime's queue.
- `vkr_malloc`/`vkr_upload` errors propagate; on any failure the model is
  destroyed and `*pModel` left NULL.

### Error cleanup
- Every partial state path (header parse, metadata, tensor infos, upload,
  pool/CB creation) unwinds through `vkmodel_destroy()`, so a failed load
  leaks nothing.

### Safetensors upload path (reuses shared uploader)
| Condition                                        | Expected                          | Actual |
|--------------------------------------------------|-----------------------------------|--------|
| data_offsets end-start == nelems × esize         | size = end-start, uploaded verbatim | ✅ |
| data_offsets start non-32-aligned                | bytes read at 8+N+start verbatim  | ✅ |
| end <= start or size mismatch                    | VK_ERROR_INITIALIZATION_FAILED    | ✅ |
| unknown dtype                                   | VK_ERROR_INITIALIZATION_FAILED    | ✅ |
| __metadata__ string value                       | exposed as VKModelKV (STRING)     | ✅ |
| __metadata__ non-string value                   | skipped (get_kv_string → NULL)    | ✅ |
| shape > 4 dims / dim == 0 / overflow            | VK_ERROR_INITIALIZATION_FAILED    | ✅ |
| JSON node free (children in array slots)        | keys/strings/child arrays freed once, no free() of slots | ✅ |

## Truth tables before code

### Metadata value parsing
| Value type | Read width | Stored in              |
|------------|-----------|------------------------|
| UINT8/16/32/64 | 1/2/4/8 LE | `v.u64` (zero-extended) |
| INT8/16/32/64  | 1/2/4/8 LE | `v.u64` (sign-preserved) |
| FLOAT32        | 4 LE       | `v.f64` (promoted)     |
| FLOAT64        | 8 LE       | `v.f64`                |
| BOOL           | 1          | `v.u64` 0/1            |
| STRING         | u64 len + bytes | `v.str` NUL-terminated |
| ARRAY          | u32 elem + u64 count | `v.array` recursive |

### Tensor size table (representative rows)
| ggml_type | blck_elems | blck_bytes | 64-elem tensor bytes |
|-----------|-----------|------------|----------------------|
| F32 (0)   | 1         | 4          | 256                  |
| F16 (1)   | 1         | 2          | 128                  |
| Q8_0 (8)  | 32        | 36         | 72                   |
| Q4_K (12) | 256       | 144        | 144 (1 super-block)  |
| I8 (24)   | 1         | 1          | 64                   |

### Upload path (per tensor)
| Condition                                   | Expected                          | Actual |
|---------------------------------------------|-----------------------------------|--------|
| size == 0 (no-op)                           | return VK_SUCCESS, no buffer      | ✅     |
| vkr_malloc fails                            | propagate, destroy model          | ✅     |
| fread short (offset+size past EOF)          | VK_ERROR_INITIALIZATION_FAILED    | ✅     |
| chunk slice vkr_upload fails                | propagate, destroy model          | ✅     |
| multiple chunks                             | slices written at ascending offsets | ✅   |
| cmd reset between vkr_upload calls          | vkr_upload resets cmd itself      | ✅     |

### VERDICT: parser + upload paths proven by tests/test_vkmodel.c (GGUF load,
metadata, tensor info, block map, F32 download memcmp, 33-byte odd-size
offset-verbatim download; safetensors load of F32/F16 tensors + __metadata__
KV exposure + dtype name/enum mapping + 2-tensor byte-identical download +
non-aligned odd-offset I8 [33] verbatim download; OpenVINO IR v11 load of a
5-tensor .xml+.bin — f32 Conv [2,3,4], scalar f16, bf16 [8], legacy
<weights> f32 [8,8] resolved via output-port precision/dims, opaque u8 [16] —
with dtype/dtype_name/nelems/size asserts and byte-exact vkr_download
round-trips per tensor, plus rejection of a size-mismatch IR and a sub-byte
i4 IR).

### OpenVINO rejection truth table
| Input IR                                             | Expected                            | Actual |
|------------------------------------------------------|-------------------------------------|--------|
| `<data size>` ≠ element_size × nelems (f32 4×4 size=100) | VK_ERROR_INITIALIZATION_FAILED   | ✅ test |
| element_type `i4` (sub-byte, no esize)               | VK_ERROR_INITIALIZATION_FAILED      | ✅ test |
| `offset+size` past .bin EOF                          | VK_ERROR_INITIALIZATION_FAILED      | ✅ code path |
| Const with `<data>` + no output port                 | dtype/shape from `<data>`, size validated | ✅ code path |
| layer with `<weights>` but no resolvable dtype/dims (no port) | VK_ERROR_INITIALIZATION_FAILED | ✅ code path |
| `.bin` larger than needed (unused tail bytes)        | accepted (only per-tensor span checked) | ✅ code path |
| `<data>` without `size` attribute (size = 0)         | rejected (0 != nelems×esize)        | ✅ code path |

## Files
| File | Purpose |
|------|---------|
| `vkmodel_internal.h` | Model struct, value/tensor/KV storage, constants |
| `vkmodel.c` | GGUF reader/parser, self-contained JSON parser, safetensors loader, OpenVINO IR v11 loader, streamed upload, public API, type tables |
