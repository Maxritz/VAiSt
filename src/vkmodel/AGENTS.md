# VKModel Source — Local Contract

Child of root `AGENTS.md` and `include/vkmodel/AGENTS.md`.

## Implementation Rules

### Host parser (no Vulkan)
- Pure C99 `FILE*` reading through a small little-endian byte reader
  (`vkmodel_reader_t`). All multi-byte fields are assembled byte-by-byte, so
  the parser is endian-explicit and alignment-safe.
- Header: magic `0x46554747` ("GGUF"), version 2..3 accepted (v2/v3 both use
  u64 counts; v1 used u32 counts and is rejected), tensor_count + kv_count.
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

### Tensor byte size
- `size = ceil(nelems / blck_elems) * blck_bytes` from the two ggml_type
  tables in `vkmodel.c`. Block byte sizes follow the stack's VKQuant canonical
  layout (f32-scale Q4_0=20/Q8_0=36, Q4_K=144, Q6_K=210, IQ4_XS=136) and the
  classic ggml layout for the remaining types. Unknown/removed enum values
  yield size 0 → load fails.

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

### VERDICT: parser + upload paths proven by tests/test_vkmodel.c (load,
metadata, tensor info, block map, F32 download memcmp, 33-byte odd-size
offset-verbatim download).

## Files
| File | Purpose |
|------|---------|
| `vkmodel_internal.h` | Model struct, value/tensor/KV storage, constants |
| `vkmodel.c` | Reader, parser, streamed upload, public API, type tables |
