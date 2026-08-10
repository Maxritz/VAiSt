# Common Issues in Vulkan LLM Inference Engine Development

Catalog of recurring GPU-side failures observed during Vulkan compute-based
LLM runtime development. Each entry: symptom, root cause template, and the
correct fix pattern (harness-validated).

---

## 1. GPU hangs — unbounded fence wait

### Symptom
Process freezes indefinitely at a compute submit; no CPU progress; `vulkaninfo`
may later report `VK_ERROR_DEVICE_LOST` if the driver times out.

### Root-cause template
`vkWaitForFences(..., UINT64_MAX /*infinite*/)` on a fence that is never
signaled because a GPU-side dispatch loop never completes or the command
buffer is malformed.

### Fix (validated pattern)
- Bound the wait: `vkWaitForFences(dev, 1, &fence, 10'000'000'000ULL /* 10 s */, VK_FALSE)`.
- On `VK_TIMEOUT` → treat as `VK_ERROR_DEVICE_LOST`; reset device.
- Keep debug probes **off by default**; any `printf`/I/O inside the hot
  dispatch loop can starve the queue.

---

## 2. GPU hangs — per-op submit serialization

### Symptom
Throughput collapses to a fraction of peak even on a fast GPU; profiler shows
CPU↔GPU round-trips dominating.

### Root-cause template
`vkQueueSubmit` + `vkWaitForFences` once per dispatch (e.g. ~600 submits per
token on a 32-layer model) serializes the CPU on every GPU barrier.

### Fix (validated pattern)
- Record **all dispatches for one token** into a single `VkCommandBuffer`.
- Use **compute→compute `vkCmdPipelineBarrier`** between dispatches that
  share buffers (no host-side fence between them).
- Submit **once per token** (not once per op).  Keep readback submissions as
  separate batched submits only at flush points (sampling, debug probes).

---

## 3. GPU hangs / DEVICE_LOST — fused-shader dispatch faults

### Symptom
A fused shader (e.g. fused QKV / fused MLP) runs in isolation but crashes the
whole device when dispatched in the full pipeline.

### Root-cause templates
- **Wrong pipeline fetched** — binding `pipeline_plain` where `pipeline_fused`
  was intended, so a buffer that is never written reads as 0 → infinite loop.
- **Dimension assumption mismatch** — fused shader assumes Q8_0-packed
  layout but is dispatched for F32 / K-quant tensors → reads garbage sizes →
  OOB buffer access → page fault.
- **Shared-memory overflow** — fused kernel loads the entire input tile into
  `shared[]` but the tile is sized for the smallest test case; on larger K
  the shared array spills past LDS limit → hard hang.

### Fix (validated pattern)
- **Gate the fused path**: only dispatch fused shader when
  `(format == Q8_0 && dims_aligned && use_fused_path_)`.  All other formats /
  mis-aligned dims fall back to the verified per-kernel `DispatchGemm`.
- Validate pipeline fetches with a static `assert`/enum-to-pipeline map; no
  loose indices.
- Size shared-memory arrays via `min(K, TILE_K)` and pass real K as a push
  constant so the shader bounds its shared load.

---

## 4. GPU hangs — shader cache false-negative

### Symptom
A shader edit appears to have **no effect**: the runtime keeps producing old
output even though the `.comp` source changed.

### Root-cause template
Shader-cache key is computed from compile-time `#define`s only and ignores
the source-file mtime/size → stale SPIR-V is re-used.

### Fix (validated pattern)
- Cache key = `hash(defines) + file_mtime + file_size`.  Any edit to a
  `.comp` file changes mtime+size → cache miss → recompile.

---

## 5. GPU correctness bugs — sign-extension in quantized weight decode

### Symptom
Negatives in Q8_0 / Q6_K / Q5_K weights read as large positives (e.g. a real
−1.2e-3 becomes +3.3e-1).  Model generates fluent but wrong text.

### Root-cause template
GLSL `uint(data_W[...])` reads bytes as unsigned 0..255; `int8_t` cast is
missing, so `int8_t(0xFF)` becomes `255.0f` instead of `-1.0f`.

### Fix (validated pattern)
```glsl
int8_t q = int8_t(data_W[base + 4u + e]);   // explicit signed cast
float w = d * float(q);
```
Apply to **all** quantized GEMM shaders; do not hand-roll a `f16_to_f32`
helper that omits the subnormal exponent shift (use hardware
`unpackHalf2x16` where possible).

---

## 6. GPU correctness bugs — layout assumption (ggml block packing)

### Symptom
Certain quant formats (Q2_K / Q3_K / Q5_K / Q6_K) produce NaN or wildly wrong
outputs even though the sign-extend fix is applied.

### Root-cause template
Assuming ggml block bytes are consecutively packed; real ggml layouts are
strided or quadrant-interleaved:
- Q2_K: byte `b` holds elements `{b, b+32, b+64, b+96}`.
- Q3_K: 2-bit levels strided across 4 bytes; scales are 6-bit packed with kmask.
- Q6_K: 6-bit `ql` quadrant layout; nibble = `quad >> 1`; qh shift = `quad*2`.

### Fix (validated pattern)
Port the **exact** ggml `ggml_eval_*_quant` byte-index math verbatim — do
not re-derive.  Validate byte-for-byte against a Python reference of the GGUF
tensor bytes.

---

## 7. VRAM / RAM budget — device-lost from OOM

### Symptom
`vkAllocateMemory` returns `VK_ERROR_OUT_OF_DEVICE_MEMORY`; subsequent
`vkQueueSubmit` is a no-op → fence never signals → hang (see #1).

### Root-cause template
Loading the whole model into VRAM then cloning it into a CPU staging buffer;
working set exceeds VRAM.

### Fix (validated pattern)
- Load **layer-by-layer**; keep off-screen layers in file-backed `mmap`
  (evictable, not a VRAM copy).
- Enforce `model_size × 1.10 < live_vram_budget`; layers beyond budget stay
  mmap-backed and stream per-token via `EnsureLayerWeights`.

---

## 8. CPU-GPU data race — readback before GPU finish

### Symptom
Logits read back for sampling are stale or zero; token probabilities are flat.

### Root-cause template
`vkMapMemory` on the readback buffer without a preceding fence wait.

### Fix (validated pattern)
- Submit the dispatch + `vkCmdCopyBufferToBuffer` to the readback buffer in
  one command buffer.
- `vkQueueSubmit` → `vkWaitForFences(timeout=10 s)` → **then** `vkMapMemory`.
- Never map a buffer that the GPU might still be writing (validation layer
  `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION` will flag it).

---

## Reference: harness validation procedure

After any change:
1. `cmake --build build --config Debug --target rdna4_llama -- -m:4`
2. Toy sanity: `stories260k.gguf` + `"Once upon a time"` → MUST print the Lily
   story.
3. Scale sanity: Llama-3-8B Q8_0 + `"The capital of France is"` → MUST
   mention Paris.
4. Quant battery: Q2_K 7B → coherent English.
5. Debug probes (`debug_probe_=true`, run `-n 2`) to isolate the first
   stage whose range/NaN goes insane.
