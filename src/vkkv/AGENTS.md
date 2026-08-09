# VKKV Source — Local Contract

Child of root `AGENTS.md` and `include/vkkv/AGENTS.md`.

## Implementation Rules

### Fit is host-side (double precision)
- `vkkv_fit_cpu` builds, per head: G = X^T X (src_dim x src_dim) and
  B = X^T Y (src_dim x tgt_dim) in C99 double, adds `ridge_lambda` to the
  diagonal of G, then solves G W = B by Gauss-Jordan elimination with partial
  pivoting on the augmented matrix [G | B] (d x (d + m) doubles).
- A singular system (pivot below 1e-300) returns
  `VK_ERROR_INITIALIZATION_FAILED`. The diagonal ridge makes singularity
  practically impossible for lambda > 0.
- The stacked float W buffer [n_heads][src_dim*tgt_dim] is uploaded with one
  `vkr_upload` through the transfer's internal command buffer. Re-fitting
  overwrites the device buffer.

### Apply is GPU-side (single shader)
- `vkkv_apply` records: bind pipeline → push 16-byte constants (num_elements =
  n*tgt_dim, tgt_dim, src_dim, w_offset = h*src_dim*tgt_dim) → push descriptors
  (0 = src read, 1 = W read, 2 = dst write) → `vkCmdDispatch(ceil(n*tgt_dim/256))`.
- `apply.comp` computes `dst[i]` with `j = i % tgt_dim`, `r = i / tgt_dim`,
  `acc = sum_k src[r*src_dim+k] * w[w_offset + k*tgt_dim+j]`. Bounds are
  guarded (`if (i >= pc.num_elements) return`). local_size 256, Vulkan 1.4
  core only (no int64/subgroup/coopmatrix).

### Pipeline + descriptor layout
- Pipeline created lazily on the first `vkkv_apply` and cached in
  `ctx->pipeline` (single shader → no hashmap cache needed).
- Descriptor set layout: 3 SSBO bindings (0, 1, 2) with
  `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT` +
  `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT` when the device exposes
  `vkCmdPushDescriptorSetKHR`; otherwise a descriptor pool fallback.
- Push constant range: 16 bytes, compute stage.

### Internal runtime / command buffer
- `vkkv_create_transfer` fetches the compute queue via
  `vkGetDeviceQueue(dev, 0, 0)` (device must expose compute on family 0,
  matching every test harness in the stack) and creates an internal
  VkRuntime + command pool (RESET_COMMAND_BUFFER_BIT) + one command buffer.
- The internal command buffer is used only by `vkr_upload`/`vkr_download`
  (each takes it over for one submission and resets it). It is never used to
  record `vkkv_apply`.

### W storage layout (authoritative)
Device buffer of `n_heads * src_dim * tgt_dim` floats, row-major:
`W[(h*src_dim + k)*tgt_dim + j] == W_h[k][j]`. `w_offset = h*src_dim*tgt_dim`.
Apply: `dst[r*tgt_dim + j] = sum_k src[r*src_dim + k] * W[w_offset + k*tgt_dim + j]`.

## Truth tables before code

Reference trace for the apply index math (the part that must be bit-exact):

### Input State
| Variable  | Value | Source                        |
|-----------|-------|-------------------------------|
| tgt_dim   | 8     | vkkv_create_transfer          |
| src_dim   | 8     | vkkv_create_transfer          |
| n         | 32    | vkkv_apply arg                |
| w_offset  | 0     | h*src_dim*tgt_dim for h = 0   |
| i         | 17    | gl_GlobalInvocationID.x       |

### Decision Tree
```
if (i >= pc.num_elements)     → TRUE: return (bounds guard)
j = i % tgt_dim = 17 % 8 = 1
r = i / tgt_dim = 17 / 8 = 2
acc = sum_{k=0}^{7} src[2*8 + k] * w[w_offset + k*8 + 1]
dst[17] = acc
```

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| i < n*tgt_dim | row r, col j computed | r=2, j=1 | ✅ |
| i >= n*tgt_dim | guard return (no OOB write) | early return | ✅ |
| w[w_offset + k*tgt_dim + j] | W_h[k][j] | W[(h*src_dim+k)*tgt_dim + j] | ✅ |
| src[r*src_dim + k] | X_h[r][k] (row-major) | src[r*src_dim + k] | ✅ |
| n*tgt_dim == 0 (n=0) | guarded by vkkv_apply validation | rejected | ✅ |

### Race Conditions
- [x] No read-before-write on shared data — src/W are readonly, dst written once per element, disjoint i.
- [x] Barriers between producer and consumer — caller-owned (documented in vkkv.h).
- [x] No buffer overflow — bounds guard; dispatch groups = ceil(n*tgt_dim/256) covers [0, n*tgt_dim).

### VERDICT: PASS — apply index math is correct.

## Files

| File | Purpose |
|------|---------|
| `vkkv_internal.h` | Push constants, context struct, helper decls |
| `vkkv.c` | Context lifecycle, host ridge fit, GPU apply, pipeline creation |
| `shaders_spv.h` | Auto-generated SPIR-V blob for `apply.comp` |
