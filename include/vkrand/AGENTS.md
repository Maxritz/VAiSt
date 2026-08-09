# VKRAND Public API — Local Contract

Child of root `AGENTS.md` and the VKMath subtree contract (`include/vkmath/`,
`src/vkmath/AGENTS.md`), whose pattern this library mirrors exactly.

## Scope

Stateless counter-based PRNG ops executed as Vulkan compute dispatches.
Ops: Philox4x32-10 -> uniform f32 in [0,1), ThreeFry2x32-20 -> uniform f32 in
[0,1), Philox Box-Muller -> N(0,1) f32, and Philox -> raw uint32 in [0, 2^32).

## Public API

```
VkResult vkrand_create_context(VkPhysicalDevice, VkDevice, VkRandContext**);
void     vkrand_destroy_context(VkRandContext*);
VkResult vkrand_uniform_f32(VkRandContext*, VkCommandBuffer,
                            uint32_t seed, uint32_t count, VkBuffer output);
VkResult vkrand_threefry_uniform_f32(VkRandContext*, VkCommandBuffer,
                                     uint32_t seed, uint32_t count, VkBuffer output);
VkResult vkrand_normal_f32(VkRandContext*, VkCommandBuffer,
                           uint32_t seed, uint32_t count, VkBuffer output);
VkResult vkrand_uniform_uint32(VkRandContext*, VkCommandBuffer,
                               uint32_t seed, uint32_t count, VkBuffer output);
uint32_t vkrand_get_arch_index(VkRandContext*);
const char* vkrand_get_arch_name(VkRandContext*);
void     vkrand_flush_pipelines(VkRandContext*);
VkPipelineLayout vkrand_get_pipeline_layout(VkRandContext*);
```

## Design rules

- **Opaque context**: `VkRandContext` is an opaque handle; layout lives in
  `src/vkrand/vkrand_internal.h`.
- **Stateless**: `vkrand_uniform_f32` derives the Philox counter from the
  global invocation index and the seed. Same (seed, count) -> identical bits.
- **Push descriptors everywhere**: `vkCmdPushDescriptorSetKHR` with a
  descriptor-pool fallback (mirror `vkmath.c`; in the fallback path set
  `writes[i].dstSet = ds` BEFORE `vkUpdateDescriptorSets`).
- **No heap allocation in hot paths**: pipeline cache is a fixed-size
  open-addressing array (256 slots, linear probing, hash of kernel+tier).
- **Push constants**: 16 bytes, std140, only `uint32_t`/`float` members.
  Static assert on exact size in the internal header.
- **No int64 in shaders**: shaders must NOT require `shaderInt64`.
  Philox hi-word multiplies use `umulExtended` (native 32-bit).

## Descriptor set layout (set=0)

| binding | type | access  | stage   |
|---------|------|---------|---------|
| 2       | SSBO | write   | compute |

The runtime layout declares exactly the bindings the shader uses (binding 2
only); the shader's SSBO binding must match.

## Truth tables before code

Per root AGENTS.md: any generator/mapping/layout change requires a decision
tree + truth table trace first, validated against the Random123 known-answer
test vectors:

```
philox4x32-10, ctr={0,0,0,0}, key={0,0}      -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8
philox4x32-10, ctr={ffffffff...}, key={ff..} -> 408f276d 41c83b0e a20bc7c6 6d5451fd
philox4x32-10, ctr=pi digits, key=pi digits  -> d16cfe09 94fdcceb 5001e420 24126ea1
threefry2x32-20, ctr={0,0}, key={0,0}        -> 6b200159 99ba4efe
threefry2x32-20, ctr={ffffffff}, key={ffffffff} -> 1cb996fc bb002be7
threefry2x32-20, ctr=pi digits, key=pi digits -> c4923a9c 483df7a0
```

Note for ThreeFry2x32: Random123 injects the key after every 4-round group
INCLUDING the final group (`Nrounds>19`), so the 20-round output includes the
r=5 injection; a loop that skips the post-final injection will NOT match the
KATs (observed: X1 off by +5).

## Files

| File | Purpose |
|------|---------|
| `include/vkrand/vkrand.h` | Public API (DOX doc comments) |
| `src/vkrand/vkrand_internal.h` | Push constants, pipeline cache, context struct |
| `src/vkrand/vkrand.c` | Context lifecycle, op dispatch, pipeline creation |
| `src/vkrand/shaders_spv.h` | Auto-generated SPIR-V blob arrays |
| `shaders/vkrand/` | GLSL compute shaders (see its AGENTS.md) |
| `tests/test_vkrand.c` | Public-API test harness |
