// VKMath shaders / baseline — Tier 0: portable Vulkan 1.4 core
//
// Push constant layout (std140):
//   uint  num_elements;    offset 0
//   uint  num_rows;        offset 4
//   uint  num_cols;        offset 8
//   float alpha;           offset 12
//   float beta;            offset 16
//   uint  _pad0,_pad1,_pad2; offsets 20,24,28
//   uint64_t stride_a;      offset 32
//   uint64_t stride_b;      offset 40
//   uint64_t stride_out;    offset 48
//   uint64_t batch_count;   offset 56
//   uint64_t _pad3;         offset 64
// Total: 72 bytes.
