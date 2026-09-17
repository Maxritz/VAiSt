# version 450

// Cooperatve matrix tier for VKMath.
// Elementwise ops (ReLU, SiLU, GELU, etc.) and reductions do not benefit
// from cooperative matrix intrinsics, so no coopmatrix variants are provided.
//
// Reductions could use matrix-based approaches for very wide reductions,
// but this is planned for a future iteration.
