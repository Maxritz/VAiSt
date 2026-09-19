from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DeviceCaps, TileConfig
from .spirv import analyze_shader


@dataclass
class TuneResult:
    """Result of a tile tuning run."""

    config: TileConfig
    estimated_ms: float
    bound_type: str
    notes: str = ""


def tune_gemm_tile(
    spirv_path: str | Path,
    caps: DeviceCaps,
    m: int,
    n: int,
    k: int,
    candidates: list[TileConfig] | None = None,
) -> TuneResult:
    """Tune GEMM tile size by analyzing the shader's shared memory / barrier profile.

    Uses the HipKittens pattern: pick the tile whose smem footprint
    maximizes occupancy while keeping smem under budget.
    """
    from .config import TILE_PRESETS

    if candidates is None:
        candidates = TILE_PRESETS

    analysis = analyze_shader(str(spirv_path))

    best = None
    best_score = -1.0

    for cfg in candidates:
        if cfg.total_threads > caps.max_threads_per_block:
            continue
        if cfg.smem_per_block > caps.max_smem_per_block * 0.8:  # 80% smem budget
            continue

        # Score: high occupancy + low barrier count + good ALU:memory ratio
        block_per_sm = caps.max_blocks_per_sm_vmem(cfg.smem_per_block)
        occupancy = block_per_sm * cfg.total_threads / (caps.subgroup_size * caps.waves_for(cfg.total_threads) * 32)
        # Avoid division by zero: estimate occupancy directly from threads
        occupancy = min(1.0, block_per_sm * cfg.total_threads / 1024.0)

        barrier_penalty = analysis.barrier_count / max(cfg.total_threads, 1)
        smem_efficiency = min(1.0, caps.max_smem_per_block / max(cfg.smem_per_block, 1))

        score = occupancy * smem_efficiency * (1.0 - barrier_penalty * 0.5)

        if score > best_score:
            best_score = score
            best = cfg

    if best is None:
        best = TileConfig(64, 64, 32)  # fallback

    analysis = analyze_shader(str(spirv_path))
    bound = analysis.estimated_throughput_bound

    # Rough MS estimate: FLOPs / peak_flops
    flops = 2 * m * n * k
    peak_flops = _estimate_peak_flops(caps)
    est_ms = (flops / peak_flops) * 1000.0

    return TuneResult(
        config=best,
        estimated_ms=est_ms,
        bound_type=bound,
        notes=f"occupancy_score={best_score:.3f}, barriers={analysis.barrier_count}",
    )


def tune_attention_tile(
    spirv_path: str | Path,
    caps: DeviceCaps,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    candidates: list[TileConfig] | None = None,
) -> TuneResult:
    """Tune attention kernel tile size using the FreeToken split-K pattern.

    For attention, the 'K' dimension is seq_len. Small tiles benefit from
    split-K across heads (following FreeToken's _GEMV_SPLITK_TARGET=1024).
    """
    from .config import TILE_PRESETS

    if candidates is None:
        candidates = TILE_PRESETS

    analysis = analyze_shader(str(spirv_path))

    # Attention is typically memory-bound; smaller tiles help L2 reuse
    best = None
    best_score = -1.0

    for cfg in candidates:
        if cfg.total_threads > caps.max_threads_per_block:
            continue
        if cfg.smem_per_block > caps.max_smem_per_block * 0.7:
            continue

        # For attention, prefer tiles where seq_len % tile_k == 0
        k_aligned = seq_len % cfg.tile_k == 0
        n_aligned = num_heads * head_dim % cfg.tile_n == 0

        alignment_bonus = (1.0 if k_aligned else 0.3) * (1.0 if n_aligned else 0.7)
        block_per_sm = caps.max_blocks_per_sm_vmem(cfg.smem_per_block)
        occupancy = min(1.0, block_per_sm * cfg.total_threads / 1024.0)
        smem_eff = min(1.0, caps.max_smem_per_block / max(cfg.smem_per_block, 1))

        score = occupancy * smem_eff * alignment_bonus

        if score > best_score:
            best_score = score
            best = cfg

    if best is None:
        best = TileConfig(64, 64, 32)

    bound = "memory"  # attention is typically memory-bound
    flops = 2 * num_heads * seq_len * seq_len * head_dim  # QK^T + AV
    peak_flops = _estimate_peak_flops(caps)
    est_ms = (flops / peak_flops) * 1000.0

    return TuneResult(
        config=best,
        estimated_ms=est_ms,
        bound_type=bound,
        notes=f"attention_tile, score={best_score:.3f}, seq_len={seq_len}, heads={num_heads}",
    )


def _estimate_peak_flops(caps: DeviceCaps) -> float:
    """Estimate peak FP32 FLOP/s based on GPU name heuristics.

    Following GEAK roofline_tools' peak table approach,
    using known RDNA/RDNA4/CDNA peak values.
    """
    name = caps.gpu_name.lower()
    if "gfx12" in name or "rx 9" in name:  # RDNA4
        return 2.0 * 175e9  # 2 FMA/cycle * 175 GHz effective
    elif "gfx11" in name or "rx 7" in name:  # RDNA3
        return 2.0 * 123e9
    elif "gfx10" in name or "rx 6" in name:  # RDNA2
        return 2.0 * 64e9
    elif "mi300" in name or "gfx9" in name:  # CDNA3
        return 2.0 * 151e9
    elif "mi200" in name or "gfx103" in name:  # CDNA2
        return 2.0 * 91e9
    return 50e9  # conservative default
