#!/usr/bin/env python3
"""vaist-optim: Vulkan shader/kernels optimizer CLI.

Usage:
  vaist-optim analyze <shader.spv>
  vaist-optim tune-gemm <shader.spv> --m 1024 --n 4096 --k 8192 [--cache-dir ./shader_cache]
  vaist-optim tune-attn <shader.spv> --seq 2048 --heads 32 --dim 128
  vaist-optim roofline --ms 0.5 --flops 2.6e9 --bytes-rd 1.0e7 --bytes-wr 2.0e6
  vaist-optim cache-stats [--cache-dir ./shader_cache]
"""
from __future__ import annotations

import argparse
import sys

from .cache import ShaderCache
from .config import DeviceCaps
from .profiler import RooflineProfiler
from .spirv import analyze_shader
from .tuner import tune_gemm_tile, tune_attention_tile


def cmd_analyze(spirv_path: str) -> int:
    a = analyze_shader(spirv_path)
    print(f"=== SPIR-V Analysis: {spirv_path} ===")
    print(f"  Workgroup size: {a.target_workgroup_size}")
    print(f"  Shared memory:  {a.shared_memory_bytes} bytes")
    print(f"  Barrier count:  {a.barrier_count}")
    print(f"  Estimated regs: {a.estimated_register_pressure}")
    print(f"  Has atomics:    {a.has_atomics}")
    print(f"  Has float64:    {a.has_float64}")
    print(f"  Has int64:      {a.has_int64}")
    print(f"  Has subgroup:   {a.has_subgroup_ops}")
    print(f"  Bound type:     {a.estimated_throughput_bound}")
    top_ops = sorted(a.op_count.items(), key=lambda x: -x[1])[:10]
    print(f"  Top ops:        {[(k, v) for k, v in top_ops]}")
    return 0


def cmd_tune_gemm(spirv_path: str, m: int, n: int, k: int, cache_dir: str | None = None) -> int:
    caps = DeviceCaps(65536, (1024, 1024, 64), (1024, 1024, 64), 1024, 32, "gfx1201")
    result = tune_gemm_tile(spirv_path, caps, m, n, k)
    print(f"=== GEMM Tile Tuning ({m}x{n}x{k}) ===")
    print(f"  Recommended tile: {result.config}")
    print(f"  Estimated ms:     {result.estimated_ms:.3f}")
    print(f"  Bound:            {result.bound_type}")
    print(f"  Notes:            {result.notes}")
    if cache_dir:
        cache = ShaderCache(cache_dir)
        print(f"  Cache stats:      {cache.stats()}")
    return 0


def cmd_tune_attn(spirv_path: str, seq: int, heads: int, dim: int) -> int:
    caps = DeviceCaps(65536, (1024, 1024, 64), (1024, 1024, 64), 1024, 32, "gfx1201")
    result = tune_attention_tile(spirv_path, caps, seq, heads, dim)
    print(f"=== Attention Tile Tuning (seq={seq}, heads={heads}, dim={dim}) ===")
    print(f"  Recommended tile: {result.config}")
    print(f"  Estimated ms:     {result.estimated_ms:.3f}")
    print(f"  Bound:            {result.bound_type}")
    print(f"  Notes:            {result.notes}")
    return 0


def cmd_roofline(ms: float, flops: float, bytes_rd: float, bytes_wr: float) -> int:
    caps = DeviceCaps(65536, (1024, 1024, 64), (1024, 1024, 64), 1024, 32, "gfx1201")
    profiler = RooflineProfiler(None, caps)
    result = profiler.analyze_kernel(ms, flops, bytes_rd, bytes_wr, (16, 16, 1))
    print(f"=== Roofline Analysis ===")
    print(f"  Duration:    {result.duration_ms:.3f} ms")
    print(f"  FLOPs:       {result.estimated_flops:.3e}")
    print(f"  Bytes:       {result.estimated_bytes:.3e}")
    print(f"  AI:          {result.roofline_ai} FLOP/byte")
    print(f"  Compute:     {result.compute_util_pct:.1f}%")
    print(f"  Memory:      {result.memory_util_pct:.1f}%")
    print(f"  Bound:       {result.bound}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vaist-optim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze")
    p.add_argument("shader")
    p.set_defaults(func=lambda a: cmd_analyze(a.shader))

    p = sub.add_parser("tune-gemm")
    p.add_argument("shader")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--cache-dir", default=None)
    p.set_defaults(func=lambda a: cmd_tune_gemm(a.shader, a.m, a.n, a.k, a.cache_dir))

    p = sub.add_parser("tune-attn")
    p.add_argument("shader")
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.set_defaults(func=lambda a: cmd_tune_attn(a.shader, a.seq, a.heads, a.dim))

    p = sub.add_parser("roofline")
    p.add_argument("--ms", type=float, required=True)
    p.add_argument("--flops", type=float, required=True)
    p.add_argument("--bytes-rd", type=float, required=True)
    p.add_argument("--bytes-wr", type=float, required=True)
    p.set_defaults(func=lambda a: cmd_roofline(a.ms, a.flops, a.bytes_rd, a.bytes_wr))

    p = sub.add_parser("cache-stats")
    p.add_argument("--cache-dir", default="./shader_cache")
    p.set_defaults(func=lambda a: cmd_cache_stats(a.cache_dir))

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_cache_stats(cache_dir: str) -> int:
     cache = ShaderCache(cache_dir)
     stats = cache.stats()
     print(f"=== Shader Cache Stats ===")
     print(f"  Entries:     {stats['entries']}/{stats['max_entries']}")
     print(f"  Bytes:       {stats['total_bytes']}/{stats['max_bytes']} ({stats['utilization']:.1%})")
     print(f"  Dir:         {cache_dir}")
     return 0


if __name__ == "__main__":
    sys.exit(main())
