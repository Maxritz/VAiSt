from __future__ import annotations

"""Shader compilation cache for vaist-optim.

Mirrors FreeToken's freetoken-kernel-cache: a disk-backed LRU cache that
stores compiled Vulkan shader variants keyed by (target, tile_config).

Cache layout (per specforge/configs pattern):
  shader_cache/
    <shader_name>/
      <hash(tiling+target).spv    -- compiled SPIR-V
      <hash(tiling+target).meta   -- perf data (ns, bound type, throughput est.)
    cache_index.json              -- LRU manifest, eviction policy
"""

import hashlib
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ShaderCacheEntry:
    """A cached compiled shader with its performance metadata."""
    spv_path: str
    meta_path: str
    compile_ns: float  # wall time of best run (0 if not measured)
    bound_type: str    # "compute" | "memory" | "latency"
    throughput_est: float  # ops/sec estimate
    tile_config: dict = field(default_factory=dict)
    gpu_arch: str = ""


class ShaderCache:
    """Disk-backed LRU cache for compiled Vulkan shaders.

    Follows FreeToken's cache pattern: hash(shader_source + compile_args) -> spv.
    The cache manifest (cache_index.json) tracks LRU order and perf data.
    """

    MAX_ENTRIES = 256  # cap on cache file count (disk bounded)
    MAX_BYTES = 256 * 1024 * 1024  # 256 MB total shader cache

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "cache_index.json"
        self._index: OrderedDict[str, dict] = OrderedDict()
        self._current_bytes = 0
        self._load_index()

    def _load_index(self) -> None:
        """Load the LRU manifest from disk (FreeToken pattern)."""
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "r") as f:
                data = json.load(f)
                self._index = OrderedDict((k, v) for k, v in data.get("entries", {}).items())
                self._current_bytes = data.get("total_bytes", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_index(self) -> None:
        """Persist LRU manifest to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": dict(self._index),
            "total_bytes": self._current_bytes,
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _hash_key(self, shader_source: str, compile_args: str, tile_config: dict | None = None) -> str:
        """Compute a deterministic hash of (source + args + tile) -> cache key.

        Follows FreeToken's _cuda_version_suffix approach: deterministic tags
        so cache hits span compiler versions when relevant.
        """
        parts = [shader_source, compile_args]
        if tile_config:
            parts.append(json.dumps(tile_config, sort_keys=True))
        h = hashlib.sha256("".join(parts).encode()).hexdigest()[:16]
        return h

    def _entry_path(self, key: str, shader_name: str) -> tuple[Path, Path]:
        """Get .spv and .meta paths for a cache key."""
        subdir = self.cache_dir / shader_name
        spv = subdir / f"{key}.spv"
        meta = subdir / f"{key}.meta"
        return spv, meta

    def has(self, shader_source: str, compile_args: str, tile_config: dict | None = None) -> bool:
        """Check if a compiled variant exists in the cache."""
        if not self._index:
            return False
        key = self._hash_key(shader_source, compile_args, tile_config)
        shader_name = _extract_shader_name(shader_source)
        spv, _ = self._entry_path(key, shader_name)
        return spv.exists()

    def get(self, shader_source: str, compile_args: str,
            tile_config: dict | None = None) -> Optional[ShaderCacheEntry]:
        """Retrieve a cached shader entry, updating LRU order."""
        key = self._hash_key(shader_source, compile_args, tile_config or {})
        if key not in self._index:
            return None

        shader_name = _extract_shader_name(shader_source)
        spv, meta = self._entry_path(key, shader_name)
        if not spv.exists():
            del self._index[key]
            self._save_index()
            return None

        entry_data = self._index[key]
        self._index.move_to_end(key)  # LRU update
        self._save_index()

        return ShaderCacheEntry(
            spv_path=str(spv),
            meta_path=str(meta),
            compile_ns=entry_data.get("compile_ns", 0),
            bound_type=entry_data.get("bound_type", "unknown"),
            throughput_est=entry_data.get("throughput_est", 0),
            tile_config=entry_data.get("tile_config", {}),
            gpu_arch=entry_data.get("gpu_arch", ""),
        )

    def put(self, shader_source: str, compile_args: str, spv_bytes: bytes,
            tile_config: dict | None = None, perf_data: dict | None = None) -> ShaderCacheEntry:
        """Store a compiled shader in the cache with perf metadata."""
        key = self._hash_key(shader_source, compile_args, tile_config or {})
        shader_name = _extract_shader_name(shader_source)
        spv, meta = self._entry_path(key, shader_name)
        spv.parent.mkdir(parents=True, exist_ok=True)

        old_size = spv.stat().st_size if spv.exists() else 0
        spv.write_bytes(spv_bytes)
        new_size = len(spv_bytes)
        self._current_bytes += (new_size - old_size)

        perf = perf_data or {}
        self._index[key] = {
            "shader_name": shader_name,
            "compile_ns": perf.get("compile_ns", 0),
            "bound_type": perf.get("bound_type", "unknown"),
            "throughput_est": perf.get("throughput_est", 0),
            "tile_config": tile_config or {},
            "gpu_arch": perf.get("gpu_arch", ""),
            "size_bytes": new_size,
        }

        # Eviction: if over cap, evict LRU
        while len(self._index) > self.MAX_ENTRIES or self._current_bytes > self.MAX_BYTES:
            oldest_key, oldest_val = self._index.popitem(last=False)
            old_spv, old_meta = self._entry_path(oldest_key, oldest_val.get("shader_name", "unknown"))
            if old_spv.exists():
                old_spv.unlink()
                self._current_bytes -= old_spv.stat().st_size
            if old_meta.exists():
                old_meta.unlink()

        self._save_index()
        return ShaderCacheEntry(
            spv_path=str(spv),
            meta_path=str(meta),
            compile_ns=perf.get("compile_ns", 0),
            bound_type=perf.get("bound_type", "unknown"),
            throughput_est=perf.get("throughput_est", 0),
            tile_config=tile_config or {},
            gpu_arch=perf.get("gpu_arch", ""),
        )

    def evict(self, key: str) -> bool:
        """Remove a specific entry from the cache."""
        if key not in self._index:
            return False
        entry = self._index.pop(key)
        spv, meta = self._entry_path(key, entry.get("shader_name", "unknown"))
        if spv.exists():
            self._current_bytes -= spv.stat().st_size
            spv.unlink()
        if meta.exists():
            meta.unlink()
        self._save_index()
        return True

    def clear(self) -> int:
        """Clear the entire cache. Returns count of files removed."""
        count = 0
        if self.cache_dir.exists():
            for spv_file in self.cache_dir.rglob("*.spv"):
                spv_file.unlink()
                count += 1
            for meta_file in self.cache_dir.rglob("*.meta"):
                meta_file.unlink()
                count += 1
        self._index.clear()
        self._current_bytes = 0
        if self.index_path.exists():
            self.index_path.unlink()
        return count

    def stats(self) -> dict:
        """Cache statistics for reporting."""
        return {
            "entries": len(self._index),
            "total_bytes": self._current_bytes,
            "max_entries": self.MAX_ENTRIES,
            "max_bytes": self.MAX_BYTES,
            "utilization": self._current_bytes / self.MAX_BYTES if self.MAX_BYTES else 0,
        }


def _extract_shader_name(source: str) -> str:
    """Extract a friendly name from a shader path or source."""
    # If it's a path, use the filename stem
    if "/" in source or "\\" in source or source.endswith(".comp"):
        import os
        return os.path.basename(source).rsplit(".", 1)[0]
    return "shader"
