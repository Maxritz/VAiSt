from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeviceCaps:
    """Minimal Vulkan device capabilities needed for tile/tuning decisions.

    Mirrors the fields used by GEAK roofline_tools.py and FreeToken's
    _gemm_wave device-query logic, adapted to Vulkan physical-device limits.
    """

    max_compute_shared_memory_size: int
    max_compute_workgroup_size: tuple[int, int, int]  # x, y, z
    max_compute_workgroups_per_shader_stage: tuple[int, int, int]
    max_compute_workgroup_invocations: int
    subgroup_size: int
    gpu_name: str = "unknown"

    @property
    def max_smem_per_block(self) -> int:
        return self.max_compute_shared_memory_size

    @property
    def max_threads_per_block(self) -> int:
        return self.max_compute_workgroup_invocations

    @property
    def wave_size(self) -> int:
        return self.subgroup_size

    def waves_for(self, block_threads: int) -> int:
        """Number of warps/wave per block (ceiling division)."""
        return (block_threads + self.wave_size - 1) // self.wave_size

    def max_blocks_per_sm_vmem(self, block_smem: int, block_regs: int = 0) -> int:
        """Max resident blocks per unit (SM-equivalent) given smem budget.

        Simplified: ignores register pressure (Vulkan has no per-thread
        register query in core; can be extended with VK_KHR_shader_module
        extended properties). Mirrors FreeToken's _gemm_wave logic.
        """
        if block_smem <= 0:
            return 4  # register-bound default (HipKittens uses 4)
        smem_based = self.max_smem_per_block // block_smem
        return max(1, min(smem_based, 4))


@dataclass
class TileConfig:
    """Tunable tile dimensions for a compute shader.

    Follows HipKittens' pattern: MxN subtile with K-step,
    mapped to Vulkan local_size_x/y workgroups.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    threads_x: int = 16
    threads_y: int = 16
    stages: int = 2  # ping-pong buffering stages in shared memory
    prefetch: bool = True

    @property
    def total_threads(self) -> int:
        return self.threads_x * self.threads_y

    @property
    def smem_per_block(self) -> int:
        """Approximate shared memory in bytes (2 * tile_m * tile_k + 2 * tile_k * tile_n) * 2 bytes bf16."""
        return 2 * self.tile_m * self.tile_k * 2 + 2 * self.tile_k * self.tile_n * 2

    def waves(self, caps: DeviceCaps) -> int:
        return caps.waves_for(self.total_threads)

    def __str__(self) -> str:
        return f"Tile({self.tile_m}x{self.tile_n}x{self.tile_k},threads={self.threads_x}x{self.threads_y},stages={self.stages})"


# Preset tile configurations following HipKittens / ThunderKittens patterns
TILE_PRESETS = [
    TileConfig(64, 64, 32),
    TileConfig(128, 64, 32),
    TileConfig(64, 128, 32),
    TileConfig(128, 128, 32),
    TileConfig(256, 128, 64),
    TileConfig(128, 256, 64),
    TileConfig(256, 256, 64),
    TileConfig(256, 256, 128),
]
