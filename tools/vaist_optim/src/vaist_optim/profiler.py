from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RooflineResult:
    """Roofline analysis of a kernel execution."""

    duration_ms: float
    estimated_flops: float
    estimated_bytes: float
    compute_util_pct: float
    memory_util_pct: float
    bound: str  # "compute", "memory", "latency"
    roofline_ai: float  # achieved arithmetic intensity (FLOP/byte)


class RooflineProfiler:
    """Vulkan-based roofline profiler.

    Uses VK_QUERY_TYPE_TIMESTAMP + VK_QUERY_TYPE_PERFORMANCE for AMD
    Radeon GPU Profiler (RGP) counter collection, following the GEAK
    roofline_tools.py classification logic.

    Requires: amdgpu-top or RGP for hardware counters.
    """

    def __init__(self, compute_queue, device_limits):
        self.queue = compute_queue
        self.limits = device_limits
        self._peak_flops: float | None = None
        self._peak_bw: float | None = None

    def set_peaks(self, peak_flops: float, peak_bw: float) -> None:
        """Manually set peak FLOP/s and memory bandwidth."""
        self._peak_flops = peak_flops
        self._peak_bw = peak_bw

    def estimate_peaks(self) -> tuple[float, float]:
        """Estimate peaks from device properties.

        Confidence: low — display only, per GEAK roofline_tools.py design.
        """
        if self._peak_flops is None:
            # Conservative estimate: 2 FMA/cycle * freq * num_CU
            # This is a placeholder; real impl queries VkPhysicalDeviceProperties
            self._peak_flops = 1.0e12  # 1 TFLOP/s conservative
        if self._peak_bw is None:
            self._peak_bw = 512e9  # 512 GB/s conservative (HBM2e minimum)
        return self._peak_flops, self._peak_bw

    def analyze_kernel(
        self,
        duration_ms: float,
        flops: float,
        bytes_rd: float,
        bytes_wr: float,
        workgroup_size: tuple[int, int, int],
    ) -> RooflineResult:
        """Classify a kernel's bound type and utilization.

        Follows GEAK roofline_tools.py thresholds:
        - LAUNCH_OVERHEAD_S = 5e-6 (5us) dispatch floor
        - UTIL_BOUND_THRESHOLD = 0.60
        - Target efficiency: gemm=0.90, elementwise=0.875
        """
        peak_flops, peak_bw = self.estimate_peaks()

        total_bytes = bytes_rd + bytes_wr
        ai = flops / total_bytes if total_bytes > 0 else 0.0

        compute_util = (flops / (duration_ms * 1e-3)) / peak_flops * 100 if duration_ms > 0 else 0
        mem_util = (total_bytes / (duration_ms * 1e-3)) / peak_bw * 100 if duration_ms > 0 else 0

        threads_per_block = workgroup_size[0] * workgroup_size[1] * workgroup_size[2]
        launch_overhead_ms = 5e-6 * 1000  # 5us in ms

        if duration_ms <= launch_overhead_ms * 2:
            bound = "latency"
        elif compute_util > 60 and compute_util >= mem_util * (ai / max(1.0, peak_flops / peak_bw)):
            bound = "compute"
        elif mem_util > 60:
            bound = "memory"
        else:
            bound = "latency"

        return RooflineResult(
            duration_ms=duration_ms,
            estimated_flops=flops,
            estimated_bytes=total_bytes,
            compute_util_pct=round(compute_util, 2),
            memory_util_pct=round(mem_util, 2),
            bound=bound,
            roofline_ai=round(ai, 3),
        )
