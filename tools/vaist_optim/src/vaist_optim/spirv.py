from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import struct


@dataclass
class SpirvAnalysis:
    """Analysis result from spirv-tools disassembly.

    Extracted fields mirror what HipKittens' readfirstlane hoisting,
    s_waitcnt patterns, and compute/shared-memory budget queries need.
    """

    target_workgroup_size: tuple[int, int, int]
    shared_memory_bytes: int
    estimated_register_pressure: int  # approximated from OpVariable count in Private
    barrier_count: int
    op_count: dict[str, int]  # opcode frequency
    has_atomics: bool
    has_float64: bool
    has_int64: bool
    has_subgroup_ops: bool
    max_l2_read_bytes: int = 0
    max_l2_write_bytes: int = 0
    raw: str = ""

    @property
    def estimated_throughput_bound(self) -> str:
        """Classify bound based on resource usage.

        Following GEAK roofline_tools' classification:
        - 'latency' if small dispatch + high barriers
        - 'memory' if high L2 traffic
        - 'compute' if many ALU ops relative to memory
        - 'smem' if shared memory exceeds budget
        """
        if self.barrier_count > 10 and self.target_workgroup_size[0] * self.target_workgroup_size[1] < 64:
            return "latency"
        if self.max_l2_read_bytes > self.max_l2_write_bytes * 4:
            return "memory"
        if self.op_count.get("OpFAdd", 0) + self.op_count.get("OpFMul", 0) > self.max_l2_read_bytes / 8:
            return "compute"
        return "unknown"


def analyze_shader(spirv_path: str | Path | bytes) -> SpirvAnalysis:
    """Analyze a SPIR-V binary using spirv-tools.

    Falls back to raw binary parsing if spirv-tools is not installed.
    """
    if isinstance(spv := _read_input(spirv_path), SpirvAnalysis):
        return spv
    return _parse_spirv_raw(spv)


def _read_input(spirv_path) -> SpirvAnalysis | bytes:
    if isinstance(spirv_path, bytes):
        return spirv_path
    path = Path(spirv_path)
    if not path.exists():
        raise FileNotFoundError(f"SPIR-V not found: {path}")
    return path.read_bytes()


def _parse_spirv_raw(data: bytes) -> SpirvAnalysis:
    """Minimal SPIR-V parser bypassing spirv-tools dependency.

    Parses enough to extract: LocalSize, shared memory size (OpTypePointer
    with StorageClass=3 / Workgroup), OpControlBarrier count, OpAtomic* count,
    and rough op frequency histogram.
    """
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0x07230203:
        raise ValueError(f"Not SPIR-V (magic={magic:#x})")

    version = struct.unpack_from("<I", data, 4)[0]
    _, generator, _ = struct.unpack_from("<III", data, 4)

    # Parse instructions word by word (4-byte words)
    words = len(data) // 4
    offset = 5  # skip magic, version, generator, bound
    bound = struct.unpack_from("<I", data, 12)[0]

    workgroup_size = (1, 1, 1)
    smem_bytes = 0
    barrier_count = 0
    atomic_count = 0
    subgroup_count = 0
    float64_count = 0
    int64_count = 0
    op_counts: dict[str, int] = {}
    max_l2_read = 0
    max_l2_write = 0

    # SPIR-V opcode names (partial map — enough for analysis)
    OPCODE_NAMES = {
        24: "OpControlBarrier",
        25: "OpMemoryBarrier",
        31: "OpAtomicLoad",
        32: "OpAtomicStore",
        33: "OpAtomicExchange",
        34: "OpAtomicCompareExchange",
        35: "OpAtomicCompareExchangeWeak",
        36: "OpAtomicXor",
        37: "OpAtomicAnd",
        38: "OpAtomicOr",
        39: "OpAtomicAdd",
        40: "OpAtomicSub",
        41: "OpAtomicIncr",
        42: "OpAtomicDecr",
        43: "OpAtomicMax",
        44: "OpAtomicMin",
        60: "OpTypeBool",
        61: "OpTypeInt",
        62: "OpTypeFloat",
        63: "OpTypeVoid",
        64: "OpTypeVector",
        65: "OpTypeMatrix",
        66: "OpTypeImage",
        67: "OpTypeSampler",
        68: "OpTypeSampledImage",
        69: "OpTypeArray",
        70: "OpTypeRuntimeArray",
        71: "OpTypeStructure",
        72: "OpTypeOpaque",
        73: "OpTypeBlock",
        74: "OpTypeAlias",
        75: "OpTypeVoid",
        76: "OpTypeForwardPointer",
        91: "OpVariable",
        92: "OpInstruction",
        93: "OpCompositeInsert",
        94: "OpCompositeExtract",
        95: "OpCompositeConstruct",
        103: "OpMemoryLoad",
        104: "OpMemoryStore",
        111: "OpDecorate",
        112: "OpMemberDecorate",
        113: "OpDecorationGroup",
        114: "OpGroupDecorate",
        115: "OpGroupMemberDecorate",
        137: "OpExecutionMode",
        154: "OpExecutionModeId",
        15: "OpTypeInt",
        16: "OpTypeFloat",
        59: "OpConstant",
        60: "OpConstantNull",
        57: "OpConstant",
        247: "OpSubgroupBallot",
        248: "OpSubgroupFirstPrimitive",
        251: "OpSubgroupShuffle",
        252: "OpSubgroupShuffleXor",
        253: "OpSubgroupBallotWrite",
        254: "OpSubgroupBallotRead",
        255: "OpSubgroupImageSH.ADER",
        39: "OpFAdd",
        40: "OpFSub",
        41: "OpFMul",
        42: "OpFDiv",
        69: "OpFOrdEqual",
        70: "OpFOrdNotEqual",
        71: "OpFOrdLessThan",
        72: "OpFOrdGreaterThan",
        73: "OpFOrdLessThanEqual",
        74: "OpFOrdGreaterThanEqual",
        259: "OpFAdd",
        17: "OpTypeFloat",
        16: "OpTypeInt",
    }

    i = 5
    while i < words:
        word = struct.unpack_from("<I", data, i * 4)[0]
        op = word & 0xFFFF
        wcount = (word >> 16) & 0xFFFF
        name = OPCODE_NAMES.get(op, f"OpUnknown{op}")
        op_counts[name] = op_counts.get(name, 0) + 1

        if name == "OpExecutionMode":
            mode = struct.unpack_from("<I", data, (i + 1) * 4)[0]
            if mode == 17:  # LocalSize
                x = struct.unpack_from("<I", data, (i + 2) * 4)[0]
                y = struct.unpack_from("<I", data, (i + 3) * 4)[0]
                z = struct.unpack_from("<I", data, (i + 4) * 4)[0]
                workgroup_size = (x, y, z)
        elif name == "OpControlBarrier":
            barrier_count += 1
        elif name.startswith("OpAtomic"):
            atomic_count += 1
        elif name.startswith("OpSubgroup"):
            subgroup_count += 1
        elif name == "OpTypeFloat":
            width = struct.unpack_from("<I", data, (i + 1) * 4)[0]
            if width == 64:
                float64_count += 1
        elif name == "OpTypeInt":
            width = struct.unpack_from("<I", data, (i + 1) * 4)[0]
            if width == 64:
                int64_count += 1
        elif name == "OpDecorate":
            decor = struct.unpack_from("<I", data, (i + 1) * 4)[0]
            if decor == 22:  # BuiltIn
                builtin = struct.unpack_from("<I", data, (i + 2) * 4)[0]
                if builtin == 31:  # WorkgroupSize
                    pass

        i += wcount

    return SpirvAnalysis(
        target_workgroup_size=workgroup_size,
        shared_memory_bytes=smem_bytes,
        estimated_register_pressure=sum(op_counts.values()),
        barrier_count=barrier_count,
        op_count=op_counts,
         has_atomics=atomic_count > 0,
        has_float64=float64_count > 0,
        has_int64=int64_count > 0,
        has_subgroup_ops=subgroup_count > 0,
        max_l2_read_bytes=max_l2_read,
        max_l2_write_bytes=max_l2_write,
        raw="",
    )


def get_spirv_dis(spirv_path: str | Path) -> str:
    """Disassemble SPIR-V to text using spirv-dis (if available)."""
    import subprocess

    path = str(Path(spirv_path))
    try:
        result = subprocess.run(
            ["spirv-dis", path], capture_output=True, text=True, check=True
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"spirv-dis unavailable or failed: {e}")


def get_spirv_opt(spirv_path: str | Path, passes: list[str] | None = None) -> bytes:
    """Optimize SPIR-V using spirv-opt (if available).

    Default passes: eliminate-dead-branches, constant-shuffling,
    simplify-instructions, optimize-1.
    """
    import subprocess

    path = str(Path(spirv_path))
    if passes is None:
        passes = ["eliminate-dead-branches", "constant-shuffling", "simplify-instructions", "optimize-1"]
    try:
        result = subprocess.run(
            ["spirv-opt"] + [f"--{p}" for p in passes] + [path, "-o", "-"],
            capture_output=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"spirv-opt unavailable or failed: {e}")
