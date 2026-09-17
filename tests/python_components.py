import ctypes
import importlib
import os
import sys
from pathlib import Path

MODULES = (
    "core", "runtime", "tensor", "compute", "quant", "graph",
    "model", "nn", "llm", "engine", "ai", "distributed", "blas", "linalg"
)


def artifact(build_dir: Path, platform: str, module: str) -> Path:
    if platform == "windows":
        names = [f"vaist_{module}.dll"]
    else:
        names = [f"libvaist_{module}.so"]
    candidates = [build_dir / n for n in names] + [build_dir / "Release" / n for n in names]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python_components.py <build-dir> <windows|linux>")
    build_dir = Path(sys.argv[1]).resolve()
    platform = sys.argv[2]
    python_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(python_root))
    for module in MODULES:
        package = importlib.import_module(module)
        loader = importlib.import_module(f"{module}.vaist_{module}")
        expected = f"vaist_{module}.dll" if platform == "windows" else f"libvaist_{module}.so"
        assert loader.library_name() == expected
        path = artifact(build_dir, platform, module)
        assert path.is_file(), path
        lib = loader.load(path)
        assert lib is not None
        assert package.load is loader.load
        del lib
    print(f"python component package/load: {len(MODULES)}/{len(MODULES)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
