import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODULES = ['core', 'runtime', 'tensor', 'compute', 'quant', 'graph', 'model', 'nn', 'llm', 'engine', 'ai', 'distributed', 'blas', 'linalg']

def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: clean_staging.py <build-dir> <platform> <cmake>")
    build = Path(sys.argv[1]).resolve()
    platform = sys.argv[2]
    cmake = sys.argv[3]
    stage = build / "vaist-clean-stage"
    if stage.exists():
        shutil.rmtree(stage)
    subprocess.run([cmake, "--install", str(build), "--prefix", str(stage)], check=True)
    libdir = stage / ("bin" if platform == "windows" else "lib")
    for name in MODULES:
        artifact = libdir / (f"vaist_{name}.dll" if platform == "windows" else f"libvaist_{name}.so")
        if not artifact.is_file():
            raise AssertionError(f"missing staged artifact: {artifact}")
        lib = ctypes.CDLL(str(artifact))
        del lib
    print(f"clean staging native load: {len(MODULES)}/{len(MODULES)} PASS")

if __name__ == "__main__":
    main()
