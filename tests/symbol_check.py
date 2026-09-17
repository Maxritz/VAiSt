import re
import subprocess
import sys
from pathlib import Path

MODULES = ("core", "runtime", "tensor", "compute", "quant", "graph", "model", "nn", "llm", "engine", "ai", "distributed", "blas", "linalg")

def expected_symbols(header: Path):
    text = header.read_text(encoding="utf-8")
    return set(re.findall(r"VAIST_API\s+(?:[A-Za-z_][A-Za-z0-9_\s\*]*?)\s+(vaist_[A-Za-z0-9_]+)\s*\(", text))

def main():
    build = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    total = 0
    for module in MODULES:
        lib = build / f"libvaist_{module}.so"
        header = root / "include" / "vaist" / f"vaist_{module}.h"
        if not lib.is_file():
            raise AssertionError(lib)
        output = subprocess.check_output(["nm", "-D", "--defined-only", str(lib)], text=True)
        symbols = set(re.findall(r"\b(vaist_[A-Za-z0-9_]+)$", output, re.M))
        expected = expected_symbols(header)
        missing = expected - symbols
        if missing:
            raise AssertionError(f"{module}: missing exported symbols: {sorted(missing)}")
        total += len(expected)
    print(f"linux C99 exported-symbol check: {len(MODULES)}/{len(MODULES)} PASS, {total} expected symbols present")

if __name__ == "__main__":
    main()
