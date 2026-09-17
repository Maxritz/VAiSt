import importlib
import os
from pathlib import Path

MODULES = ['core', 'runtime', 'tensor', 'compute', 'quant', 'graph', 'model', 'nn', 'llm', 'engine', 'ai', 'distributed']

def main():
    build = Path(os.environ.get("VAIST_BUILD_DIR", Path(__file__).resolve().parents[3] / "build"))
    for name in MODULES:
        mod = importlib.import_module(f"{name}.vaist_{name}")
        expected = f"vaist_{name}.dll" if os.name == "nt" else f"libvaist_{name}.so"
        assert mod.library_name() == expected
        path = build / expected
        assert path.is_file(), path
        assert mod.load(path) is not None
    print("python components: 12/12 PASS")

if __name__ == "__main__":
    main()
