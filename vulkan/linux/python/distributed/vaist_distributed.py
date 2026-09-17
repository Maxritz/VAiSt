import ctypes
import os
from pathlib import Path

MODULE_NAME = "vaist_distributed"

def library_name():
    if os.name == "nt":
        return f"{MODULE_NAME}.dll"
    return f"lib{MODULE_NAME}.so"

def _default_path():
    explicit = os.environ.get("VAIST_BUILD_DIR")
    if explicit:
        return Path(explicit) / library_name()
    return Path(library_name())

def load(path=None):
    return ctypes.CDLL(str(path or _default_path()))
