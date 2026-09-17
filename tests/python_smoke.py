import ctypes, os, pathlib
root=pathlib.Path(__file__).resolve().parents[1]
libdir=root/'build'
core=libdir/('vaist_core.dll' if os.name=='nt' else 'libvaist_core.so')
assert core.exists(), core
lib=ctypes.CDLL(str(core))
lib.vaist_get_version.restype=ctypes.c_char_p
assert lib.vaist_get_version().decode()=='1.0.0'
lib.vaist_get_abi_version.restype=ctypes.c_uint32
assert lib.vaist_get_abi_version()!=0
print('python core smoke PASS')
