"""Python bindings for the vaist_tokens component."""
import ctypes
import os

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "..", "..", "libvaist_tokens.so"))

VAIST_TOKENIZER_BYTES = 1
VAIST_TOKENIZER_BPE = 2
VAIST_TOKENIZER_SENTENCEPIECE = 3

_vaist_tokenizer_create = _lib.vaist_tokenizer_create
_vaist_tokenizer_create.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
_vaist_tokenizer_create.restype = ctypes.c_int

_vaist_tokenizer_destroy = _lib.vaist_tokenizer_destroy
_vaist_tokenizer_destroy.argtypes = [ctypes.c_void_p]

_vaist_tokenize = _lib.vaist_tokenize
_vaist_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_vaist_tokenize.restype = ctypes.c_int

_vaist_detokenize = _lib.vaist_detokenize
_vaist_detokenize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_vaist_detokenize.restype = ctypes.c_int


class Tokenizer:
    """Wrapper around the vaist_tokens C library for Python use."""

    def __init__(self, tokenizer_type=VAIST_TOKENIZER_BPE, vocab_path=None):
        self._handle = ctypes.c_void_p()
        status = _vaist_tokenizer_create(
            tokenizer_type, vocab_path.encode() if vocab_path else None, ctypes.byref(self._handle))
        if status != 0:
            raise RuntimeError(f"vaist_tokenizer_create failed: {status}")

    def __del__(self):
        if self._handle:
            _vaist_tokenizer_destroy(self._handle)

    def encode(self, text, max_tokens=128):
        """Encode text to token ids."""
        tokens = (ctypes.c_uint32 * max_tokens)()
        count = ctypes.c_size_t()
        status = _vaist_tokenize(self._handle, text.encode(), tokens, max_tokens, ctypes.byref(count))
        if status != 0:
            raise RuntimeError(f"vaist_tokenize failed: {status}")
        return list(tokens[:count.value])

    def decode(self, token_ids):
        """Decode token ids to text."""
        arr = (ctypes.c_uint32 * len(token_ids))(*token_ids)
        buf = ctypes.create_string_buffer(4096)
        written = ctypes.c_size_t()
        status = _vaist_detokenize(self._handle, arr, len(token_ids), buf, 4096, ctypes.byref(written))
        if status != 0:
            raise RuntimeError(f"vaist_detokenize failed: {status}")
        return buf.raw[:written.value].decode()
