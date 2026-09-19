# vaist_tokens

## PURPOSE
BPE, SentencePiece (unigram/BPE), and byte-level tokenization for LLM inference.
Loads `vocab.json`/`merges.txt` (BPE) and `.model` protobuf (SentencePiece),
exposes greedy greedy and unigram lattice sampling, and reverses tokenization
to UTF-8 text.

## MANDATORY USE
Canonical public boundary for this responsibility. Higher layers (llm/engine)
use this boundary instead of model-specific tokenizers.

## PUBLIC API
See `include/vaist/vaist_tokens.h`. C99 is authoritative; C++ and Python
preserve its semantics.

## INTERNAL COMPONENTS
C99 implementation, C++ convenience layer, Python ctypes binding.

## DEPENDENCIES
`vaist_core` for the common ABI/status contract; platform C runtime only.
File I/O uses the portable `fopen`/`fread`/`fseek`/`ftell`/`fclose` pattern
from `vaist_model.c`.

## OWNERSHIP
`vaist_tokenizer_create` transfers object ownership to the caller;
`vaist_tokenizer_destroy` releases all owned resources (vocab hash tables,
dense id->string tables, merge list, SentencePiece piece array, flat buffers).
Borrowed pointers remain parent-owned.

## LIFETIME
Independent handles are independent objects. `vaist_tokenizer_info` copies a
snapshot; changes to the handle are not reflected in a prior snapshot.

## THREADING
Independent objects are reentrant. No implicit worker threads. Callers
serialize access to a single handle if mutated externally; concurrent reads of
distinct handles are safe.

## SYNCHRONIZATION
No hidden synchronization. Tokenization is synchronous CPU work.

## CAPABILITIES
- `VAIST_TOKENIZER_BPE`: byte-level GPT-2 BPE with `vocab.json` +
  `merges.txt`. Pre-tokenizes on whitespace, applies greedy highest-priority
  merge, handles the byte→unicode mapping table.
- `VAIST_TOKENIZER_SENTENCEPIECE`: parses the `.model` protobuf wire format;
  reads `TrainerSpec.model_type`; greedy longest-match for unigram/BPE;
  lattice-beam sampling via `vaist_tokenize_sample`.
- `VAIST_TOKENIZER_BYTES`: one token per byte (`id = byte + offset`).
- Special tokens (`<s>`, `</s>`, `<unk>`, `<pad>`) are detected for both BPE
  (`<|bos|>`/`bos`, etc.) and SentencePiece (by name + `type` field).
- `max_length` truncation when set.
Unsupported combinations return `VAIST_UNSUPPORTED`.

## DISPATCH
Capability-driven. The `type` argument selects the algorithm family at
creation; unsupported types return `VAIST_UNSUPPORTED`.

## FALLBACKS
Bytes tokenizer is the guaranteed fallback (no file I/O required) and always
succeeds. BPE/SentencePiece fall back to `VAIST_IO_ERROR` when assets are
missing and `VAIST_MODEL_ERROR`/`VAIST_UNSUPPORTED` for malformed data.

## ERRORS
Common `VaistStatus` semantics from `vaist_core`. Specific codes:
- `VAIST_INVALID_ARGUMENT` — NULL handles/args.
- `VAIST_IO_ERROR` — file open/read failure.
- `VAIST_MODEL_ERROR` — bad JSON, bad protobuf, missing/malformed magic.
- `VAIST_OUT_OF_MEMORY` — heap exhaustion or insufficient caller buffer
  (caller capacity is reported via `count`/`len`).
- `VAIST_UNSUPPORTED` — unknown tokenizer type or non-unigram sample request.

## TESTS
`tests/tokens_test.c` covers create/destroy, byte tokenize/detokenize,
piece→id lookup, BPE round-trip (vocab.json + merges.txt), error paths
(NULL args, missing files, unsupported types), and SentencePiece load +
greedy + sampling determinism. Build via CMake `vaist_tokens` target and run
through the stack smoke harness.

## PERFORMANCE
Hot loops are allocation-free after load (greedy BPE uses stack word-id
buffers; SentencePiece greedy is O(pos × pieces) with longest-match scan).
The unigram sampler allocates per-beam-path but bounds the beam to `nbest`.
Byte-level lookup tables are initialized once.

## PLATFORM DIFFERENCES
Windows uses DLL conventions (`VAIST_API = __declspec(dllexport)`); Linux
uses ELF visibility. Public semantics remain identical. File paths use
platform-agnostic forward-slash separators (`fopen` accepts them on both).
