#ifndef VAIST_TOKENS_H
#define VAIST_TOKENS_H
#include "vaist_core.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/** \brief Tokenizer algorithm family selected at creation. */
typedef enum VaistTokenizerType {
    VAIST_TOKENIZER_BYTES        = 1, /**< Fallback: one token per byte (id = byte + offset). */
    VAIST_TOKENIZER_BPE          = 2, /**< Byte-Pair Encoding (GPT-2 style with merges.txt).  */
    VAIST_TOKENIZER_SENTENCEPIECE= 3, /**< SentencePiece (unigram / BPE model).                */
} VaistTokenizerType;
/** \brief Alias matching the create() parameter name used in the task contract. */
typedef enum VaistTokenizerType TokenizerType;

/** \brief Fixed-size byte->unicode map used by the byte-level BPE pre-tokenizer. */
#define VAIST_BYTE_MAP_SIZE 256u

/** \brief Token descriptor returned by introspection APIs. */
typedef struct VaistToken {
    uint32_t    id;                 /**< Vocabulary id assigned to this piece.          */
    float       score;              /**< Model score (e.g. unigram logprob); 0 if N/A. */
    uint8_t     type;               /**< SentencePiece type (1=normal..4=user_defined).  */
    char        piece[64];          /**< UTF-8 piece text, NUL-terminated (truncated).   */
} VaistToken;

/** \brief Read-only summary of an opened tokenizer. */
typedef struct VaistTokenizerInfo {
    uint32_t    vocab_size;      /**< Number of distinct token ids.                */
    uint32_t    max_length;      /**< Recommended context length (0 = unlimited).  */
    int32_t     bos_id;          /**< Beginning-of-sequence id (-1 if absent).     */
    int32_t     eos_id;          /**< End-of-sequence id (-1 if absent).           */
    int32_t     unk_id;          /**< Unknown-token id (-1 if absent).             */
    int32_t     pad_id;          /**< Padding id (-1 if absent).                   */
    uint32_t    byte_offset;     /**< Base offset for the byte tokenizer (>=0).    */
    TokenizerType type;          /**< Algorithm family backing this handle.        */
} VaistTokenizerInfo;

/** \brief Opaque handle to a loaded tokenizer. Lifetime is managed by the
 *         create/destroy pair. All other functions borrow it and are reentrant
 *         across distinct handles. */
typedef struct VaistTokenizer VaistTokenizer;

/** \brief Create a tokenizer of the requested algorithm family.
 *
 *  \param type      Algorithm family to instantiate (BPE / SentencePiece / Bytes).
 *  \param vocab_path  For BPE: directory containing vocab.json + merges.txt.
 *                     For SentencePiece: path to the .model protobuf file.
 *                     For Bytes: may be NULL (byte_offset defaults to 0).
 *  \param out       Receives the new handle on success.
 *  \retval VAIST_OK                Tokenizer loaded and ready.
 *  \retval VAIST_INVALID_ARGUMENT  \p out is NULL, or \p vocab_path missing for BPE/SP.
 *  \retval VAIST_IO_ERROR          A required file could not be opened/read.
 *  \retval VAIST_MODEL_ERROR       A file parsed but violated its format (bad JSON,
 *                                  bad protobuf, mismatched vocab/merges).
 *  \retval VAIST_UNSUPPORTED       \p type is unknown.
 *  \retval VAIST_OUT_OF_MEMORY     Heap exhausted while loading.
 */
VAIST_API VaistStatus vaist_tokenizer_create(TokenizerType type,
                                             const char *vocab_path,
                                             VaistTokenizer **out);

/** \brief Release a tokenizer created by vaist_tokenizer_create().
 *
 *  \param t  Handle previously returned (may be NULL; no-op).
 */
VAIST_API void vaist_tokenizer_destroy(VaistTokenizer *t);

/** \brief Tokenize a UTF-8 text string into a sequence of token ids.
 *
 *  Greedy BPE/WordPiece for BPE, longest-match lattice for SentencePiece
 *  (unigram Viterbi on first call path; full sampling requires an explicit
 *  sample API pass-by-pass). The output is truncated to the tokenizer's
 *  \c max_length when one is configured.
 *
 *  \param t      Tokenizer handle (borrowed).
 *  \param text   NUL-terminated UTF-8 input (may be empty).
 *  \param tokens Output buffer for ids; may be NULL if \p cap is 0.
 *  \param cap    Capacity of \p tokens (in elements).
 *  \param count  Receives the number of ids written.
 *  \retval VAIST_OK               Tokenization succeeded (output may be 0 ids).
 *  \retval VAIST_INVALID_ARGUMENT \p t / \p text / \p count is NULL.
 *  \retval VAIST_OUT_OF_MEMORY    \p cap was too small; \p count holds the needed size.
 */
VAIST_API VaistStatus vaist_tokenize(VaistTokenizer *t,
                                     const char *text,
                                     uint32_t *tokens,
                                     size_t cap,
                                     size_t *count);

/** \brief Detokenize a sequence of token ids back into UTF-8 text.
 *
 *  Reverses the tokenization: BPE decodes the unicode-mapped pieces back to
 *  bytes via the inverse byte->unicode map; SentencePiece concatenates piece
 *  surfaces; Bytes casts each id to its byte value.
 *
 *  \param t    Tokenizer handle (borrowed).
 *  \param tokens Input id array.
 *  \param n      Number of ids in \p tokens.
 *  \param out    Output buffer for UTF-8 text (NUL-terminated). May be NULL if
 *                \p cap is 0.
 *  \param cap    Capacity of \p out (in bytes, including the NUL).
 *  \param len    Receives the number of bytes written (excluding the NUL).
 *  \retval VAIST_OK               Detokenization succeeded.
 *  \retval VAIST_INVALID_ARGUMENT \p t / \p tokens is NULL, or \p n and \p out agree to write.
 *  \retval VAIST_OUT_OF_MEMORY    \p cap too small; \p len holds the needed size.
 */
VAIST_API VaistStatus vaist_detokenize(VaistTokenizer *t,
                                       const uint32_t *tokens,
                                       size_t n,
                                       char *out,
                                       size_t cap,
                                       size_t *len);

/** \brief Report the vocabulary size of a tokenizer.
 *
 *  \param t  Tokenizer handle (borrowed; may be NULL).
 *  \retval Number of distinct token ids (0 if \p t is NULL).
 */
VAIST_API uint32_t vaist_tokenizer_vocab_size(VaistTokenizer *t);

/** \brief Fill a VaistTokenizerInfo summary for an open tokenizer.
 *
 *  \param t    Tokenizer handle (borrowed).
 *  \param info Output summary (must not be NULL).
 *  \retval VAIST_OK               Summary written.
 *  \retval VAIST_INVALID_ARGUMENT \p t or \p info is NULL.
 */
VAIST_API VaistStatus vaist_tokenizer_info(const VaistTokenizer *t,
                                           VaistTokenizerInfo *info);

/** \brief Look up a single token's id by piece text.
 *
 *  \param t     Tokenizer handle (borrowed).
 *  \param piece NUL-terminated UTF-8 piece (e.g. a word or unicode-mapped token).
 *  \param id    Receives the id; set to (uint32_t)-1 if not found.
 *  \retval VAIST_OK               Lookup completed (not-found is reported via *id).
 *  \retval VAIST_INVALID_ARGUMENT \p t / \p piece / \p id is NULL.
 */
VAIST_API VaistStatus vaist_tokenizer_piece_to_id(VaistTokenizer *t,
                                                  const char *piece,
                                                  uint32_t *id);

/** \brief Sample-based tokenization for SentencePiece unigram models.
 *
 *  Performs full unigram lattice sampling (not just greedy Viterbi), drawing
 *  the top-\p nbest candidate sequence from the normalized piece-score lattice.
 *  Only meaningful for SentencePiece tokenizers; other types return the greedy
 *  result or VAIST_UNSUPPORTED. \p seed drives a local xorshift PRNG so results
 *  are reproducible for a given input.
 *
 *  \param t      Tokenizer handle (borrowed).
 *  \param text   NUL-terminated UTF-8 input.
 *  \param seed   PRNG seed for sampling.
 *  \param nbest  Number of best sample paths to consider (clamped to 1..64).
 *  \param tokens Output buffer for ids.
 *  \param cap    Capacity of \p tokens (in elements).
 *  \param count  Receives the number of ids written.
 *  \retval VAIST_OK               Sampling succeeded.
 *  \retval VAIST_INVALID_ARGUMENT \p t / \p text / \p tokens / \p count is NULL.
 *  \retval VAIST_UNSUPPORTED      Tokenizer is not a SentencePiece unigram model.
 *  \retval VAIST_OUT_OF_MEMORY    \p cap too small; \p count holds the needed size.
 */
VAIST_API VaistStatus vaist_tokenize_sample(VaistTokenizer *t,
                                            const char *text,
                                            uint64_t seed,
                                            uint32_t nbest,
                                            uint32_t *tokens,
                                            size_t cap,
                                            size_t *count);

#ifdef __cplusplus
}
#endif
#endif
