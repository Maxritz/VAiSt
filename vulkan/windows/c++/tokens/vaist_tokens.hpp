#pragma once
#include "vaist_tokens.h"
namespace vaist {
inline VaistStatus create(TokenizerType type, const char* vocab_path, VaistTokenizer** t) {
    return vaist_tokenizer_create(type, vocab_path, t);
}
inline void destroy(VaistTokenizer* t) {
    vaist_tokenizer_destroy(t);
}
inline VaistStatus tokenize(VaistTokenizer* t, const char* text, uint32_t* tokens, size_t cap, size_t* count) {
    return vaist_tokenize(t, text, tokens, cap, count);
}
inline VaistStatus detokenize(VaistTokenizer* t, const uint32_t* tokens, size_t n, char* buf, size_t buf_size, size_t* out_written) {
    return vaist_detokenize(t, tokens, n, buf, buf_size, out_written);
}
}
