#include "vaist_tokens.hpp"
int main() {
    VaistTokenizer* t = nullptr;
    vaist::create(VAIST_TOKENIZER_BYTES, nullptr, &t);
    uint32_t tokens[64];
    size_t count = 0;
    const char* text = "abc";
    vaist::tokenize(t, text, tokens, 64, &count);
    char buf[128];
    size_t written = 0;
    vaist::detokenize(t, tokens, count, buf, sizeof(buf), &written);
    int ok = (count == 3 && written == 3 && buf[0] == 'a');
    vaist::destroy(t);
    return ok ? 0 : 1;
}
