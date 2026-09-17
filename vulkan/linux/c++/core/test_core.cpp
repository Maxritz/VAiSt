#include "vaist_core.hpp"
int main(){ VaistVersion v=vaist::abi(); return v.major==VAIST_ABI_MAJOR?0:1; }
