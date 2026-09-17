#include "vaist_tensor.hpp"
int main(){ vaist::Tensor t; const uint64_t s[1]={4}; VaistStatus st=t.create(VAIST_F32,s,1); return st==VAIST_OK && t.get()!=nullptr?0:1; }
