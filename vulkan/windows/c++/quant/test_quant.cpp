#include "vaist_quant.hpp"
#include <cstddef>
int main(){ float x[32]={}; unsigned char d[256]={}; size_t u=0; VaistStatus st=vaist::quantize(VAIST_Q8_0,x,32,d,sizeof(d),&u); return st==VAIST_OK && u>0?0:1; }
