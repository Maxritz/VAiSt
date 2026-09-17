#include "vaist_ai.hpp"
int main(){ float a[1]={1},b[1]={1},o=0; VaistStatus st=vaist::cosine(a,b,1,&o); return st==VAIST_OK?0:1; }
