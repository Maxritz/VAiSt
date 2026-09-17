#include "vaist_compute.hpp"
int main(){ float a[1]={2},b[1]={3},o[1]={0}; VaistStatus st=vaist::matmul(a,b,o,1,1,1); return st==VAIST_OK && o[0]==6.0f?0:1; }
