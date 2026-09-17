#include "vaist_nn.hpp"
int main(){ float x[2]={-1,2}; VaistStatus st=vaist::relu(x,2); return st==VAIST_OK && x[0]==0.0f?0:1; }
