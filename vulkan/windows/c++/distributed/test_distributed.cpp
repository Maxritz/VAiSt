#include "vaist_distributed.hpp"
int main(){ VaistCommunicator c={0,0}; VaistStatus st=vaist::init(&c,0,1); return st==VAIST_OK && c.size==1?0:1; }
