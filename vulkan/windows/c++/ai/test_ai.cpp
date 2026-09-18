#include "vaist_ai.hpp"
#include <cstdio>
int main(){ float a[1]={1},b[1]={1},o=0; VaistStatus st=vaist::cosine(a,b,1,&o); int rc=st==VAIST_OK?0:1; std::fprintf(stderr,"[T] %s:test_ai rc=%d st=%d o=%f\n",rc?"FAIL":"PASS",rc,(int)st,o); return rc; }
