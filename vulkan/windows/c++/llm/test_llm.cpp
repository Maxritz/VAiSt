#include "vaist_llm.hpp"
#include <cstdio>
int main(){ vaist::KV k; VaistStatus s=k.create(1,2); int rc=s==VAIST_OK?0:1; std::fprintf(stderr,"[T] %s:test_llm rc=%d s=%d\n",rc?"FAIL":"PASS",rc,(int)s); return rc; }
