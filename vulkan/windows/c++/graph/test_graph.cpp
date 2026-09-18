#include "vaist_graph.hpp"
#include <cstdio>
int main(){ vaist::Graph g; int rc=g.get()!=nullptr?0:1; std::fprintf(stderr,"[T] %s:test_graph rc=%d p=%p\n",rc?"FAIL":"PASS",rc,(void*)g.get()); return rc; }
