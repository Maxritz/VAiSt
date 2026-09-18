#include "vaist_graph.h"
#include <cstdio>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { std::fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)
#endif
namespace vaist { class Graph { VaistGraph*p_; public: Graph():p_(nullptr){vaist_graph_create(&p_); DBG_TRACE("Graph ctor p=%p",(void*)p_);} ~Graph(){DBG_TRACE("Graph dtor p=%p",(void*)p_); vaist_graph_destroy(p_);} VaistGraph*get()const{return p_;} }; }
