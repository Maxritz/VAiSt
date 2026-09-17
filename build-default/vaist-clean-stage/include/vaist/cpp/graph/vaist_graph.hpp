#include "vaist_graph.h"
namespace vaist { class Graph { VaistGraph*p_; public: Graph():p_(nullptr){vaist_graph_create(&p_);} ~Graph(){vaist_graph_destroy(p_);} VaistGraph*get()const{return p_;} }; }