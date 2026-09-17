#ifndef VAIST_GRAPH_H
#define VAIST_GRAPH_H
#include "vaist_core.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VaistGraph VaistGraph;
typedef uint32_t VaistNodeId;
typedef VaistStatus (*VaistGraphOp)(const float*,const float*,float*,size_t);
VAIST_API VaistStatus vaist_graph_create(VaistGraph **out);
VAIST_API void vaist_graph_destroy(VaistGraph*g);
VAIST_API VaistStatus vaist_graph_add_binary(VaistGraph*g,VaistGraphOp op,VaistNodeId *out);
VAIST_API VaistStatus vaist_graph_compile(VaistGraph*g);
VAIST_API VaistStatus vaist_graph_execute(VaistGraph*g,const float*a,const float*b,float*out,size_t n);
VAIST_API uint32_t vaist_graph_node_count(const VaistGraph*g);
#ifdef __cplusplus
}
#endif
#endif