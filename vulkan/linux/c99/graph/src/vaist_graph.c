#include "vaist_graph.h"
#include <stdlib.h>
struct VaistGraph{VaistGraphOp*ops;size_t count,cap;uint32_t compiled;};
VAIST_API VaistStatus vaist_graph_create(VaistGraph**o){VaistGraph*g;if(!o)return VAIST_INVALID_ARGUMENT;g=(VaistGraph*)calloc(1,sizeof(*g));if(!g)return VAIST_OUT_OF_MEMORY;*o=g;return VAIST_OK;}
VAIST_API void vaist_graph_destroy(VaistGraph*g){if(g){free(g->ops);free(g);}}
VAIST_API VaistStatus vaist_graph_add_binary(VaistGraph*g,VaistGraphOp op,VaistNodeId*id){VaistGraphOp*p;if(!g||!op||!id)return VAIST_INVALID_ARGUMENT;if(g->compiled)return VAIST_INVALID_STATE;if(g->count==g->cap){size_t nc=g->cap?g->cap*2:8;p=(VaistGraphOp*)realloc(g->ops,nc*sizeof(*p));if(!p)return VAIST_OUT_OF_MEMORY;g->ops=p;g->cap=nc;}g->ops[g->count]=op;*id=(VaistNodeId)g->count;g->count++;return VAIST_OK;}
VAIST_API VaistStatus vaist_graph_compile(VaistGraph*g){if(!g||!g->count)return VAIST_INVALID_ARGUMENT;g->compiled=1;return VAIST_OK;}
VAIST_API VaistStatus vaist_graph_execute(VaistGraph*g,const float*a,const float*b,float*o,size_t n){size_t i;if(!g||!g->compiled||!a||!b||!o||!n)return VAIST_INVALID_STATE;if(g->count==1)return g->ops[0](a,b,o,n);{float*tmp=(float*)malloc(n*sizeof(float));if(!tmp)return VAIST_OUT_OF_MEMORY;if(g->ops[0](a,b,tmp,n)!=VAIST_OK){free(tmp);return VAIST_RUNTIME_ERROR;}for(i=1;i<g->count;i++){VaistStatus s=g->ops[i](tmp,b,o,n);if(s!=VAIST_OK){free(tmp);return s;}for(size_t j=0;j<n;j++)tmp[j]=o[j];}for(i=0;i<n;i++)o[i]=tmp[i];free(tmp);}return VAIST_OK;}
VAIST_API uint32_t vaist_graph_node_count(const VaistGraph*g){return g?(uint32_t)g->count:0;}
