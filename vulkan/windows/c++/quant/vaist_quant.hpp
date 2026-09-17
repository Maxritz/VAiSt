#include "vaist_quant.h"
namespace vaist { inline VaistStatus quantize(VaistQuantType q,const float*s,size_t n,void*d,size_t c,size_t*u){return vaist_quantize_f32(q,s,n,d,c,u);} }