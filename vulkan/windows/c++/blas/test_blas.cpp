#include "vaist_blas.hpp"
int main(){
    float x[2]={1.0f,2.0f}; int8_t s[2]={1,1}; float sc[1]={1.0f}, y[1]={0};
    VaistRuntime*rt=0; vaist_runtime_create(VAIST_BACKEND_CPU,&rt);
    VaistStatus st=vaist::matvec_ternary(rt,s,sc,2,1,x,y);
    int ok = (st==VAIST_OK && (y[0]==3.0f));
    vaist_runtime_destroy(rt);
    return ok?0:1;
}
