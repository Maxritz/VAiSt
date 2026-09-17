#include "vaist_runtime.hpp"
int main(){ vaist::Runtime r(VAIST_BACKEND_CPU); return r.get()!=nullptr?0:1; }
