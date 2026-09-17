#include "vaist_core.hpp"
#include "vaist_runtime.hpp"
#include "vaist_tensor.hpp"
#include "vaist_engine.hpp"
#include <iostream>
int main(){vaist::Runtime r(VAIST_BACKEND_CPU); if(!r.get()) return 1; vaist::Engine e; std::cout<<vaist::version()<<"\n"; return 0;}