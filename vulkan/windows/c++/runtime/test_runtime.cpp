#include "vaist_runtime.hpp"
#include <string>
#include <cstddef>
#include <cstdint>

int main(int argc, char **argv){
#if defined(_WIN32)
    if(argc>=2 && std::string(argv[1])=="VAIST_VK_PROBE=1"){
        int ok=0;
        vaist::Runtime rt(vaist::Runtime::Backend::VAIST_BACKEND_VULKAN);
        void *dev=NULL,*q=NULL; uint32_t qf=0;
        ok = (vaist::runtime_vk_state(rt.get(),&dev,&q,&qf)==VAIST_OK) ? 1:0;
        rt.reset();
        return ok?0:1;
    }
#endif

    vaist::Runtime rt(vaist::Runtime::Backend::VAIST_BACKEND_CPU);
    return rt.get()!=nullptr?0:1;
}
