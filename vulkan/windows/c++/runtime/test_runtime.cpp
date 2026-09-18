#include "vaist_runtime.hpp"
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main(int argc, char **argv){
#if defined(_WIN32)
    if(argc>=2 && std::string(argv[1])=="VAIST_VK_PROBE=1"){
        std::fprintf(stderr,"[T] probe-child enter\n");
        int ok=0;
        vaist::Runtime rt(vaist::Runtime::VULKAN);
        void *dev=NULL, *q=NULL; uint32_t qf=0;
        ok = (vaist::runtime_vk_state(rt.get(), &dev, &q, &qf) == VAIST_OK) ? 1 : 0;
        std::fprintf(stderr,"[T] %s:probe-child ok=%d dev=%p q=%p\n",ok?"PASS":"FAIL",ok,dev,q);
        rt.reset();
        return ok ? 0 : 1;
    }
#endif

    vaist::Runtime rt(vaist::Runtime::CPU);
    int rc = rt.get() != nullptr ? 0 : 1;
    std::fprintf(stderr,"[T] %s:test_runtime rc=%d\n",rc?"FAIL":"PASS",rc);
    return rc;
}
