#include "vaist_runtime.h"
#include <cstdio>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { std::fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)
#endif
namespace vaist {
class Runtime {
    VaistRuntime* p_;
public:
    enum Backend { AUTO = VAIST_BACKEND_AUTO, CPU = VAIST_BACKEND_CPU, VULKAN = VAIST_BACKEND_VULKAN };
    explicit Runtime(Backend b = AUTO) : p_(nullptr) {
        VaistStatus s = vaist_runtime_create((VaistBackend)b, &p_);
        DBG_TRACE("Runtime ctor backend=%d s=%d p=%p", (int)b, (int)s, (void*)p_);
        if (s != VAIST_OK) p_ = nullptr;
    }
    ~Runtime() { DBG_TRACE("Runtime dtor p=%p", (void*)p_); vaist_runtime_destroy(p_); }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    void reset() { DBG_TRACE("Runtime reset p=%p", (void*)p_); vaist_runtime_destroy(p_); p_ = nullptr; }
    VaistRuntime* get() const { return p_; }
    VaistRuntimeInfo info() const { VaistRuntimeInfo i{}; vaist_runtime_info(p_, &i); return i; }
};
inline int runtime_vk_state(VaistRuntime* rt, void** dev, void** q, uint32_t* qf) {
    int r = vaist_runtime_vk_state(rt, dev, q, qf);
    DBG_TRACE("runtime_vk_state r=%d dev=%p q=%p", r, dev ? *dev : nullptr, q ? *q : nullptr);
    return r;
}
}
