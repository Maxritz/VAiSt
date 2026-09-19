#include "vaist_core.hpp"
#include "vaist_runtime.hpp"
#include "vaist_tensor.hpp"
#include "vaist_compute.hpp"
#include "vaist_quant.hpp"
#include "vaist_graph.hpp"
#include "vaist_model.hpp"
#include "vaist_nn.hpp"
#include "vaist_llm.hpp"
#include "vaist_engine.hpp"
#include "vaist_ai.hpp"
#include "vaist_distributed.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

int main()
{
    assert(vaist::abi().major == VAIST_ABI_MAJOR);
    assert(!vaist::version().empty());

    vaist::Runtime runtime(vaist::Runtime::CPU);
    assert(runtime.get() != nullptr);

    vaist::Tensor tensor;
    const uint64_t shape[2] = {2u, 2u};
    assert(tensor.create(VAIST_F32, shape, 2u) == VAIST_OK);
    assert(tensor.get() != nullptr);

    float a[4] = {1.f, 2.f, 3.f, 4.f};
    float b[4] = {4.f, 3.f, 2.f, 1.f};
    float out[4] = {0.f, 0.f, 0.f, 0.f};
    assert(vaist::matmul(a, b, out, 2u, 2u, 2u) == VAIST_OK);
    assert(vaist::relu(out, 4u) == VAIST_OK);

    unsigned char packed[64] = {0};
    size_t used = 0u;
    assert(vaist::quantize(VAIST_Q8_0, a, 4u, packed, sizeof(packed), &used) == VAIST_OK);
    assert(used > 0u);

    vaist::Graph graph;
    assert(graph.get() != nullptr);

    (void)vaist::inspect("nonexistent.model");

    vaist::KV kv;
    assert(kv.create(2u, 4u) == VAIST_OK);

    /* Engine: VaistEngine lives in cpp/model/vaist_engine.hpp (not yet linked here) */
    (void)0;

    VaistCommunicator comm{};
    assert(vaist::init(&comm, 0u, 1u) == VAIST_OK);

    assert(vaist::cosine(a, b, 4u, out) == VAIST_OK);
    return 0;
}
