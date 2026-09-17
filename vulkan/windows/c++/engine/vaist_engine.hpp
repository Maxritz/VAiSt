#include "vaist_engine.h"
namespace vaist { class Engine { VaistEngine*p_; public: Engine():p_(nullptr){vaist_engine_create(&p_);} ~Engine(){vaist_engine_destroy(p_);} }; }