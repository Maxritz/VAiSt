#include "vaist_model.h"
namespace vaist { inline VaistModelInfo inspect(const char*p){VaistModelInfo i{};vaist_model_inspect(p,&i);return i;} }