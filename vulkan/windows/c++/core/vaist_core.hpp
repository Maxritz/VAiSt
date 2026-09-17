#include "vaist_core.h"
#include <string>
namespace vaist { inline std::string version(){return vaist_get_version();} inline VaistVersion abi(){return vaist_get_abi_version();} }