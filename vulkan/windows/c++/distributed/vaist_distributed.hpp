#include "vaist_distributed.h"
namespace vaist { inline VaistStatus init(VaistCommunicator*c,uint32_t r,uint32_t s){return vaist_comm_init(c,r,s);} }