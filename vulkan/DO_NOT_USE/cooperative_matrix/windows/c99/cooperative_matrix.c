#include <stdint.h>
typedef enum { VAIST_COOPMAT_DO_NOT_USE=0, VAIST_COOPMAT_DISABLED=1 } VaistCooperativeMatrixStatus;
VaistCooperativeMatrixStatus vaist_cooperative_matrix_status(void){return VAIST_COOPMAT_DISABLED;}
