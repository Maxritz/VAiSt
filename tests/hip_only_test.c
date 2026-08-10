#define __HIP_PLATFORM_AMD__ 1
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Starting HIP minimal test...\n");

    void* ptr = NULL;
    hipError_t e = hipHostMalloc(&ptr, 1024, hipHostMallocMapped);
    printf("hipHostMalloc result: %d, ptr=%p\n", e, ptr);
    if (e != hipSuccess) {
        printf("FAIL: hipHostMalloc\n");
        return 1;
    }
    printf("PASS: HIP allocated 1024 bytes\n");

    float* fp = (float*)ptr;
    fp[0] = 42.0f;
    printf("PASS: HIP write OK\n");

    hipDeviceSynchronize();
    printf("PASS: HIP sync OK\n");

    hipHostFree(ptr);
    printf("PASS: HIP free OK\n");
    printf("DONE\n");
    return 0;
}
