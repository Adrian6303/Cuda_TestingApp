#include <stdio.h>

__global__ void testCUDA() {
    printf("CUDA is working! Thread %d\n", threadIdx.x);
}

int main() {
    testCUDA<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
// Compile and run
// nvcc test_cuda.cu -o test_cuda
// ./test_cuda