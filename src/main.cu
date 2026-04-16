#include <iostream>
#include <cuda_runtime.h>

__global__ void testKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx;
}

int main() {
    const int N = 16;
    int *d_data;
    int h_data[N];

    cudaMalloc(&d_data, N * sizeof(int));

    testKernel<<<1, N>>>(d_data);

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_data);
    return 0;
}
