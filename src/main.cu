#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include "../include/radix_sort.cuh"

int main() {
    const int N = 8;
    uint32_t h_input[N] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32_t *d_input, *d_hist;
    uint32_t h_hist[RADIX_SIZE];

    cudaMalloc(&d_input, N * sizeof(uint32_t));
    cudaMalloc(&d_hist, RADIX_SIZE * sizeof(uint32_t));

    cudaMemcpy(d_input, h_input, N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    histogram_kernel(d_input, d_hist, N, 0);

    cudaMemcpy(h_hist, d_hist, RADIX_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < RADIX_SIZE; i++) {
        std::cout << "bin " << i << ": " << h_hist[i] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_hist);
    return 0;
}
