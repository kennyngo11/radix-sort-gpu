#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include "../include/radix_sort.cuh"

__global__ void histogram_kernel_cuda(const uint32_t *input, uint32_t *hist, int N, int shift) {
    __shared__ uint32_t localHist[RADIX_SIZE];

    if (threadIdx.x < RADIX_SIZE) {
        localHist[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t key = input[idx];
        uint32_t digit = (key >> shift) & 0xF;
        atomicAdd(&localHist[digit], 1);
    }
    __syncthreads();

    if (threadIdx.x < RADIX_SIZE) {
        atomicAdd(&hist[threadIdx.x], localHist[threadIdx.x]);
    }
}

void histogram_kernel(const uint32_t *input, uint32_t *hist, int N, int shift) {
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    cudaMemset(hist, 0, RADIX_SIZE * sizeof(uint32_t));
    histogram_kernel_cuda<<<gridSize, blockSize>>>(input, hist, N, shift);
    cudaDeviceSynchronize();
}
