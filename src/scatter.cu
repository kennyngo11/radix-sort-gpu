#include <cuda_runtime.h>
#include <cstdint>
#include "../include/radix_sort.cuh"

__global__ void scatter_kernel_cuda(const uint32_t *input,
                                    uint32_t *output,
                                    uint32_t *offsets,
                                    int N,
                                    int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t key = input[idx];
        uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);

        uint32_t pos = atomicAdd(&offsets[digit], 1);
        output[pos] = key;
    }
}

void scatter_kernel(const uint32_t *input,
                    uint32_t *output,
                    uint32_t *offsets,
                    int N,
                    int shift) {

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    scatter_kernel_cuda<<<gridSize, blockSize>>>(input, output, offsets, N, shift);
    cudaDeviceSynchronize();
}
