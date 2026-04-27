#include <cuda_runtime.h>
#include <cstdint>
#include "../include/radix_sort.cuh"

__global__ void scatter_kernel_cuda(const uint32_t *input,
                                     uint32_t *output,
                                     const uint32_t *globalOffsets,
                                     const uint32_t *blockOffsets,
                                     int N,
                                     int shift) {
    __shared__ uint32_t local_offsets[RADIX_SIZE];

    // Load this block's offsets into shared memory
    if (threadIdx.x < RADIX_SIZE)
        local_offsets[threadIdx.x] = blockOffsets[blockIdx.x * RADIX_SIZE + threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t key = input[idx];
        uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);
        uint32_t pos = globalOffsets[digit] + local_offsets[digit];
        atomicAdd(&((uint32_t *)blockOffsets)[blockIdx.x * RADIX_SIZE + digit], 1);
        output[pos] = key;
    }
}

void scatter_kernel(const uint32_t *input,
                    uint32_t *output,
                    uint32_t *globalOffsets,
                    uint32_t *blockOffsets,
                    int N,
                    int shift,
                    int blockSize) {
    int gridSize = (N + blockSize - 1) / blockSize;
    scatter_kernel_cuda<<<gridSize, blockSize>>>(input, output, globalOffsets, blockOffsets, N, shift);
    cudaDeviceSynchronize();
}