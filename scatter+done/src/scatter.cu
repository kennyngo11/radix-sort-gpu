#include <cuda_runtime.h>
#include <cstdint>
#include "../include/radix_sort.cuh"

// Counts how many keys of each digit appear in each thread block.
// This is used by scatter to keep the LSD radix sort stable.
__global__ void block_histogram_kernel(const uint32_t *input,
                                       uint32_t *blockHist,
                                       int N,
                                       int shift) {
    __shared__ uint32_t localHist[RADIX_SIZE];

    if (threadIdx.x < RADIX_SIZE) {
        localHist[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t digit = (input[idx] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&localHist[digit], 1);
    }
    __syncthreads();

    if (threadIdx.x < RADIX_SIZE) {
        blockHist[blockIdx.x * RADIX_SIZE + threadIdx.x] = localHist[threadIdx.x];
    }
}

// Stable scatter.
// final position = global bucket offset + previous-block count + in-block rank.
__global__ void scatter_kernel_cuda(const uint32_t *input,
                                    uint32_t *output,
                                    const uint32_t *offsets,
                                    const uint32_t *blockOffsets,
                                    int N,
                                    int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    uint32_t key = input[idx];
    uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);

    uint32_t localRank = 0;
    int blockStart = blockIdx.x * blockDim.x;

    // Count matching digits earlier in this same block only.
    // Combined with blockOffsets, this preserves the original order.
    for (int j = blockStart; j < idx; ++j) {
        uint32_t otherDigit = (input[j] >> shift) & (RADIX_SIZE - 1);
        if (otherDigit == digit) {
            ++localRank;
        }
    }

    uint32_t blockBase = blockOffsets[blockIdx.x * RADIX_SIZE + digit];
    uint32_t outPos = offsets[digit] + blockBase + localRank;
    output[outPos] = key;
}

void build_block_histogram(const uint32_t *input,
                           uint32_t *blockHist,
                           int N,
                           int shift,
                           int blockSize) {
    int gridSize = (N + blockSize - 1) / blockSize;
    block_histogram_kernel<<<gridSize, blockSize>>>(input, blockHist, N, shift);
    cudaDeviceSynchronize();
}

void scatter_kernel(const uint32_t *input,
                    uint32_t *output,
                    uint32_t *offsets,
                    uint32_t *blockOffsets,
                    int N,
                    int shift,
                    int blockSize) {
    int gridSize = (N + blockSize - 1) / blockSize;
    scatter_kernel_cuda<<<gridSize, blockSize>>>(input, output, offsets, blockOffsets, N, shift);
    cudaDeviceSynchronize();
}
