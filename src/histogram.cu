#include <cuda_runtime.h>
#include <cstdint>
#include "../include/radix_sort.cuh"

// Per-block histogram kernel
// Each block counts digit frequencies into shared memory, then adds to global histogram
__global__ void histogram_kernel_cuda(const uint32_t *input,
                                       uint32_t *hist,
                                       int N,
                                       int shift) {
    __shared__ uint32_t local_hist[RADIX_SIZE];

    // Each thread initializes part of shared histogram
    if (threadIdx.x < RADIX_SIZE)
        local_hist[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t digit = (input[idx] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write shared histogram to global
    if (threadIdx.x < RADIX_SIZE)
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
}

// Per-block histogram (used for stable scatter offsets)
__global__ void block_histogram_kernel(const uint32_t *input,
                                        uint32_t *blockHist,
                                        int N,
                                        int shift,
                                        int blockSize) {
    __shared__ uint32_t local_hist[RADIX_SIZE];

    if (threadIdx.x < RADIX_SIZE)
        local_hist[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t digit = (input[idx] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    if (threadIdx.x < RADIX_SIZE)
        blockHist[blockIdx.x * RADIX_SIZE + threadIdx.x] = local_hist[threadIdx.x];
}

void histogram_kernel(const uint32_t *input, uint32_t *hist, int N, int shift) {
    // Zero out histogram first
    cudaMemset(hist, 0, RADIX_SIZE * sizeof(uint32_t));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    histogram_kernel_cuda<<<gridSize, blockSize>>>(input, hist, N, shift);
    cudaDeviceSynchronize();
}

void build_block_histogram(const uint32_t *input, uint32_t *blockHist,
                            int N, int shift, int blockSize) {
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(blockHist, 0, gridSize * RADIX_SIZE * sizeof(uint32_t));
    block_histogram_kernel<<<gridSize, blockSize>>>(input, blockHist, N, shift, blockSize);
    cudaDeviceSynchronize();
}