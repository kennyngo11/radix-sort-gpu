#include "radix_sort.cuh"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void blelloch_scan_kernel(uint32_t *data, int n) {
    extern __shared__ uint32_t temp[];

    int tid = threadIdx.x;
    temp[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();

    // Up sweep (reduce phase)
    for (int stride = 1; stride < n; stride <<= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n)
            temp[idx] += temp[idx - stride];
        __syncthreads();
    }

    // Set last element to identity (exclusive scan)
    if (tid == 0) temp[n - 1] = 0;
    __syncthreads();

    // Down sweep phase
    for (int stride = n >> 1; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n) {
            uint32_t left = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += left;
        }
        __syncthreads();
    }

    if (tid < n) data[tid] = temp[tid];
}

void scan_kernel(uint32_t *d_data, int size) {
    // size = 16 fits in one block, launch exactly that many threads
    int threads = size; // 16
    size_t sharedMem = size * sizeof(uint32_t);
    blelloch_scan_kernel<<<1, threads, sharedMem>>>(d_data, size);
    cudaDeviceSynchronize();
}