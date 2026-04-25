#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "../include/radix_sort.cuh"

static void check_cuda(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
    }
}

static void build_block_offsets_on_host(uint32_t *d_blockHist,
                                        uint32_t *d_blockOffsets,
                                        int numBlocks) {
    std::vector<uint32_t> h_blockHist(numBlocks * RADIX_SIZE);
    std::vector<uint32_t> h_blockOffsets(numBlocks * RADIX_SIZE, 0);

    check_cuda(cudaMemcpy(h_blockHist.data(),
                          d_blockHist,
                          h_blockHist.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               "copy block histograms to host");

    for (int digit = 0; digit < RADIX_SIZE; ++digit) {
        uint32_t running = 0;
        for (int block = 0; block < numBlocks; ++block) {
            int pos = block * RADIX_SIZE + digit;
            h_blockOffsets[pos] = running;
            running += h_blockHist[pos];
        }
    }

    check_cuda(cudaMemcpy(d_blockOffsets,
                          h_blockOffsets.data(),
                          h_blockOffsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "copy block offsets to device");
}

// Full LSD radix sort pipeline.
// Uses the provided histogram_kernel from src/histogram.cu and scan_kernel from src/scan.cu.
void radix_sort_gpu(uint32_t *d_input, uint32_t *d_output, int N) {
    if (N <= 0) {
        return;
    }

    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    uint32_t *d_hist = nullptr;
    uint32_t *d_offsets = nullptr;
    uint32_t *d_blockHist = nullptr;
    uint32_t *d_blockOffsets = nullptr;
    uint32_t *d_temp = nullptr;

    check_cuda(cudaMalloc(&d_hist, RADIX_SIZE * sizeof(uint32_t)), "malloc histogram");
    check_cuda(cudaMalloc(&d_offsets, RADIX_SIZE * sizeof(uint32_t)), "malloc offsets");
    check_cuda(cudaMalloc(&d_blockHist, numBlocks * RADIX_SIZE * sizeof(uint32_t)), "malloc block histogram");
    check_cuda(cudaMalloc(&d_blockOffsets, numBlocks * RADIX_SIZE * sizeof(uint32_t)), "malloc block offsets");
    check_cuda(cudaMalloc(&d_temp, N * sizeof(uint32_t)), "malloc temp buffer");

    uint32_t *readBuf = d_input;
    uint32_t *writeBuf = d_temp;

    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        // P1: global digit histogram, using the provided implementation.
        histogram_kernel(readBuf, d_hist, N, shift);

        // P2: exclusive prefix scan of the global histogram.
        check_cuda(cudaMemcpy(d_offsets,
                              d_hist,
                              RADIX_SIZE * sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice),
                   "copy histogram into offsets");
        scan_kernel(d_offsets, RADIX_SIZE);

        // Part 3: stable scatter needs per-block prefix counts too.
        build_block_histogram(readBuf, d_blockHist, N, shift, blockSize);
        build_block_offsets_on_host(d_blockHist, d_blockOffsets, numBlocks);
        scatter_kernel(readBuf, writeBuf, d_offsets, d_blockOffsets, N, shift, blockSize);

        uint32_t *tmp = readBuf;
        readBuf = writeBuf;
        writeBuf = tmp;
    }

    // There are 8 passes with RADIX_BITS = 4. Copy final data to d_output.
    check_cuda(cudaMemcpy(d_output,
                          readBuf,
                          N * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice),
               "copy final sorted data to output");

    cudaFree(d_hist);
    cudaFree(d_offsets);
    cudaFree(d_blockHist);
    cudaFree(d_blockOffsets);
    cudaFree(d_temp);
}
