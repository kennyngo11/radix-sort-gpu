#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "../include/radix_sort.cuh"

int main() {
    const int N = 1 << 15;
    std::vector<uint32_t> h_input(N);
    std::vector<uint32_t> h_output(N);
    std::vector<uint32_t> h_expected(N);

    std::mt19937 rng(12345);
    for (int i = 0; i < N; ++i) {
        h_input[i] = rng();
    }
    h_expected = h_input;
    std::sort(h_expected.begin(), h_expected.end());

    uint32_t *d_input = nullptr;
    uint32_t *d_output = nullptr;

    cudaMalloc(&d_input, N * sizeof(uint32_t));
    cudaMalloc(&d_output, N * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    radix_sort_gpu(d_input, d_output, N);

    cudaMemcpy(h_output.data(), d_output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    bool ok = (h_output == h_expected);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    if (!ok) {
        for (int i = 0; i < N; ++i) {
            if (h_output[i] != h_expected[i]) {
                std::cout << "first mismatch at " << i
                          << ": gpu=" << h_output[i]
                          << " cpu=" << h_expected[i] << std::endl;
                break;
            }
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return ok ? 0 : 1;
}