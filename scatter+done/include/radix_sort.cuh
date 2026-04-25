#pragma once
#include <cstdint>

#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)

void radix_sort_gpu(uint32_t *d_input, uint32_t *d_output, int N);

void histogram_kernel(const uint32_t *input, uint32_t *hist, int N, int shift);
void scan_kernel(uint32_t *data, int size);
void scatter_kernel(const uint32_t *input, uint32_t *output, uint32_t *offsets, int N, int shift);