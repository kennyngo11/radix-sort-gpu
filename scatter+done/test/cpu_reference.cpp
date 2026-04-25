#include <algorithm>
#include <cstdint>
#include <vector>

void cpu_radix_sort_reference(std::vector<uint32_t> &data) {
    std::sort(data.begin(), data.end());
}
