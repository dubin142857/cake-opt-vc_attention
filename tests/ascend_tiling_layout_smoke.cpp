// Host C++ contract check. Does not compile or execute an Ascend C kernel.
// c++ -std=c++17 -Iascend/csrc tests/ascend_tiling_layout_smoke.cpp -o tiling_smoke
#include "vc_tiling_host.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

static_assert(offsetof(vc910::Tiling, workspace_bytes) == 44, "Reuse the reserved slot");
static_assert(offsetof(vc910::Tiling, tasks) == 48, "Preserve serialized task offset");

int main() {
    for (const uint32_t dim : {64u, 128u}) {
        for (const uint32_t cores : {1u, 24u, 32u}) {
            vc910::Tiling host{};
            host.dim = dim;
            host.cores = cores;
            host.workspace_bytes = uint32_t(vc910::WorkspaceBytes(dim));
            host.tasks = (uint64_t(1) << 32) + 17;  // tasks must remain 64-bit
            unsigned char serialized[sizeof(host)];
            std::memcpy(serialized, &host, sizeof(host));
            vc910::Tiling device{};
            std::memcpy(&device, serialized, sizeof(device));
            assert(device.tasks == host.tasks && device.dim == dim);
            assert(device.workspace_bytes == (dim == 64 ? 32768u : 40960u));

            // Every byte is claimed exactly once by S, P, or one AIV's PV half.
            // Independent tile sizes verify no gap/overlap across cores/halves.
            std::vector<unsigned char> claimed(size_t(cores) * host.workspace_bytes, 0);
            const auto claim = [&](uint64_t begin, uint64_t bytes) {
                assert(begin + bytes <= claimed.size());
                for (uint64_t i = begin; i < begin + bytes; ++i) {
                    assert(claimed[i] == 0);
                    claimed[i] = 1;
                }
            };
            for (uint32_t core = 0; core < cores; ++core) {
                const uint64_t base = uint64_t(core) * device.workspace_bytes;
                assert(base % 512 == 0);
                claim(base, 32 * 128 * 4);               // INT32 S
                claim(base + 32 * 128 * 4, 32 * 128 * 2); // FP16 P
                for (uint32_t sub = 0; sub < 2; ++sub)
                    claim(base + 32 * 128 * 6 + sub * 16 * dim * 4, 16 * dim * 4);
            }
            for (auto byte : claimed) assert(byte == 1);
        }
    }
    std::cout << "Tiling serialization and D64/D128 workspace partitions: passed\n";
}
