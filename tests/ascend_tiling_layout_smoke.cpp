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
  for (const uint32_t tile_m : {32u, 64u}) {
    for (const uint32_t dim : {64u, 128u}) {
        for (const uint32_t cores : {1u, 24u, 32u}) {
            vc910::Tiling host{};
            host.dim = dim;
            host.cores = cores;
            host.workspace_bytes = uint32_t(vc910::WorkspaceBytes(dim, tile_m));
            host.tasks = (uint64_t(1) << 32) + 17;  // tasks must remain 64-bit
            unsigned char serialized[sizeof(host)];
            std::memcpy(serialized, &host, sizeof(host));
            vc910::Tiling device{};
            std::memcpy(&device, serialized, sizeof(device));
            assert(device.tasks == host.tasks && device.dim == dim);
            assert(device.workspace_bytes == (dim == 64 ? 32768u : 40960u) * (tile_m / 32));
            assert(vc910::VectorBytes(0, tile_m) == (tile_m == 32 ? 39424u : 77824u));

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
                claim(base, tile_m * 128 * 4);                       // INT32 S
                claim(base + tile_m * 128 * 4, tile_m * 128 * 2);     // FP16 P
                for (uint32_t sub = 0; sub < 2; ++sub)
                    claim(base + tile_m * 128 * 6 + sub * (tile_m / 2) * dim * 4,
                          (tile_m / 2) * dim * 4);
            }
            for (auto byte : claimed) assert(byte == 1);
        }
    }
    // Each AIV must use exactly one ORIGINAL Q32 scale, including tail padding
    // and non-contiguous tasks assigned to a physical core across head changes.
    for (uint32_t queries : {1u, 17u, 32u, 33u, 63u, 64u, 65u, 1025u, 37741u}) {
      for (uint32_t heads : {1u, 2u, 65u}) {
        const uint32_t qpad = (queries + tile_m - 1) / tile_m * tile_m;
        const uint32_t tasks = heads * (qpad / tile_m);
        const uint32_t cores = tasks < 24 ? tasks : 24;
        std::vector<unsigned char> rows(size_t(heads) * queries, 0);
        for (uint32_t core = 0; core < cores; ++core) {
          for (uint32_t task = core; task < tasks; task += cores) {
            const uint32_t head = task / (qpad / tile_m);
            for (uint32_t sub = 0; sub < 2; ++sub) {
              const uint32_t qr = (task % (qpad / tile_m)) * tile_m + sub * (tile_m / 2);
              const uint32_t qscale = head * (qpad / vc910::Q_SCALE_BLOCK) + qr / vc910::Q_SCALE_BLOCK;
              assert(qscale < heads * (qpad / 32));
              for (uint32_t r = 0; r < tile_m / 2; ++r) {
                assert(qr / 32 == (qr + r) / 32);
                if (qr + r < queries) assert(rows[head * queries + qr + r]++ == 0);
              }
            }
          }
        }
        for (auto count : rows) assert(count == 1);
      }
    }
  }
    // Diagnostic writers own whole 64-byte cache lines, including across roles.
    static_assert(vc910::PROFILE_FIELDS * sizeof(uint64_t) % 64 == 0, "No cache-line sharing");
    std::cout << "M32/M64 tiling, workspace, Q32 scale ownership and tail coverage: passed\n";
}
