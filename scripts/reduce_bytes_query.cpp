//  查 `reduce_bytes` —— ReduceMax/ReduceSum 的 UB 暂存尺寸, UB 预算里唯一的运行时量。
//
//  ★为什么单独写它: 这个数此前只知道"运行时由 GetReduceSumMaxMinTmpSize 定", 数值从未记过,
//    而 tile_m=128 的 UB 预算差的就是它。走内核构建去问要 4 分钟, 这样问只要 10 秒。
//
//  编译 (本机实测的最小链接集, 少一个就报 DSO missing):
//    C=/usr/local/Ascend/ascend-toolkit/latest
//    g++ -std=c++17 -o /tmp/rb scripts/reduce_bytes_query.cpp \
//        -I$C/include -I$C/aarch64-linux/include -I$C/aarch64-linux/asc/include \
//        -L$C/aarch64-linux/lib64 -ltiling_api -lplatform -lgraph -lgraph_base \
//        -lregister -lunified_dlog -lascendalog
//    LD_LIBRARY_PATH=$C/aarch64-linux/lib64:$LD_LIBRARY_PATH /tmp/rb
//
//  2026-10-30 本机结果 (N=128): tile_m=64 -> 8192 · tile_m=128 -> **16384**, 即 ROWS*N*2。
//  口径与 vc_torch.cpp 的 ForwardImpl 完全一致 (同一对 API、同一取整与下限)。
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>
#include <algorithm>
int main() {
    for (uint32_t tile_m : {64u, 128u}) {
        uint32_t max_tmp = 0, min_tmp = 0, sum_tmp = 0;
        ge::Shape shape({tile_m / 2, 128});
        AscendC::GetReduceMaxMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                           true, false, max_tmp, min_tmp);
        AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                           true, false, sum_tmp, min_tmp);
        uint32_t rb = (std::max(max_tmp, sum_tmp) + 31) / 32 * 32;
        rb = std::max(rb, uint32_t(32));
        printf("tile_m=%u  rows=%u  max_tmp=%u sum_tmp=%u -> reduce_bytes=%u\n",
               tile_m, tile_m/2, max_tmp, sum_tmp, rb);
    }
    return 0;
}
