// vc28 candidate: balanced INT32 row-max tree. Apache-2.0.
// Included inside sage_vc after work_reduction.cuh.
#pragma once

template<uint32_t NK>
__device__ __forceinline__ int vc_integer_row_max_balanced(
    const int32_t (&scores)[NK][8], int r) {
  static_assert(NK==8);
  // Row order is k-major, four values per fragment at [2r,2r+1,2r+4,2r+5].
  // 32 leaves -> ten triples + two leaves -> three triples + one triple
  // -> one triple -> final pair. Fifteen max3 and one max2, as in vc28,
  // but the longest source dependency chain is four max nodes, not six.
  int partial[10];
  #pragma unroll
  for (int t=0;t<10;++t) {
    constexpr int stride=3;
    int a=t*stride,b=a+1,c=a+2;
    partial[t]=__vimax3_s32(
        scores[a/4][r*2+(a%4/2)*4+a%2],
        scores[b/4][r*2+(b%4/2)*4+b%2],
        scores[c/4][r*2+(c%4/2)*4+c%2]);
  }
  int a=__vimax3_s32(partial[0],partial[1],partial[2]);
  int b=__vimax3_s32(partial[3],partial[4],partial[5]);
  int c=__vimax3_s32(partial[6],partial[7],partial[8]);
  int d=__vimax3_s32(partial[9],scores[7][r*2+4],scores[7][r*2+5]);
  return max(__vimax3_s32(a,b,c),d);
}
