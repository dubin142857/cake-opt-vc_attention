// Split issue/wait for the lookahead schedule. Retained kernels still use
// the original blocking primitive. Semantics: NVIDIA PTX wgmma.wait_group;
// compiler operand fencing follows the approach documented by CUTLASS.
#pragma once

__device__ __forceinline__ void wgmma_issue(float (&c)[16][4], uint8_t* a, uint8_t* b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    // Commit one product; its output is inaccessible until wgmma_ready.
    asm volatile(
        "{\n.reg .pred p0, p1;\nsetp.ne.b32 p0, 0, 0;\nsetp.ne.b32 p1, 1, 0;\n"
        "wgmma.fence.sync.aligned;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p0, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %66, %67, p1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %68, %69, p1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %70, %71, p1, 1, 1;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "}\n"
        : "=&f"(c[0][0]),
          "=&f"(c[0][1]),
          "=&f"(c[0][2]),
          "=&f"(c[0][3]),
          "=&f"(c[1][0]),
          "=&f"(c[1][1]),
          "=&f"(c[1][2]),
          "=&f"(c[1][3]),
          "=&f"(c[2][0]),
          "=&f"(c[2][1]),
          "=&f"(c[2][2]),
          "=&f"(c[2][3]),
          "=&f"(c[3][0]),
          "=&f"(c[3][1]),
          "=&f"(c[3][2]),
          "=&f"(c[3][3]),
          "=&f"(c[4][0]),
          "=&f"(c[4][1]),
          "=&f"(c[4][2]),
          "=&f"(c[4][3]),
          "=&f"(c[5][0]),
          "=&f"(c[5][1]),
          "=&f"(c[5][2]),
          "=&f"(c[5][3]),
          "=&f"(c[6][0]),
          "=&f"(c[6][1]),
          "=&f"(c[6][2]),
          "=&f"(c[6][3]),
          "=&f"(c[7][0]),
          "=&f"(c[7][1]),
          "=&f"(c[7][2]),
          "=&f"(c[7][3]),
          "=&f"(c[8][0]),
          "=&f"(c[8][1]),
          "=&f"(c[8][2]),
          "=&f"(c[8][3]),
          "=&f"(c[9][0]),
          "=&f"(c[9][1]),
          "=&f"(c[9][2]),
          "=&f"(c[9][3]),
          "=&f"(c[10][0]),
          "=&f"(c[10][1]),
          "=&f"(c[10][2]),
          "=&f"(c[10][3]),
          "=&f"(c[11][0]),
          "=&f"(c[11][1]),
          "=&f"(c[11][2]),
          "=&f"(c[11][3]),
          "=&f"(c[12][0]),
          "=&f"(c[12][1]),
          "=&f"(c[12][2]),
          "=&f"(c[12][3]),
          "=&f"(c[13][0]),
          "=&f"(c[13][1]),
          "=&f"(c[13][2]),
          "=&f"(c[13][3]),
          "=&f"(c[14][0]),
          "=&f"(c[14][1]),
          "=&f"(c[14][2]),
          "=&f"(c[14][3]),
          "=&f"(c[15][0]),
          "=&f"(c[15][1]),
          "=&f"(c[15][2]),
          "=&f"(c[15][3])
        : "l"(descriptor(a + 0)), "l"(descriptor(b + 0)),
          "l"(descriptor(a + 32)), "l"(descriptor(b + 32)),
          "l"(descriptor(a + 64)), "l"(descriptor(b + 64)),
          "l"(descriptor(a + 96)), "l"(descriptor(b + 96))
        : "memory");
#else
    asm volatile("trap;");  // Production must contain an sm_90a image.
#endif
}

template<int Pending>
__device__ __forceinline__ void wgmma_ready(float (&c)[16][4]) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(Pending) : "memory");
    // Bind every accumulator across the wait: a memory clobber alone cannot
    // constrain compiler movement of register-only consumers.
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) asm volatile("" : "+f"(c[n][j]) :: "memory");
    }
#else
    asm volatile("trap;");
#endif
}
