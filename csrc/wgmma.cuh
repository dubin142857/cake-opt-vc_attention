// Native Hopper WGMMA. Own inline PTX; register and shared-memory layouts
// follow NVIDIA's documented m64n128k32 F32/E4M3 K-major 128-byte swizzle.
#pragma once

__device__ __forceinline__ uint64_t descriptor(uint8_t* p) {
    // Each backing array is 1024-byte aligned; each K slice starts in row 0.
    // start[13:0], LBO=16 bytes, SBO=8*128 bytes, base offset=0, swizzle=128B.
    const uint64_t address = __cvta_generic_to_shared(p);
    return ((address & 0x3ffff) >> 4) | (uint64_t(1) << 16)
         | (uint64_t(64) << 32) | (uint64_t(1) << 62);
}

__device__ __forceinline__ void wgmma_product(float (&c)[16][4], uint8_t* a, uint8_t* b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    // All 128 threads execute this after the CTA barrier. The proxy fence
    // publishes ordinary shared stores to WGMMA's asynchronous proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    // Keep the whole group and its wait in ONE asm so consumers cannot be
    // scheduled before completion. Early-clobber outputs preserve descriptors.
    asm volatile(
        "{\n.reg .pred p0, p1;\nsetp.ne.b32 p0, 0, 0;\nsetp.ne.b32 p1, 1, 0;\n"
        "wgmma.fence.sync.aligned;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p0, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %66, %67, p1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %68, %69, p1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %70, %71, p1, 1, 1;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "wgmma.wait_group.sync.aligned 0;\n}\n"
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
