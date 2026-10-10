"""N256 INT8 QK experiment; two sequential, unchanged K128 ExpCast/PV steps.

The last one or two logical tiles retain N128 QK and the original final-tile
arithmetic. No K256 probability reduction or change to quantization scales.
"""
import hashlib
from attention_phase_probe import ROOT, VC, replace_exact
import scale_prefetch_native as base

EXACT_OUTPUT = True
BASELINE = "vc28_scale_prefetch"


def wide_mma_source():
    # Extend the pinned Apache-2.0 Sage N128 inline-PTX wrapper to N256.
    # PTX 8.4 integer WGMMA uses 128 S32 accumulators per thread for N256;
    # the first/second 64 accumulators describe the first/second 128 columns.
    operands = []
    constraints = []
    for group in range(16):
        operands.append('    "' + ', '.join(f'%{group*8+i}' for i in range(8)) +
                        (', "' if group < 15 else '}, %128, %129, %130;\\n"'))
        constraints.append('    ' + ', '.join(
            f'"+r"(d[{group//8}][0][{group%8}][{i}])' for i in range(8)))
    return '''
// Adapted from SageAttention csrc/wgmma.cuh (Apache-2.0, notice above).
template<int ScaleD, int D>
__device__ __forceinline__ void paired_qk_mma(int32_t (&d)[2][1][8][8], int8_t* q, int8_t* k) {
  uint64_t qa=wgmma::make_smem_desc<D>(q), ka=wgmma::make_smem_desc<D>(k);
  asm volatile(
    "{\\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 {"
''' + '\n'.join(operands) + '\n    "}\\n"\n    :\n' + ',\n'.join(constraints) + '''
    : "l"(qa), "l"(ka), "n"(int32_t(ScaleD)));
}
'''


def candidate_kernel():
    source = base.candidate_kernel()
    source = replace_exact(source, "sage_vc_attention_kernel_scale_prefetch(",
                           "sage_vc_attention_kernel_paired_qk(")
    source = replace_exact(source,
        "RegisterLimit==128 || RegisterLimit==144 || RegisterLimit==168 || RegisterLimit==192",
        "RegisterLimit==240")
    source = replace_exact(source, "  static_assert(NUM_THREADS == 128);",
        "  static_assert(NUM_THREADS == 128 && CTA_Q==64 && CTA_K==128);")
    source = replace_exact(source,
        "CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t)",
        "CTA_Q * head_dim * sizeof(int8_t) + 2 * CTA_K * head_dim * sizeof(int8_t)")
    source = replace_exact(source,
        "expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);",
        "expect_bytes<(2 * CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);", count=2)
    start = source.index("  int p = 1;")
    tail_marker = "  {\n    p ^= 1;"
    main_end = source.rindex(tail_marker)
    original_main = source[start:main_end]
    # All arithmetic, the PV issue/wait, next-scale prefetch and next-V load
    # come directly from the baseline. The iteration is still a K128 index.
    work_start = original_main.index("    // Integer-max candidates")
    work = original_main[work_start:original_main.rindex("  }")]
    narrow_start = original_main.index("    // compute QK^T")
    narrow_end = original_main.index("    // load K")
    narrow_qk = original_main[narrow_start:narrow_end]
    loop = '''  // Pair only MAIN tiles. Leave the final 1--2 tiles on N128 QK.
  // K's barrier phase advances once per pair; V's once per logical K128 tile.
  const uint32_t pair_count = (num_iterations - 1) / 2;
  for (uint32_t pair = 0; pair < pair_count; ++pair) {
    wait(&barrier_K, pair & 1);
    int32_t paired_scores[2][1][8][8];
    wgmma::warpgroup_arrive();
    paired_qk_mma<0,head_dim>(paired_scores,sQ,sK);
    #pragma unroll
    for (int ki=1; ki<head_dim/32; ++ki)
      paired_qk_mma<1,head_dim>(paired_scores,sQ+ki*32,sK+ki*32);
    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();
    // Both scores are now in registers, so the K buffer can be reused.
    if (threadIdx.x == 0) {
      expect_bytes<256 * head_dim>(&barrier_K);
      load_async_4D(sK,&tensorMapK,&barrier_K,0,(pair+1)*256,kv_head_id,batch_id);
    }
    #pragma unroll
    for (int slot=0; slot<2; ++slot) {
      auto& RS = paired_scores[slot];
      const uint32_t iter=pair*2+slot+1;
      const int p=(iter-1)&1;
      float dequant_scale=q_scale*current_k_scale;
      sm_scale=original_sm_scale*dequant_scale;
''' + work + '''
    }
  }
  wait(&barrier_K, pair_count & 1);
  if ((num_iterations - 1) & 1) {
    const uint32_t iter=num_iterations-1;
    const int p=(iter-1)&1;
    float dequant_scale=q_scale*current_k_scale;
    sm_scale=original_sm_scale*dequant_scale;
''' + narrow_qk + work + '''
  }
  // The final K128 block can be the second half of the last K256 TMA box.
  sK += ((num_iterations - 1) & 1) * 128 * head_dim;
  int p=(num_iterations-1)&1;
'''
    tail = source[main_end:]
    tail = replace_exact(tail, tail_marker, "  {")
    tail = replace_exact(tail, "    wait(&barrier_K, p);",
                         "    // Last K256 load was waited before the remainder above.")
    return wide_mma_source() + source[:start] + loop + tail


def generate_source():
    source = base.generate_source()
    source = replace_exact(source, base.candidate_kernel(), base.candidate_kernel()+candidate_kernel())
    old = '''if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scale_prefetch<64,128,128,D,
    QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
    MaskMode::kNone,false,true,true,false,false,true,0,168,false,9>;'''
    source = replace_exact(source, old,
        old.replace("kernel_scale_prefetch", "kernel_paired_qk").replace("0,168,false,9", "0,240,false,9"))
    source = replace_exact(source, "else return sage_vc::sage_vc_attention_kernel_sage_layout<",
                           "else return sage_vc::sage_vc_attention_kernel_scale_prefetch<")
    source = replace_exact(source, "create_tensor_map_4D<128,D>(k.data_ptr<int8_t>()",
                           "create_tensor_map_4D<(Trace ? 256 : 128),D>(k.data_ptr<int8_t>()")
    # Both the resource report and actual launch must include the larger K box.
    source = source.replace("320*D", "(Trace ? 448 : 320)*D")
    source = source.replace("Scale prefetch experiment requires ExpCast", "Paired QK experiment requires ExpCast")
    probe = (ROOT/"experiments/expcast/paired_qk_check.cuh").read_text(encoding="utf-8")
    return replace_exact(source, "PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {",
        probe+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n  m.def("check_paired_qk",&check_paired_qk);')


def load_native():
    return base.base.load_source(generate_source(), "_vc_paired_qk_")


def build_audit(ext, vc, build, dim, output_type):
    result = base.build_audit(ext, vc, build, dim, output_type)
    result["baseline_source"] = "Unmodified scale_prefetch_native candidate"
    result["baseline_source_sha256"] = hashlib.sha256(base.candidate_kernel().encode()).hexdigest()
    result["candidate_launch"] = {"query_tile": 64, "qk_key_tile": 256,
        "probability_key_tile": 128, "pv_key_tile": 128, "threads": 128,
        "dynamic_shared_bytes": 448*dim, "register_limit": 240}
    return result


forward = base.forward


def validate_qk(ext, device, progress):
    import torch
    rows = []
    gen = torch.Generator().manual_seed(4100)
    for dim in (64, 128):
        for pattern in ("random", "extreme", "column_mapping"):
            q = torch.randint(-127,128,(3,64,dim),dtype=torch.int8,generator=gen)
            k = torch.randint(-127,128,(3,256,dim),dtype=torch.int8,generator=gen)
            if pattern == "extreme":
                q.fill_(127);k[:,::2,:]=127;k[:,1::2,:]=-127
            elif pattern == "column_mapping":
                q.zero_();q[:,:,0]=1
                k.zero_();k[:,:,0]=torch.arange(-128,128,dtype=torch.int8)
            # Integer CPU oracle: no TF32 or floating dot-product approximation.
            expected = q.int() @ k.int().transpose(-1,-2)
            row = {"dim": dim, "pattern": pattern, "status": "running"}
            rows.append(row);progress(rows)
            print("Validate N256 QK mapping",dim,pattern,flush=True)
            out = ext.check_paired_qk(q.to(device),k.to(device)).cpu()
            torch.testing.assert_close(out,expected,rtol=0,atol=0)
            row["status"]="passed";progress(rows)
    return rows


def validate_pipeline(ext, vc, reference, device, progress):
    """Odd/even K256 boxes, masked tails, distinct scales, and reuse/streams."""
    import torch
    from validate_scale_prefetch import assert_exact
    from profile_int8_h200 import metrics
    rows=[]
    gen=torch.Generator(device=device).manual_seed(4101)
    lengths=(1,127,128,129,255,256,257,383,384,385,511,512,513,769,37741)
    for dtype in (torch.float16,torch.bfloat16):
        for dim in (64,128):
            for n in lengths:
                qkv=tuple(torch.randn(2,2,length,dim,device=device,dtype=dtype,generator=gen)
                          for length in (65,n,n))
                state=vc.prepare(*qkv,expcast=True,v_smooth=False)
                ks=state["ks"].clone().view(2,2,-1,4)
                factors=torch.tensor([.5,1.,1.5,.75,2.],device=device)
                ks*=factors[torch.arange(ks.shape[2],device=device)%5][None,None,:,None]
                ks*=torch.tensor([.5,.75,1.25,1.5],device=device)
                state={**state,"ks":ks.reshape_as(state["ks"])}
                # Make padded V nonzero: only the probability mask should stop
                # it contributing. Do not accidentally rely on zero V padding.
                if n%128:
                    permutation=reference.value_permutation(state["v"].shape[-1],device)
                    v=state["v"].clone().view(torch.uint8)
                    v[...,permutation[n:]]=0x7e  # largest finite positive E4M3
                    state={**state,"v":v.view(state["v"].dtype)}
                row={"dtype":str(dtype),"dim":dim,"keys":n,"status":"running"}
                rows.append(row);progress(rows)
                print("Validate paired QK/tail",str(dtype),dim,n,flush=True)
                old=forward(ext,state,False);out=forward(ext,state,True)
                row["same_payload"]=assert_exact(out,old)
                row["decoded_reference"]=metrics(out,reference.reference(state,chunk=32).to(dtype),limit=.01)
                stream=torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):side=forward(ext,state,True)
                stream.synchronize()
                row["nondefault_stream"]=assert_exact(side,out)
                row["status"]="passed";progress(rows)
    return rows
