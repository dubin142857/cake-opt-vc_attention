"""TileLang 0.1.15 fused INT8 QK / paper ExpCast / FP8 PV on Hopper.

Consumes naturally ordered, channel-major FP8 V without Sage's byte permutation.
No Python tile loop, no NxN tensor, no dequantized QK/PV fallback.
Developed using Tile-AI's MIT-licensed attention/GEMM examples; see
experiments/expcast/TILELANG_LICENSE and SAGE_EXPCAST_TILELANG.md.
"""
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V_LAYOUT = "channel_major_natural"


def make_primfunc(batch, heads, queries, keys, dim, output_dtype="bfloat16", stages=1):
    import tilelang.language as T
    if dim not in (64,128) or stages not in (1,2):
        raise ValueError("Supported: D64/128, one or two pipeline stages")
    if min(batch,heads,queries,keys)<1 or output_dtype not in ("float16","bfloat16"):
        raise ValueError("Expected nonempty FP16/BF16 MHA")
    bm,bn=64,128
    blocks=(keys+127)//128
    qs_len=(queries+63)//64*32
    prelude=(ROOT/"experiments/expcast/paper_math.cuh").read_text(encoding="utf-8")
    # Keep caller's actual scale, not a compile-time approximation of 1/sqrt(D).
    @T.prim_func
    def kernel(Q:T.Tensor((batch,heads,queries,dim),T.int8),
               K:T.Tensor((batch,heads,keys,dim),T.int8),
               V:T.Tensor((batch,heads,dim,blocks*128),T.float8_e4m3fn),
               QS:T.Tensor((batch,heads,qs_len),T.float32),
               KS:T.Tensor((batch,heads,blocks*4),T.float32),
               VS:T.Tensor((batch,heads,dim),T.float32),
               O:T.Tensor((batch,heads,queries,dim),output_dtype),
               scale:T.float32):
        T.func_attr({"global_symbol":"vc_tilelang_paper"})
        with T.Kernel(T.ceildiv(queries,bm),heads,batch,threads=128,prelude=prelude) as (bx,h,b):
            q=T.alloc_shared((bm,dim),T.int8)
            k=T.alloc_shared((bn,dim),T.int8)
            v=T.alloc_shared((dim,bn),T.float8_e4m3fn)
            dots=T.alloc_fragment((bm,bn),T.int32)
            logits=T.alloc_fragment((bm,bn),T.float32)
            prob=T.alloc_fragment((bm,bn),T.float8_e4m3fn)
            p_shared=T.alloc_shared((bm,bn),T.float8_e4m3fn)
            pv=T.alloc_fragment((bm,dim),T.float32)
            accum=T.alloc_fragment((bm,dim),T.float32)
            maximum=T.alloc_fragment((bm,),T.float32)
            previous=T.alloc_fragment((bm,),T.float32)
            alpha=T.alloc_fragment((bm,),T.float32)
            mass=T.alloc_fragment((bm,),T.float32)
            denom=T.alloc_fragment((bm,),T.float32)
            T.copy(Q[b,h,bx*bm:(bx+1)*bm,:],q)
            T.fill(maximum,-5e6)
            T.clear(accum)
            T.clear(denom)
            for tile in T.Pipelined(blocks,num_stages=stages):
                T.copy(K[b,h,tile*bn:(tile+1)*bn,:],k)
                T.copy(V[b,h,:,tile*bn:(tile+1)*bn],v)
                T.gemm(q,k,dots,transpose_B=True,clear_accum=True,policy=T.GemmWarpPolicy.FullRow)
                for i,j in T.Parallel(bm,bn):
                    row=bx*bm+i
                    dequant=T.call_extern("float32","vc_paper_mul",QS[b,h,(row//16)*8+row%8],KS[b,h,tile*4+(j%8)//2])
                    if tile==blocks-1:
                        score=T.call_extern("float32","vc_paper_mul",T.cast(dots[i,j],T.float32),dequant)
                        logits[i,j]=T.call_extern("float32","vc_paper_mul",score,scale)
                    else:
                        factor=T.call_extern("float32","vc_paper_mul",scale,dequant)
                        logits[i,j]=T.call_extern("float32","vc_paper_mul",T.cast(dots[i,j],T.float32),factor)
                    if tile*bn+j>=keys:
                        logits[i,j]=-T.infinity(T.float32)
                T.copy(maximum,previous)
                T.reduce_max(logits,maximum,dim=1,clear=False)
                for i in T.Parallel(bm):
                    alpha[i]=T.call_extern("float32","vc_paper_alpha",previous[i],maximum[i])
                for i,j in T.Parallel(bm,bn):
                    code=T.call_extern("uint8","vc_paper_code",logits[i,j],maximum[i])
                    prob[i,j]=T.reinterpret(T.float8_e4m3fn,code)
                    logits[i,j]=T.cast(prob[i,j],T.float32)
                T.reduce_sum(logits,mass,dim=1)
                # INT8-QK accumulators and FP8-PV register operands have different
                # lane layouts. Shared P makes the conversion explicit and safe.
                T.copy(prob,p_shared)
                # A fresh FP32 PV partial per 128-key tile, then FP32 history.
                T.gemm(p_shared,v,pv,transpose_B=True,clear_accum=True,policy=T.GemmWarpPolicy.FullRow)
                for i in T.Parallel(bm):
                    denom[i]=T.call_extern("float32","vc_paper_add",T.call_extern("float32","vc_paper_mul",denom[i],alpha[i]),mass[i])
                for i,d in T.Parallel(bm,dim):
                    accum[i,d]=T.call_extern("float32","vc_paper_add",T.call_extern("float32","vc_paper_mul",accum[i,d],alpha[i]),pv[i,d])
            for i,d in T.Parallel(bm,dim):
                O[b,h,bx*bm+i,d]=T.call_extern("float32","vc_paper_mul",accum[i,d]/denom[i],VS[b,h,d])
    return kernel


@lru_cache(maxsize=64)
def compile_kernel(batch,heads,queries,keys,dim,dtype,stages):
    import tilelang
    if tilelang.__version__.split("+")[0]!="0.1.15":
        raise RuntimeError("This experiment pins TileLang 0.1.15; run scripts/run_expcast_tilelang_h200.sh")
    return tilelang.compile(make_primfunc(batch,heads,queries,keys,dim,dtype,stages),
        out_idx=[6],target={"kind":"cuda","arch":"sm_90a"},
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})


def forward_prepared(state,stages=1):
    import torch
    from profile_int8_h200 import require_int8_state
    require_int8_state(state)
    if state.get("v_layout")!=V_LAYOUT:
        raise ValueError("TileLang requires natural V from tilelang_expcast.prepare; Sage-permuted V is incompatible")
    q,k=state["q"],state["k"]
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)!=(9,0):
        raise ValueError("Hopper required; no CPU/FP16-matmul fallback")
    if state["block_values"] or state["means"].numel() or state["permutation"] is not None:
        raise ValueError("TileLang experiment requires V smoothing/grouping OFF")
    if any(not state[n].is_contiguous() or state[n].device!=q.device for n in ("q","k","v","qs","ks","vs")):
        raise ValueError("Expected contiguous, same-device prepared tensors")
    b,h,m,d=q.shape
    dtype="float16" if state["output_type"]==0 else "bfloat16"
    with torch.cuda.device(q.device):
        fn=compile_kernel(b,h,m,k.shape[2],d,dtype,stages)
        return fn(*(state[n] for n in ("q","k","v","qs","ks","vs")),state["scale"])


def prepare_values(v):
    """Quantize directly in token order; no Sage permutation or inverse gather.

    The existing channel quantizer is position-independent. Feed it a padded
    [B,H,D,N] tensor directly, bypassing transpose_pad_permute_cuda. This retains
    its exact scale/FP8 rounding contract. The upstream helper uses stream 0.
    """
    import torch
    from sageattention_local import _fused
    if v.ndim!=4 or not v.is_cuda or v.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError("Expected CUDA FP16/BF16 V in HND layout")
    with torch.no_grad(),torch.cuda.device(v.device):
        if torch.cuda.current_stream(v.device)!=torch.cuda.default_stream(v.device):
            raise ValueError("V preprocessing requires the default CUDA stream")
        b,h,n,d=v.shape
        if min(b,h,n,d)<1 or d not in (64,128):
            raise ValueError("Expected nonempty D64/128 V")
        pad=(-n)%128
        if pad:
            v=torch.cat((v,torch.zeros(b,h,pad,d,device=v.device,dtype=v.dtype)),dim=2)
        # Ordinary matrix transpose only; token j remains token j.
        transposed=v.transpose(2,3).contiguous()
        v8=torch.empty(transposed.shape,device=v.device,dtype=torch.float8_e4m3fn)
        scale=torch.empty((b,h,d),device=v.device,dtype=torch.float32)
        _fused.scale_fuse_quant_cuda(transposed,v8,scale,n+pad,448.0,1)
        return v8,scale


def prepare(q,k,v):
    """ExpCast only, INT8 QK and natural FP8 V; all preprocessing is fresh."""
    import torch
    from sageattention_local import vc
    from sageattention_local.triton.quant_per_thread import per_thread_int8
    vc.validate_inputs(q,k,v,"HND",False,False,True,False,8,2)
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)!=(9,0):
        raise ValueError("Hopper required")
    with torch.no_grad(),torch.cuda.device(q.device):
        if torch.cuda.current_stream(q.device)!=torch.cuda.default_stream(q.device):
            raise ValueError("Preprocessing requires the default CUDA stream")
        q,k,v=(x.contiguous() for x in (q,k,v))
        km=k.mean(dim=2,keepdim=True)
        q8,qs,k8,ks=per_thread_int8(q,k,km,BLKQ=64,WARPQ=16,BLKK=128,WARPK=128,tensor_layout="HND")
        v8,vs=prepare_values(v)
        return {"q":q8,"k":k8,"v":v8,"qs":qs,"ks":ks,"vs":vs,
                "means":torch.empty(0,device=q.device,dtype=torch.float16),
                "expcast":True,"block_values":False,"permutation":None,"v_layout":V_LAYOUT,
                "output_type":0 if q.dtype==torch.float16 else 1,"scale":q.shape[-1]**-.5}


def make_encoder(n):
    import tilelang.language as T
    prelude=(ROOT/"experiments/expcast/paper_math.cuh").read_text(encoding="utf-8")
    @T.prim_func
    def encode(X:T.Tensor((n,),T.float32),Y:T.Tensor((n,),T.uint8)):
        with T.Kernel(T.ceildiv(n,256),threads=128,prelude=prelude) as block:
            for i in T.Parallel(256):
                Y[block*256+i]=T.call_extern("uint8","vc_paper_code",X[block*256+i],T.float32(0))
    return encode
