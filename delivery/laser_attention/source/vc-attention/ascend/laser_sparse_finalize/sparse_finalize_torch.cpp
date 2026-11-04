#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <algorithm>
#include <climits>
#include <string>
#include <tuple>
extern "C" uint32_t aclrtlaunch_vc_sparse_finalize_fp16(uint32_t,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
namespace {
using Outputs=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
void Check(const at::Tensor& v,const at::Tensor& c,const at::Tensor& s,const at::Tensor& p,const at::Tensor& g){
 TORCH_CHECK(v.dim()==2 && v.size(0)>0 && v.size(0)<=INT32_MAX && (v.size(1)==64||v.size(1)==128),"Half V[globalKeys,D64/128] required");
 TORCH_CHECK(c.dim()==3 && c.size(0)>0 && c.size(1)>=32 && c.size(1)<=512 && c.size(1)%32==0 && c.size(2)>0 && c.size(2)%128==0,"Candidate[H,C32..512,Qwork128] required");
 TORCH_CHECK(c.size(0)<=INT32_MAX/c.size(1)/c.size(2),"Candidate int32 extent exceeded");
 const int64_t rows=c.size(0)*c.size(2);
 TORCH_CHECK(s.dim()==2 && s.size(0)==rows && s.size(1)>=1 && s.size(1)<=8 && p.sizes()==s.sizes(),"Matching selected[H*Qwork,rank1..8] required");
 TORCH_CHECK(g.sizes()==at::IntArrayRef({1056}),"Finalize geometry[1056] required");
 TORCH_CHECK(v.scalar_type()==at::kHalf && c.scalar_type()==at::kInt && s.scalar_type()==at::kHalf && p.scalar_type()==at::kLong && g.scalar_type()==at::kInt,"Finalize dtype mismatch");
 for(const auto& x:{v,c,s,p,g})TORCH_CHECK(x.device()==v.device()&&x.is_contiguous()&&!x.requires_grad(),"Contiguous matching inference inputs required");
}
at::Tensor Empty(at::IntArrayRef dims,const at::TensorOptions& options){return at_npu::native::empty_with_format(dims,options,ACL_FORMAT_ND);}
Outputs Allocate(const at::Tensor& v,const at::Tensor& c,const at::Tensor& s,const at::Tensor& p,const at::Tensor& g){
 Check(v,c,s,p,g);const int64_t rows=c.size(0)*c.size(2),rank=s.size(1),cores=std::min<int64_t>(48,rows/16);
 return {at::empty({rows,v.size(1)},v.options().dtype(at::kFloat)),at::empty({rows,rank},v.options().dtype(at::kInt)),at::empty({rows,rank},v.options()),
         at::empty({rows/16,16,16},v.options().dtype(at::kInt)),at::empty({rows/16,16,16},v.options()),at::empty({cores,64},v.options().dtype(at::kLong))};
}
Outputs Finalize(const at::Tensor& v,const at::Tensor& c,const at::Tensor& s,const at::Tensor& p,const at::Tensor& g){
 TORCH_CHECK(v.is_privateuseone(),"NPU required; no fallback");c10::OptionalDeviceGuard guard(v.device());Check(v,c,s,p,g);
 for(const auto& x:{v,c,s,p,g}){auto f=at_npu::native::get_npu_format(x);TORCH_CHECK((f==ACL_FORMAT_ND||f==ACL_FORMAT_NCHW)&&reinterpret_cast<uintptr_t>(x.data_ptr())%32==0,"Aligned base ND inputs required");}
 const char* soc=aclrtGetSocName();TORCH_CHECK(soc&&std::string(soc)==VC_SPARSE_FINALIZE_SOC_VERSION&&std::string(soc)=="Ascend910B2","Exact 910B2 runtime required");
 auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform&&platform->GetCoreNumAiv()==48,"48 AIV required");
 const uint32_t rows=c.size(0)*c.size(2),rank=s.size(1),cores=std::min<uint32_t>(48,rows/16);
 auto out=Empty({rows,v.size(1)},v.options().dtype(at::kFloat));auto ids=Empty({rows,rank},v.options().dtype(at::kInt));auto weights=Empty({rows,rank},v.options());
 auto pi=Empty({rows/16,16,16},v.options().dtype(at::kInt));auto pw=Empty({rows/16,16,16},v.options());
 auto storage=Empty({(cores+1)*64},v.options().dtype(at::kLong));const auto offset=((512-reinterpret_cast<uintptr_t>(storage.data_ptr())%512)%512)/8;
 auto count=storage.narrow(0,offset,cores*64).view({cores,64});auto stream=c10_npu::getCurrentNPUStream(v.device().index());
 for(const auto& x:{v,c,s,p,g,out,ids,weights,pi,pw,storage})c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(),stream);
 auto ret=aclrtlaunch_vc_sparse_finalize_fp16(cores,stream.stream(true),v.data_ptr(),c.data_ptr(),s.data_ptr(),p.data_ptr(),g.data_ptr(),out.data_ptr(),ids.data_ptr(),weights.data_ptr(),pi.data_ptr(),pw.data_ptr(),count.data_ptr(),
      c.size(0),c.size(2),c.size(1),v.size(0),v.size(1),rank,cores);
 TORCH_CHECK(ret==ACL_SUCCESS,"Finalize launch failed: ",ret);return {out,ids,weights,pi,pw,count};
}
std::string Metadata(){return "{\"soc\":\"" VC_SPARSE_FINALIZE_SOC_VERSION "\",\"operation\":\"mapped_softmax_sparse_pv\",\"contract_version\":1,\"rows\":16,\"rank_max\":8,\"columns_max\":512,\"geometry_words\":1056,\"ub_bytes\":94560,\"softmax\":\"fp32_exp_sum_div_half\",\"output\":\"float32\"}";}
}
TORCH_LIBRARY(vc_laser_sparse_finalize,m){m.def("finalize(Tensor value, Tensor candidates, Tensor scores, Tensor positions, Tensor geometry) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");m.def("metadata() -> str",TORCH_FN(Metadata));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_finalize,PrivateUse1,m){m.impl("finalize",TORCH_FN(Finalize));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_finalize,Meta,m){m.impl("finalize",TORCH_FN(Allocate));}
