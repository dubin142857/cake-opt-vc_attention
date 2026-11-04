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
#include <cmath>
#include <string>
#include <tuple>
extern "C" uint32_t aclrtlaunch_vc_sparse_inline_pv_fp16(uint32_t,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,float);
extern "C" uint32_t aclrtlaunch_vc_sparse_inline_pv_selector_probe(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t);
namespace {
using Outputs=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
using Pair=std::tuple<at::Tensor,at::Tensor>;
void TensorCheck(const at::Tensor& q,const at::Tensor& x){TORCH_CHECK(x.scalar_type()==at::kHalf&&x.device()==q.device()&&x.is_contiguous()&&!x.requires_grad(),"Matching contiguous inference half tensors required");}
void DeviceCheck(const at::Tensor& q){
 TORCH_CHECK(q.is_privateuseone(),"NPU required; no fallback");const char* soc=aclrtGetSocName();
 TORCH_CHECK(soc&&std::string(soc)==VC_CANDIDATE_SOC_VERSION&&std::string(soc)=="Ascend910B2","Exact910B2 required");
 auto p=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(p&&p->GetCoreNumAic()==24&&p->GetCoreNumAiv()==48,"24AIC/48AIV required");
}
void Check(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,int64_t queries,int64_t keys,int64_t dimension){
 TORCH_CHECK(q.dim()==4&&k.dim()==4&&v.sizes()==k.sizes()&&q.size(0)>0&&q.size(1)>0&&q.size(0)==k.size(0)&&q.size(1)==k.size(1)&&q.size(3)==128&&k.size(3)==128,"Padded BHSD128 QKV required");
 TORCH_CHECK(queries>0&&keys>0&&queries<=INT32_MAX-255&&keys<=INT32_MAX-255&&q.size(2)==(queries+255)/256*256&&k.size(2)==(keys+255)/256*256&&(dimension==64||dimension==128),"Logical D64/128 and S256 padding required");
 TORCH_CHECK(g.sizes()==at::IntArrayRef({64,128}),"Mask geometry[64,128] required");for(const auto& t:{q,k,v,g})TensorCheck(q,t);
 auto h=q.size(0)*q.size(1),qr=(queries+127)/128*128;TORCH_CHECK(h<=INT32_MAX/k.size(2)&&h<=INT32_MAX/32/qr,"int32 extent exceeded");
}
at::Tensor Empty(at::IntArrayRef sizes,const at::TensorOptions& options,bool native){return native?at_npu::native::empty_with_format(sizes,options,ACL_FORMAT_ND):at::empty(sizes,options);}
Outputs Make(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,int64_t queries,int64_t keys,int64_t dimension,bool native){
 Check(q,k,v,g,queries,keys,dimension);const int64_t h=q.size(0)*q.size(1),qr=(queries+127)/128*128,r=h*qr,c=std::min<int64_t>(24,r/128),rank=std::min<int64_t>(8,(keys+127)/128);auto opt=q.options();
 return {Empty({r,8},opt,native),Empty({h,32,qr},opt.dtype(at::kInt),native),Empty({3*c,64},opt.dtype(at::kLong),native),
 Empty({r,128},opt.dtype(at::kFloat),native),Empty({2*c,64},opt.dtype(at::kLong),native),Empty({r,rank},opt.dtype(at::kInt),native),Empty({r,rank},opt,native),
 Empty({r/16,16,16},opt.dtype(at::kInt),native),Empty({r/16,16,16},opt,native)};
}
Outputs Allocate(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,int64_t queries,int64_t keys,int64_t dimension){return Make(q,k,v,g,queries,keys,dimension,false);}
Outputs Produce(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,int64_t queries,int64_t keys,int64_t dimension){
 c10::OptionalDeviceGuard guard(q.device());Check(q,k,v,g,queries,keys,dimension);DeviceCheck(q);
 for(const auto& t:{q,k,v,g}){auto f=at_npu::native::get_npu_format(t);TORCH_CHECK((f==ACL_FORMAT_ND||f==ACL_FORMAT_NCHW)&&reinterpret_cast<uintptr_t>(t.data_ptr())%32==0,"Aligned base ND format required");}
 const int64_t h=q.size(0)*q.size(1),qr=(queries+127)/128*128,c=std::min<int64_t>(24,h*qr/128),rank=std::min<int64_t>(8,(keys+127)/128);
 auto out=Make(q,k,v,g,queries,keys,dimension,true);auto ring=Empty({c,2,8,128,128},q.options(),true);auto storage=Empty({(5*c+1)*64},q.options().dtype(at::kLong),true);
 auto offset=((512-reinterpret_cast<uintptr_t>(storage.data_ptr())%512)%512)/8;
 std::get<2>(out)=storage.narrow(0,offset,3*c*64).view({3*c,64});std::get<4>(out)=storage.narrow(0,offset+3*c*64,2*c*64).view({2*c,64});
 auto stream=c10_npu::getCurrentNPUStream(q.device().index());
 for(const auto& t:{q,k,v,g,ring,storage,std::get<0>(out),std::get<1>(out),std::get<3>(out),std::get<5>(out),std::get<6>(out),std::get<7>(out),std::get<8>(out)})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
 auto rc=aclrtlaunch_vc_sparse_inline_pv_fp16(c,stream.stream(true),q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr(),std::get<3>(out).data_ptr(),ring.data_ptr(),std::get<1>(out).data_ptr(),std::get<0>(out).data_ptr(),std::get<5>(out).data_ptr(),std::get<6>(out).data_ptr(),std::get<7>(out).data_ptr(),std::get<8>(out).data_ptr(),std::get<2>(out).data_ptr(),std::get<4>(out).data_ptr(),h,q.size(2),k.size(2),queries,keys,dimension,c,32,rank,1/std::sqrt(float(dimension)));
 TORCH_CHECK(rc==ACL_SUCCESS,"Inline producer/PV launch failed: ",rc);return out;
}
Pair Probe(const at::Tensor& x,const at::Tensor& g,int64_t rows,int64_t valid){
 c10::OptionalDeviceGuard guard(x.device());DeviceCheck(x);TensorCheck(x,x);TensorCheck(x,g);
 TORCH_CHECK(rows>0&&rows<=48*64&&valid>=0&&valid<=128&&x.sizes()==at::IntArrayRef({(rows+63)/64*64,128})&&g.sizes()==at::IntArrayRef({64,128}),"Padded probe extents");
 const int64_t n=(rows+63)/64*64;auto a=Empty({2,n},x.options(),true),b=Empty({2,n},x.options().dtype(at::kInt),true);auto st=c10_npu::getCurrentNPUStream(x.device().index());
 for(const auto& t:{x,g,a,b})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),st);
 auto rc=aclrtlaunch_vc_sparse_inline_pv_selector_probe(n/64,st.stream(true),x.data_ptr(),g.data_ptr(),a.data_ptr(),b.data_ptr(),rows,valid);TORCH_CHECK(rc==ACL_SUCCESS,"Probe launch failed");return {a,b};
}
std::string Metadata(){return "{\"soc\":\"" VC_CANDIDATE_SOC_VERSION "\",\"operation\":\"block1_window_top8_inline_pv\",\"contract_version\":1,\"q_rows\":128,\"k_columns\":128,\"vector_rows\":64,\"ub_bytes\":170624,\"score_slots\":2,\"candidates_per_block\":1,\"score_chunk_blocks\":8,\"selection_window_blocks\":32,\"output_rank\":8,\"output_columns\":32,\"producer_counter_version\":7,\"pv_counter_version\":2,\"inline_pv\":true,\"selection\":\"window32_top8_then_resident_top8\",\"tie\":\"Sort32 observed input-stable; validate value/ID relation\"}";}
}
TORCH_LIBRARY(vc_laser_sparse_inline_pv,m){m.def("produce(Tensor q, Tensor k, Tensor v, Tensor geometry, int queries, int keys, int dimension) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");m.def("probe(Tensor scores, Tensor geometry, int rows, int valid) -> (Tensor, Tensor)");m.def("metadata() -> str",TORCH_FN(Metadata));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_inline_pv,PrivateUse1,m){m.impl("produce",TORCH_FN(Produce));m.impl("probe",TORCH_FN(Probe));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_inline_pv,Meta,m){m.impl("produce",TORCH_FN(Allocate));}
