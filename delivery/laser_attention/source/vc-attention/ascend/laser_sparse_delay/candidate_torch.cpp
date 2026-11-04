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
#include <dlfcn.h>
#include <tuple>
extern "C" uint32_t aclrtlaunch_vc_sparse_delay_fp16(uint32_t,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,float);
extern "C" uint32_t aclrtlaunch_vc_sparse_pair_fp16(uint32_t,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,float);
extern "C" uint32_t aclrtlaunch_vc_sparse_delay_selector_probe(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_vc_delay_qk_tile(uint32_t,void*,void*,void*,void*);
extern "C" uint32_t aclrtlaunch_vc_sparse_delay_reduce_probe(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,float);
extern "C" uint32_t aclrtlaunch_vc_sparse_delay_math_probe(uint32_t,void*,void*,void*,void*,void*,void*,void*,uint32_t,uint32_t);
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
 TORCH_CHECK(k.size(2)%256==0,"Pair batching requires even physical K128 blocks");
 auto h=q.size(0)*q.size(1),qr=(queries+127)/128*128;TORCH_CHECK(h<=INT32_MAX/k.size(2)&&h<=INT32_MAX/32/qr,"int32 extent exceeded");
}
at::Tensor Empty(at::IntArrayRef sizes,const at::TensorOptions& options,bool native){return native?at_npu::native::empty_with_format(sizes,options,ACL_FORMAT_ND):at::empty(sizes,options);}
Outputs Make(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,int64_t queries,int64_t keys,int64_t dimension,bool native,bool raw=false){
 Check(q,k,v,g,queries,keys,dimension);const int64_t h=q.size(0)*q.size(1),qr=(queries+127)/128*128,r=h*qr,c=std::min<int64_t>(24,r/128),rank=std::min<int64_t>(8,(keys+127)/128);auto opt=q.options();
 return {Empty({r,8},opt.dtype(raw?at::kHalf:at::kFloat),native),Empty({h,32,qr},opt.dtype(at::kInt),native),Empty({3*c,64},opt.dtype(at::kLong),native),
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
 auto rc=aclrtlaunch_vc_sparse_delay_fp16(c,stream.stream(true),q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr(),std::get<3>(out).data_ptr(),ring.data_ptr(),std::get<1>(out).data_ptr(),std::get<0>(out).data_ptr(),std::get<5>(out).data_ptr(),std::get<6>(out).data_ptr(),std::get<7>(out).data_ptr(),std::get<8>(out).data_ptr(),std::get<2>(out).data_ptr(),std::get<4>(out).data_ptr(),h,q.size(2),k.size(2),queries,keys,dimension,c,32,rank,1/std::sqrt(float(dimension)));
 TORCH_CHECK(rc==ACL_SUCCESS,"Inline producer/PV launch failed: ",rc);return out;
}
Outputs RawReference(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& g,int64_t queries,int64_t keys,int64_t dimension,const std::string& library){
 c10::OptionalDeviceGuard guard(q.device());Check(q,k,v,g,queries,keys,dimension);DeviceCheck(q);
 for(const auto& t:{q,k,v,g}){auto f=at_npu::native::get_npu_format(t);TORCH_CHECK((f==ACL_FORMAT_ND||f==ACL_FORMAT_NCHW)&&reinterpret_cast<uintptr_t>(t.data_ptr())%32==0,"Aligned base ND format required");}
 const int64_t h=q.size(0)*q.size(1),qr=(queries+127)/128*128,c=std::min<int64_t>(24,h*qr/128),rank=std::min<int64_t>(8,(keys+127)/128);
 auto out=Make(q,k,v,g,queries,keys,dimension,true,true);auto ring=Empty({c,2,8,128,128},q.options(),true);auto storage=Empty({(5*c+1)*64},q.options().dtype(at::kLong),true);
 auto offset=((512-reinterpret_cast<uintptr_t>(storage.data_ptr())%512)%512)/8;
 std::get<2>(out)=storage.narrow(0,offset,3*c*64).view({3*c,64});std::get<4>(out)=storage.narrow(0,offset+3*c*64,2*c*64).view({2*c,64});
 auto stream=c10_npu::getCurrentNPUStream(q.device().index());
 for(const auto& t:{q,k,v,g,ring,storage,std::get<0>(out),std::get<1>(out),std::get<3>(out),std::get<5>(out),std::get<6>(out),std::get<7>(out),std::get<8>(out)})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
 void* handle=dlopen(library.c_str(),RTLD_NOW|RTLD_NOLOAD);TORCH_CHECK(handle,"Rawreference provider must already be loaded");
 auto raw=reinterpret_cast<decltype(&aclrtlaunch_vc_sparse_pair_fp16)>(dlsym(handle,"aclrtlaunch_vc_sparse_pair_fp16"));
 Dl_info info{};TORCH_CHECK(raw&&dladdr(reinterpret_cast<void*>(raw),&info)&&info.dli_fname&&library==info.dli_fname,"Rawreference symbol provider identity differs");
 auto rc=raw(c,stream.stream(true),q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr(),std::get<3>(out).data_ptr(),ring.data_ptr(),std::get<1>(out).data_ptr(),std::get<0>(out).data_ptr(),std::get<5>(out).data_ptr(),std::get<6>(out).data_ptr(),std::get<7>(out).data_ptr(),std::get<8>(out).data_ptr(),std::get<2>(out).data_ptr(),std::get<4>(out).data_ptr(),h,q.size(2),k.size(2),queries,keys,dimension,c,32,rank,1.0f);
 dlclose(handle);TORCH_CHECK(rc==ACL_SUCCESS,"Inline producer/PV launch failed: ",rc);return out;
}
Pair Probe(const at::Tensor& x,const at::Tensor& g,int64_t rows,int64_t valid){
 c10::OptionalDeviceGuard guard(x.device());DeviceCheck(x);TensorCheck(x,x);TensorCheck(x,g);
 TORCH_CHECK(rows>0&&rows<=48*64&&valid>=0&&valid<=128&&x.sizes()==at::IntArrayRef({(rows+63)/64*64,128})&&g.sizes()==at::IntArrayRef({64,128}),"Padded probe extents");
 const int64_t n=(rows+63)/64*64;auto a=Empty({2,n},x.options(),true),b=Empty({2,n},x.options().dtype(at::kInt),true);auto st=c10_npu::getCurrentNPUStream(x.device().index());
 for(const auto& t:{x,g,a,b})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),st);
 auto rc=aclrtlaunch_vc_sparse_delay_selector_probe(n/64,st.stream(true),x.data_ptr(),g.data_ptr(),a.data_ptr(),b.data_ptr(),rows,valid);TORCH_CHECK(rc==ACL_SUCCESS,"Probe launch failed");return {a,b};
}
at::Tensor QkTile(const at::Tensor& q,const at::Tensor& k){
 c10::OptionalDeviceGuard guard(q.device());DeviceCheck(q);TensorCheck(q,q);TensorCheck(q,k);
 TORCH_CHECK(q.sizes()==at::IntArrayRef({128,128})&&k.sizes()==q.sizes(),"QK audit requires half128x128 tiles");
 auto out=Empty({128,128},q.options(),false);auto stream=c10_npu::getCurrentNPUStream(q.device().index());
 for(const auto& t:{q,k,out})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
 auto rc=aclrtlaunch_vc_delay_qk_tile(1,stream.stream(true),q.data_ptr(),k.data_ptr(),out.data_ptr());TORCH_CHECK(rc==ACL_SUCCESS,"QK source audit launch failed:",rc);return out;
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> FmaProbe(const at::Tensor& x,const at::Tensor& w,const at::Tensor& a){
 c10::OptionalDeviceGuard guard(x.device());DeviceCheck(x);TensorCheck(x,x);TensorCheck(x,w);
 TORCH_CHECK(x.dim()==2&&x.size(0)>0&&x.size(0)%16==0&&x.size(0)<=INT32_MAX/128&&x.size(1)==128&&w.sizes()==at::IntArrayRef({x.size(0)}),"Probe half rowsR16x128/rowweightsR required");
 TORCH_CHECK(a.scalar_type()==at::kFloat&&a.device()==x.device()&&a.sizes()==x.sizes()&&a.is_contiguous()&&!a.requires_grad(),"Matching float32 accumulator required");
 for(const auto& t:{x,w,a}){auto f=at_npu::native::get_npu_format(t);TORCH_CHECK((f==ACL_FORMAT_ND||f==ACL_FORMAT_NCHW)&&reinterpret_cast<uintptr_t>(t.data_ptr())%32==0,"Aligned ND probe inputs required");}
 auto old=Empty(x.sizes(),x.options().dtype(at::kFloat),false),fp=Empty(x.sizes(),x.options().dtype(at::kFloat),false),mixed=Empty(x.sizes(),x.options().dtype(at::kFloat),false);
 auto stream=c10_npu::getCurrentNPUStream(x.device().index());for(const auto& t:{x,w,a,old,fp,mixed})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
 const uint32_t cores=std::min<int64_t>(48,x.size(0)/16);auto rc=aclrtlaunch_vc_sparse_delay_math_probe(cores,stream.stream(true),x.data_ptr(),w.data_ptr(),a.data_ptr(),old.data_ptr(),fp.data_ptr(),mixed.data_ptr(),x.size(0),cores);
 TORCH_CHECK(rc==ACL_SUCCESS,"FMA primitive probe launch failed:",rc);return {old,fp,mixed};
}
Pair PairProbe(const at::Tensor& x,const at::Tensor& g,int64_t valid0,int64_t valid1,int64_t base,double scale){
 c10::OptionalDeviceGuard guard(x.device());DeviceCheck(x);TensorCheck(x,x);TensorCheck(x,g);
 TORCH_CHECK(x.sizes()==at::IntArrayRef({128,128})&&g.sizes()==at::IntArrayRef({64,128})&&valid0>=0&&valid0<=128&&valid1>=0&&valid1<=128&&base>=0&&base<=INT32_MAX-256&&std::isfinite(scale)&&scale>0,"Pair probe domain");
 auto a=Empty({128},x.options(),true),b=Empty({128},x.options().dtype(at::kInt),true);auto st=c10_npu::getCurrentNPUStream(x.device().index());
 for(const auto& t:{x,g,a,b})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),st);
 auto rc=aclrtlaunch_vc_sparse_delay_reduce_probe(1,st.stream(true),x.data_ptr(),g.data_ptr(),a.data_ptr(),b.data_ptr(),valid0,valid1,base,float(scale));TORCH_CHECK(rc==ACL_SUCCESS,"Pair primitive launch failed:",rc);return {a,b};
}
std::string Metadata(){return R"json({"soc":"Ascend910B2","operation":"block1_raw_top8_center_scale_pv","contract_version":1,"q_rows":128,"k_columns":128,"vector_rows":64,"ub_bytes":155520,"score_slots":2,"candidates_per_block":1,"score_chunk_blocks":8,"selection_window_blocks":32,"output_rank":8,"output_columns":32,"producer_counter_version":10,"pv_counter_version":2,"inline_pv":true,"pv_arithmetic":"fp32_vmla_half","delayed_temperature":true,"scale_elements_per_query_tile":1536,"scale_order":"raw_Top8_then_FP32_center_scale","score_dtype":"float32_scaled","raw_reference":"original_E119_binary_temperature1","selection":"raw_window32_top8_then_resident_top8","tie":"Raw half order; scaled-half ties may select different V; raw reference exact source contract"})json";}
}
TORCH_LIBRARY(vc_laser_sparse_delay,m){m.def("raw_reference(Tensor q, Tensor k, Tensor v, Tensor geometry, int queries, int keys, int dimension, str library) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");m.def("pair_probe(Tensor scores, Tensor geometry, int valid0, int valid1, int base, float scale) -> (Tensor, Tensor)");m.def("fma_probe(Tensor values, Tensor weights, Tensor accum) -> (Tensor, Tensor, Tensor)");m.def("qk_tile(Tensor q, Tensor k) -> Tensor");m.def("produce(Tensor q, Tensor k, Tensor v, Tensor geometry, int queries, int keys, int dimension) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");m.def("probe(Tensor scores, Tensor geometry, int rows, int valid) -> (Tensor, Tensor)");m.def("metadata() -> str",TORCH_FN(Metadata));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_delay,PrivateUse1,m){m.impl("raw_reference",TORCH_FN(RawReference));m.impl("pair_probe",TORCH_FN(PairProbe));m.impl("fma_probe",TORCH_FN(FmaProbe));m.impl("qk_tile",TORCH_FN(QkTile));m.impl("produce",TORCH_FN(Produce));m.impl("probe",TORCH_FN(Probe));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_delay,Meta,m){m.impl("produce",TORCH_FN(Allocate));}
