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
extern "C" uint32_t aclrtlaunch_vc_sparse_candidates_score_chunk_fp16(uint32_t,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
    uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,float,uint32_t);
extern "C" uint32_t aclrtlaunch_vc_sparse_candidate_score_chunk_probe(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_vc_sparse_candidate_score_chunk_diagnostic(uint32_t,void*,void*,void*,void*,uint32_t);
namespace {
using Outputs=std::tuple<at::Tensor,at::Tensor,at::Tensor>;
using Pair=std::tuple<at::Tensor,at::Tensor>;
void TensorCheck(const at::Tensor& q,const at::Tensor& x){
    TORCH_CHECK(x.scalar_type()==at::kHalf && x.device()==q.device() && x.is_contiguous() && !x.requires_grad(),"Contiguous matching-device inference half tensors required");
}
void Check(const at::Tensor& q,const at::Tensor& k,const at::Tensor& geometry,int64_t queries,int64_t keys,int64_t dimension){
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && q.size(0)>0 && q.size(1)>0 && q.size(0)==k.size(0) && q.size(1)==k.size(1) && q.size(3)==128 && k.size(3)==128,"BHSD padded D128 required");
    TORCH_CHECK(queries>0 && keys>0 && queries<=INT32_MAX-255 && keys<=INT32_MAX-255 && q.size(2)==(queries+255)/256*256 && k.size(2)==(keys+255)/256*256 && (dimension==64||dimension==128),"S256 padding and logical D64/128 required");
    TORCH_CHECK(geometry.sizes()==at::IntArrayRef({64,128}),"Half position geometry[64,128] required");
    for(const auto& x:{q,k,geometry})TensorCheck(q,x);
    const int64_t heads=q.size(0)*q.size(1),qwork=(queries+127)/128*128,cols=(k.size(2)/128+31)/32*32;
    TORCH_CHECK(heads<=INT32_MAX/k.size(2) && heads<=INT32_MAX/cols/qwork,"Candidate int32 indexing extent exceeded");
}
void DeviceCheck(const at::Tensor& q){
    TORCH_CHECK(q.is_privateuseone(),"NPU required; no fallback");
    const char* soc=aclrtGetSocName();TORCH_CHECK(soc && std::string(soc)==VC_CANDIDATE_SOC_VERSION && std::string(soc)=="Ascend910B2","Exact 910B2 runtime required");
    auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform && platform->GetCoreNumAic()==24 && platform->GetCoreNumAiv()==48,"24 AIC / 48 AIV required");
}
at::Tensor Empty(at::IntArrayRef sizes,const at::TensorOptions& options){return at_npu::native::empty_with_format(sizes,options,ACL_FORMAT_ND);}
Outputs Allocate(const at::Tensor& q,const at::Tensor& k,const at::Tensor& geometry,int64_t queries,int64_t keys,int64_t dimension){
    Check(q,k,geometry,queries,keys,dimension);const int64_t h=q.size(0)*q.size(1),qr=(queries+127)/128*128,cols=(k.size(2)/128+31)/32*32,c=std::min<int64_t>(24,h*(qr/128));
    return {at::empty({h,cols,qr},q.options()),at::empty({h,cols,qr},q.options().dtype(at::kInt)),at::empty({3*c,64},q.options().dtype(at::kLong))};
}
Outputs Produce(const at::Tensor& q,const at::Tensor& k,const at::Tensor& geometry,int64_t queries,int64_t keys,int64_t dimension){
    c10::OptionalDeviceGuard guard(q.device());Check(q,k,geometry,queries,keys,dimension);DeviceCheck(q);
    for(const auto& x:{q,k,geometry}){auto format=at_npu::native::get_npu_format(x);TORCH_CHECK((format==ACL_FORMAT_ND||format==ACL_FORMAT_NCHW)&&reinterpret_cast<uintptr_t>(x.data_ptr())%32==0,"Aligned base format required");}
    const int64_t h=q.size(0)*q.size(1),qr=(queries+127)/128*128,cols=(k.size(2)/128+31)/32*32,c=std::min<int64_t>(24,h*(qr/128));
    auto values=Empty({h,cols,qr},q.options());auto indices=Empty({h,cols,qr},q.options().dtype(at::kInt));
    const int64_t realCols=k.size(2)/128;
    if(cols>realCols)values.narrow(1,realCols,cols-realCols).fill_(-INFINITY);
    auto ring=Empty({c,2,8,128,128},q.options());auto storage=Empty({(3*c+1)*64},q.options().dtype(at::kLong));
    const auto offset=((512-reinterpret_cast<uintptr_t>(storage.data_ptr())%512)%512)/8;
    auto counts=storage.narrow(0,offset,3*c*64).view({3*c,64});
    auto stream=c10_npu::getCurrentNPUStream(q.device().index());
    for(const auto& x:{q,k,geometry,values,indices,ring,storage})c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(),stream);
    auto ret=aclrtlaunch_vc_sparse_candidates_score_chunk_fp16(c,stream.stream(true),q.data_ptr(),k.data_ptr(),geometry.data_ptr(),values.data_ptr(),ring.data_ptr(),indices.data_ptr(),values.data_ptr(),ring.data_ptr(),counts.data_ptr(),h,q.size(2),k.size(2),queries,keys,dimension,c,cols,1/std::sqrt(float(dimension)),0);
    TORCH_CHECK(ret==ACL_SUCCESS,"Candidate producer launch failed: ",ret);return {values,indices,counts};
}
Pair Probe(const at::Tensor& x,const at::Tensor& geometry,int64_t rows,int64_t valid){
    c10::OptionalDeviceGuard guard(x.device());DeviceCheck(x);TensorCheck(x,x);TensorCheck(x,geometry);
    TORCH_CHECK(rows>0 && rows<=48*64 && valid>=0 && valid<=128 && x.sizes()==at::IntArrayRef({(rows+63)/64*64,128}) && geometry.sizes()==at::IntArrayRef({64,128}),"Padded probe rows / valid-key extent invalid");
    const int64_t padded=(rows+63)/64*64;auto values=Empty({2,padded},x.options());auto indices=Empty({2,padded},x.options().dtype(at::kInt));
    auto stream=c10_npu::getCurrentNPUStream(x.device().index());for(const auto& t:{x,geometry})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
    auto ret=aclrtlaunch_vc_sparse_candidate_score_chunk_probe(padded/64,stream.stream(true),x.data_ptr(),geometry.data_ptr(),values.data_ptr(),indices.data_ptr(),rows,valid);
    TORCH_CHECK(ret==ACL_SUCCESS,"Candidate probe launch failed: ",ret);return {values,indices};
}
at::Tensor Diagnostic(const at::Tensor& x,const at::Tensor& geometry,int64_t valid){
    c10::OptionalDeviceGuard guard(x.device());DeviceCheck(x);TensorCheck(x,x);TensorCheck(x,geometry);
    TORCH_CHECK(x.sizes()==at::IntArrayRef({64,128})&&geometry.sizes()==at::IntArrayRef({64,128})&&valid>=0&&valid<=128,"Diagnostic extents");
    auto out=Empty({10,8192},x.options());out.fill_(0);auto stream=c10_npu::getCurrentNPUStream(x.device().index());
    for(const auto&t:{x,geometry})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
    auto ret=aclrtlaunch_vc_sparse_candidate_score_chunk_diagnostic(1,stream.stream(true),x.data_ptr(),geometry.data_ptr(),out.data_ptr(),valid);
    TORCH_CHECK(ret==ACL_SUCCESS,"Diagnostic launch failed");return out;
}
std::string Metadata(){return "{\"soc\":\"" VC_CANDIDATE_SOC_VERSION "\",\"operation\":\"block1_score_chunk\",\"contract_version\":1,\"q_rows\":128,\"k_columns\":128,\"vector_rows\":64,\"ub_bytes\":54656,\"score_slots\":2,\"candidates_per_block\":1,\"geometry\":\"mask_words_in_half_storage\",\"lookup_bytes\":6176,\"output_slots\":8,\"defer_first_ack\":false,\"readiness_events\":2,\"score_ack\":\"after_mte2_read\",\"score_ack_pipe\":\"MTE2\",\"output_bulk_columns\":8,\"output_batch_buffers\":1,\"producer_counter_version\":3,\"score_chunk_blocks\":8,\"bulk_dma_api\":\"DataCopyPadExt\"}";}
}
TORCH_LIBRARY(vc_laser_sparse_candidates_score_chunk,m){m.def("produce(Tensor q, Tensor k, Tensor geometry, int queries, int keys, int dimension) -> (Tensor, Tensor, Tensor)");m.def("probe(Tensor scores, Tensor geometry, int rows, int valid) -> (Tensor, Tensor)");m.def("diagnostic(Tensor scores, Tensor geometry, int valid) -> Tensor");m.def("metadata() -> str",TORCH_FN(Metadata));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_candidates_score_chunk,PrivateUse1,m){m.impl("produce",TORCH_FN(Produce));m.impl("probe",TORCH_FN(Probe));m.impl("diagnostic",TORCH_FN(Diagnostic));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_candidates_score_chunk,Meta,m){m.impl("produce",TORCH_FN(Allocate));}
