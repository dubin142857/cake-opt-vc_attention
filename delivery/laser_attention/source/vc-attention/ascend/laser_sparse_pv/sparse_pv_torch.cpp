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
extern "C" uint32_t aclrtlaunch_vc_sparse_pv_fp16(uint32_t,void*,void*,void*,void*,void*,void*,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
namespace {
using Outputs=std::tuple<at::Tensor,at::Tensor>;
void Check(const at::Tensor& value,const at::Tensor& indices,const at::Tensor& weights,int64_t rows,int64_t rank) {
    TORCH_CHECK(value.dim()==2 && value.size(0)>0 && (value.size(1)==64||value.size(1)==128),"Half V[keys,D64/128] required");
    TORCH_CHECK(rows>0 && rows<=UINT32_MAX-15 && value.size(0)<=INT32_MAX && rank>=1 && rank<=16,"Invalid sparse PV extents");
    TORCH_CHECK(value.scalar_type()==at::kHalf && indices.scalar_type()==at::kInt && weights.scalar_type()==at::kHalf,"Half values/weights and int32 indices required");
    TORCH_CHECK(indices.dim()==3 && indices.size(0)==(rows+15)/16 && indices.size(1)==16 && indices.size(2)==16 && weights.sizes()==indices.sizes(),"Rank-major group/rank/row packet required");
    for(const auto& x:{value,indices,weights}) {
        TORCH_CHECK(x.device()==value.device() && x.is_contiguous() && !x.requires_grad(),"Contiguous matching inference inputs required");
    }
}
Outputs Allocate(const at::Tensor& value,const at::Tensor& indices,const at::Tensor& weights,int64_t rows,int64_t rank) {
    Check(value,indices,weights,rows,rank);const int64_t cores=std::min<int64_t>((rows+15)/16,48);
    return {at::empty({rows,value.size(1)},value.options().dtype(at::kFloat)),at::empty({cores,64},value.options().dtype(at::kLong))};
}
Outputs Weighted(const at::Tensor& value,const at::Tensor& indices,const at::Tensor& weights,int64_t rows,int64_t rank) {
    TORCH_CHECK(value.is_privateuseone(),"NPU required; no fallback");c10::OptionalDeviceGuard guard(value.device());Check(value,indices,weights,rows,rank);
    for(const auto& x:{value,indices,weights}) {
        auto format=at_npu::native::get_npu_format(x);
        TORCH_CHECK(format==ACL_FORMAT_ND||format==ACL_FORMAT_NCHW,"Base ND format required");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%32==0,"32B input alignment required");
    }
    const char* soc=aclrtGetSocName();TORCH_CHECK(soc && std::string(soc)==VC_SPARSE_PV_SOC_VERSION,"Compiled/runtime SoC mismatch");
    auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform && platform->GetSocVersion()==platform_ascendc::SocVersion::ASCEND910B && platform->GetCoreNumAiv()==48,"910B2 48-AIV platform required");
    const uint32_t cores=std::min<int64_t>((rows+15)/16,platform->GetCoreNumAiv());
    auto output=at_npu::native::empty_with_format({rows,value.size(1)},value.options().dtype(at::kFloat),ACL_FORMAT_ND);
    auto status=at_npu::native::empty_with_format({cores,64},value.options().dtype(at::kLong),ACL_FORMAT_ND);
    auto stream=c10_npu::getCurrentNPUStream(value.device().index());
    for(const auto& x:{value,indices,weights})c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(),stream);
    auto ret=aclrtlaunch_vc_sparse_pv_fp16(cores,stream.stream(true),value.data_ptr(),indices.data_ptr(),weights.data_ptr(),output.data_ptr(),status.data_ptr(),
        rows,value.size(0),value.size(1),rank,cores);
    TORCH_CHECK(ret==ACL_SUCCESS,"Sparse PV launch failed: ",ret);return {output,status};
}
std::string Metadata(){return "{\"soc\":\"" VC_SPARSE_PV_SOC_VERSION "\",\"operation\":\"weighted_sparse_pv\",\"contract_version\":1,\"rows\":16,\"rank_max\":16,\"ub_bytes\":84032,\"output\":\"float32\"}";}
}
TORCH_LIBRARY(vc_laser_sparse_pv,m){m.def("weighted(Tensor value, Tensor indices, Tensor weights, int rows, int rank) -> (Tensor, Tensor)");m.def("metadata() -> str",TORCH_FN(Metadata));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_pv,PrivateUse1,m){m.impl("weighted",TORCH_FN(Weighted));}
TORCH_LIBRARY_IMPL(vc_laser_sparse_pv,Meta,m){m.impl("weighted",TORCH_FN(Allocate));}
