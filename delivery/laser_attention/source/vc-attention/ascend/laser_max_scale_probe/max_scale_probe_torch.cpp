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
extern "C" uint32_t aclrtlaunch_vc_max_scale_probe(uint32_t,void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,float);
namespace {
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> Probe(const at::Tensor& x,double scale){
 TORCH_CHECK(x.is_privateuseone(),"NPU required");c10::OptionalDeviceGuard guard(x.device());
 TORCH_CHECK(x.dim()==2&&x.size(0)>0&&x.size(0)%64==0&&x.size(0)<=INT32_MAX/128&&x.size(1)==128,"Padded [R64,128] required");
 TORCH_CHECK(x.scalar_type()==at::kHalf&&x.is_contiguous()&&!x.requires_grad(),"Contiguous FP16 inference input required");
 TORCH_CHECK(std::isfinite(scale)&&scale>0,"Finite positive scale required");
 const char* soc=aclrtGetSocName();TORCH_CHECK(soc&&std::string(soc)==VC_MAX_SCALE_PROBE_SOC_VERSION&&std::string(soc)=="Ascend910B2","Exact910B2 required");
 auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform&&platform->GetCoreNumAiv()==48,"48 AIV required");
 auto f=at_npu::native::get_npu_format(x);TORCH_CHECK((f==ACL_FORMAT_ND||f==ACL_FORMAT_NCHW)&&reinterpret_cast<uintptr_t>(x.data_ptr())%32==0,"Aligned ND input required");
 auto y=at_npu::native::empty_with_format(x.sizes(),x.options(),ACL_FORMAT_ND);auto old=at_npu::native::empty_with_format({x.size(0),2},x.options(),ACL_FORMAT_ND);auto raw=at_npu::native::empty_with_format({x.size(0),2},x.options(),ACL_FORMAT_ND);auto maximum=at_npu::native::empty_with_format({x.size(0)},x.options(),ACL_FORMAT_ND);
 const uint32_t cores=std::min<int64_t>(48,x.size(0)/64);auto stream=c10_npu::getCurrentNPUStream(x.device().index());for(const auto& t:{x,y,old,raw,maximum})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
 auto rc=aclrtlaunch_vc_max_scale_probe(cores,stream.stream(true),x.data_ptr(),y.data_ptr(),old.data_ptr(),raw.data_ptr(),maximum.data_ptr(),x.size(0),cores,float(scale));TORCH_CHECK(rc==ACL_SUCCESS,"Scale probe launch failed:",rc);return {y,old,raw,maximum};
}
}
TORCH_LIBRARY(vc_laser_max_scale_probe,m){m.def("run(Tensor input, float scale) -> (Tensor, Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(vc_laser_max_scale_probe,PrivateUse1,m){m.impl("run",TORCH_FN(Probe));}
