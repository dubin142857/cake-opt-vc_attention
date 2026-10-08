// Copyright 2026 vc-attention contributors. SPDX-License-Identifier: Apache-2.0
// Included after vc_sm90.cu definitions; the INT8 entry/dispatch stays separate.
namespace {
template<int D, typename Out, bool Cast>
auto fp8_kernel() {
  return sage_vc::sage_vc_attention_kernel_sage_layout<64,128,128,D,
      QuantGranularity::kPerBlock,QuantGranularity::kPerBlock,Out,
      MaskMode::kNone,false,true,Cast,false,false,Cast,0,168,false,(Cast?2:0),true>;
}

template<int D, typename Out, bool Cast>
void launch_fp8(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
                torch::Tensor ks, torch::Tensor vs, torch::Tensor out, float scale) {
  auto qm=sage_vc::create_tensor_map_4D<64,D>(reinterpret_cast<int8_t*>(q.data_ptr()),
      q.size(0),q.size(1),q.size(2),D,q.stride(0),q.stride(1),q.stride(2));
  auto km=sage_vc::create_tensor_map_4D<128,D>(reinterpret_cast<int8_t*>(k.data_ptr()),
      k.size(0),k.size(1),k.size(2),D,k.stride(0),k.stride(1),k.stride(2));
  auto vm=sage_vc::create_tensor_map_4D<D,128>(reinterpret_cast<int8_t*>(v.data_ptr()),
      v.size(0),v.size(1),D,v.size(3),v.stride(0),v.stride(1),v.stride(2));
  auto kernel=fp8_kernel<D,Out,Cast>();
  constexpr int bytes=(64+128+128)*D;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  kernel<<<dim3((q.size(2)+63)/64,q.size(1),q.size(0)),128,bytes,
      c10::cuda::getCurrentCUDAStream(q.get_device()).stream()>>>(
      qm,km,vm,qs.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),
      reinterpret_cast<Out*>(out.data_ptr()),nullptr,out.stride(0),out.stride(1),out.stride(2),
      q.size(2),k.size(2),1,scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int D, typename Out>
void dispatch_fp8(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
                  torch::Tensor ks, torch::Tensor vs, torch::Tensor out, float scale, bool cast) {
  if (cast) launch_fp8<D,Out,true>(q,k,v,qs,ks,vs,out,scale);
  else launch_fp8<D,Out,false>(q,k,v,qs,ks,vs,out,scale);
}

template<int D, typename Out, bool Cast>
std::map<std::string,int64_t> fp8_resources() {
  auto kernel=fp8_kernel<D,Out,Cast>();
  constexpr int bytes=320*D;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  cudaFuncAttributes a{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
  int active=0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,128,bytes));
  return {{"registers_per_thread",a.numRegs},{"local_bytes_per_thread",int64_t(a.localSizeBytes)},
      {"static_shared_bytes",int64_t(a.sharedSizeBytes)},{"dynamic_shared_bytes",bytes},
      {"threads_per_block",128},{"predicted_active_blocks_per_sm",active}};
}
template<int D, typename Out>
std::map<std::string,std::map<std::string,int64_t>> fp8_resource_pair() {
  return {{"fp8_control",fp8_resources<D,Out,false>()},{"fp8_expcast",fp8_resources<D,Out,true>()}};
}
void require_fp8_hopper(int device) {
  int major,minor;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
  TORCH_CHECK(major==9 && minor==0,"FP8 QK attention requires Hopper; no fallback");
}
}

std::map<std::string,std::map<std::string,int64_t>> vc_fp8_kernel_resources(int64_t output_type, int64_t dim) {
  TORCH_CHECK((output_type==0 || output_type==1) && (dim==64 || dim==128),"Expected output_type=0/1, D=64/128");
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  require_fp8_hopper(device);
  if (dim==64) {
    if (output_type==0) return fp8_resource_pair<64,half>();
    return fp8_resource_pair<64,nv_bfloat16>();
  }
  if (output_type==0) return fp8_resource_pair<128,half>();
  return fp8_resource_pair<128,nv_bfloat16>();
}

torch::Tensor vc_forward_fp8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, bool expcast, int64_t output_type, double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.numel()>0,"Q requires nonempty CUDA BHND");
  for (const auto& x : {q,k,v,qs,ks,vs})
    TORCH_CHECK(x.device()==q.device() && x.is_contiguous(),"Payloads must be contiguous on the same CUDA device");
  for (const auto& x : {q,k,v})
    TORCH_CHECK(x.scalar_type()==at::ScalarType::Float8_e4m3fn,"FP8 Q/K/V require E4M3");
  for (const auto& x : {qs,ks,vs})
    TORCH_CHECK(x.scalar_type()==at::kFloat,"Scales require FP32");
  TORCH_CHECK(k.dim()==4 && k.numel()>0 && v.dim()==4,"Expected four-dimensional K/V");
  int64_t b=q.size(0),h=q.size(1),nq=q.size(2),nk=k.size(2),d=q.size(3),tiles=(nk+127)/128;
  TORCH_CHECK((d==64 || d==128) && k.size(0)==b && k.size(1)==h && k.size(3)==d,"B/H/D mismatch; D=64/128; no GQA");
  TORCH_CHECK(v.sizes()==torch::IntArrayRef({b,h,d,tiles*128}),"Incorrect packed V shape");
  TORCH_CHECK(qs.sizes()==torch::IntArrayRef({b,h,(nq+63)/64}) &&
      ks.sizes()==torch::IntArrayRef({b,h,tiles}) && vs.sizes()==torch::IntArrayRef({b,h,d}),"Incorrect FP8 block scales");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(vs.data_ptr())%alignof(float2)==0,"V scales require float2 pointer alignment");
  TORCH_CHECK(q.numel()<INT32_MAX && k.numel()<INT32_MAX && v.numel()<INT32_MAX && b<=65535 && h<=65535,
      "Shape exceeds kernel bounds");
  TORCH_CHECK(output_type==0 || output_type==1,"Output type must be 0 (FP16) or 1 (BF16)");
  TORCH_CHECK(std::isfinite(scale) && scale>0 && scale<=FLT_MAX && float(scale)>0,"Scale must be positive finite FP32");
  const c10::cuda::CUDAGuard guard(q.device());
  require_fp8_hopper(q.get_device());
  auto out=torch::empty(q.sizes(),q.options().dtype(output_type==0?at::kHalf:at::kBFloat16));
  if (d==64) {
    if (output_type==0) dispatch_fp8<64,half>(q,k,v,qs,ks,vs,out,scale,expcast);
    else dispatch_fp8<64,nv_bfloat16>(q,k,v,qs,ks,vs,out,scale,expcast);
  } else {
    if (output_type==0) dispatch_fp8<128,half>(q,k,v,qs,ks,vs,out,scale,expcast);
    else dispatch_fp8<128,nv_bfloat16>(q,k,v,qs,ks,vs,out,scale,expcast);
  }
  return out;
}
