// Appended to a generated diagnostic copy of the pinned Sage/VC loop.
// The generated prefix retains the original Apache-2.0 license notice.
namespace {
template<int D, typename Out, bool Cast, bool Trace>
auto phase_kernel() {
  return sage_vc::sage_vc_attention_kernel_sage_layout<64,128,128,D,
    QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
    MaskMode::kNone,false,true,Cast,false,false,Cast,0,168,false,
    (Cast ? 9 : 0),false,false,false,Trace>;
}

template<int D, typename Out, bool Cast, bool Trace>
void launch_phase(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs,
    torch::Tensor out, torch::Tensor clocks, float scale) {
  auto qm=sage_vc::create_tensor_map_4D<64,D>(q.data_ptr<int8_t>(),
    q.size(0),q.size(1),q.size(2),D,q.stride(0),q.stride(1),q.stride(2));
  auto km=sage_vc::create_tensor_map_4D<128,D>(k.data_ptr<int8_t>(),
    k.size(0),k.size(1),k.size(2),D,k.stride(0),k.stride(1),k.stride(2));
  auto vm=sage_vc::create_tensor_map_4D<D,128>(reinterpret_cast<int8_t*>(v.data_ptr()),
    v.size(0),v.size(1),D,v.size(3),v.stride(0),v.stride(1),v.stride(2));
  auto kernel=phase_kernel<D,Out,Cast,Trace>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,320*D));
  dim3 grid((q.size(2)+63)/64,q.size(1),q.size(0));
  kernel<<<grid,128,320*D,c10::cuda::getCurrentCUDAStream(q.get_device()).stream()>>>(
    qm,km,vm,qs.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),
    reinterpret_cast<Out*>(out.data_ptr()),Trace ? clocks.data_ptr() : nullptr,
    out.stride(0),out.stride(1),out.stride(2),q.size(2),k.size(2),1,scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int D, typename Out> void dispatch_phase(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, torch::Tensor clocks,
    float scale, bool cast, bool trace) {
  if (cast && trace) launch_phase<D,Out,true,true>(q,k,v,qs,ks,vs,out,clocks,scale);
  else if (cast) launch_phase<D,Out,true,false>(q,k,v,qs,ks,vs,out,clocks,scale);
  else if (trace) launch_phase<D,Out,false,true>(q,k,v,qs,ks,vs,out,clocks,scale);
  else launch_phase<D,Out,false,false>(q,k,v,qs,ks,vs,out,clocks,scale);
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, int64_t output_type,
    double scale, bool cast, bool trace) {
  TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.is_contiguous(),"Expected contiguous CUDA HND Q");
  c10::cuda::CUDAGuard guard(q.device());
  cudaDeviceProp props; C10_CUDA_CHECK(cudaGetDeviceProperties(&props,q.get_device()));
  TORCH_CHECK(props.major==9 && props.minor==0,"Diagnostic kernel requires SM90a");
  for (auto t: {k,v,qs,ks,vs})
    TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous(),"CUDA device/layout mismatch");
  auto B=q.size(0), H=q.size(1), M=q.size(2), D=q.size(3);
  TORCH_CHECK(B>0 && H>0 && M>0 && (D==64 || D==128) && k.dim()==4 &&
    k.size(0)==B && k.size(1)==H && k.size(3)==D && k.size(2)>0,"Q/K shape mismatch");
  auto N=k.size(2), blocks=(N+127)/128;
  TORCH_CHECK(q.scalar_type()==torch::kInt8 && k.scalar_type()==torch::kInt8 &&
    v.scalar_type()==torch::kFloat8_e4m3fn,"Expected INT8 QK and E4M3 V");
  TORCH_CHECK(v.dim()==4 && v.size(0)==B && v.size(1)==H && v.size(2)==D &&
    v.size(3)==blocks*128,"Expected padded transposed FP8 V");
  TORCH_CHECK(qs.scalar_type()==torch::kFloat32 && ks.scalar_type()==torch::kFloat32 &&
    vs.scalar_type()==torch::kFloat32 && qs.numel()==B*H*((M+63)/64)*32 &&
    ks.numel()==B*H*blocks*4 && vs.numel()==B*H*D,"Invalid per-thread/channel scales");
  TORCH_CHECK((output_type==0 || output_type==1) && std::isfinite(scale) && scale>0,"Invalid output/scale");
  auto out=torch::empty(q.sizes(),q.options().dtype(output_type==0 ? torch::kFloat16 : torch::kBFloat16));
  auto count=(B*H*((M+63)/64)+PHASE_CTA_STRIDE-1)/PHASE_CTA_STRIDE;
  auto slots=(blocks-1+PHASE_TILE_STRIDE-1)/PHASE_TILE_STRIDE+1;
  // Sentinel initialization makes missing stores detectable even with allocator reuse.
  // This diagnostic allocation/memset is included in the reported probe API overhead.
  auto clocks=torch::zeros({trace ? count : 0, slots, int64_t(PHASE_RECORD_WIDTH)},q.options().dtype(torch::kInt64));
  if (D==64 && output_type==0) dispatch_phase<64,half>(q,k,v,qs,ks,vs,out,clocks,scale,cast,trace);
  else if (D==64) dispatch_phase<64,nv_bfloat16>(q,k,v,qs,ks,vs,out,clocks,scale,cast,trace);
  else if (output_type==0) dispatch_phase<128,half>(q,k,v,qs,ks,vs,out,clocks,scale,cast,trace);
  else dispatch_phase<128,nv_bfloat16>(q,k,v,qs,ks,vs,out,clocks,scale,cast,trace);
  return {out,clocks};
}

template<int D,typename Out,bool Cast,bool Trace>
std::map<std::string,int64_t> resources_for() {
  cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncGetAttributes(&a,phase_kernel<D,Out,Cast,Trace>()));
  return {{"registers_per_thread",a.numRegs},{"local_bytes",a.localSizeBytes},
          {"static_shared_bytes",a.sharedSizeBytes},{"dynamic_shared_bytes",320*D}};
}
template<int D,typename Out>
std::map<std::string,std::map<std::string,int64_t>> resources_typed() {
  return {{"control_plain",resources_for<D,Out,false,false>()},
          {"control_probe",resources_for<D,Out,false,true>()},
          {"expcast_plain",resources_for<D,Out,true,false>()},
          {"expcast_probe",resources_for<D,Out,true,true>()}};
}
auto resources(int dim,int output_type) {
  TORCH_CHECK((dim==64 || dim==128) && (output_type==0 || output_type==1),"Unsupported resource query");
  if (dim==64 && output_type==0) return resources_typed<64,half>();
  if (dim==64) return resources_typed<64,nv_bfloat16>();
  if (output_type==0) return resources_typed<128,half>();
  return resources_typed<128,nv_bfloat16>();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&forward);
  m.def("resources",&resources);
}
