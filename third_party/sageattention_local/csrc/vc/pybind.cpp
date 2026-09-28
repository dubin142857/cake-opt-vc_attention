// VC additions to SageAttention, Apache-2.0; see NOTICE.
#include "vc_api.h"
#include <c10/cuda/CUDAGuard.h>

// Same deterministic Lloyd initialization/iteration policy as vc_attention.
std::vector<torch::Tensor> group_values_legacy_debug(torch::Tensor v, int64_t clusters, int64_t iterations) {
  TORCH_CHECK(v.is_cuda() && v.dim() == 4 && v.numel() > 0, "Clustering requires nonempty CUDA BHND");
  TORCH_CHECK(clusters > 0 && clusters <= 256 && iterations > 0 && iterations <= 100, "Invalid clustering parameters");
  const c10::cuda::CUDAGuard guard(v.device());
  const at::NoGradGuard no_grad;
  auto x = v.to(at::kFloat);
  int64_t count = std::min(clusters, x.size(2));
  auto ids = at::linspace(0, x.size(2) - 1, count, x.options()).to(at::kLong);
  auto centers = x.index_select(2, ids).clone();
  auto labels = at::empty({x.size(0),x.size(1),x.size(2)}, x.options().dtype(at::kLong));
  auto assignment_centers = centers;
  for (int64_t iter = 0; iter < iterations; ++iter) {
    assignment_centers = centers;
    auto sums = at::zeros_like(centers);
    auto counts = at::zeros({x.size(0),x.size(1),count,1}, x.options());
    auto norms = centers.square().sum(-1).unsqueeze(-2);
    for (int64_t start = 0; start < x.size(2); start += 2048) {
      auto rows = x.slice(2,start,std::min(start + 2048,x.size(2)));
      auto distances = rows.square().sum(-1,true) + norms - 2 * at::matmul(rows,centers.transpose(-2,-1));
      auto groups = distances.argmin(-1);
      labels.slice(2,start,start + rows.size(2)).copy_(groups);
      sums.scatter_add_(2,groups.unsqueeze(-1).expand_as(rows),rows);
      counts.scatter_add_(2,groups.unsqueeze(-1),at::ones_like(rows.slice(3,0,1)));
    }
    centers = at::where(counts > 0,sums / counts.clamp_min(1),centers);
  }
  return {at::argsort(labels,true,-1,false), labels, assignment_centers};
}

torch::Tensor group_values_legacy(torch::Tensor v, int64_t clusters, int64_t iterations) {
  return group_values_legacy_debug(v,clusters,iterations)[0];
}

torch::Tensor group_values(torch::Tensor v, int64_t clusters, int64_t iterations) {
  // Preserve the wider public cluster-count range through the legacy path.
  if (v.dim() == 4 && std::min(clusters,v.size(2)) <= 8)
    return vc_group_values_fused(v,clusters,iterations)[0];
  return group_values_legacy(v,clusters,iterations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("sage_vc_api_version") = 1;
  m.attr("sage_vc_kernel_version") = 6;
  m.attr("sage_vc_preprocess_version") = 1;
  m.def("forward", &vc_forward, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
      pybind11::arg("qs"), pybind11::arg("ks"), pybind11::arg("vs"), pybind11::arg("means"),
      pybind11::arg("expcast"), pybind11::arg("block_values"), pybind11::arg("output_type"),
      pybind11::arg("scale"), pybind11::arg("optimized") = true, pybind11::arg("packed_probability") = true, pybind11::arg("compact_values") = true);
  m.def("kernel_resources", &vc_kernel_resources, pybind11::arg("output_type") = 1);
  m.def("prepare_values", &vc_prepare_values, pybind11::arg("v"), pybind11::arg("channel_scale") = true);
  m.def("expcast_codes", &vc_expcast_codes);
  m.def("group_values", &group_values);
  m.def("group_values_legacy", &group_values_legacy);
  m.def("group_values_debug", &vc_group_values_fused);
  m.def("group_values_legacy_debug", &group_values_legacy_debug);
}
