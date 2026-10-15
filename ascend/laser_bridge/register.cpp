// Minimal eager-only registration around the upstream MindIE Laser PTA adapter.
#include <torch/library.h>
#include <ATen/ATen.h>
#include "la.h"

TORCH_LIBRARY(vc_laser, m) {
    m.def("la(Tensor query, Tensor key, Tensor value, Tensor? atten_mask=None, "
          "Tensor? alibi_mask=None, Tensor? drop_mask=None, float scale_value=1.0, "
          "int head_num=2, str input_layout='BNSD', float keep_prob=1.0, "
          "int pre_tokens=2147483647, int next_tokens=1, bool is_highPrecision=True) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(vc_laser, PrivateUse1, m) {
    m.impl("la", TORCH_FN(la_mindie_sd_impl_npu));
}

// Shape inference only; graph compilation/torchair integration is not validated.
std::tuple<at::Tensor, at::Tensor> vc_laser_meta(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& alibi,
    const c10::optional<at::Tensor>& drop, double scale, int64_t heads,
    std::string layout, double keep, int64_t pre, int64_t next, bool precise) {
    TORCH_CHECK(q.dim() == 4, "Laser expects rank-four BNSD tensors");
    return {at::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat)),
            at::empty(q.sizes(), q.options().dtype(at::kFloat))};
}

TORCH_LIBRARY_IMPL(vc_laser, Meta, m) {
    m.impl("la", TORCH_FN(vc_laser_meta));
}
