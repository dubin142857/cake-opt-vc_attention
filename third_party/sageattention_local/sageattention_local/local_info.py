"""Provenance for the independent source build; no import of upstream sageattention."""
import json
from pathlib import Path
import torch


def local_build_info():
    # Explicit imports expose a missing/broken binary rather than a fallback.
    from . import _fused, _qattn_sm90, sm90_compile
    from . import core

    if not core.SM90_ENABLED:
        raise RuntimeError("Local Hopper extension is unavailable; rebuild with scripts/build_sage_local.sh")
    op = sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    if not isinstance(op, torch.library.CustomOpDef):
        raise RuntimeError("Local SM90 entry is not the registered CUDA operator")
    provenance = json.loads(Path(__file__).with_name("UPSTREAM.json").read_text(encoding="utf-8"))
    return {
        "upstream_repository": provenance["repository"], "upstream_commit": provenance["commit"],
        "upstream_tag": provenance["tag"], "local_version": provenance["local_version"],
        "package_file": str(Path(__file__).with_name("__init__.py")),
        "extensions": {"qattn_sm90": _qattn_sm90.__file__, "fused": _fused.__file__},
        "torch_operator": "sageattention_local_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
        "hopper_path": "Q/K per-thread INT8 (Triton), P/V FP8 (CUDA), fp32+fp32 accumulation",
        "kernel_source_modified": True,
        "kernel_math_modified": False,
        "build_only_patches": ["fused.cu: remove unused CUDAContext.h include"],
    }
