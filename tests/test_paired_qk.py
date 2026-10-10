"""Guards for the N256 native boundary, immutable math, and remote oracle."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import paired_qk_native as native
import benchmark_cuda_expcast as bench


def test_candidate_is_opt_in_and_uses_scale_prefetch_baseline():
    args=bench.parse_args(["--experiment","paired_qk"])
    assert args.output.name=="h200_cuda_expcast_paired_qk.json"
    assert bench.EXPERIMENTS[args.experiment][0] is native
    assert bench.parse_args([]).experiment=="balanced_max"
    assert bench.variant_modes(native,"vc28_paired_qk",native.BASELINE)=={
        "vc28_scale_prefetch":False,"vc28_paired_qk":True}
    assert native.EXACT_OUTPUT


def test_generated_native_boundary_matches_memory_footprint():
    s=native.generate_source()
    assert native.base.candidate_kernel() in s
    assert "else return sage_vc::sage_vc_attention_kernel_scale_prefetch<" in s
    assert "create_tensor_map_4D<(Trace ? 256 : 128),D>(k.data_ptr<int8_t>()" in s
    assert s.count("(Trace ? 448 : 320)*D")==3  # attribute, launch, resource report
    assert "0,240,false,9>" in s
    assert "clock64" not in s and "torch::zeros" not in s


def test_exact_probability_pv_and_epilogue_regions_preserved():
    old=native.base.candidate_kernel();new=native.candidate_kernel()
    start=old.index("    // Integer-max candidates")
    end=old.index("\n  }\n\n  {\n    p ^= 1;",start)
    # Once inside the paired loop, once for an unpaired MAIN tile.
    assert new.count(old[start:end])==2
    tail=old[old.index("    // convert RS to float"):]
    assert new.endswith(tail)
    # All K overwrite happens after its pending QK batch completes.
    wait=new.index("wgmma::warpgroup_wait<0>();",new.index("paired_scores[2]"))
    overwrite=new.index("(pair+1)*256",wait)
    assert wait<overwrite<new.index("auto& RS = paired_scores[slot]")


def test_wide_mma_operands_cover_each_accumulator_once():
    import re
    s=native.wide_mma_source()
    assert "m64n256k32.s32.s8.s8" in s
    refs=re.findall(r'"\+r"\(d\[(\d)\]\[0\]\[(\d)\]\[(\d)\]\)',s)
    assert len(refs)==len(set(refs))==128
    assert refs==[(str(h),str(f),str(r)) for h in range(2) for f in range(8) for r in range(8)]


def test_integer_oracle_validates_real_tensor_values():
    seen=[]
    def check(q,k):
        seen.append((q.shape,k.shape))
        return q.int() @ k.int().transpose(-1,-2)
    rows=native.validate_qk(SimpleNamespace(check_paired_qk=check),torch.device("cpu"),lambda rows:None)
    assert len(seen)==6 and all(r["status"]=="passed" for r in rows)


def test_integer_oracle_rejects_wrong_column_mapping():
    def broken(q,k):return (q.int() @ k.int().transpose(-1,-2)).flip(-1)
    with pytest.raises(AssertionError):
        native.validate_qk(SimpleNamespace(check_paired_qk=broken),torch.device("cpu"),lambda rows:None)
