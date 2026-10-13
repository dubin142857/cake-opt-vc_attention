"""Independent fragment-map proof and remote fail-closed validation checks."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import tensor_mass_native as native
import benchmark_cuda_expcast as bench


@pytest.mark.parametrize("bits",[8,16])
def test_mma_selector_preserves_each_lanes_own_partial_sum(bits):
    # Reconstruct A/B/C from the PTX fragment coordinates, independently of
    # CUDA code. Distinct values distinguish every source lane and register.
    width=4 if bits==8 else 2
    fragment=torch.arange(32*4*width,dtype=torch.float64).reshape(32,4,width)+1
    a=torch.zeros(16,8*width,dtype=torch.float64)
    b=torch.zeros(8*width,8,dtype=torch.float64)
    for lane in range(32):
        group,t=lane//4,lane%4
        for reg in range(4):
            for i in range(width):
                a[group+(reg%2)*8,t*width+i+(reg//2)*4*width]=fragment[lane,reg,i]
        for reg in range(2):
            for i in range(width):
                b[t*width+i+reg*4*width,group]=int(t==lane//8)
    c=a@b
    for lane in range(32):
        for row in range(2):
            expected=fragment[lane,row::2].sum()
            assert c[lane//4+row*8,(lane%4)*2]==expected
            assert c[lane//4+row*8,(lane%4)*2+1]==expected


def test_each_mma_and_fp32_merge_fit_exact_integer_units():
    units=[c if c<8 else ((c&7)+8)<<(c//8-1) for c in range(121)]
    assert min(units)==0 and max(units)==2**17
    assert max(units)*8==2**20 and max(units)*32==2**22


def test_candidate_uses_measured_baseline_and_retains_tail_and_fallback():
    assert native.base.candidate_kernel() in native.generate_source()
    original=native.base.candidate_kernel()
    candidate=native.candidate_kernel()
    # Keep the entire pipeline (including original masked tail) identical;
    # only the probability helper call and C++ symbol change.
    restored=candidate.replace("kernel_tensor_mass(","kernel_scale_prefetch(")
    restored=restored.replace("vc_probability_prepare_tensor_mass<", "vc_probability_prepare_integer<")
    assert restored==original
    a=bench.parse_args(["--experiment","tensor_mass"])
    assert a.output.name=="h200_cuda_expcast_tensor_mass.json"
    assert bench.EXPERIMENTS["tensor_mass"][0] is native
    assert native.EXACT_OUTPUT


@pytest.mark.parametrize("field,value",[("registers_per_thread",169),("local_bytes",4),
    ("dynamic_shared_bytes",57344),("static_shared_bytes",256)])
def test_remote_resource_regression_stops_candidate(field,value):
    def resources(dim,dtype):
        old=dict(registers_per_thread=168,local_bytes=0,dynamic_shared_bytes=320*dim,static_shared_bytes=128)
        return {"expcast_plain":old,"expcast_probe":{**old,field:value}}
    with pytest.raises(RuntimeError):native.check_resources(SimpleNamespace(resources=resources))


def test_byte_cases_cover_subnormals_all_codes_and_all_fragment_positions():
    from validate_tensor_mass import byte_cases
    cases=dict(byte_cases(torch.device("cpu")))
    assert len(cases)==193
    assert all(f"constant_{c}" in cases for c in range(121))
    assert all(f"impulse_{c}" in cases for c in range(64))
    assert all(f"max_and_subnormal_{c}" in cases for c in range(1,8))
