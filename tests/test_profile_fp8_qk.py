"""Comparison callbacks must time fresh APIs and retain each dtype's payloads."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import profile_fp8_qk as profile


def test_callbacks_share_payloads_and_keep_full_preprocessing():
    seen=[]
    def int_core(state,*,kernel_mode):
        seen.append(("int_core",state,kernel_mode))
    def fp8_core(state): seen.append(("fp8_core",state))
    def full(*qkv,**kwargs): seen.append(("full",qkv,kwargs))
    vc=SimpleNamespace(forward_prepared=int_core,sageattn_vc=full)
    fp8=SimpleNamespace(forward_prepared=fp8_core,sageattn_vc_fp8=full)
    int_state={"q":torch.zeros(1,1,1,64,dtype=torch.int8),"block_values":False}
    fp8_state={"q":torch.ones(1,1,1,64).to(torch.float8_e4m3fn),"block_values":False}
    original=tuple(object() for _ in range(3))
    core,apis,states=profile.make_calls(vc,fp8,full,None,original,int_state,fp8_state)
    for name in states:
        core[name]()
        assert seen[-1][1] is states[name]
        apis[name]();apis[name]()
        assert seen[-1][0]=="full" and seen[-1][1]==original
        assert seen[-1][2]["expcast"]==states[name]["expcast"]
    assert states["fp8_control"]["q"] is states["fp8_expcast"]["q"] is fp8_state["q"]
    assert states["int8_control_vc16"]["q"] is states["int8_expcast_vc22"]["q"] is int_state["q"]
    assert states["fp8_control"]["q"] is not states["int8_control_vc16"]["q"]
    assert "expcast" not in fp8_state and "expcast" not in int_state


def test_default_shape_and_invalid_measurement_arguments():
    args=profile.parse_args([])
    assert (args.batch,args.queries,args.heads,args.dim)==(1,37741,28,128)
    assert args.tokens==37741 and args.dtype=="bfloat16"
    assert profile.parse_args(["--check-queries","0"]).check_queries==0
    with pytest.raises(SystemExit): profile.parse_args(["--repeats","0"])
    with pytest.raises(SystemExit): profile.parse_args(["--check-queries","-1"])
