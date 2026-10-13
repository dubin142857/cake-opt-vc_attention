"""Safety checks for the candidate-local fence correction on physical TIR."""
from pathlib import Path
import sys

import pytest

pytest.importorskip("tilelang")
from tvm import IRModule, ir, tirx
from tvm.tirx.stmt_functor import post_order_visit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tilelang_q128_lowering import _fix_q128_fences


def module_with_fences(pv_extent=64, dots_count=128, offset=0,
                       pointer_name="dots", include_pv=True):
    dots = tirx.decl_buffer((64,), "int32", name=pointer_name, scope="local")
    pv = tirx.decl_buffer((pv_extent,), "float32", name="pv", scope="local")
    op = ir.Op.get("tl.warpgroup_fence_operand")

    def fence(buf, count):
        return tirx.Evaluate(tirx.Call("handle", op, [
            tirx.StringImm(str(buf.dtype)), buf.data,
            tirx.IntImm("int32", offset), tirx.IntImm("int32", count),
        ]))

    body = [tirx.AllocBuffer(dots), tirx.AllocBuffer(pv), fence(dots, dots_count)]
    if include_pv:
        body.append(fence(pv, pv_extent * 2))
    func = tirx.PrimFunc([], tirx.SeqStmt(body)).with_attr("global_symbol", "candidate")
    gv = ir.GlobalVar("candidate")
    return IRModule({gv: func}, attrs={"metadata_marker": "preserve me"}), gv


def fence_counts(mod):
    counts = []

    def visit(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", None) == "tl.warpgroup_fence_operand":
            counts.append((node.args[1].name, int(node.args[3])))

    for func in mod.functions.values():
        post_order_visit(func.body, visit)
    return counts


@pytest.mark.parametrize("pv_extent", [32, 64])
def test_physical_fences_repair_and_preserve_module(pv_extent):
    original, gv = module_with_fences(pv_extent=pv_extent)
    fixed = _fix_q128_fences(original)
    assert fence_counts(fixed) == [("dots", 64), ("pv", pv_extent)]
    assert fence_counts(original) == [("dots", 128), ("pv", pv_extent * 2)]
    assert next(iter(fixed.functions)).same_as(gv)
    assert ir.structural_equal(fixed.attrs, original.attrs)
    assert ir.structural_equal(fixed[gv].attrs, original[gv].attrs)
    # A compiler that already emitted correct counts stays unchanged.
    assert ir.structural_equal(_fix_q128_fences(fixed), fixed)


@pytest.mark.parametrize("kwargs,match", [
    ({"dots_count": 127}, "fence count"),
    ({"offset": 1}, "literal zero"),
    ({"pointer_name": "unexpected"}, "exact dots or pv"),
])
def test_unrecognized_fences_are_rejected(kwargs, match):
    mod, _ = module_with_fences(**kwargs)
    with pytest.raises(ValueError, match=match):
        _fix_q128_fences(mod)


def test_missing_operand_fences_are_rejected():
    mod, _ = module_with_fences(include_pv=False)
    with pytest.raises(ValueError, match="both dots and pv"):
        _fix_q128_fences(mod)
