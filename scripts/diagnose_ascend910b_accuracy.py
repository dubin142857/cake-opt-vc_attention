"""Separate long-K accumulation from repeated query-tile tasks; no benchmark."""
import argparse
import json
from pathlib import Path
import sys
import torch
from ascend_accuracy_diagnostics import check_payload, diagnose_failure, payload_difference, print_check
from validate_ascend910b import to_cpu, error


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="results/ascend910b_accuracy_diagnosis.json")
    args = parser.parse_args()
    import torch_npu
    import vc_attention_ascend as vc
    from vc_attention_ascend.reference import payload_reference
    torch.npu.set_device(args.device)
    # These are CPU accuracy references, not timed CPU/NPU workloads.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if hasattr(torch.npu, "matmul"):
        torch.npu.matmul.allow_hf32 = False
    report = {"status": "running", "device": torch.npu.get_device_name(args.device),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "vc_version": vc.__version__, "implementation_gate": .01,
              "repeat_gate": .001, "model_quality_claim": False,
              "cases": [], "failure_detail": None}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def save_detail(detail):
        report["failure_detail"] = detail
        save()

    save()
    try:
        generator = torch.Generator().manual_seed(91023)
        # Short Q / long K versus many Q tasks / short K, then the combination.
        cases = [(32, 513), (32, 4097), (32, 37741), (1025, 129), (1025, 37741)]
        failed_payload = None
        for nq, nk in cases:
            inputs = [torch.randn(1, 2, n, 128, generator=generator).bfloat16().to(f"npu:{args.device}")
                      for n in (nq, nk, nk)]
            p = vc.prepare(*inputs, layout="HND")
            fresh = vc.prepare(*inputs, layout="HND")
            preprocessing = payload_difference(p, fresh)
            ids = torch.linspace(0, nq - 1, min(nq, 32)).long().unique()
            for cast in (False, True):
                print(f"Checking long-sequence diagnostic: Q={nq}, K={nk}, H=2, D=128, expcast={cast}", flush=True)
                row, sample = check_payload(vc, p, cast, ids)
                fresh_out = vc.run_prepared(fresh, expcast=cast)
                row["fresh_preprocessing"] = preprocessing
                row["fresh_vs_fixed_sample"] = error(fresh_out.index_select(2, ids.to(fresh_out.device)), sample)
                row["passed"] = row["passed"] and row["fresh_vs_fixed_sample"]["relative_rmse"] <= .001
                report["cases"].append(row)
                print_check(row)
                save()
                if not row["passed"] and failed_payload is None:
                    failed_payload = (p, ids, cast, sample)
        if failed_payload is not None:
            p, ids, cast, sample = failed_payload
            expected = payload_reference(to_cpu(p), expcast=cast, query_indices=ids).to(p["dtype"])
            diagnose_failure(vc, p, sample, expected, ids, cast, save_detail)
        report["status"] = "passed" if all(r["passed"] for r in report["cases"]) else "failed"
        report["scope"] = "Focused shapes only; passing does not certify the original [1,37741,28,128] workload"
        save()
    except Exception as exc:
        report.update(status="error", error=repr(exc))
        save()
        raise
    print("Saved:", path, "status:", report["status"], flush=True)
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
