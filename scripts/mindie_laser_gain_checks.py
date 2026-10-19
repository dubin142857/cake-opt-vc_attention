"""Device correctness gates for per-K-tile gain preparation; never timed."""
from __future__ import annotations


def validate_block_gains(adapters, device, report, save):
    """Stress adjacent Q/K scales, tails and repeated use of the same storage.

    Values are finite normal scales. The two operators see identical INT8
    tensors and scale pointers. Comparing every padded FP32 output element
    detects a wrong Q0/Q1 selection without the final FP16/BF16 cast hiding it.
    """
    import torch
    from profile_mindie_laser_quant import compare_all_rows

    rows = report.setdefault("alternating_block_gains", [])
    for dtype in (torch.float16, torch.bfloat16):
        generator = torch.Generator().manual_seed(830)
        inputs = [torch.randn(shape, generator=generator).to(dtype).to(device) for shape in
                  ((2, 3, 257, 64), (2, 3, 4097, 64), (2, 3, 4097, 64))]
        payload = adapters["baseline"].prepare(*inputs)
        saved = (payload.q_scale.clone(), payload.k_scale.clone())
        pointers = (payload.q_scale.data_ptr(), payload.k_scale.data_ptr())
        # Each flattened block also changes phase between heads/batches.
        for iteration in range(4):
            for i, target in enumerate((payload.q_scale, payload.k_scale)):
                pattern = torch.tensor((0.25, 2.0, 0.5, 1.5), dtype=torch.float32, device=device)
                ids = (torch.arange(target.numel(), device=device) + iteration + i) % 4
                target.copy_(saved[i] * pattern[ids].reshape(target.shape))
            order = ("baseline", "reuse") if iteration % 2 else ("reuse", "baseline")
            # No host synchronization between scale writes and attention launches.
            outputs = {name: adapters[name].run_native(payload) for name in order}
            error = compare_all_rows(outputs["reuse"], outputs["baseline"])
            unchanged = pointers == (payload.q_scale.data_ptr(), payload.k_scale.data_ptr())
            rows.append(dict(dtype=str(dtype), iteration=iteration, order=order,
                             pointers_unchanged=unchanged, **error))
            save()
            if not unchanged or not error["exact_match"]:
                raise AssertionError("Precomputed block gains changed padded native outputs")
            del outputs
