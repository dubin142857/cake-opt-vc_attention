"""CPU diagnostics and structured implementation checks for quantized Laser.

The general decoded reference uses a global score maximum and is a diagnostic,
not an exact emulator of Laser's online softmax. Structured test inputs put a
zero maximum in every native key tile, eliminating online-history ambiguity.
"""
import dataclasses

import torch

from mindie_laser_expcast_reference import expcast_probability
from mindie_laser_quant_backend import QK_BLOCK
from mindie_laser_quant_preprocessing import POLICY, check_preprocessing


def cpu_payload(payload):
    return dataclasses.replace(payload, **{
        field.name: getattr(payload, field.name).detach().cpu()
        for field in dataclasses.fields(payload)
        if isinstance(getattr(payload, field.name), torch.Tensor)})


@torch.no_grad()
def decoded_reference(payload, query_indices=None, *, expcast=False, chunk=4, structured=False):
    """INT8 operands, per-block gains, FP16 scores/PV and original V scales.

    Native accumulation/reduction rounding is not emulated. Structured mode
    verifies a zero maximum in every 128-key block before using its oracle.
    """
    if payload.q.device.type != "cpu":
        raise ValueError("The explicit decoded Laser oracle is CPU-only")
    ids = (torch.arange(payload.shape[2]) if query_indices is None else query_indices.cpu())
    results = []
    for head in range(payload.q.shape[1]):
        key = payload.k[:, head, :payload.keys].float().transpose(-1, -2)
        value = payload.v[:, head, :payload.keys].float()
        key_ids = torch.arange(payload.keys) // QK_BLOCK
        key_scale = payload.k_scale[:, head].index_select(1, key_ids)
        rows = []
        for selected in ids.split(chunk):
            query = payload.q[:, head].index_select(1, selected).float()
            query_scale = payload.q_scale[:, head].index_select(1, selected // QK_BLOCK)
            dots = query @ key  # Integer dot products fit exactly in FP32.
            # DEQF16 FIX accepts the FP32 dequant gain's bits 31:13. It
            # truncates the low 13 mantissa bits before INT32 -> FP16 QK.
            gain = (query_scale[:, :, None] * key_scale[:, None, :]).contiguous()
            gain = (gain.view(torch.int32) & -8192).view(torch.float32)
            scores = (dots * gain).half()
            attention_scale = torch.tensor(payload.scale, dtype=torch.float16)
            scores = (scores.float() * attention_scale.float()).half()
            if structured:
                for begin in range(0, payload.keys, QK_BLOCK):
                    maximum = scores[..., begin:begin + QK_BLOCK].amax(-1)
                    if not torch.equal(maximum, torch.zeros_like(maximum)):
                        raise ValueError("Structured oracle needs zero maximum in every native key tile")
            shifted = (scores.float() - scores.amax(-1, keepdim=True).float()).half()
            if expcast:
                probability = expcast_probability(shifted).float()
                mass = probability.sum(-1, keepdim=True)
            else:
                unrounded = shifted.float().exp()
                probability = unrounded.half().float()
                # Laser process_calc_sum reads the same FP16 probability
                # buffer consumed by PV, then accumulates its mass in FP32.
                mass = probability.sum(-1, keepdim=True)
            out = (probability @ value) / mass
            out *= payload.v_scale[:, head, None, :]
            rows.append(out[..., :payload.shape[3]])
        results.append(torch.cat(rows, dim=1))
    return torch.stack(results, dim=1).half().to(payload.dtype)


def structured_case(dtype, dim, queries, keys, *, zero=False):
    """Vary quantization scales across query/key blocks, heads, and V channels."""
    q = torch.zeros((1, 2, queries, dim), dtype=dtype)
    k = torch.zeros((1, 2, keys, dim), dtype=dtype)
    v = torch.zeros_like(k)
    if zero:
        return q, k, v
    query_ids, key_ids = torch.arange(queries), torch.arange(keys)
    q_amplitudes = torch.tensor([.5, 1, 2], dtype=dtype)[query_ids % 3]
    q_blocks = torch.tensor([.25, 1, 4], dtype=dtype)[(query_ids // 128) % 3]
    k_blocks = torch.tensor([.25, 1, 4], dtype=dtype)[(key_ids // 128) % 3]
    features = [feature for feature in (0, 3, 31, 32, 63, 64, 95, 96, 127) if feature < dim]
    for index, feature in enumerate(features):
        q_feature = (index + 1 + (query_ids * 3 + index) % 5).to(dtype)
        k_feature = (len(features) - index + (key_ids * 7 + index * 3) % 11).to(dtype)
        q[..., feature] = q_amplitudes * q_blocks * q_feature / len(features)
        k[..., feature] = -(key_ids % 128).to(dtype) * k_blocks * k_feature / (128 * len(features))
    q[:, 1] *= .5
    k[:, 1] *= 2
    # Distinct variable channels prevent a constant-V-only test from missing
    # an incorrect V scale or an unchanged probability algorithm.
    values = ((key_ids[:, None] * 13 + torch.arange(dim)[None, :] * 7) % 29 - 14).float() / 14
    channel_scales = torch.pow(2.0, (torch.arange(dim) % 9).float() - 4)
    v[:] = (values * channel_scales).to(dtype)
    v[:, 1] *= .5
    v[..., 0], v[..., 1] = 0, 2
    return q, k, v


def validate_fused_reuse(adapter, device, report, save):
    """Exercise direct-launch stream ordering, reused inputs and strided D64."""
    if adapter.metadata.get("qk_preprocessing") != "fused_ascendc":
        return
    records = report["fused_preprocessing_reuse"] = []
    generator = torch.Generator().manual_seed(91128)
    # Six heads cross scale-group boundaries; unequal Q/K lengths include
    # full tiles, partial tiles and wholly zero physical padding blocks.
    for dtype in (torch.float16, torch.bfloat16):
        storage = [torch.empty((2, 3, n, 128), dtype=dtype, device=device)
                   for n in (137, 259, 259)]
        inputs = [x[..., ::2] for x in storage]
        for iteration, amplitude in enumerate((.25, 4., 0., 1.)):
            original = [torch.randn(x.shape, generator=generator).mul_(amplitude).to(dtype)
                        for x in inputs]
            for destination, source in zip(inputs, original):
                destination.copy_(source)  # Ordered NPU writes, no manual synchronize.
            payload = adapter.prepare(*inputs)
            comparison = check_preprocessing(original, cpu_payload(payload))
            row = {"dtype": str(dtype), "iteration": iteration, "amplitude": amplitude,
                   "inputs_reused": True, "input_strided": True, "preprocessing": comparison}
            records.append(row)
            save()
            if not comparison["passed"]:
                raise AssertionError(f"Fused preprocessing reuse/stride check failed: {comparison['errors']}")
            del payload


def validate_small(adapter, device, report, save, torch_module):
    """Native checks: zero, tails, block-scale variation, and section boundary."""
    import benchmark_cann_fas_local as common
    shapes = ((17, 129, 64), (256, 256, 128), (257, 385, 128), (17, 4097, 64), (1, 1, 128))
    report["small_accuracy"] = []
    report["small_accuracy_contract"] = {
        "decoded_implementation_gate": "Every 128-key tile has zero maximum; all rows/channels except independent constant channels",
        "decoded_limit": .01, "full_prepared_limit": .001,
        "dense_scope": "Original unquantized FP32 attention diagnostic, separate from implementation correctness",
        "native_fp8": False, "k_smoothing": False, "v_reordering": False,
        "preprocessing": dict(POLICY)}
    report["preprocessing_rounding_probes"] = []
    save()
    for dtype in (torch.float16, torch.bfloat16):
        # Absmax=127 gives an exact scale of 1. This independently checks
        # round-to-nearest-even at positive/negative ties on the actual device.
        ties = torch.zeros(1, 1, 1, 64, dtype=dtype)
        ties[..., :7] = torch.tensor([127, .5, 1.5, 2.5, -.5, -1.5, -2.5], dtype=dtype)
        probe_inputs = (ties, ties.clone(), torch.zeros_like(ties))
        probe = cpu_payload(adapter.prepare(*(x.to(device) for x in probe_inputs)))
        probe_check = check_preprocessing(probe_inputs, probe)
        probe_check["exact_unit_qk_scales"] = all(
            torch.equal(x, torch.ones_like(x)) for x in (probe.q_scale, probe.k_scale))
        if not probe_check["exact_unit_qk_scales"]:
            probe_check["passed"] = False
            probe_check["errors"].append("The absmax=127 RNE probe requires exact Q/K scale=1")
        report["preprocessing_rounding_probes"].append({"dtype": str(dtype), **probe_check})
        save()
        if not probe_check["passed"]:
            raise AssertionError(f"Exact-scale RNE probe failed for {dtype}: {probe_check['errors']}")
        for nq, nk, dim in shapes:
            for zero in ((False, True) if (nq, nk) == (17, 129) else (False,)):
                label = f"{dtype} Q{nq} K{nk} D{dim} zero={zero}"
                print("Quantized Laser structured validation: " + label, flush=True)
                row = {"case": label, "status": "running"}
                report["small_accuracy"].append(row)
                save()
                original = structured_case(dtype, dim, nq, nk, zero=zero)
                inputs = [tensor.to(device) for tensor in original]
                payload = adapter.prepare(*inputs)
                host_payload = cpu_payload(payload)
                # Validate backend preprocessing separately from native output.
                row["payload"] = adapter.describe_payload(payload)
                preprocessing = check_preprocessing(original, host_payload)
                row["preprocessing_cpu_comparison"] = preprocessing
                if not preprocessing["passed"]:
                    row["status"] = "failed"
                save()
                if not preprocessing["passed"]:
                    raise AssertionError(f"{label}: preprocessing failed: {preprocessing['errors']}")
                for name in ("q", "k"):
                    count = preprocessing["fields"][name]["accepted_boundary_differences"]
                    if count:
                        print(f"  {name}: {count} INT8 rounding-boundary differences accepted "
                              "within 1 FP32 ULP; details saved", flush=True)
                prepared = adapter.run_prepared(payload).cpu()
                full = adapter(*inputs).cpu()
                reference = decoded_reference(host_payload, expcast=adapter.expcast, structured=True)
                dense = common.dense_reference(*original, torch.arange(nq))
                for name, actual, target, limit in (
                    ("full_vs_prepared", full, prepared, .001),
                    ("decoded_reference_nonconstant_channels", full[..., 2:], reference[..., 2:], .01),
                ):
                    row[name] = common.check_error(actual, target, limit)
                    save()
                    if not row[name]["passed"]:
                        raise AssertionError(f"{label}: {name} failed: {row[name]}")
                row["dense_fp32_all_rows_all_keys"] = common.error_metrics(full, dense)
                torch.testing.assert_close(full[..., 0], torch.zeros_like(full[..., 0]), rtol=0, atol=0)
                constant = 0 if zero else 2
                torch.testing.assert_close(full[..., 1], torch.full_like(full[..., 1], constant), rtol=.004, atol=.003)
                if adapter.expcast and nk > 1 and not zero:
                    ordinary = decoded_reference(host_payload, expcast=False, structured=True)
                    row["ordinary_vs_expcast_oracle"] = common.error_metrics(ordinary[..., 2:], reference[..., 2:])
                    if row["ordinary_vs_expcast_oracle"]["relative_rmse"] <= .01:
                        raise AssertionError("Structured case cannot distinguish unchanged softmax from ExpCast")
                row["constant_value_padding_check"], row["status"] = "passed", "passed"
                save()
    validate_fused_reuse(adapter, device, report, save)
    return report["small_accuracy"]
