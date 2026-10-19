"""Provider isolation and numerical checks for the independent scale-reuse A/B."""
from __future__ import annotations

import os
import json
from pathlib import Path
import sys

import benchmark_mindie_laser_quant as quant

common = quant.common


def paired_environment(base, providers, directory):
    """Start from a scrubbed environment, then allow only two audited vendors."""
    roots = [str(Path(provider.vendor_root).resolve()) for provider in providers.values()]
    if len(roots) != 2 or roots[0] == roots[1]:
        raise ValueError("Scale A/B requires two distinct vendor roots")
    env = quant.baseline.worker_environment(base, "local", roots[0], directory)
    env["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(roots)
    # The baseline scrubber adds its own op_api; replace it with the two roots.
    paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
             if p and not any(common.within(p, root) for root in roots)]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(Path(root) / "op_api/lib") for root in roots] + paths)
    for name in list(env):
        if name.startswith("VC_") and name.endswith("MARKER_FILE"):
            env.pop(name)
    for name, provider in providers.items():
        env[provider.marker_env] = str(Path(directory) / "local" / (name + "_markers.jsonl"))
    return env


def verify_provider_paths(paths, providers, *, background=()):
    """An allowlist of complete, independently verified vendor roots, not a bypass."""
    relevant = set(background)
    for name, provider in providers.items():
        other_roots = [other.vendor_root for key, other in providers.items() if key != name]
        filtered = [path for path in paths if path not in background
                    and not any(common.within(path, root) for root in other_roots)]
        relevant.update(common.verify_libraries(filtered, "custom", provider.vendor_root))
        bridge = str(Path(provider.bridge_path).resolve())
        if bridge not in paths:
            raise RuntimeError(f"{name} bridge absent from process mappings")
        relevant.add(bridge)
    return sorted(relevant)


def inspect_providers(providers, report, directory, *, cann_root, soc):
    if any(name == "mindiesd" or name.startswith("mindiesd.") for name in sys.modules):
        raise RuntimeError("Paired local worker must not import mindiesd")
    text = Path("/proc/self/maps").read_text(encoding="utf-8")
    (directory / "proc_self_maps.txt").write_text(text, encoding="utf-8")
    paths = common.mapped_libraries(text)
    background = quant.baseline.inspect_batch_invariant(paths, cann_root, soc)
    relevant = verify_provider_paths(paths, providers, background=background["libraries"])
    # Require the mapped vendor libraries to be exactly those in the audited
    # build, including content. Directory membership alone is not provenance.
    for provider in providers.values():
        build = json.loads(Path(provider.build_manifest).read_text(encoding="utf-8"))
        expected = {str(Path(item["path"]).resolve()): item["sha256"]
                    for item in build["host_libraries"]}
        expected[str(Path(provider.bridge_path).resolve())] = build["bridge_sha256"]
        for path in relevant:
            if common.within(path, provider.vendor_root) or path == str(Path(provider.bridge_path).resolve()):
                if path not in expected or common.sha256_file(path) != expected[path]:
                    raise RuntimeError("Mapped custom library differs from build manifest: " + path)
    report["mapped_library_paths"] = paths
    report["ambient_registered_provider"] = background["metadata"]
    report["loaded_library_sha256"] = {path: common.sha256_file(path) for path in relevant}


def make_calls(adapters, payload, inputs):
    """Both fixed timers receive the very same object, including scale pointers."""
    return {
        "native": {name: lambda a=adapter: a.run_native(payload) for name, adapter in adapters.items()},
        "prepared": {name: lambda a=adapter: a.run_prepared(payload) for name, adapter in adapters.items()},
        "full": {name: lambda a=adapter: a(*inputs) for name, adapter in adapters.items()},
    }


def validate_scale_reuse(adapters, device, report, save):
    """Rewrite scale storage on-device then launch immediately on the same stream."""
    import torch
    from profile_mindie_laser_quant import compare_all_rows, FIELDS
    report["scale_storage_reuse"] = []
    for dtype in (torch.float16, torch.bfloat16):
        generator = torch.Generator().manual_seed(829)
        # Multiple heads/batches/Q blocks, plus a K section boundary and tail.
        inputs = [torch.randn(shape, generator=generator).to(dtype).to(device) for shape in
                  ((2, 3, 257, 64), (2, 3, 4097, 64), (2, 3, 4097, 64))]
        fixed = adapters["baseline"].prepare(*inputs)
        pointers = {name: getattr(fixed, name).data_ptr() for name in FIELDS}
        adapters["reuse"].run_native(fixed)  # prime the candidate's scalar read cache
        for iteration, (q_gain, k_gain) in enumerate(((.25, 1.), (4., 1.), (4., .125), (0., 0.), (1., 1.))):
            fresh = adapters["baseline"].prepare(inputs[0] * q_gain, inputs[1] * k_gain, inputs[2])
            for name in FIELDS:
                getattr(fixed, name).copy_(getattr(fresh, name))
            # No host synchronization between the new scale writes and native reads.
            order = ("reuse", "baseline") if iteration % 2 == 0 else ("baseline", "reuse")
            out = {name: adapters[name].run_native(fixed) for name in order}
            error = compare_all_rows(out["reuse"], out["baseline"])
            unchanged = all(getattr(fixed, name).data_ptr() == pointers[name] for name in FIELDS)
            report["scale_storage_reuse"].append(dict(dtype=str(dtype), iteration=iteration,
                q_gain=q_gain, k_gain=k_gain, order=order, pointers_unchanged=unchanged, **error))
            save()
            if not unchanged or not error["exact_match"]:
                raise AssertionError("Scale reuse changed output after in-place payload update")
            del out, fresh
