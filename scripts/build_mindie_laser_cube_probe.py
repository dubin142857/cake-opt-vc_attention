#!/usr/bin/env python3
"""Build one isolated original-FP16 or INT8 Cube diagnostic provider, locally only."""
from __future__ import annotations

import argparse
from functools import partial
import importlib
from pathlib import Path
import traceback
import sys

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SOURCE = ROOT / "ascend/laser_cube_probe/bridge"
SCHEMAS = {
    "int8": "la(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, float scale, int keys, bool expcast, int probe_mode) -> (Tensor, Tensor)",
    "vscale": "la(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, float scale, int keys, bool expcast, int probe_mode) -> (Tensor, Tensor)",
    "pv": "la(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, float scale, int keys, bool expcast, int probe_mode) -> (Tensor, Tensor)",
    "fp16": "la(Tensor q, Tensor k, Tensor v, float scale, int keys, int probe_mode) -> (Tensor, Tensor)",
}


def _module(name):
    return importlib.import_module(("scripts." if __package__ else "") + name)


def identity(kind):
    return _module("prepare_mindie_laser_cube_probe_source").identity(kind)


def marker_token(kind):
    return _module("prepare_mindie_laser_cube_probe_source").marker_token(kind)


def marker_environment(kind):
    return _module("prepare_mindie_laser_cube_probe_source").marker_environment(kind)


def bridge_name(kind):
    return f'lib{identity(kind)["vendor"]}_bridge.so'


def build_bridge(source, cann, bridge_dir, jobs, output, *, kind="int8", runner=None, runtime=None):
    native = identity(kind)
    base = _module("build_mindie_laser_quant")
    # Preserve the existing int8/fp16 bridge source hashes and build cache.
    bridge_source = BRIDGE_SOURCE / kind if kind in ("vscale", "pv") else BRIDGE_SOURCE
    report = base.build_bridge(source, cann, bridge_dir, jobs, output, runner=runner, runtime=runtime,
                              identity=native, torch_namespace=native["vendor"], bridge_source=bridge_source,
                              bridge_name=bridge_name(kind))
    report.update(torch_schema=SCHEMAS[kind], probe_kind=kind)
    base.write_json(output, report)
    return report


def build(source, reference, source_manifest, cann, soc, jobs, install_dir, bridge_dir, output,
          *, kind="int8", preparer=None, source_validator=None, operator_builder=None,
          bridge_builder=None, runner=None):
    preparation = _module("prepare_mindie_laser_cube_probe_source")
    native = identity(kind)
    return _module("build_mindie_laser_quant").build(
        source, reference, source_manifest, cann, soc, jobs, install_dir, bridge_dir, output,
        preparer=partial(preparation.prepare, kind=kind) if preparer is None else preparer,
        source_validator=partial(preparation.validate_prepared_source, kind=kind) if source_validator is None else source_validator,
        operator_builder=operator_builder,
        bridge_builder=partial(build_bridge, kind=kind) if bridge_builder is None else bridge_builder,
        runner=runner, identity=native, marker_token=marker_token(kind), marker_environment=marker_environment(kind))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--kind", choices=("int8", "fp16", "vscale", "pv"), required=True)
    result.add_argument("--cann-root", type=Path, required=True)
    result.add_argument("--soc", required=True)
    result.add_argument("--jobs", type=int, default=4)
    result.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    for option in ("source-dir", "source-manifest", "install-dir", "bridge-dir", "output"):
        result.add_argument("--" + option, type=Path)
    return result


def resolve_defaults(args):
    prefix = f"mindie_laser_cube_{args.kind}_probe"
    values = {"source_dir": ROOT / "build" / (prefix + "_source"),
              "source_manifest": ROOT / "build" / (prefix + "_source_manifest.json"),
              "install_dir": ROOT / "build" / (prefix + "_install"),
              "bridge_dir": ROOT / "build" / (prefix + "_bridge"),
              "output": ROOT / "results" / (prefix + "_build.json")}
    for key, value in values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None):
    args = resolve_defaults(parser().parse_args(argv))
    try:
        report = build(args.source_dir, args.reference_dir, args.source_manifest, args.cann_root,
                       args.soc, args.jobs, args.install_dir, args.bridge_dir, args.output, kind=args.kind)
    except Exception:
        traceback.print_exc()
        print(f"Failure evidence: {args.output.resolve()}", file=sys.stderr, flush=True)
        return 1
    print(f"Cube {args.kind} probe vendor: {report['vendor_root']}", flush=True)
    print(f"Cube {args.kind} probe bridge: {report['bridge_path']}", flush=True)
    print(f"Manifest: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
