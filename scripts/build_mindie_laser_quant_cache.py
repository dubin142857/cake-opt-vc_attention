#!/usr/bin/env python3
"""Build the separately named INT8-QK Laser scale-cache experiment.

The production and scale-reuse source, installation and bridge caches are
not modified. The candidate uses the same audited build machinery, with its own
source validator, operator identity, vendor directory and torch namespace.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "op_name": "vc_laser_quant_cache_attention",
    "vendor": "vc_laser_quant_cache",
    "op_type": "VcLaserQuantCacheAttention",
    "native_api": "aclnnVcLaserQuantCacheAttention",
}
MARKER_TOKEN = "vc_mindie_laser_quant_cache_v1"
MARKER_ENVIRONMENT = "VC_MINDIE_LASER_QUANT_CACHE_MARKER_FILE"
BRIDGE_SOURCE = ROOT / "ascend/laser_quant_cache/bridge"
BRIDGE_NAME = "libvc_laser_quant_cache_bridge.so"
TORCH_NAMESPACE = "vc_laser_quant_cache"


def _module(name):
    return importlib.import_module(("scripts." if __package__ else "") + name)


def build_bridge(source, cann, bridge_dir, jobs, output, *, runner=None, runtime=None):
    return _module("build_mindie_laser_quant").build_bridge(
        source, cann, bridge_dir, jobs, output, runner=runner, runtime=runtime,
        identity=IDENTITY, torch_namespace=TORCH_NAMESPACE,
        bridge_source=BRIDGE_SOURCE, bridge_name=BRIDGE_NAME)


def build(source, reference, source_manifest, cann, soc, jobs, install_dir, bridge_dir, output,
          *, preparer=None, source_validator=None, operator_builder=None, bridge_builder=None,
          runner=None):
    if preparer is None or source_validator is None:
        preparation = _module("prepare_mindie_laser_quant_cache_source")
        if preparation.IDENTITY != IDENTITY:
            raise ValueError("Scale-cache preparation and build identities differ")
        preparer = preparation.prepare if preparer is None else preparer
        source_validator = (preparation.validate_prepared_source
                            if source_validator is None else source_validator)
    return _module("build_mindie_laser_quant").build(
        source, reference, source_manifest, cann, soc, jobs, install_dir, bridge_dir, output,
        preparer=preparer, source_validator=source_validator, operator_builder=operator_builder,
        bridge_builder=build_bridge if bridge_builder is None else bridge_builder, runner=runner,
        identity=IDENTITY, marker_token=MARKER_TOKEN, marker_environment=MARKER_ENVIRONMENT)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cann-root", type=Path, required=True)
    result.add_argument("--soc", required=True)
    result.add_argument("--jobs", type=int, default=4)
    result.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_cache_source")
    result.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    result.add_argument("--source-manifest", type=Path,
                        default=ROOT / "build/mindie_laser_quant_cache_source_manifest.json")
    result.add_argument("--install-dir", type=Path, default=ROOT / "build/mindie_laser_quant_cache_install")
    result.add_argument("--bridge-dir", type=Path, default=ROOT / "build/mindie_laser_quant_cache_bridge")
    result.add_argument("--output", type=Path, default=ROOT / "results/mindie_laser_quant_cache_build.json")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        report = build(arguments.source_dir, arguments.reference_dir, arguments.source_manifest,
                       arguments.cann_root, arguments.soc, arguments.jobs, arguments.install_dir,
                       arguments.bridge_dir, arguments.output)
    except Exception:
        traceback.print_exc()
        print(f"Failure evidence: {arguments.output.resolve()}", file=sys.stderr, flush=True)
        return 1
    print(f"Scale-cache Laser vendor: {report['vendor_root']}", flush=True)
    print(f"Scale-cache Laser bridge: {report['bridge_path']}", flush=True)
    print(f"Manifest: {arguments.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
