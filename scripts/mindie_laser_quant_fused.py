"""Load the separately built Ascend C quantizer, with no implicit fallback.

Manifest inspection deliberately has no torch import so the parent benchmark
can check build evidence before starting isolated NPU workers.
"""
import hashlib
import json
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "ascend/laser_quant_preprocess"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preprocess_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    build = json.loads(manifest_path.read_text(encoding="utf-8"))
    if build.get("status") != "passed" or build.get("contract_version") != 1:
        raise ValueError("Fused preprocessing needs a passed contract-v1 build manifest")
    library = Path(build["library_path"]).resolve()
    if not library.is_file() or _sha256(library) != build.get("library_sha256"):
        raise ValueError("Fused preprocessing library SHA256 mismatch")
    if not build.get("soc"):
        raise ValueError("Fused preprocessing manifest lacks the exact compiled SoC")
    records = build.get("source_files")
    if not isinstance(records, list) or not records:
        raise ValueError("Fused preprocessing manifest lacks source hashes")
    expected_paths = {path.resolve() for path in SOURCE_ROOT.rglob("*")
                      if path.name == "CMakeLists.txt" or path.suffix in (".cpp", ".h", ".asc")}
    recorded_paths = set()
    for record in records:
        path = Path(record["path"]).resolve()
        if path in recorded_paths or path not in expected_paths:
            raise ValueError(f"Unexpected/duplicate fused preprocessing source: {path}")
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"Fused preprocessing source changed; rebuild: {path}")
        recorded_paths.add(path)
    if not expected_paths or recorded_paths != expected_paths:
        raise ValueError("Fused preprocessing source manifest is incomplete")
    return {"build_manifest": str(manifest_path), "build_manifest_sha256": _sha256(manifest_path),
            "library_path": str(library), "library_sha256": build["library_sha256"],
            "source_files": records, "soc": build["soc"], "contract_version": 1}


def load_fused_quantizer(manifest_path):
    import torch
    import torch_npu

    evidence = validate_preprocess_manifest(manifest_path)
    torch.ops.load_library(evidence["library_path"])
    native = torch.ops.vc_laser_quant_preprocess.quantize.default
    schema = native._schema
    if ([str(arg.type) for arg in schema.arguments] != ["Tensor"]
            or [str(ret.type) for ret in schema.returns] != ["Tensor", "Tensor"]):
        raise RuntimeError(f"Unrecognized fused Q/K quantizer schema: {schema}")
    metadata = json.loads(torch.ops.vc_laser_quant_preprocess.metadata())
    if metadata.get("contract_version") != 1 or metadata.get("soc") != evidence["soc"]:
        raise RuntimeError("Loaded fused preprocessing metadata disagrees with the build manifest")

    def quantize(x):
        if x.device.type != "npu":
            raise ValueError("Fused Laser preprocessing requires NPU input; no CPU fallback")
        # Strided public inputs are copied explicitly and included in timing.
        prepared = torch_npu.npu_format_cast(x.contiguous(), 2)
        if prepared.data_ptr() % 32:
            prepared = prepared.clone()  # A contiguous offset view can still be unaligned.
        quant, scale = native(prepared)
        b, h, n, d = x.shape
        padded = (n + 255) // 256 * 256
        if (quant.shape != (b, h, padded, 128) or quant.dtype != torch.int8
                or scale.shape != (b, h, padded // 128) or scale.dtype != torch.float32
                or quant.device != x.device or scale.device != x.device
                or not quant.is_contiguous() or not scale.is_contiguous()):
            raise RuntimeError("Fused Laser quantizer returned invalid padded INT8/FP32 scale tensors")
        return quant, scale

    return quantize, {"qk_preprocessing": "fused_ascendc", "preprocessing_build": evidence,
                      "preprocessing_native": metadata, "preprocessing_schema": str(schema)}
