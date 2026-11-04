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


def validate_preprocess_manifest(manifest_path, *, operation="quantize"):
    manifest_path = Path(manifest_path).resolve()
    build = json.loads(manifest_path.read_text(encoding="utf-8"))
    if operation not in ("quantize", "merge", "pack", "global", "global_probe", "global_half_sum", "global_pipe", "global_q64", "global_q128", "global_stream", "global_stream_q64", "global_stream_onepass", "global_stream_overlap", "sparse_pv", "sparse_candidates", "sparse_candidates_lookup", "sparse_candidates_dma_overlap", "sparse_candidates_block1", "sparse_candidates_read_ack", "sparse_candidates_bulk_output", "sparse_candidates_score_chunk", "sparse_candidates_score_chunk32", "sparse_finalize", "sparse_delay", "sparse_pair", "sparse_fma_float", "sparse_fma_half", "sparse_postmax_scale", "max_scale_probe", "sort32_probe", "sparse_online_top8", "sparse_window_top8", "sparse_inline_pv") or build.get("operation", "quantize") != operation:
        raise ValueError("Direct-launch build operation mismatch")
    if build.get("status") != "passed" or build.get("contract_version") != (6 if operation=="global_stream_overlap" else 5 if operation in ("global_stream_q64","global_stream_onepass") else 4 if operation=="global_stream" else 3 if operation in ("global_q64","global_q128") else 2 if operation in ("merge","global_half_sum","global_pipe") else 1):
        raise ValueError("Direct-launch build needs a passed matching-contract manifest")
    source_root = SOURCE_ROOT if operation == "quantize" else Path(__file__).resolve().parents[1] / ("ascend/laser_" + operation)
    if operation == "global_probe":
        try:
            from .prepare_mindie_laser_global_probe_source import validate
        except ImportError:
            from prepare_mindie_laser_global_probe_source import validate
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"))
    if operation == "global_half_sum":
        try:
            from .prepare_mindie_laser_global_half_sum_source import validate
        except ImportError:
            from prepare_mindie_laser_global_half_sum_source import validate
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"))
    if operation == "global_pipe":
        try:
            from .prepare_mindie_laser_global_pipe_source import validate
        except ImportError:
            from prepare_mindie_laser_global_pipe_source import validate
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"))
    if operation in ("global_q64", "global_q128"):
        try:
            from .prepare_mindie_laser_global_geometry_source import validate,geometry
        except ImportError:
            from prepare_mindie_laser_global_geometry_source import validate,geometry
        rows=int(operation.removeprefix("global_q"))
        if build.get("geometry")!=geometry(rows):raise ValueError("Q-geometry manifest mismatch")
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"),rows)
    if operation == "global_stream":
        try:
            from .prepare_mindie_laser_global_stream_source import validate
        except ImportError:
            from prepare_mindie_laser_global_stream_source import validate
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"))
    if operation in ("global_stream_q64", "global_stream_onepass"):
        try:
            from .prepare_mindie_laser_stream_onepass_source import validate,geometry
        except ImportError:
            from prepare_mindie_laser_stream_onepass_source import validate,geometry
        if build.get("geometry")!=geometry(operation):raise ValueError("One-pass geometry mismatch")
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"),operation)
    if operation == "global_stream_overlap":
        try:
            from .prepare_mindie_laser_stream_overlap_source import validate
        except ImportError:
            from prepare_mindie_laser_stream_overlap_source import validate
        source_root = Path(build["definitions"]["CMAKE_HOME_DIRECTORY"]).resolve()
        validate(source_root,build.get("generator_inputs"))
    library = Path(build["library_path"]).resolve()
    if not library.is_file() or _sha256(library) != build.get("library_sha256"):
        raise ValueError("Fused preprocessing library SHA256 mismatch")
    if not build.get("soc"):
        raise ValueError("Fused preprocessing manifest lacks the exact compiled SoC")
    records = build.get("source_files")
    if not isinstance(records, list) or not records:
        raise ValueError("Fused preprocessing manifest lacks source hashes")
    expected_paths = {path.resolve() for path in source_root.rglob("*")
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
            "source_files": records, "soc": build["soc"], "contract_version": build["contract_version"]}


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


def load_fused_merger(manifest_path):
    import torch
    evidence = validate_preprocess_manifest(manifest_path, operation="merge")
    torch.ops.load_library(evidence["library_path"])
    metadata = json.loads(torch.ops.vc_laser_merge.metadata())
    if metadata.get("soc") != evidence["soc"] or metadata.get("splits") != 2 or metadata.get("contract_version") != 2 or metadata.get("operation") != "split_kv_merge":
        raise RuntimeError("Loaded merge metadata disagrees with its build")
    native = torch.ops.vc_laser_merge.merge.default
    if [str(a.type) for a in native._schema.arguments] != ["Tensor","Tensor","int","int","int","bool","bool"]:
        raise RuntimeError("Unrecognized fused merge schema")
    def merge(result, payload):
        p = payload.original
        if payload.splits != 2 or p.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Fused merge requires split2 with FP16/BF16 output")
        output = native(result[0], result[1], p.shape[2], p.shape[3], payload.extra_zero_keys,
                        p.dtype == torch.bfloat16, getattr(payload, "raw_stats", False))
        if output.shape != p.shape or output.dtype != p.dtype or not output.is_contiguous():
            raise RuntimeError("Invalid fused merge output contract")
        return output
    return merge, evidence


def load_fused_packer(manifest_path):
    import torch
    import torch_npu
    evidence = validate_preprocess_manifest(manifest_path,operation="pack")
    torch.ops.load_library(evidence["library_path"])
    native = torch.ops.vc_laser_pack.pack.default
    if ([str(a.type) for a in native._schema.arguments] != ["Tensor"]*3 or
            [str(r.type) for r in native._schema.returns] != ["Tensor"]*3):
        raise RuntimeError("Unrecognized split-KV pack schema")
    metadata = json.loads(torch.ops.vc_laser_pack.metadata())
    if metadata.get("soc") != evidence["soc"] or metadata.get("splits") != 2 or metadata.get("operation") != "split_kv_pack" or metadata.get("contract_version") != 1:
        raise RuntimeError("Loaded pack metadata disagrees with its build")
    def pack(q,k,v):
        prepared = []
        for x in (q,k,v):
            if x.device.type != "npu":
                raise ValueError("Split-KV packing requires NPU inputs; no fallback")
            x = x.contiguous()
            if torch_npu.get_npu_format(x) not in (0,2):
                x = torch_npu.npu_format_cast(x,2)
            if x.data_ptr() % 32:
                x = x.clone()
            prepared.append(x)
        outputs = native(*prepared)
        b,h,n,d = q.shape
        chunk = (k.shape[2]+1)//2
        expected = [(2*b,h,(seq+255)//256*256,128) for seq in (n,chunk,chunk)]
        if len(outputs)!=3 or any(x.shape!=shape or x.dtype!=torch.float16 or x.device!=q.device or not x.is_contiguous()
                                 for x,shape in zip(outputs,expected)):
            raise RuntimeError("Invalid fused split-KV pack output contract")
        return tuple(outputs)
    return pack,dict(evidence,metadata=metadata,schema=str(native._schema))
