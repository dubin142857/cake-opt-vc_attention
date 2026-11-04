"""Explicit MindIE-SD Laser baseline. No automatic attention fallback.

API inspected at Ascend/MindIE-SD 71098c20ff90838d1fd1ca0a40515c751a3869b0.
Uses the installed upstream preprocessing/postprocessing, not a copied kernel.
"""
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import torch


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_npu(q):
    if q.device.type != "npu":
        raise ValueError("MindIE Laser benchmark requires NPU inputs")


@dataclass
class LaserPayload:
    tensors: tuple
    dtype: torch.dtype
    shape: tuple
    scale: float
    pre_tokens: int


@dataclass
class AttentionContract:
    """Public shape/dtype only; fused packing never materializes unsplit QKV."""
    dtype: torch.dtype
    shape: tuple
    scale: float


def validate_mha_inputs(q,k,v):
    _require_npu(q)
    if any(x.ndim != 4 for x in (q,k,v)):
        raise ValueError("Laser adapter expects HND/BNSD rank-four tensors")
    if q.dtype not in (torch.float16,torch.bfloat16):
        raise TypeError("Laser adapter supports FP16/BF16 inputs")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k,v)):
        raise ValueError("Q/K/V dtype and device must match")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Dense MHA requires matching B/H/D and K/V shapes")
    if min(*q.shape,*k.shape) <= 0 or q.shape[-1] not in (64,128):
        raise ValueError("Nonempty D64/128 inputs required")


def describe_contract(p):
    return dict(input_shape_hnd=list(p.shape),input_dtype=str(p.dtype),native_dtype=str(torch.float16),
                scale=p.scale,next_tokens=1,is_high_precision=True,causal=False,quantized=False)


class MindIELaser:
    def __init__(self, preprocess, postprocess, native, metadata):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.native = native
        self.metadata = metadata

    def prepare(self, q, k, v):
        validate_mha_inputs(q,k,v)
        tensors = tuple(self.preprocess(q, k, v))
        b, h, nq, d = q.shape
        nk = k.shape[2]
        expected = [(b, h, ((n + 255) // 256) * 256, 128) for n in (nq, nk, nk)]
        if len(tensors) != 3 or any(tuple(x.shape) != shape or x.dtype != torch.float16
                                    or x.device != q.device or not x.is_contiguous()
                                    for x, shape in zip(tensors, expected)):
            raise RuntimeError("Installed MindIE Laser preprocessing changed; expected contiguous FP16 S256/D128 padding")
        # Laser's pre_tokens is tail-padding length here, NOT CANN's causal window.
        padding = tensors[1].shape[2] - nk
        return LaserPayload(tensors, q.dtype, tuple(q.shape), 1 / math.sqrt(d),
                            padding if padding else 2147483647)

    def run_prepared(self, payload):
        q, k, v = payload.tensors
        result = self.native(q, k, v, None, None, None, payload.scale,
                             payload.shape[1], "BNSD", 1.0, payload.pre_tokens, 1, True)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("mindiesd::la must return (softmax statistics, attention output)")
        # Output is the SECOND tensor. Upstream handles BF16 round-trip and trim.
        out = self.postprocess(result[1], payload.dtype, payload.shape[2], payload.shape[3])
        if tuple(out.shape) != payload.shape or out.dtype != payload.dtype or out.device != q.device:
            raise RuntimeError("MindIE Laser output shape/dtype/device differs from the requested contract")
        return out.contiguous()  # same contiguous HND output contract as VC

    def __call__(self, q, k, v):
        return self.run_prepared(self.prepare(q, k, v))

    def describe_payload(self, payload):
        return {"input_shape_hnd": list(payload.shape), "input_dtype": str(payload.dtype),
                "native_shapes": [list(x.shape) for x in payload.tensors],
                "native_dtype": str(payload.tensors[0].dtype), "scale": payload.scale,
                "pre_tokens": payload.pre_tokens, "next_tokens": 1,
                "is_high_precision": True, "causal": False, "quantized": False,
                "fixed_core_scope": "preconverted/padded QKV; includes allocation, native dispatch, output cast/trim/contiguous"}


@dataclass
class SplitKVPayload:
    tensors: tuple
    original: LaserPayload | AttentionContract
    splits: int
    pre_tokens: int
    extra_zero_keys: int
    raw_stats: bool = False
    masked_tail_keys: int = 0
    duplicated_single_key: bool = False


def merge_split_kv(lse, outputs, extra_zero_keys=0):
    """Stable normalized partition merge; discount unmasked zero K/V padding."""
    shift = lse.amax(dim=0)
    weights = (lse - shift).exp()
    denominator = weights.sum(dim=0)
    if extra_zero_keys:
        denominator = denominator - extra_zero_keys * (-shift).exp()
    return (outputs * weights.unsqueeze(-1)).sum(dim=0) / denominator.unsqueeze(-1)


def merge_split_kv_raw(stats, outputs, extra_zero_keys=0):
    """Merge unnormalized numerators with per-head planar max/FP32 rowsum."""
    nq = outputs.shape[-2]
    if stats.shape[:-1] != outputs.shape[:-2] or stats.shape[-1] != 2 * nq:
        raise ValueError("Raw stats must have two planar query extents")
    maxima, sums = stats[..., :nq], stats[..., nq:]
    shift = maxima.amax(dim=0)
    weights = (maxima - shift).exp()
    denominator = (weights * sums).sum(dim=0)
    if extra_zero_keys:
        denominator = denominator - extra_zero_keys * (-shift).exp()
    return (outputs * weights.unsqueeze(-1)).sum(dim=0) / denominator.unsqueeze(-1)


def encode_partition_padding(common_pad, extra_tail):
    """Internal raw_masked control, consumed before max/exp on device."""
    if type(common_pad) is not int or not 0 <= common_pad <= 255 or extra_tail not in (0,1):
        raise ValueError("Invalid common padding or tail flag")
    encoded = common_pad + 256 * extra_tail
    return encoded if encoded else 2147483647


class SplitKVLaser:
    """Explicit noncausal scheduling candidate requiring the audited LSE variant.

    Native outputs stay FP32 until merge. This changes reduction ordering.
    Preparation duplicates Q and partitions KV; its cost is measured separately.
    """
    def __init__(self, adapter, splits, *, fused_merge=None, raw_stats=False, masked_padding=False, fused_pack=None):
        if splits < 1:
            raise ValueError("Positive split count required")
        self.adapter, self.splits = adapter, splits
        self.fused_merge = fused_merge
        self.raw_stats = bool(raw_stats)
        self.masked_padding = bool(masked_padding)
        self.fused_pack = fused_pack
        if self.masked_padding and (not raw_stats or splits != 2):
            raise ValueError("Per-partition masking requires raw split2")
        if fused_pack is not None and not self.masked_padding:
            raise ValueError("Fused packing requires raw_masked split2")
        self.expected_device_index = None
        self.metadata = dict(adapter.metadata, scheduling="split-KV", splits=splits,
                             requires_source_variant="raw_masked" if masked_padding else "raw_stats" if raw_stats else "lse",
                             padding="per-partition pre-softmax" if masked_padding else "common padding + dummy correction",
                             packing="fused_ascendc" if fused_pack is not None else "torch",
                             merge="unnormalized max/rowsum" if raw_stats else "FP32 LSE weighted")

    def prepare(self, q, k, v):
        if self.expected_device_index is not None:
            if any(x.device.type != "npu" or x.device.index != self.expected_device_index for x in (q,k,v)):
                raise ValueError("Inputs must use the configured NPU device")
            if any(x.requires_grad for x in (q,k,v)):
                raise ValueError("Raw-stats adapter is inference-only")
        if self.fused_pack is not None:
            validate_mha_inputs(q,k,v)
            tensors = self.fused_pack(q,k,v)
            nk = k.shape[2]
            chunk = (nk+1)//2
            common = tensors[1].shape[2] - chunk
            contract = AttentionContract(q.dtype,tuple(q.shape),1/math.sqrt(q.shape[-1]))
            return SplitKVPayload(tensors,contract,2,common or 2147483647,0,True,
                                  nk%2 if nk>1 else 0,nk==1)
        p = self.adapter.prepare(q, k, v)
        b, h, nk, _ = k.shape
        duplicate_single = self.masked_padding and nk == 1
        if self.splits > nk and not duplicate_single:
            raise ValueError("Split count exceeds valid key count")
        chunk = (nk + self.splits - 1) // self.splits
        native_chunk = ((chunk + 255) // 256) * 256
        extra = 0 if duplicate_single else chunk * self.splits - nk
        parts = []
        for tensor in p.tensors[1:]:
            valid = tensor[:, :, :nk]
            if duplicate_single:
                partitioned = valid.repeat(self.splits,1,1,1)
            else:
                padded = torch.nn.functional.pad(valid, (0, 0, 0, extra))
                partitioned = padded.reshape(b, h, self.splits, chunk, 128).permute(2, 0, 1, 3, 4)
                partitioned = partitioned.reshape(self.splits * b, h, chunk, 128)
            parts.append(torch.nn.functional.pad(partitioned, (0, 0, 0, native_chunk - chunk)).contiguous())
        repeated_q = p.tensors[0].repeat(self.splits, 1, 1, 1)
        padding = native_chunk - chunk
        return SplitKVPayload((repeated_q, *parts), p, self.splits,
                              padding if padding else 2147483647, 0 if self.masked_padding else extra,
                              self.raw_stats, extra if self.masked_padding else 0, duplicate_single)

    def run_native(self, payload):
        if self.expected_device_index is not None and payload.tensors[0].device.index != self.expected_device_index:
            raise ValueError("Prepared payload uses a different NPU device")
        p = payload.original
        tokens = payload.pre_tokens
        if self.masked_padding:
            common = 0 if tokens == 2147483647 else tokens
            tokens = encode_partition_padding(common,payload.masked_tail_keys)
        return self.adapter.native(*payload.tensors, None, None, None, p.scale,
                                   p.shape[1], "BNSD", 1.0, tokens, 1, True)

    def run_prepared(self, payload):
        return self.merge_output(self.run_native(payload), payload)

    def merge_output(self, result, payload):
        if self.fused_merge is not None:
            return self.fused_merge(result, payload)
        lse, raw = result
        b,h = payload.original.shape[:2]
        nq = payload.tensors[0].shape[2]
        raw = raw.reshape(payload.splits, b, h, nq, 128)
        if payload.raw_stats:
            stats = lse.reshape(payload.splits, b, h, 2*nq)
            out = merge_split_kv_raw(stats, raw, payload.extra_zero_keys)
        else:
            lse = lse.reshape(payload.splits, b, h, nq)
            out = merge_split_kv(lse, raw, payload.extra_zero_keys)
        p = payload.original
        return self.adapter.postprocess(out, p.dtype, p.shape[2], p.shape[3]).contiguous()

    def __call__(self, q, k, v):
        return self.run_prepared(self.prepare(q, k, v))

    def describe_payload(self, payload):
        common = 0 if payload.pre_tokens == 2147483647 else payload.pre_tokens
        native_tokens = (encode_partition_padding(common,payload.masked_tail_keys)
                         if self.masked_padding else payload.pre_tokens)
        return dict(describe_contract(payload.original), splits=payload.splits,
                    native_shapes=[list(t.shape) for t in payload.tensors],
                    pre_tokens=native_tokens, common_padding=common,
                    extra_unmasked_zero_keys=payload.extra_zero_keys,
                    raw_stats=payload.raw_stats,masked_tail_keys=payload.masked_tail_keys,
                    duplicated_single_key=payload.duplicated_single_key,
                    fixed_core_scope="native partition API + FP32 merge + final output postprocess; no preparation")



def load_local_raw_stats(config):
    """Load the audited eager candidate in an already configured process."""
    import os,json
    from types import SimpleNamespace
    import benchmark_mindie_laser_local as baseline
    from benchmark_mindie_laser_quant import physical_device
    from mindie_laser_quant_fused import load_fused_merger, load_fused_packer
    device = config["device"]
    if physical_device(device) == 0:
        raise ValueError("Physical NPU 0 is reserved for administrator CI")
    provider = SimpleNamespace(**config["provider"])
    paths = [p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p]
    if len(paths) != 1 or Path(paths[0]).resolve() != Path(provider.vendor_root).resolve():
        raise ValueError("Configure the single raw_stats vendor before runtime import")
    variant = "raw_stats" if config.get("allow_legacy_unsafe",False) else "raw_masked"
    evidence = baseline.provenance(provider, expected_variant=variant)
    build = evidence["build_manifest"]
    merge_build = json.loads(Path(config["merge_manifest"]).read_text())
    cann = build.get("cann_root") or build.get("definitions",{}).get("CUSTOM_ASCEND_CANN_PACKAGE_PATH")
    if not cann or Path(os.environ.get("ASCEND_HOME_PATH", "")).resolve() != Path(cann).resolve() or Path(merge_build.get("cann_root", "")).resolve() != Path(cann).resolve():
        raise ValueError("Runtime/attention/merge CANN roots disagree")
    import torch_npu
    torch.npu.set_device(device)
    base, transforms = baseline.load_adapter(provider,evidence["source_manifest"])
    merger, merge_evidence = load_fused_merger(config["merge_manifest"])
    packer,pack_evidence = None,None
    if config.get("pack_manifest"):
        pack_build = json.loads(Path(config['pack_manifest']).read_text())
        if Path(pack_build.get('cann_root','')).resolve() != Path(cann).resolve():
            raise ValueError("Pack/attention CANN roots disagree")
        packer,pack_evidence = load_fused_packer(config['pack_manifest'])
    adapter = SplitKVLaser(base,2,fused_merge=merger,raw_stats=True,masked_padding=variant == "raw_masked",fused_pack=packer)
    adapter.expected_device_index = device
    adapter.metadata.update(source_transform_validation=transforms,merge_build=merge_evidence,
        pack_build=pack_evidence,
        source_manifest_sha256=evidence["source_manifest_sha256"],public_loader=True,
        logical_device=device,physical_device=physical_device(device))
    return adapter


def load_mindie_laser():
    """Import before first NPU workload so MindIE can set its custom OPP paths."""
    try:
        package = importlib.import_module("mindiesd")
        module = importlib.import_module("mindiesd.layers.flash_attn.ascend_laser_attention")
        cls = module.AscendLaserAttention
        native = torch.ops.mindiesd.la.default
        schema = native._schema
        expected = ["query", "key", "value", "atten_mask", "alibi_mask", "drop_mask",
                    "scale_value", "head_num", "input_layout", "keep_prob", "pre_tokens", "next_tokens"]
        names = [a.name for a in schema.arguments]
        if len(names) != 13 or names[:12] != expected or names[-1] not in ("is_highPrecision", "is_high_precision") or len(schema.returns) != 2:
            raise RuntimeError(f"Unrecognized mindiesd::la schema: {schema}")
        source = Path(module.__file__).resolve()
        metadata = {"package": "mindiesd", "version": distribution_version("mindiesd") or getattr(package, "__version__", "unknown"),
                    "package_path": str(Path(package.__file__).resolve()), "wrapper_source": str(source),
                    "wrapper_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "native_schema": str(schema), "dispatch": "torch.ops.mindiesd.la.default",
                    "auto_dispatch": False, "public_attention_forward": False,
                    "upstream_repository": "https://github.com/Ascend/MindIE-SD",
                    "inspected_commit": "71098c20ff90838d1fd1ca0a40515c751a3869b0",
                    "loaded_libraries": sorted(str(x) for x in torch.ops.loaded_libraries if "mindie" in str(x).lower() or "PTAExtension" in str(x)),
                    "full_scope": "installed Laser preprocess + explicit la + installed postprocess + contiguous output"}
        return MindIELaser(cls.la_preprocess_input, cls.la_postprocess_output, native, metadata)
    except Exception as exc:
        raise RuntimeError("MindIE Laser unavailable. Install MindIE-SD (Python package mindiesd) with its LaserAttention native ops; "
                           "MindIE-LLM alone is insufficient. Match the target CANN/torch_npu versions. "
                           f"Original error: {type(exc).__name__}: {exc}") from exc


@torch.no_grad()
def validate_laser(adapter, device, dense_reference, error):
    """Real-NPU tail/scale/dtype checks; all keys, sampled queries, outside timing."""
    results = []
    generator = torch.Generator().manual_seed(91024)
    for dtype in (torch.float16, torch.bfloat16):
        for nq, nk, d in ((256, 256, 128), (257, 385, 64)):
            original = [torch.randn((1, 2, n, d), generator=generator).to(dtype) for n in (nq, nk, nk)]
            original[2][..., 0] = 0
            original[2][..., 1] = 2
            q, k, v = [x.to(device) for x in original]
            print(f"Validate MindIE Laser: {dtype}, Q={nq}, K={nk}, D={d}", flush=True)
            prepared = adapter.prepare(q, k, v)
            out = adapter.run_prepared(prepared).cpu()
            full = adapter(q, k, v).cpu()
            consistency = error(out, full)
            if consistency["relative_rmse"] > .001:
                raise AssertionError(f"Laser full/prepared inconsistency: {consistency}")
            ids = torch.tensor([0, nq // 2, nq - 1])
            dense = dense_reference(*original, query_indices=ids)
            metric = error(out.index_select(2, ids), dense)
            if metric["relative_rmse"] > .01:
                raise AssertionError(f"Laser dense-reference implementation check failed: {metric}")
            torch.testing.assert_close(out[..., 0], torch.zeros_like(out[..., 0]), rtol=0, atol=0)
            torch.testing.assert_close(out[..., 1], torch.full_like(out[..., 1], 2), rtol=.004, atol=.003)
            results.append({**adapter.describe_payload(prepared), "dense_fp32": metric, "full_vs_prepared": consistency})
    return results
