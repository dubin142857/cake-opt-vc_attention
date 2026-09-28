"""Verify vendored files against their pinned upstream/local source hashes."""
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "third_party/sageattention_local"
    manifest = json.loads((root / "UPSTREAM.json").read_text(encoding="utf-8"))
    native = 0
    build_patches = []
    for row in manifest["files"]:
        data = (root / row["local_path"]).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != row["local_sha256"]:
            raise RuntimeError(f"Source hash mismatch: {row['local_path']}")
        if row["upstream_path"].startswith(("csrc/", "sageattention/triton/")):
            if row["local_sha256"] != row["upstream_sha256"]:
                # Only this exact include removal is allowed; reversing it must
                # recover the pinned upstream bytes, including the kernel body.
                marker = b"// Modified for vc-attention: omit unused CUDAContext.h (avoids cuSPARSE headers)."
                if row["upstream_path"] != "csrc/fused/fused.cu" or data.count(marker) != 1:
                    raise RuntimeError(f"Upstream kernel changed: {row['local_path']}")
                restored = data.replace(marker, b"#include <ATen/cuda/CUDAContext.h>")
                if hashlib.sha256(restored).hexdigest() != row["upstream_sha256"]:
                    raise RuntimeError(f"Kernel body differs from upstream: {row['local_path']}")
                build_patches.append(row["local_path"])
            else:
                native += 1
    packaged = json.loads((root / "sageattention_local/UPSTREAM.json").read_text(encoding="utf-8"))
    if packaged != manifest:
        raise RuntimeError("Packaged provenance differs from source provenance")
    return {"status": "passed", "commit": manifest["commit"], "verified_files": len(manifest["files"]),
            "unchanged_kernel_files": native, "build_only_patches": build_patches}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
