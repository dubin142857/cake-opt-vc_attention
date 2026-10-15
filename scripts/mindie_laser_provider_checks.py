"""Narrow ownership checks for an already installed batch_invariant provider.

An existing provider is not evidence that it is a CANN built-in. This helper
only exempts the exact observed tiling library after configuration/ownership
checks; every other provider remains subject to the caller's strict checks.
"""
import hashlib
import json
from pathlib import Path
import re


_TILING_PATHS = tuple(
    Path("op_impl/ai_core/tbe/op_tiling/lib/linux") / arch / "libcust_opmaster_rt2.0.so"
    for arch in ("aarch64", "x86_64")
)


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _laser_registration(value):
    """Inspect JSON keys and values, including nested lists and spellings."""
    if isinstance(value, str):
        normalized = re.sub(r"[^a-z0-9]", "", value.lower())
        return value if "laserattention" in normalized else None
    if isinstance(value, dict):
        for key, item in value.items():
            found = _laser_registration(key) or _laser_registration(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _laser_registration(item)
            if found is not None:
                return found
    return None


def inspect_batch_invariant(paths, cann_root, soc):
    """Return eligible canonical tiling paths and their provenance metadata.

    ``cann_root`` must come from the selected build/environment manifest, not
    from an arbitrary mapped library. The caller must retain its target API,
    tiling marker and output checks and reject unmatched custom libraries.
    """
    empty = {"libraries": [], "metadata": {}}
    paths = list(paths)
    if not paths:
        return empty
    cann = Path(cann_root).resolve()
    opp = (cann / "opp").resolve()  # CANN/latest and opp symlinks are normal.
    vendors = (opp / "vendors").resolve()
    vendor = vendors / "batch_invariant"
    vendor_real = vendor.resolve()
    expected = {vendor / suffix: suffix for suffix in _TILING_PATHS}
    expected_real = {path: path.resolve() for path in expected}
    candidates = []
    for supplied in paths:
        raw = Path(supplied).absolute()
        resolved = raw.resolve()
        for location, canonical in expected_real.items():
            if resolved == canonical or raw == location:
                candidates.append((location, canonical, expected[location]))
                break
    if not candidates:
        return empty  # Do not require an unrelated provider/configuration.

    if not _within(vendors, opp):
        raise ValueError(f"batch_invariant vendors root escapes the selected OPP root: {vendors}; OPP={opp}")
    if vendor_real.parent != vendors or vendor_real.name != "batch_invariant":
        raise ValueError(f"batch_invariant vendor symlink escapes its canonical location: {vendor} -> {vendor_real}")
    libraries = []
    for location, canonical, suffix in candidates:
        if canonical != vendor_real / suffix or not _within(canonical, vendor_real):
            raise ValueError(f"batch_invariant tiling library escapes its canonical location: {location} -> {canonical}")
        if not canonical.is_file():
            raise ValueError(f"batch_invariant mapped tiling library does not exist: {canonical}")
        libraries.append(str(canonical))
    libraries = sorted(set(libraries))
    if soc not in ("ascend910b", "ascend910_93"):
        raise ValueError(f"Unsupported batch_invariant configuration SoC {soc!r}; vendor={vendor_real}")
    config = vendor_real / "op_impl/ai_core/tbe/config" / soc / f"aic-{soc}-ops-info.json"
    config_real = config.resolve()
    if not _within(config_real, vendor_real):
        raise ValueError(f"batch_invariant configuration escapes its vendor: {config} -> {config_real}")
    if not config_real.is_file():
        raise ValueError(f"batch_invariant configuration is missing: {config}; mapped libraries=" + ", ".join(libraries))
    try:
        configuration = json.loads(config_real.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read batch_invariant configuration {config_real}: {exc}") from exc
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError(f"batch_invariant configuration must be a nonempty JSON object: {config_real}")
    conflicting = _laser_registration(configuration)
    if conflicting is not None:
        raise ValueError(f"batch_invariant configuration contains a LaserAttention/VcLaserAttention registration "
                         f"at {config_real}: {conflicting!r}")
    return {"libraries": libraries, "metadata": {
        "classification": "environment_existing_provider",
        "provider": "batch_invariant", "cann_root": str(cann), "opp_root": str(opp),
        "vendor_root": str(vendor_real), "soc": soc,
        "config_path": str(config_real), "config_sha256": _sha256(config_real),
        "op_names": sorted(configuration),
        "library_sha256": {path: _sha256(Path(path)) for path in libraries},
        "scope": "Only the canonical batch_invariant tiling library; no API-library exemption. "
                 "Configuration contains no LaserAttention/VcLaserAttention identifiers. "
                 "Existing environment ownership does not establish CANN built-in provenance.",
    }}
