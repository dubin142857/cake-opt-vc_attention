"""Install the pinned official AscendC backend without replacing Torch/CANN.

This bootstrap runs on the target Linux NPU machine. Importing this module on a
CPU machine is safe; hardware probes only run from main().
"""
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/tile-ai/tilelang-ascend.git"
SOURCE_BRANCH = "ascendc_pto"
SOURCE_COMMIT = "1d7525399141b242435e6e1c60cc006df0a3d7a6"
VENV = ROOT / ".venv-tilelang-npu"
SOURCE = ROOT / "build" / ("tilelang-ascend-npu-" + SOURCE_COMMIT[:12])
REPORT = ROOT / "results" / "tilelang_attention_npu_setup.json"
MARKER = ".tilelang-attention-npu-owned.json"
OWNER = "ordinary-tilelang-attention-npu-v1"
JSON_PREFIX = "__TILELANG_NPU_SETUP_JSON__"
BACKEND_ENV_OVERRIDES = (
    "TL_ROOT", "TVM_IMPORT_PYTHON_PATH", "TVM_LIBRARY_PATH", "TVM_PREBUILD_PATH",
    "TVM_SOURCE_DIR", "TL_TEMPLATE_PATH", "TL_CUTLASS_PATH", "TL_COMPOSABLE_KERNEL_PATH",
    "SKIP_LOADING_TILELANG_SO", "PYPI_BUILD",
)

# No requirement below depends on Torch. --no-deps additionally prevents pip
# from changing the inherited, hardware-specific torch / torch_npu stack.
PYTHON_REQUIREMENTS = [
    "cmake>=3.26", "packaging", "setuptools>=61", "wheel", "patchelf",
    "Cython", "numpy>=1.23.5", "tqdm>=4.62.3", "typing_extensions>=4.10.0",
    "cloudpickle", "ml_dtypes", "psutil", "pybind11", "PyYAML", "decorator",
    "absl-py", "attrs", "scipy", "tornado",
]

TORCH_PROBE = r'''
import json, pathlib, sys
import torch
import torch_npu
device = int(sys.argv[1])
if not torch.npu.is_available():
    raise RuntimeError("torch.npu.is_available() is false")
torch.npu.set_device(device)
x = torch.ones((8,), device="npu:%d" % device)
observed = (x * x).sum().item()
torch.npu.synchronize()
if observed != 8:
    raise RuntimeError("NPU smoke operation returned %r" % observed)
result = {
    "python": sys.executable, "prefix": sys.prefix,
    "torch_version": torch.__version__,
    "torch_file": str(pathlib.Path(torch.__file__).resolve()),
    "torch_npu_version": getattr(torch_npu, "__version__", None),
    "torch_npu_file": str(pathlib.Path(torch_npu.__file__).resolve()),
    "device": device, "device_name": torch.npu.get_device_name(device),
    "npu_smoke_passed": True,
}
print("__TILELANG_NPU_SETUP_JSON__" + json.dumps(result))
'''

BACKEND_PROBE = r'''
import inspect, json, pathlib, re, subprocess
import torch
import torch_npu
import tilelang
from tilelang import language as T
required = ("gemm_v0", "alloc_L1", "alloc_L0C", "alloc_ub", "Scope",
            "set_cross_flag", "wait_cross_flag", "barrier_all")
missing = [name for name in required if not callable(getattr(T, name, None))]
if missing:
    raise RuntimeError("Not the AscendC TileLang backend; missing " + repr(missing))
if "is_npu" not in inspect.signature(T.Kernel).parameters:
    raise RuntimeError("T.Kernel does not support is_npu")
parameters = inspect.signature(tilelang.jit).parameters
if any(name not in parameters for name in ("platform", "target", "execution_backend")):
    raise RuntimeError("tilelang.jit does not expose the AscendC platform/backend API")
for name in ("TL_ASCEND_AUTO_SYNC", "TL_ASCEND_MEMORY_PLANNING",
             "TL_ASCEND_AUTO_CV_COMBINE", "TL_ASCEND_AUTO_CV_SYNC"):
    if not hasattr(tilelang.PassConfigKey, name):
        raise RuntimeError("Missing Ascend pass configuration: " + name)
result = {"version": str(tilelang.__version__),
          "file": str(pathlib.Path(tilelang.__file__).resolve()),
          "library": str(getattr(tilelang, "_LIB_PATH", "")),
          "ascendc_api_verified": True}
if not result["library"] or not pathlib.Path(result["library"]).is_file():
    raise RuntimeError("The TileLang native shared library is not loaded")
module_dir = pathlib.Path(tilelang.__file__).resolve().parent
evidence = []
if (module_dir.parent / ".git").exists():
    evidence.append(subprocess.check_output(
        ["git", "-C", str(module_dir.parent), "rev-parse", "HEAD"], text=True).strip())
for version_file in (module_dir / "VERSION", module_dir.parent / "VERSION"):
    if version_file.is_file():
        evidence.extend(re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])",
                                   version_file.read_text()))
result["commit_evidence"] = evidence
print("__TILELANG_NPU_SETUP_JSON__" + json.dumps(result))
'''


def write_report(value, path=REPORT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(argv, *, env=None, cwd=None, capture=False):
    display = list(map(str, argv))
    if len(display) > 2 and display[1] == "-c":
        display[2] = "<environment probe>"
    print("[TileLang NPU setup] " + " ".join(display), flush=True)
    result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env,
                            text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if result.returncode:
        detail = (result.stdout or "")[-12000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {display!r}\n{detail}")
    return result.stdout if capture else ""


def parse_probe_output(output):
    for line in reversed(output.splitlines()):
        if line.startswith(JSON_PREFIX):
            return json.loads(line[len(JSON_PREFIX):])
    raise RuntimeError("Python environment probe did not produce its JSON result")


def probe(python, code, *args, env=None):
    return parse_probe_output(run([python, "-c", code, *args], env=env, capture=True))


def assert_torch_preserved(before, after):
    keys = ("torch_version", "torch_file", "torch_npu_version", "torch_npu_file")
    changed = {key: [before.get(key), after.get(key)] for key in keys
               if before.get(key) != after.get(key)}
    if changed:
        raise RuntimeError("Setup must preserve the existing Torch/NPU stack: " + repr(changed))


def require_owned_venv(path):
    path = Path(path)
    if not path.exists():
        return False
    marker = path / MARKER
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Refusing to modify existing unowned venv: {path}") from error
    if data.get("owner") != OWNER or data.get("root") != str(ROOT.resolve()):
        raise RuntimeError(f"Refusing to modify venv with a different owner: {path}")
    if not (path / "pyvenv.cfg").is_file() or not (path / "bin" / "python").is_file():
        raise RuntimeError(f"Owned venv is incomplete; no existing files were removed: {path}")
    return True


def ensure_venv(base_python, path=VENV):
    path = Path(path)
    if not require_owned_venv(path):
        run([base_python, "-m", "venv", "--system-site-packages", path])
        (path / MARKER).write_text(json.dumps({"owner": OWNER, "root": str(ROOT.resolve())}),
                                   encoding="utf-8")
    python = path / "bin" / "python"
    prefix = probe(python, 'import json, sys; print("' + JSON_PREFIX +
                   '" + json.dumps({"prefix": sys.prefix}))')["prefix"]
    if Path(prefix).resolve() != path.resolve():
        raise RuntimeError("The managed Python does not belong to the expected venv")
    return python


def expose_existing_stack(python, before, path=VENV):
    """Also support a base interpreter that itself belongs to another venv.

    --system-site-packages alone only exposes the underlying system, not an
    outer venv's packages. Add its already-probed module roots AFTER this venv's
    own site-packages; never copy or reinstall the Torch binaries.
    """
    site_dir = Path(probe(python, 'import json, sysconfig; print("' + JSON_PREFIX +
                          '" + json.dumps({"purelib": sysconfig.get_path("purelib")}))')["purelib"])
    if not site_dir.resolve().is_relative_to(Path(path).resolve()):
        raise RuntimeError("Refusing to write a package path outside the managed venv")
    roots = list(dict.fromkeys(str(Path(before[key]).resolve().parent.parent)
                              for key in ("torch_file", "torch_npu_file")))
    if any("\n" in root or "\r" in root for root in roots):
        raise RuntimeError("The existing package directory contains an invalid path separator")
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "tilelang_attention_existing_npu_stack.pth").write_text(
        "\n".join(roots) + "\n", encoding="utf-8")


def ensure_source(path=SOURCE):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", "--branch",
             SOURCE_BRANCH, SOURCE_URL, path])
        run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=path)
    if not (path / ".git").exists():
        raise RuntimeError(f"Existing source path is not a Git checkout: {path}")
    remote = run(["git", "remote", "get-url", "origin"], cwd=path, capture=True).strip()
    if remote.rstrip("/").removesuffix(".git") != SOURCE_URL.removesuffix(".git"):
        raise RuntimeError(f"Existing source has an unexpected origin: {remote}")
    head = run(["git", "rev-parse", "HEAD"], cwd=path, capture=True).strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"Existing source is not pinned at {SOURCE_COMMIT}: {head}")
    # Never reset a checkout or remove user files. Build outputs are untracked.
    changed = run(["git", "status", "--porcelain", "--untracked-files=no",
                   "--ignore-submodules=all"], cwd=path, capture=True).strip()
    if changed:
        raise RuntimeError("Pinned source has tracked modifications; refusing to build: " + changed)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=path)
    return run(["git", "submodule", "status", "--recursive"], cwd=path, capture=True)


def limit_build_jobs(argv, jobs=4):
    """Limit upstream's hardcoded cmake --build -jN without changing its source."""
    argv = list(argv)
    if not argv or argv[0] != "--build":
        return argv
    result, index = [], 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return result + ["--parallel", str(jobs)] + argv[index:]
        if argument in ("-j", "--parallel"):
            index += 1
            if index < len(argv) and argv[index].isdigit():
                index += 1
            continue
        if ((argument.startswith("-j") and argument[2:].isdigit()) or
                argument.startswith("--parallel=")):
            index += 1
            continue
        result.append(argument)
        index += 1
    return result + ["--parallel", str(jobs)]


def build_environment(python, source=SOURCE):
    env = dict(os.environ)
    for name in BACKEND_ENV_OVERRIDES:
        env.pop(name, None)
    env.update(USE_ASCEND="true", USE_LLVM="false", USE_ROCM="false",
               ACL_OP_INIT_MODE="1", PYTHONNOUSERSITE="1", TL_RUN_MODE="npu")
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    real_cmake = shutil.which("cmake", path=env["PATH"])
    if not real_cmake:
        raise RuntimeError("CMake is unavailable after installing build requirements")
    tools = Path(source) / ".setup-tools"
    tools.mkdir(parents=True, exist_ok=True)
    wrapper = tools / "cmake"
    wrapper.write_text("#!/usr/bin/env python3\nimport subprocess, sys\n" +
                       inspect.getsource(limit_build_jobs) +
                       "\nsys.exit(subprocess.call([" + repr(real_cmake) +
                       "] + limit_build_jobs(sys.argv[1:], 4)))\n", encoding="utf-8", newline="\n")
    wrapper.chmod(wrapper.stat().st_mode | 0o111)
    env["PATH"] = str(tools) + os.pathsep + env["PATH"]
    return env


def check_local_backend(backend, venv=VENV):
    if not Path(backend["file"]).resolve().is_relative_to(Path(venv).resolve()):
        raise RuntimeError("TileLang was inherited from another environment instead of the NPU venv")
    evidence = backend.get("commit_evidence", [])
    if not evidence or any(commit != SOURCE_COMMIT for commit in evidence):
        raise RuntimeError("The managed backend does not identify the pinned source commit")


def main():
    started = time.time()
    report = {"status": "starting", "source_url": SOURCE_URL, "source_branch": SOURCE_BRANCH,
              "source_commit": SOURCE_COMMIT, "source_path": str(SOURCE),
              "venv": str(VENV), "hardware_validation": "not run"}
    try:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("TileLang-Ascend setup requires Linux; run the launcher on the NPU host")
        if sys.version_info < (3, 10):
            raise RuntimeError("TileLang-Ascend requires Python >=3.10")
        cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME")
        if not cann or not Path(cann).is_dir():
            raise RuntimeError("Source the installed CANN set_env.sh before running setup")
        os.environ["ASCEND_HOME_PATH"] = cann
        os.environ["ACL_OP_INIT_MODE"] = "1"
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ["TL_RUN_MODE"] = "npu"
        for name in BACKEND_ENV_OVERRIDES:
            os.environ.pop(name, None)
        if not shutil.which("bisheng"):
            raise RuntimeError("CANN Bisheng is unavailable; source the selected CANN environment")
        report["cann_home"] = cann
        report["bisheng"] = run(["bisheng", "--version"], capture=True).strip()
        device = str(int(os.environ.get("NPU_DEVICE", "0")))
        override = os.environ.get("TILELANG_NPU_PYTHON")
        if override:
            python = shutil.which(override) or override
            report["external_override"] = True
            report["torch_before"] = probe(python, TORCH_PROBE, device)
            report["backend"] = probe(python, BACKEND_PROBE)
            evidence = report["backend"].get("commit_evidence", [])
            if not evidence or any(commit != SOURCE_COMMIT for commit in evidence):
                raise RuntimeError("TILELANG_NPU_PYTHON must use the pinned AscendC source commit")
            report["pinned_source_verified"] = True
        else:
            before = probe(sys.executable, TORCH_PROBE, device)
            report["torch_before"] = before
            python = ensure_venv(sys.executable)
            expose_existing_stack(python, before)
            assert_torch_preserved(before, probe(python, TORCH_PROBE, device))
            try:
                backend = probe(python, BACKEND_PROBE)
                check_local_backend(backend)
            except RuntimeError:
                if not shutil.which("git") or not (shutil.which("g++") or shutil.which("c++")):
                    raise RuntimeError("Source builds require Git and a C++ compiler (g++ or c++)")
                run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
                     *PYTHON_REQUIREMENTS])
                report["submodules"] = ensure_source()
                environment = build_environment(python)
                distribution = SOURCE / "dist"
                distribution.mkdir(exist_ok=True)
                run([python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                     "--wheel-dir", distribution, "."], env=environment, cwd=SOURCE)
                wheels = sorted(distribution.glob("tilelang-*.whl"), key=lambda path: path.stat().st_mtime)
                if not wheels:
                    raise RuntimeError("The pinned source build did not produce a TileLang wheel")
                wheel = wheels[-1]
                # --ignore-installed installs locally even when a newer CUDA
                # tilelang is visible through --system-site-packages.
                run([python, "-m", "pip", "install", "--no-deps", "--ignore-installed", wheel])
                import hashlib
                report["wheel"] = {"path": str(wheel), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
                backend = probe(python, BACKEND_PROBE)
                check_local_backend(backend)
            report["backend"] = backend
            report["pinned_source_verified"] = True
            after = probe(python, TORCH_PROBE, device)
            assert_torch_preserved(before, after)
            report["torch_after"] = after
        report.update(status="ready", python_executable=str(python),
                      hardware_validation="Torch NPU smoke passed; attention kernel not yet tested")
    except Exception as error:
        report.update(status="failed", error=str(error))
        print("[TileLang NPU setup] " + str(error), file=sys.stderr, flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 3)
    write_report(report)
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
