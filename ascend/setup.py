"""Build using CANN and the target machine's existing torch/torch_npu."""
import os
import re
from pathlib import Path
import subprocess
import sys
from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


class NativeBuild(build_py):
    def run(self):
        if sys.platform != "linux":
            raise RuntimeError("Build on Linux with the Ascend C Toolkit and torch_npu installed")
        super().run()
        source = Path(__file__).resolve().parent
        destination = Path(self.build_lib).resolve() / "vc_attention_ascend"
        cann = os.environ.get("ASCEND_HOME_PATH")
        if not cann:
            raise RuntimeError("Source the CANN set_env.sh first (ASCEND_HOME_PATH is required)")
        soc = os.environ.get("SOC_VERSION")
        if not soc:
            raise RuntimeError("Set exact SOC_VERSION from preflight, or use scripts/run_ascend910b.sh")
        if not re.fullmatch(r"Ascend910(?:B(?:[1-4]|2C)|_93[0-9]{2})", soc):
            raise RuntimeError(f"Expected exact A2/A3 compiler SoC from preflight; got {soc!r}")
        # Never reuse device objects generated for a different SKU.
        native_build = source / "build" / f"native-{soc}"
        subprocess.run(["cmake", "-S", str(source), "-B", str(native_build),
                        f"-DASCEND_CANN_PACKAGE_PATH={cann}",
                        f"-DSOC_VERSION={soc}",
                        f"-DPython3_EXECUTABLE={sys.executable}",
                        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={destination}",
                        "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(native_build), "--parallel",
                        os.environ.get("MAX_JOBS", "4")], check=True)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(name="vc-attention-ascend", version="0.1.1", packages=["vc_attention_ascend"],
      python_requires=">=3.9", cmdclass={"build_py": NativeBuild}, distclass=BinaryDistribution,
      package_data={"vc_attention_ascend": ["*.so", "NOTICE", "LICENSE-CANN.txt"]},
      description="Experimental Ascend A2/A3 INT8 QK and ExpCast attention")
