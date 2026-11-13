#!/usr/bin/env python3
"""Compile and run numerical/packing regression tests with ASan and UBSan."""
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Requires native Apple Silicon with SME2 (M4+)")
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    build = ROOT/"build"
    build.mkdir(parents=True, exist_ok=True)
    common = ["src/pack.c"]
    for test, sources in (("test_pack", common),
                          ("test_gemm", common+["src/sme.c", "src/accelerate.c"])):
        target = build/test
        command = ["xcrun", "clang", "-isysroot", sdk, "-mmacosx-version-min=15.0",
                   "-std=c11", "-O1", "-g", "-mcpu=apple-m4", "-fsanitize=address,undefined",
                   "-Wall", "-Wextra", "-Werror", f"tests/{test}.c", *sources, "-framework", "Accelerate", "-o", str(target)]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([str(target)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
