#!/usr/bin/env python3
"""Compile and inspect the C intrinsics benchmark. Do not execute GEMM."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
BUILD = ROOT / "build"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Build natively on an Apple Silicon Mac")
    BUILD.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC") or subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    flags = ["-isysroot", sdk, "-mmacosx-version-min=15.0", "-std=c11", "-O3", "-DNDEBUG",
             "-mcpu=apple-m4", "-fno-fast-math", "-ffp-contract=fast", "-fno-lto",
             "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-pthread"]
    sources = [HERE / "main.c", *sorted((HERE / "src").glob("*.c"))]
    objects = []
    for source in sources:
        obj = BUILD / f"{source.stem}.o"
        command = [cc, *flags, "-c", str(source), "-o", str(obj)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        objects.append(obj)
    binary = BUILD / "sme_bench"
    command = [cc, *flags, *map(str, objects), "-framework", "Accelerate", "-o", str(binary)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    inspections = {}
    for name in ("main", "pack", "pack_opt", "runtime", "sme32", "sme_opt", "sme_pack"):
        assembly = subprocess.check_output(["xcrun", "llvm-objdump", "-d", "--mattr=+sme,+sme2",
                                           str(BUILD / f"{name}.o")], text=True)
        (BUILD / f"{name}.disasm.txt").write_text(assembly)
        tiles = sorted(set(int(tile) for tile in re.findall(r"\bfmopa\s+za(\d+)\.s\b", assembly)))
        inspections[name] = {"fmopa_tiles": tiles,
            "four_vector_loads": len(re.findall(r"\bld1w\s+\{\s*z\d+\.s,\s*z\d+\.s,\s*z\d+\.s,\s*z\d+\.s\s*\}", assembly)),
            "neon_fmla_count": len(re.findall(r"\bfmla(?:\.4s)?\s+v", assembly)),
            "matrix_or_streaming_instructions": len(re.findall(r"\b(?:fmopa|bfmopa|smopa|umopa|smstart|smstop)\b", assembly))}
        if name in ("main", "pack", "pack_opt") and inspections[name]["matrix_or_streaming_instructions"]:
            raise RuntimeError(f"Unexpected SME instructions in {name}")
    if inspections["sme32"]["fmopa_tiles"] != [0, 1, 2, 3]:
        raise RuntimeError("SME32 must use ZA0, ZA1, ZA2 and ZA3")
    if inspections["sme_opt"]["fmopa_tiles"] != [0, 1, 2, 3]:
        raise RuntimeError("Optimized SME must use four ZA tiles")
    if not inspections["sme_opt"]["four_vector_loads"]:
        raise RuntimeError("Expected SME2 four-vector loads are missing")
    tracked = [*sources, HERE / "src/gemm.h", Path(__file__).resolve()]
    settings = {name: int(re.search(rf"\b{name}\s*=\s*(\d+)", (HERE/"src/gemm.h").read_text())[1])
                for name in ("OPT_KC", "OPT_MC", "OPT_NC", "DIRECT_B_MAX")}
    manifest = {"compiler": subprocess.check_output([cc, "--version"], text=True).strip(),
                "flags": flags, "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in tracked},
                "binary_sha256": digest(binary), "inspection": inspections, "tuning": settings,
                "format": "square FP32, row-major inputs, backend-specific packing, one thread"}
    (BUILD / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("\nBuild and disassembly checks PASS. No benchmark was executed.")
    print("Next: python3 benchmark.py --suite quick --output results/quick")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        sys.exit(1)
