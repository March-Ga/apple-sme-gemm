#!/usr/bin/env python3
"""Compare optimized SME, original SME32 and Accelerate on square FP32 GEMM."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BACKENDS = ("sme32", "sme_opt", "accelerate")
SUITES = {"quick": [32, 128, 512], "full": [32, 64, 128, 256, 512, 1024, 2048, 4096]}


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def selection(text, convert=str):
    return list(dict.fromkeys(convert(part.strip()) for part in text.split(",") if part.strip()))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="full")
    parser.add_argument("--sizes", help="Override suite, e.g. 32,64,288,1024; all multiples of 32")
    parser.add_argument("--modes", default="compute,total", help="compute,total or one mode")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--rounds", type=positive, default=3)
    parser.add_argument("--samples", type=positive, default=7)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--min-sample-ms", type=float, default=30)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--verify-only", action="store_true", help="Execute correctness checks without timing samples; one round")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; execute no binaries")
    args = parser.parse_args()
    try:
        sizes = selection(args.sizes, positive) if args.sizes is not None else list(SUITES[args.suite])
        modes, backends = selection(args.modes), selection(args.backends)
        if not sizes or any(n < 32 or n > 8192 or n%32 for n in sizes):
            raise ValueError("sizes must be multiples of 32 in [32,8192]")
        if not modes or not set(modes) <= {"compute", "total"}:
            raise ValueError("unknown or empty modes")
        if not backends or not set(backends) <= set(BACKENDS):
            raise ValueError("unknown or empty backends")
        if max(args.rounds, args.samples, args.warmup) > 100:
            raise ValueError("rounds/samples/warmup must be <= 100")
        if not math.isfinite(args.min_sample_ms) or not 1 <= args.min_sample_ms <= 10000:
            raise ValueError("min-sample-ms must be in [1,10000]")
        if not 0 <= args.seed < 2**64:
            raise ValueError("seed must fit uint64")
        cases = [(size, mode) for size in sizes for mode in modes
                 if any(mode == "total" or backend != "accelerate" for backend in backends)]
        if not cases:
            raise ValueError("Accelerate is available only in total mode")
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    return args, cases, backends


def quantile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q
    lo = int(p)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(p-lo)


def csv_file(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, output):
    samples, groups, paired, checks = [], defaultdict(list), defaultdict(dict), []
    for record in records:
        common = {name: record[name] for name in ("size", "mode", "backend", "round")}
        checks.append(common | {"verified_entries": record["verified_entries"],
            "max_scaled_error": record["max_scaled_error"], "input_fingerprint": record["input_fingerprint"]})
        times = record["samples_ms"]
        if not times:
            continue
        groups[(record["size"], record["mode"], record["backend"])].extend(times)
        paired[(record["size"], record["mode"], record["round"])][record["backend"]] = statistics.median(times)
        for index, elapsed in enumerate(times, 1):
            samples.append(common | {"sample": index, "time_ms": elapsed,
                "gflops": 2*record["size"]**3/elapsed/1e6, "batch_calls": record["batch_calls"]})
    csv_file(output / "verification.csv", ["size", "mode", "backend", "round", "verified_entries",
             "max_scaled_error", "input_fingerprint"], checks)
    csv_file(output / "samples.csv", ["size", "mode", "backend", "round", "sample",
             "time_ms", "gflops", "batch_calls"], samples)
    summary = []
    for (size, mode, backend), times in sorted(groups.items()):
        median = statistics.median(times)
        summary.append({"size": size, "mode": mode, "backend": backend, "median_ms": median,
            "p25_ms": quantile(times, .25), "p75_ms": quantile(times, .75),
            "gflops": 2*size**3/median/1e6, "samples": len(times)})
    csv_file(output / "summary.csv", ["size", "mode", "backend", "median_ms", "p25_ms", "p75_ms",
             "gflops", "samples"], summary)
    ratios = defaultdict(list)
    for (size, mode, _), times in paired.items():
        for baseline, backend in (("sme32", "sme_opt"),
                                  ("accelerate", "sme32"), ("accelerate", "sme_opt")):
            if baseline in times and backend in times:
                ratios[(size, mode, baseline, backend)].append(times[baseline]/times[backend])
    speedups = [{"size": size, "mode": mode, "baseline": baseline, "backend": backend,
                 "speedup_median": statistics.median(values), "speedup_p25": quantile(values, .25),
                 "speedup_p75": quantile(values, .75), "paired_rounds": len(values)}
                for (size, mode, baseline, backend), values in sorted(ratios.items())]
    csv_file(output / "speedups.csv", ["size", "mode", "baseline", "backend", "speedup_median",
             "speedup_p25", "speedup_p75", "paired_rounds"], speedups)
    return speedups


def capture(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def hardware():
    # Keep only non-identifying fields. Serial numbers/UUIDs are never saved.
    try:
        data = json.loads(capture(["system_profiler", "SPHardwareDataType", "-json"]))["SPHardwareDataType"][0]
        return {"cpu": data.get("chip_type", "Apple Silicon"),
                "memory": data.get("physical_memory", "unavailable"),
                "model": data.get("machine_model", "unavailable"),
                "cores": data.get("number_processors", "unavailable")}
    except (ValueError, KeyError, IndexError, TypeError):
        return {"cpu": "Apple Silicon (metadata unavailable)"}


def check_build():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Run natively on Apple Silicon")
    manifest_path = BUILD / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Build first: python3 build.py")
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed ({name}); run python3 build.py")
    binary = BUILD / "sme_bench"
    if not binary.exists() or hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["binary_sha256"]:
        raise RuntimeError("Binary missing/modified; run python3 build.py")
    return manifest, binary


def main():
    args, cases, backends = parse()
    rounds = 1 if args.verify_only else args.rounds
    planned = rounds*sum(sum(mode == "total" or b != "accelerate" for b in backends) for _, mode in cases)
    print(f"Square FP32, one thread, {rounds} round(s): {planned} jobs", flush=True)
    if args.dry_run:
        for size, mode in cases:
            print(f"  N={size:4d} {mode:7s}: " + ", ".join(b for b in backends if mode == "total" or b != "accelerate"))
        print("Dry run: no binaries executed; no results created.")
        return 0
    manifest, binary = check_build()
    output = (args.output or ROOT / "results" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    environment = {"started_at": datetime.now().astimezone().isoformat(), "platform": platform.platform(),
        **hardware(),
        "power": capture(["pmset", "-g", "batt"]), "python": sys.version,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "effective_rounds": rounds, "build": manifest,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": {"dtype": "fp32", "threads": 1, "kc": 256, "panel": 16,
            "compute": "backend-specific preprocessed inputs; small optimized kernel uses original B; arithmetic and all C traffic included",
            "total": "packing required by the chosen backend on every call plus compute; Accelerate uses original row-major inputs",
            "excluded": "allocation, initialization, correctness verification, process startup and calibration",
            "sampling": "reused inputs; each sample is a batch average; not single-call tail latency",
            "parallelism": "one calling thread, USER_INITIATED QoS; Accelerate BLAS forced single-thread"}}
    environment["protocol"]["sme_opt"] = {
        **manifest["tuning"], "microkernel": "16x64 blocked or 32x32 direct B; SME2",
        "packing": "SME transpose A; small sizes use original B, large sizes use tiled B64 packing; included in total",
        "fallback": "N above DIRECT_B_MAX and not divisible by 64 uses baseline packing and sme32"}
    (output/"environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    records, fingerprints = [], {}
    rng = random.Random(args.seed)
    try:
        with (output/"raw.jsonl").open("w") as stream:
            for round_number in range(1, rounds+1):
                order = list(cases)
                rng.shuffle(order)
                for size, mode in order:
                    candidates = [b for b in backends if mode == "total" or b != "accelerate"]
                    rng.shuffle(candidates)
                    for backend in candidates:
                        print(f"[{len(records)+1}/{planned}] R{round_number} N={size} {mode} {backend}", flush=True)
                        command = [str(binary), "--backend", backend, "--mode", mode, "--size", str(size),
                            "--samples", str(args.samples), "--warmup", str(args.warmup),
                            "--min-sample-ms", str(args.min_sample_ms), "--seed", str(args.seed)]
                        if args.verify_only:
                            command.append("--verify-only")
                        result = subprocess.run(command, text=True, capture_output=True, env=env)
                        if result.returncode:
                            (output/"failed_job.json").write_text(json.dumps({"command": command,
                                "returncode": result.returncode, "stderr": result.stderr, "stdout": result.stdout}, indent=2)+"\n")
                            raise RuntimeError(f"Job failed ({result.returncode}): {result.stderr.strip()}")
                        record = json.loads(result.stdout)
                        if (record["size"], record["mode"], record["backend"], record["seed"], record["dtype"], record["threads"], record["kc"], record["panel"]) != (
                                size, mode, backend, args.seed, "fp32", 1,
                                manifest["tuning"]["OPT_KC"] if backend == "sme_opt" and
                                (size <= manifest["tuning"]["DIRECT_B_MAX"] or size%64 == 0) else 256, 16):
                            raise RuntimeError("Unexpected executable output")
                        if record["verification_only"] != args.verify_only or record["verified_entries"] < 1 or not math.isfinite(record["max_scaled_error"]):
                            raise RuntimeError("Invalid verification metadata")
                        if backend in ("sme32", "sme_opt") and record["streaming_fp32_lanes"] != 16:
                            raise RuntimeError("Unexpected SME vector length")
                        if len(record["samples_ms"]) != (0 if args.verify_only else args.samples) or any(
                                not math.isfinite(t) or t <= 0 for t in record["samples_ms"]):
                            raise RuntimeError("Invalid timing samples")
                        if fingerprints.setdefault(size, record["input_fingerprint"]) != record["input_fingerprint"]:
                            raise RuntimeError("Input fingerprint differs between compared runs")
                        record["round"] = round_number
                        records.append(record)
                        stream.write(json.dumps(record)+"\n")
                        stream.flush()
                        if args.verify_only:
                            print(f"    verification PASS ({record['verified_entries']} entries)", flush=True)
                        else:
                            elapsed = statistics.median(record["samples_ms"])
                            print(f"    {elapsed:.6f} ms, {2*size**3/elapsed/1e6:.2f} GFLOP/s; verification PASS", flush=True)
    finally:
        speedups = summarize(records, output)
        (output/"status.json").write_text(json.dumps({"complete": len(records) == planned,
            "completed_jobs": len(records), "planned_jobs": planned,
            "verification_only": args.verify_only, "finished_at": datetime.now().astimezone().isoformat()}, indent=2)+"\n")
        print(f"Results: {output}", flush=True)
    if speedups:
        print("\nSpeedups (baseline time / backend time; paired rounds):")
        for row in speedups:
            print(f"  N={row['size']:4d} {row['mode']:7s} {row['backend']} vs {row['baseline']}: {row['speedup_median']:.3f}x")
        print(f"Plot: python3 plot.py {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; completed jobs and partial reports preserved.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Run failed: {error}", file=sys.stderr)
        sys.exit(1)
