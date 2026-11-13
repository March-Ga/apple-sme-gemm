"""Reporting and CLI regression tests; synthetic timings only."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark


def record(backend, round_number, times, mode="total"):
    return {"size": 32, "mode": mode, "backend": backend, "round": round_number,
            "samples_ms": times, "batch_calls": 5 if times else 0,
            "input_fingerprint": "synthetic", "verified_entries": 1024, "max_scaled_error": 1e-8}


class ReportingTests(unittest.TestCase):
    def test_round_pairing_and_mode_separation(self):
        records = [record("sme32", 1, [9, 10, 11]), record("sme_opt", 1, [1, 2, 3]),
                   record("sme32", 2, [99, 100, 101]), record("sme_opt", 2, [9, 10, 11]),
                   record("sme32", 1, [20], "compute"), record("sme_opt", 1, [10], "compute"),
                   record("sme32", 3, [1000])]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ratios = {(r["mode"], r["baseline"], r["backend"]): r
                      for r in benchmark.summarize(records, output)}
            self.assertEqual(ratios[("total", "sme32", "sme_opt")]["speedup_median"], 7.5)
            self.assertEqual(ratios[("total", "sme32", "sme_opt")]["paired_rounds"], 2)
            self.assertEqual(ratios[("compute", "sme32", "sme_opt")]["speedup_median"], 2)
            with (output/"samples.csv").open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(len(samples), 15)
            self.assertAlmostEqual(float(samples[0]["gflops"]), 2*32**3/9/1e6)

    def test_verification_only_has_no_fake_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(benchmark.summarize([record("sme_opt", 1, [])], output), [])
            with (output/"summary.csv").open(newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            with (output/"verification.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_dry_run_launches_nothing(self):
        with patch.object(sys, "argv", ["benchmark.py", "--sizes", "32", "--dry-run"]), \
             patch.object(benchmark.subprocess, "run") as launch, \
             patch.object(benchmark.subprocess, "check_output") as capture, \
             patch("builtins.print") as output:
            self.assertEqual(benchmark.main(), 0)
            launch.assert_not_called()
            capture.assert_not_called()
            lines = [str(call.args[0]) for call in output.call_args_list]
            self.assertIn("15 jobs", lines[0])
            self.assertNotIn("accelerate", next(line for line in lines if "compute:" in line))

    def test_invalid_size_rejected_before_execution(self):
        with patch.object(sys, "argv", ["benchmark.py", "--sizes", "31", "--dry-run"]), \
             patch.object(benchmark.subprocess, "run") as launch, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                benchmark.main()
            launch.assert_not_called()

    def test_direct_comparison_with_accelerate(self):
        records = [record("accelerate", 1, [10]), record("sme32", 1, [20]),
                   record("sme_opt", 1, [5]), record("accelerate", 2, [12]), record("sme_opt", 2, [6])]
        with tempfile.TemporaryDirectory() as temporary:
            ratios = {(r["baseline"], r["backend"]): r
                      for r in benchmark.summarize(records, Path(temporary))}
        self.assertEqual(ratios[("accelerate", "sme_opt")]["speedup_median"], 2)
        self.assertEqual(ratios[("accelerate", "sme_opt")]["paired_rounds"], 2)
        self.assertEqual(ratios[("sme32", "sme_opt")]["speedup_median"], 4)


if __name__ == "__main__":
    unittest.main()
