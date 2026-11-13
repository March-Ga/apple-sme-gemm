"""Reporting and CLI regression tests; synthetic timings only."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark


def record(backend, round_number, times):
    return {"size": 32, "mode": "total", "backend": backend, "round": round_number,
            "samples_ms": times, "batch_calls": 5 if times else 0,
            "input_fingerprint": "synthetic", "verified_entries": 1024, "max_scaled_error": 1e-8}


class ReportingTests(unittest.TestCase):
    def test_paired_ratios_and_missing_round(self):
        records=[record("accelerate",1,[9,10,11]), record("sme",1,[1,2,3]),
                 record("accelerate",2,[99,100,101]), record("sme",2,[9,10,11]),
                 record("accelerate",3,[1000])]
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            ratios=benchmark.summarize(records,output)
            self.assertEqual(len(ratios),1)
            self.assertEqual(ratios[0]["speedup_median"],7.5)
            self.assertEqual(ratios[0]["speedup_p25"],6.25)
            self.assertEqual(ratios[0]["speedup_p75"],8.75)
            self.assertEqual(ratios[0]["paired_rounds"],2)
            with (output/"samples.csv").open(newline="") as stream:
                samples=list(csv.DictReader(stream))
            self.assertEqual(len(samples),13)
            self.assertAlmostEqual(float(samples[0]["gflops"]),2*32**3/9/1e6)

    def test_verification_only_has_no_fake_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            self.assertEqual(benchmark.summarize([record("sme",1,[])],output),[])
            with (output/"summary.csv").open(newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)),[])
            with (output/"verification.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),1)

    def test_dry_run_launches_nothing(self):
        with patch.object(sys,"argv",["benchmark.py","--sizes","32","--dry-run"]), \
             patch.object(benchmark.subprocess,"run") as launch, \
             patch.object(benchmark.subprocess,"check_output") as capture, \
             patch("builtins.print") as output:
            self.assertEqual(benchmark.main(),0)
            launch.assert_not_called()
            capture.assert_not_called()
            self.assertIn("6 jobs",str(output.call_args_list[0].args[0]))

    def test_invalid_size_rejected_before_execution(self):
        with patch.object(sys,"argv",["benchmark.py","--sizes","31","--dry-run"]), \
             patch.object(benchmark.subprocess,"run") as launch, patch("sys.stderr"):
            with self.assertRaises(SystemExit): benchmark.main()
            launch.assert_not_called()

    def test_unknown_backend_rejected(self):
        with patch.object(sys,"argv",["benchmark.py","--backends","other","--dry-run"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit): benchmark.main()


if __name__=="__main__":
    unittest.main()
