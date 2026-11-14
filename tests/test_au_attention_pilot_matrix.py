"""Checks for parsing and gating the AU attention pilot report."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_au_attention_pilot_matrix",
    Path(__file__).resolve().parents[1] / "scripts/run_au_attention_pilot_matrix.py",
)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)

calibration_spec = importlib.util.spec_from_file_location(
    "calibrate_au_attention",
    Path(__file__).resolve().parents[1] / "scripts/calibrate_au_attention.py",
)
calibration = importlib.util.module_from_spec(calibration_spec)
calibration_spec.loader.exec_module(calibration)


def test_pilot_report_requires_both_directions_and_enough_strong_pairs():
    train = matrix.parse_train(
        "step=1 loss=1.2 g_scale*lambda_max=[12.0, 9.0] other=x\n"
        "AMP skipped 0/1 optimizer steps\n"
    )
    calibration = matrix.parse_calibration(
        "checkpoint=x epoch=0 valid_pairs=100 strong_pairs=25 amp=True\n"
        "3e+06 0.0120 0.0110 1.00% 2.00% 0.1 0.2 True\n",
        3_000_000,
    )
    row = {"status": "completed", "train": train, "calibration": calibration}
    assert matrix.eligible(row)
    row["calibration"] = {**calibration, "p90": (0.012, 0.009)}
    assert not matrix.eligible(row)
    row["calibration"] = {**calibration, "strong_pairs": 11}
    assert not matrix.eligible(row)
    row["calibration"] = calibration
    row["train"] = {**train, "skipped": 1}
    assert not matrix.eligible(row)


def test_attention_probe_stops_only_after_minimum_batches_and_strong_pairs():
    options = {"min_batches": 8, "min_strong_pairs": 20}
    assert not calibration.enough_evidence(7, 30, **options)
    assert not calibration.enough_evidence(20, 19, **options)
    assert calibration.enough_evidence(8, 20, **options)
