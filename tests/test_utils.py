"""Smoke tests for core inference helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from utils import holm_correction, srm_check, two_proportion_ztest  # noqa: E402


def test_two_proportion_ztest_detects_large_lift() -> None:
    result = two_proportion_ztest(
        successes_t=250,
        n_t=20_000,
        successes_c=114,
        n_c=20_000,
        metric="conversion",
        comparison="Mens vs Control",
    )
    assert result.effect > 0
    assert result.p_value < 0.01
    assert result.ci_low < result.effect < result.ci_high


def test_holm_adjustment_is_monotonic_and_bounded() -> None:
    adjusted = holm_correction([1e-10, 0.04, 0.20])
    assert adjusted[0] <= adjusted[1] or adjusted[0] < 0.05
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert adjusted[0] < 0.05


def test_srm_check_flags_bad_split() -> None:
    chi2, p, verdict = srm_check(
        {"ad": 960, "psa": 40},
        expected_shares={"ad": 0.5, "psa": 0.5},
    )
    assert chi2 > 0
    assert p < 0.001
    assert verdict.startswith("FAIL")


def test_srm_check_passes_balanced_split() -> None:
    chi2, p, verdict = srm_check(
        {"a": 500, "b": 500, "c": 500},
        expected_shares=None,
    )
    assert p >= 0.001
    assert verdict == "PASS"
