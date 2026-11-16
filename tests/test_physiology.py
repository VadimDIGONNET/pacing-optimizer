"""Phase 5 : MMP, ajustement CP/W' et W'bal (donnees synthetiques)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physiology import mmp_curve, fit_cp_wprime, wbal_skiba  # noqa: E402


def test_mmp_curve_exact_on_step_signal():
    p = np.zeros(1000); p[100:400] = 400.0
    m = mmp_curve(p, durations=[60, 300, 600])
    assert m[60] == pytest.approx(400.0)
    assert m[300] == pytest.approx(400.0)
    assert m[600] == pytest.approx(200.0)


def test_fit_recovers_cp_wprime_from_hyperbolic_curve():
    cp, wp = 300.0, 20000.0
    mmp = {d: cp + wp / d for d in (180, 240, 300, 480, 600, 900, 1200)}
    cp_hat, wp_hat, r2 = fit_cp_wprime(mmp)
    assert cp_hat == pytest.approx(cp, abs=0.5) and wp_hat == pytest.approx(wp, abs=50) and r2 > 0.999


def test_wbal_depletes_and_recovers():
    cp, wp = 300.0, 20000.0
    p = np.concatenate([np.full(100, 400.0), np.full(300, 200.0)])
    wb = wbal_skiba(p, cp, wp)
    assert wb[99] == pytest.approx(wp - 100 * 100, abs=1)     # 100 s a +100 W
    assert wb[-1] > wb[99] and wb[-1] < wp                     # recuperation partielle
    assert wb.min() >= 0
