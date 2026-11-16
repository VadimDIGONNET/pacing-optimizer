"""Phase 4 : inversion des pointages et calage (parcours synthetique, pas de reseau)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pacing_model import Rider, Environment  # noqa: E402
from course_io import SegmentedCourse  # noqa: E402
from dynamics import simulate_dynamic, DynamicsParams  # noqa: E402
from calibration import (invert_sector_powers, calibrate_cp, leave_one_out,  # noqa: E402
                         scaled_cda, body_surface_area)


def _hilly(n=200, step=50.0):
    nodes = np.arange(n + 1) * step
    grade = np.where(np.arange(n) < 100, 0.05, -0.03)
    ele = np.concatenate([[300.0], 300.0 + np.cumsum(grade * step)])
    return SegmentedCourse(nodes, np.zeros(n + 1), np.zeros(n + 1), ele, np.full(n, step), grade,
                           np.zeros(n), np.full(n, np.inf), np.zeros(n), corners=[])


def test_invert_recovers_known_sector_powers():
    seg, rider, env = _hilly(), Rider(cp=320, w_prime=20000, p_max=900, p_min=0), Environment()
    params = DynamicsParams(v_start=None)
    truth = np.where(np.arange(seg.n) < 100, 380.0, 250.0)
    r = simulate_dynamic(truth, seg, rider, env, params)
    t_mid = r.seg_time_s[:100].sum(); t_end = r.time_s
    fits, powers = invert_sector_powers(seg, rider, env, [(5.0, t_mid), (10.0, t_end)], params)
    assert fits[0].power_w == pytest.approx(380.0, abs=1.0)
    assert fits[1].power_w == pytest.approx(250.0, abs=1.0)
    assert fits[0].time_s == pytest.approx(t_mid, abs=0.01)


def test_calibrate_cp_empties_wprime():
    seg, env = _hilly(), Environment()
    rider = Rider(cp=300, w_prime=20000, p_max=900, p_min=0)
    powers = np.where(np.arange(seg.n) < 100, 380.0, 250.0)
    cp = calibrate_cp(powers, seg, rider, env, DynamicsParams(v_start=None))
    from dataclasses import replace
    r = simulate_dynamic(powers, seg, replace(rider, cp=cp), env, DynamicsParams(v_start=None))
    assert abs(r.min_wbal_j) < 100.0
    assert 300 < cp < 380


def test_leave_one_out_is_exact_for_constant_power():
    """Si le coureur roule a puissance constante, omettre un pointage ne change
    rien : l'erreur de prediction doit etre nulle (la physique est coherente)."""
    seg, rider, env = _hilly(), Rider(cp=320, w_prime=20000, p_max=900, p_min=0), Environment()
    params = DynamicsParams(v_start=None)
    r = simulate_dynamic(np.full(seg.n, 330.0), seg, rider, env, params)
    cps = [(2.5, r.seg_time_s[:50].sum()), (5.0, r.seg_time_s[:100].sum()),
           (7.5, r.seg_time_s[:150].sum()), (10.0, r.time_s)]
    loo = leave_one_out(seg, rider, env, cps, params)
    assert len(loo) == 3
    assert loo.err_s.abs().max() < 0.5


def test_cda_scaling_monotonic():
    assert body_surface_area(63, 1.71) == pytest.approx(1.73, abs=0.03)
    assert scaled_cda(63, 1.71) == pytest.approx(0.21)
    assert scaled_cda(82, 1.93) > scaled_cda(63, 1.71) > scaled_cda(58, 1.70)
    assert scaled_cda(70, None) > 0.21
