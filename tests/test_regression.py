"""Tests de non-regression sur le moteur v1 (phase 0 du PLAN.md).

Les chiffres de reference ont ete mesures le 2026-11-16 avec
numpy 2.5.5 / scipy 1.18.1 (Python 3.14). SLSQP peut bouger de quelques
dixiemes de seconde d'une version de SciPy a l'autre, d'ou les tolerances.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from pacing_model import (  # noqa: E402
    Rider, Environment, Course, simulate, speed_from_power,
    constant_power_target, optimize_pacing,
)
import tdf2026_stage16  # noqa: E402


# ---------------------------------------------------------------------------
# Briques de base
# ---------------------------------------------------------------------------

def test_speed_increases_with_power():
    rider, env = Rider(), Environment()
    v = [speed_from_power(p, 0.0, rider, env) for p in (150, 250, 350)]
    assert v[0] < v[1] < v[2]


def test_speed_decreases_with_grade_and_headwind():
    rider, env = Rider(), Environment()
    flat = speed_from_power(300, 0.0, rider, env)
    climb = speed_from_power(300, 0.05, rider, env)
    headwind = speed_from_power(300, 0.0, rider, Environment(wind_ms=3.0))
    assert climb < flat
    assert headwind < flat


def test_speed_from_power_matches_martin_equation():
    """La vitesse renvoyee doit verifier l'equation de puissance a 0,1 W pres."""
    from pacing_model import G, DRIVETRAIN_EFF
    rider, env = Rider(), Environment(rho=1.2, wind_ms=1.0)
    grade, power = 0.03, 320.0
    v = speed_from_power(power, grade, rider, env)
    theta = np.arctan(grade)
    f = rider.mass * G * (np.sin(theta) + rider.crr * np.cos(theta))
    f += 0.5 * env.rho * rider.cda * (v + env.wind_ms) ** 2
    assert abs(f * v / DRIVETRAIN_EFF - power) < 0.1


def test_wbal_constant_below_cp_stays_full():
    rider, env = Rider(), Environment()
    course = Course.rolling_demo()
    res = simulate(np.full(course.n, rider.cp - 50), course, rider, env)
    assert res.min_wbal_j == pytest.approx(rider.w_prime)


def test_constant_power_target_empties_wprime_at_finish():
    rider, env = Rider(), Environment()
    course = Course.rolling_demo()
    p = constant_power_target(course, rider, env)
    res = simulate(np.full(course.n, p), course, rider, env)
    assert abs(res.wbal_j[-1]) < 50.0   # J, tolerance de brentq (xtol=0.1 W)
    assert res.min_wbal_j > -50.0


def test_course_from_elevation():
    course = Course.from_elevation([0, 1000, 2500], [100, 160, 100])
    assert course.n == 2
    assert course.length_m.tolist() == [1000.0, 1500.0]
    assert course.grade == pytest.approx([0.06, -0.04])
    assert course.total_km == pytest.approx(2.5)


def test_optimize_pacing_bounds_freeze_segments():
    rider, env = Rider(), Environment()
    course = Course.rolling_demo()
    bounds = [(rider.p_min, rider.p_max)] * course.n
    bounds[2] = (250.0, 250.0)   # descente figee
    res = optimize_pacing(course, rider, env, bounds=bounds)
    assert res.powers[2] == pytest.approx(250.0)
    assert res.min_wbal_j > -50.0


# ---------------------------------------------------------------------------
# Chiffres de reference (PLAN.md, phase 0)
# ---------------------------------------------------------------------------

def test_demo_gain_about_44s():
    """Demo 10 km : 318 W constant -> 18:19.3 ; optimise -> 17:34.8 ; gain 44,5 s."""
    course, rider, env = Course.rolling_demo(), Rider(), Environment()
    p_const = constant_power_target(course, rider, env)
    naive = simulate(np.full(course.n, p_const), course, rider, env)
    optimal = optimize_pacing(course, rider, env)
    gain = naive.time_s - optimal.time_s

    assert p_const == pytest.approx(318.0, abs=1.0)
    assert naive.time_s == pytest.approx(1099.3, abs=0.5)
    assert optimal.time_s == pytest.approx(1054.8, abs=0.5)
    assert gain == pytest.approx(44.5, abs=0.5)
    assert optimal.min_wbal_j > -50.0
    # Pousser dans les montees, relacher dans les descentes
    assert optimal.powers[1] > rider.cp and optimal.powers[4] > rider.cp
    assert optimal.powers[2] < rider.cp and optimal.powers[5] < rider.cp


def test_stage16_gain_about_22s():
    """Etape 16 TdF 2026 : temps reel 32'19\" reconstruit, gain potentiel 22,4 s."""
    res = tdf2026_stage16.analyse()
    real, best, rider = res["real"], res["best"], res["rider"]
    gain = real.time_s - best.time_s

    assert real.time_s == pytest.approx(1939.0, abs=0.5)
    assert gain == pytest.approx(22.4, abs=0.5)
    assert rider.cp == pytest.approx(438.0, abs=1.0)
    # Les pointages reels sont reproduits par construction
    cum = np.cumsum(real.seg_time_s)
    for (_, t), i in zip(tdf2026_stage16.SPLITS, [0, 1, 4, 5]):
        assert cum[i] == pytest.approx(t, abs=0.2)
    # La descente reste figee a la puissance reelle
    for i in tdf2026_stage16.SECTORS[2]:
        assert best.powers[i] == pytest.approx(res["actual"][i])
