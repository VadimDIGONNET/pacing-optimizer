"""Phase 6 : optimiseur par blocs sur le simulateur dynamique."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pacing_model import Rider, Environment, Course, optimize_pacing, constant_power_target, simulate  # noqa: E402
from dynamics import DynamicsParams, simulate_dynamic  # noqa: E402
from optimizer import optimize_dynamic, make_blocks, constant_power_dynamic, plan_table  # noqa: E402
from course_io import SegmentedCourse  # noqa: E402


def _hilly(n=400, step=50.0):
    nodes = np.arange(n + 1) * step
    grade = np.where(np.arange(n) < 160, 0.05, np.where(np.arange(n) < 280, -0.04, 0.0))
    ele = np.concatenate([[300.0], 300.0 + np.cumsum(grade * step)])
    return SegmentedCourse(nodes, np.zeros(n + 1), np.zeros(n + 1), ele, np.full(n, step), grade,
                           np.zeros(n), np.full(n, np.inf), np.zeros(n), corners=[])


def test_blocks_cover_course_and_contain_segments():
    seg = _hilly()
    block, edges = make_blocks(seg, 500.0, breakpoints_km=[8.0, 14.0])
    assert edges[0] == 0 and edges[-1] == pytest.approx(seg.dist_m[-1])
    assert len(np.unique(block)) == len(edges) - 1          # chaque bloc a au moins un segment
    assert 8.0 * 1000 in edges and 14.0 * 1000 in edges


def test_demo_gain_close_to_v1():
    """Sur la demo v1 (vitesse d'equilibre, pas de virage), l'optimiseur v2
    doit retrouver un gain du meme ordre que SLSQP v1 (44 s)."""
    demo, rider, env = Course.rolling_demo(), Rider(), Environment()
    v1 = optimize_pacing(demo, rider, env)
    pc = constant_power_target(demo, rider, env)
    gain_v1 = simulate(np.full(demo.n, pc), demo, rider, env).time_s - v1.time_s
    opt = optimize_dynamic(demo, rider, env, DynamicsParams(v_start=None), block_m=500.0, smooth_lambda=0.0)
    assert opt.result.min_wbal_j > -50.0
    # le modele dynamique (inertie) donne au mieux ~38 s sur cette demo : on exige > 60 % du gain v1
    assert opt.gain_s > 0.6 * gain_v1
    # pousser dans les montees, relacher dans les descentes
    assert opt.block_powers[1] > rider.cp and opt.block_powers[2] < rider.cp


def test_constant_reference_respects_wprime_everywhere():
    seg = _hilly()
    rider, env, params = Rider(cp=300, w_prime=20000, p_max=600, p_min=0), Environment(), DynamicsParams(v_start=None)
    p = constant_power_dynamic(seg, rider, env, params)
    r = simulate_dynamic(np.full(seg.n, p), seg, rider, env, params)
    assert abs(r.min_wbal_j) < 100.0
    # a puissance constante au-dessus de CP, W'bal decroit partout : le minimum est a l'arrivee
    assert np.all(np.diff(r.wbal_j) <= 1e-6) and abs(r.wbal_j[-1]) < 100.0


def test_optimizer_feasible_faster_and_fast_enough():
    seg = _hilly()
    rider, env, params = Rider(cp=300, w_prime=20000, p_max=600, p_min=0), Environment(), DynamicsParams(v_start=None)
    opt = optimize_dynamic(seg, rider, env, params, block_m=500.0, breakpoints_km=[8.0, 14.0])
    assert opt.result.min_wbal_j > -50.0
    assert opt.gain_s > 5.0
    assert opt.elapsed_s < 60.0
    tab = plan_table(opt, seg)
    assert len(tab) == len(opt.block_powers)
    assert tab.time_s.sum() == pytest.approx(opt.result.time_s, rel=1e-6)


def test_fixed_block_bounds_are_respected():
    seg = _hilly()
    rider, env, params = Rider(cp=300, w_prime=20000, p_max=600, p_min=0), Environment(), DynamicsParams(v_start=None)
    block, edges = make_blocks(seg, 1000.0, breakpoints_km=[8.0, 14.0])
    nb = len(edges) - 1
    bounds = [(0.0, 600.0)] * nb
    bounds[nb // 2] = (250.0, 250.0)
    opt = optimize_dynamic(seg, rider, env, params, block_m=1000.0, breakpoints_km=[8.0, 14.0], bounds=bounds)
    assert opt.block_powers[nb // 2] == pytest.approx(250.0)
