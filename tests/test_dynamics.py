"""Phase 3 : simulateur dynamique (inertie, virages, freinage)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pacing_model import Rider, Environment, Course, simulate  # noqa: E402
from dynamics import (simulate_dynamic, DynamicsParams, racing_line_radius,  # noqa: E402
                      corner_speed_limit, speed_limit_profile)
from course_io import SegmentedCourse  # noqa: E402


def _straight_with_corner(n=40, step=50.0, corner_at=20, radius=25.0, turn=90.0):
    nodes = np.arange(n + 1) * step
    seg = SegmentedCourse(
        dist_m=nodes, lat=np.zeros(n + 1), lon=np.zeros(n + 1), ele=np.zeros(n + 1),
        length_m=np.full(n, step), grade=np.zeros(n), bearing_deg=np.zeros(n),
        radius_m=np.full(n, np.inf), turn_deg=np.zeros(n),
        corners=[{"s_start_m": corner_at * step, "s_end_m": (corner_at + 1) * step,
                  "turn_deg": turn, "r_min_m": radius, "sign": 1}],
    )
    seg.radius_m[corner_at] = radius
    seg.turn_deg[corner_at] = turn
    return seg


def test_flat_course_matches_equilibrium_within_half_percent():
    rider, env = Rider(), Environment()
    flat = Course.from_segments([(100, 0.0)] * 300)      # 30 km
    p = np.full(flat.n, 300.0)
    t_eq = simulate(p, flat, rider, env).time_s
    t_dyn = simulate_dynamic(p, flat, rider, env, DynamicsParams(v_start=None)).time_s
    t_standing = simulate_dynamic(p, flat, rider, env, DynamicsParams(v_start=1.0)).time_s
    assert abs(t_dyn / t_eq - 1) < 0.001
    assert 0 < (t_standing - t_eq) and abs(t_standing / t_eq - 1) < 0.005


def test_accepts_plain_course_and_vector_environment():
    rider = Rider()
    course = Course.rolling_demo()
    env = Environment(rho=np.full(course.n, 1.2), wind_ms=np.linspace(-2, 2, course.n))
    r = simulate_dynamic(np.full(course.n, 300.0), course, rider, env)
    assert r.time_s > 0 and len(r.seg_time_s) == course.n
    assert r.min_wbal_j <= rider.w_prime


def test_corner_limits_speed_and_applied_power():
    rider, env = Rider(cp=300), Environment()
    seg = _straight_with_corner(radius=10.0)       # v_max = 7,7 m/s, bien sous l'equilibre
    params = DynamicsParams(a_lat_max=6.0, road_width_m=0.0, v_start=None)
    free = simulate_dynamic(np.full(seg.n, 350.0), seg, rider, env,
                            DynamicsParams(a_lat_max=1e6, v_start=None), keep_trace=True)
    lim = simulate_dynamic(np.full(seg.n, 350.0), seg, rider, env, params, keep_trace=True)
    v_corner = np.sqrt(6.0 * 10.0)
    assert lim.time_s > free.time_s
    assert lim.speed_ms[20] <= v_corner + 0.05
    assert lim.braking[19] or lim.braking[20]
    # la puissance appliquee est reduite la ou le plafond mord, jamais ailleurs
    assert lim.applied_power[20] < 350.0
    assert lim.applied_power[5] == pytest.approx(350.0)
    assert lim.min_wbal_j > free.min_wbal_j      # freiner coute moins de W'


def test_racing_line_radius_examples():
    assert racing_line_radius(25, 90, 6) == pytest.approx(42.5, abs=0.5)
    assert racing_line_radius(12, 180, 6) == pytest.approx(15.0, abs=0.1)
    assert racing_line_radius(30, 60, 0) == 30.0
    assert racing_line_radius(30, 10, 6) <= 300.0          # plafond 10 x


def test_speed_limit_profile_backward_pass():
    seg = _straight_with_corner(radius=20.0)
    v_node, v_corner = speed_limit_profile(seg, DynamicsParams(a_lat_max=5.0, a_brake_max=4.0,
                                                               road_width_m=0.0))
    v_c = np.sqrt(5.0 * 20.0)
    assert v_corner[20] == pytest.approx(v_c)
    # un noeud en amont ne peut depasser ce qui permet de freiner a temps
    assert v_node[19] == pytest.approx(np.sqrt(v_c ** 2 + 2 * 4.0 * 50.0), rel=1e-6)
    assert np.all(np.diff(v_node[:20]) <= 1e-9)             # decroissant vers le virage


def test_corner_speed_limit_infinite_on_straight():
    v = corner_speed_limit(np.array([np.inf, 50.0]), 6.0)
    assert np.isinf(v[0]) and v[1] == pytest.approx(np.sqrt(300.0))
