"""Phase 2 : physique de l'air et vent par segment (sans appel reseau)."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weather import (air_density, pressure_at_altitude, wind_at_height,  # noqa: E402
                     wind_components, sample_at, segment_environment, WeatherSample)
from pacing_model import Environment, Rider, Course, simulate  # noqa: E402


def test_air_density_isa():
    assert air_density(15.0, 1013.25, 0.0) == pytest.approx(1.225, abs=0.001)


def test_humidity_lowers_density():
    assert air_density(30.0, 1000.0, 100.0) < air_density(30.0, 1000.0, 0.0)


def test_pressure_drops_with_altitude():
    p = pressure_at_altitude(1013.25, 0.0, 1000.0, 15.0)
    assert 895 < p < 905   # atmosphere standard : 898.7 hPa a 1000 m


def test_wind_log_profile_factor():
    assert wind_at_height(10.0, 1.0, 0.10) == pytest.approx(5.0, abs=0.01)


def test_wind_components_sign():
    head, cross = wind_components(5.0, 0.0, 0.0)      # vent du nord, cap nord
    assert head == pytest.approx(5.0) and cross == pytest.approx(0.0)
    head, _ = wind_components(5.0, 180.0, 0.0)        # vent du sud, cap nord
    assert head == pytest.approx(-5.0)
    _, cross = wind_components(5.0, 90.0, 0.0)        # vent d'est, cap nord
    assert abs(cross) == pytest.approx(5.0)


def test_sample_at_interpolates_direction_as_vector():
    payload = {"latitude": 46.4, "longitude": 6.5, "elevation": 400.0,
               "hourly": {"time": ["2026-07-21T12:00", "2026-07-21T13:00"],
                          "temperature_2m": [20.0, 22.0], "relative_humidity_2m": [50, 40],
                          "surface_pressure": [970.0, 971.0], "pressure_msl": [1015.0, 1016.0],
                          "wind_speed_10m": [4.0, 4.0], "wind_direction_10m": [350.0, 10.0],
                          "wind_gusts_10m": [8.0, 9.0]}}
    s = sample_at(payload, datetime(2026, 7, 21, 12, 30))
    assert s.temp_c == pytest.approx(21.0)
    assert s.wind_dir_deg == pytest.approx(0.0, abs=0.5)   # pas 180


def test_segment_environment_vector_and_scalar_api():
    class Seg:
        n = 4
        mid_dist_m = np.array([50.0, 150.0, 250.0, 350.0])
        mid_ele = np.array([400.0, 600.0, 800.0, 400.0])
        bearing_deg = np.array([0.0, 90.0, 180.0, 270.0])
    s = WeatherSample(datetime(2026, 7, 21, 17), 25.0, 33.0, 975.7, 1016.0, 4.4, 0.0, 9.0, 387.0, 46.4, 6.5)
    env, df = segment_environment(Seg(), s)
    assert not env.is_scalar
    assert env.rho[2] < env.rho[0]                     # plus haut = moins dense
    assert env.wind_ms[0] > 0 and env.wind_ms[2] < 0   # face vers le nord, dos vers le sud
    assert env.at(0).is_scalar
    # simulate accepte l'environnement vectoriel
    course = Course(np.full(4, 100.0), np.zeros(4))
    r = simulate(np.full(4, 300.0), course, Rider(), env)
    assert r.seg_time_s[0] > r.seg_time_s[2]           # vent de face plus lent


def test_scalar_environment_unchanged():
    env = Environment(rho=1.2, wind_ms=1.0)
    assert env.is_scalar and env.at(3) is env
