"""Phase 1 : lecture GPX, lissage, segmentation, courbure."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from course_io import (load_gpx, smooth_elevation, segment_track, elevation_gain,  # noqa: E402
                       curvature_profile, Track, GEOD)

GPX = ROOT / "data/raw/gpx/tdf2026_stage16.gpx"


def _track_from_xy(x, y, lat0=46.38, lon0=6.54):
    lat = lat0 + y / 111320.0
    lon = lon0 + x / (111320.0 * np.cos(np.radians(lat0)))
    _, _, st = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return Track(lat, lon, np.zeros_like(lat), np.concatenate([[0.0], np.cumsum(st)]))


def test_curvature_recovers_circle_radius():
    s = np.arange(0, 400, 7.0)
    r = 50.0
    tr = _track_from_xy(r * np.sin(s / r), r * (1 - np.cos(s / r)))
    _, kappa = curvature_profile(tr)
    assert 1.0 / np.median(np.abs(kappa[5:-5])) == pytest.approx(r, rel=0.05)


def test_curvature_ignores_gps_noise_on_straight():
    s = np.arange(0, 400, 7.0)
    rng = np.random.default_rng(0)
    tr = _track_from_xy(s, rng.normal(0, 2.0, len(s)))
    _, kappa = curvature_profile(tr)
    assert 1.0 / np.abs(kappa).max() > 60.0


def test_segment_lengths_and_grades_consistent():
    s = np.linspace(0, 1000, 101)
    tr = _track_from_xy(s, np.zeros_like(s))
    tr.ele_raw = 100 + 0.05 * s
    seg = segment_track(tr, step_m=100.0, ele=tr.ele_raw)
    assert seg.n == 10
    assert seg.length_m[:-1] == pytest.approx(np.full(9, 100.0), rel=1e-3)
    assert 75.0 <= seg.length_m[-1] <= 125.0   # le dernier segment absorbe le reste
    assert seg.grade == pytest.approx(np.full(10, 0.05), abs=1e-3)
    assert np.isinf(seg.radius_m).all()
    assert seg.bearing_deg == pytest.approx(np.full(10, 90.0), abs=0.5)   # vers l'est


@pytest.mark.skipif(not GPX.exists(), reason="GPX non telecharge")
class TestStage16:
    @pytest.fixture(scope="class")
    def seg(self):
        track = load_gpx(GPX)
        return track, segment_track(track, step_m=50.0, ele=smooth_elevation(track))

    def test_distance_within_1pct(self, seg):
        _, s = seg
        assert abs(s.total_km / 26.1 - 1) < 0.01

    def test_elevation_gain_consistent_with_published(self, seg):
        track, s = seg
        # ~500 m annonces ; le GPX brut donne 470 m, le lisse 472 m
        assert 420 <= s.elevation_gain_m <= 560
        assert abs(s.elevation_gain_m - elevation_gain(track.ele_raw)) < 30

    def test_larringes_grade_close_to_published(self, seg):
        _, s = seg
        i0 = np.searchsorted(s.dist_m, 100); i1 = np.searchsorted(s.dist_m, 9640)
        gain = s.ele[i1] - s.ele[i0]
        grade = gain / (s.dist_m[i1] - s.dist_m[i0]) * 100
        assert gain == pytest.approx(403, abs=15)
        assert grade == pytest.approx(4.4, abs=0.3)

    def test_start_finish_altitudes(self, seg):
        _, s = seg
        assert s.ele[0] == pytest.approx(385, abs=10)
        assert s.ele[-1] == pytest.approx(394, abs=10)

    def test_course_view_compatible_with_v1(self, seg):
        _, s = seg
        c = s.course
        assert c.n == s.n and c.total_km == pytest.approx(s.total_km)
