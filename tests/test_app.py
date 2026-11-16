"""Test fumee du dashboard Streamlit : l'app doit se charger sans exception."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_runs_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    # Les trois indicateurs cles sont affiches
    assert len(at.metric) == 3
    assert at.metric[2].label == "Gain"
