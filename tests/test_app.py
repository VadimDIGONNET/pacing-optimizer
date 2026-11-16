"""Test fumee du dashboard Streamlit v2 : l'app doit se charger sans exception."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_runs_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert len(at.tabs) >= 1
    assert any("Plan de puissance" in str(x) for x in at.tabs)
