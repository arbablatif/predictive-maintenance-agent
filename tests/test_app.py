"""Drives the real Streamlit script in mock mode (the toggle's default), with no browser."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def start():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    return at


def test_mock_is_on_by_default():
    assert start().toggle[0].value is True


def test_demo_file_produces_a_ticket_with_no_paid_calls():
    at = start()
    at.button[0].click().run()
    assert not at.exception
    assert any(s.value.startswith("Ticket PM-") for s in at.subheader)
    assert any("0 paid API calls" in c.value for c in at.caption)
    assert any("diagnose_bearing" in e.label for e in at.expander)


def test_engine_sample_shows_rul_and_no_bearing_tool():
    at = start()
    at.selectbox(key="demo_file").set_value("Synthetic: synthetic_engine.txt").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("estimate_rul" in e.label for e in at.expander)
    assert not any("diagnose_bearing" in e.label for e in at.expander)


def test_upload_mode_waits_for_a_file():
    at = start()
    at.radio[0].set_value("Upload").run()
    assert not at.exception
    assert at.button[0].disabled
