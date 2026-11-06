"""Tools, the agent loop and the LLM client, end to end with no API."""
import json
from pathlib import Path

import numpy as np
import pytest

from pm_agent import llm as llm_module
from pm_agent.agent import TicketAgent
from pm_agent.config import SAMPLE_DIR
from pm_agent.llm import BudgetExceeded, LLMClient, LLMReply
from pm_agent.signals import SKF_6203, SignalError, load_upload
from pm_agent.ticket import RUL_NOT_AVAILABLE
from pm_agent.tools import execute


def vibration(name="synthetic_outer_race.csv", rpm=1750.0, **kwargs):
    return load_upload((SAMPLE_DIR / name).read_bytes(), name, "Test asset", 12_000, rpm, **kwargs)


def engine():
    return load_upload((SAMPLE_DIR / "synthetic_engine.txt").read_bytes(), "synthetic_engine.txt", "Test engine")


class Scripted:
    """Stands in for LLMClient and plays back a fixed list of replies."""
    api_calls = 0

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, system, messages, tools):
        self.last_messages = messages
        return self.replies.pop(0)


def call(name, signal):
    return LLMReply("", [{"id": name, "name": name, "args": {"signal_id": signal.signal_id}}])


def ticket_reply(finding, action="Confirm with a second measurement."):
    return LLMReply(json.dumps({"finding": finding, "recommended_action": action}))


# ---------------------------------------------------------------------- tools
def test_diagnosis_finds_the_synthetic_outer_race_fault(tmp_path):
    signal = vibration()
    result = execute({signal.signal_id: signal}, "diagnose_bearing", {"signal_id": signal.signal_id}, tmp_path).result
    assert result["predicted_class"] == "outer_race"
    assert result["envelope_check"] == "supports"
    assert set(result["policy"]) == {"urgency", "action_window_days", "action_code"}


def test_plot_tool_writes_a_file(tmp_path):
    signal = vibration()
    outcome = execute({signal.signal_id: signal}, "plot_envelope_spectrum", {"signal_id": signal.signal_id}, tmp_path)
    assert outcome.figures[0].exists()


def test_rul_tool_refuses_vibration(tmp_path):
    signal = vibration()
    outcome = execute({signal.signal_id: signal}, "estimate_rul", {"signal_id": signal.signal_id}, tmp_path)
    assert outcome.result["code"] == "wrong_signal_type"
    assert "rul_cycles" not in outcome.result


def test_bearing_tools_refuse_engine_history(tmp_path):
    signal = engine()
    for tool in ("diagnose_bearing", "plot_envelope_spectrum"):
        assert execute({signal.signal_id: signal}, tool, {"signal_id": signal.signal_id}, tmp_path).result[
            "code"] == "wrong_signal_type"


def test_rul_tool_gives_an_ordered_interval(tmp_path):
    signal = engine()
    outcome = execute({signal.signal_id: signal}, "estimate_rul", {"signal_id": signal.signal_id}, tmp_path)
    r = outcome.result
    assert r["interval_low_cycles"] <= r["rul_cycles"] <= r["interval_high_cycles"]
    assert outcome.figures[0].exists()


def test_unknown_tool_and_signal_are_reported_not_raised(tmp_path):
    signal = vibration()
    signals = {signal.signal_id: signal}
    assert execute(signals, "nope", {}, tmp_path).result["code"] == "unknown_tool"
    assert execute(signals, "diagnose_bearing", {"signal_id": "x"}, tmp_path).result["code"] == "unknown_signal"
    assert execute(signals, "diagnose_bearing", None, tmp_path).result["code"] == "unknown_signal"


@pytest.mark.parametrize("raw,fs", [(np.zeros(20_000), 12_000), (np.random.default_rng(0).random(500), 12_000),
                                    (np.random.default_rng(0).random(20_000), 8_000)])
def test_unusable_signals_become_tool_errors(tmp_path, raw, fs):
    text = "\n".join(f"{v:.4f}" for v in raw).encode()
    signal = load_upload(text, "bad.csv", "A", fs, 1750.0)
    result = execute({signal.signal_id: signal}, "diagnose_bearing", {"signal_id": signal.signal_id}, tmp_path).result
    assert result["code"] == "invalid_signal"


def test_out_of_range_inputs_are_flagged_and_capped(tmp_path):
    slow = vibration("synthetic_out_of_range.csv", rpm=900.0)
    result = execute({slow.signal_id: slow}, "diagnose_bearing", {"signal_id": slow.signal_id}, tmp_path).result
    assert result["warnings"] and result["confidence"] == "low"

    other = vibration(bearing=SKF_6203)
    result = execute({other.signal_id: other}, "diagnose_bearing", {"signal_id": other.signal_id}, tmp_path).result
    assert any("differs" in w for w in result["warnings"])
    assert result["confidence"] == "low"


# -------------------------------------------------------------------- uploads
def test_upload_needs_rate_and_speed():
    raw = (SAMPLE_DIR / "synthetic_healthy.csv").read_bytes()
    with pytest.raises(SignalError, match="Sample rate"):
        load_upload(raw, "a.csv", rpm=1750)
    with pytest.raises(SignalError, match="Shaft speed"):
        load_upload(raw, "a.csv", fs=12_000)


def test_upload_rejects_bad_shapes_and_types():
    with pytest.raises(SignalError, match="columns"):
        load_upload(b"1,2\n3,4\n", "a.csv", fs=12_000, rpm=1750)
    with pytest.raises(SignalError, match="Unsupported"):
        load_upload(b"x", "a.wav", fs=12_000, rpm=1750)
    with pytest.raises(SignalError, match="not numbers"):
        load_upload(b"1\nabc\n3\n", "a.csv", fs=12_000, rpm=1750)
    with pytest.raises(SignalError, match="MATLAB"):
        load_upload(b"junk", "a.mat", fs=12_000, rpm=1750)


def test_multi_engine_file_requires_a_choice():
    row = " ".join(["1"] * 26)
    two_engines = f"{row}\n{row.replace('1', '2', 1)}\n".encode()
    with pytest.raises(SignalError, match="Choose one"):
        load_upload(two_engines, "fleet.txt")
    assert load_upload(two_engines, "fleet.txt", unit=2).kind == "engine_history"


def test_signal_id_depends_on_content_and_metadata():
    assert vibration().signal_id == vibration().signal_id
    assert vibration().signal_id != vibration(rpm=1751.0).signal_id


# ---------------------------------------------------------------- whole agent
@pytest.fixture
def no_keys(monkeypatch):
    for key in llm_module.KEY_NAMES.values():
        monkeypatch.delenv(key, raising=False)


def test_mock_bearing_ticket_end_to_end_with_no_keys(no_keys, tmp_path):
    llm = LLMClient(mock=True)
    result = TicketAgent(llm, tmp_path).run(vibration())
    ticket = result.ticket
    assert ticket.status == "VERIFIED" and llm.api_calls == 0
    assert result.tools_used() == ["diagnose_bearing", "plot_envelope_spectrum"]
    assert ticket.remaining_useful_life == RUL_NOT_AVAILABLE
    assert all(Path(p).exists() for p in ticket.evidence_plots)
    assert "Asset" in ticket.to_markdown()


def test_mock_engine_ticket_end_to_end(no_keys, tmp_path):
    result = TicketAgent(LLMClient(mock=True), tmp_path).run(engine())
    assert result.ticket.status == "VERIFIED"
    assert result.ticket.action_window.endswith("cycles")
    assert result.ticket.asset_type == "turbofan engine"


def test_trap_request_gets_no_rul_for_a_bearing(no_keys, tmp_path):
    result = TicketAgent(LLMClient(mock=True), tmp_path).run(vibration(), "What is the remaining useful life?")
    assert result.ticket.remaining_useful_life == RUL_NOT_AVAILABLE
    assert "not available" in result.ticket.finding


def test_a_fabricated_number_is_rejected_then_fixed_on_retry(tmp_path):
    signal = vibration()
    llm = Scripted([call("diagnose_bearing", signal), ticket_reply("Outer race fault with 97.1% certainty."),
                    ticket_reply("Likely outer race fault.")])
    result = TicketAgent(llm, tmp_path).run(signal)
    assert result.retries == 1
    assert result.ticket.status == "VERIFIED"
    rejection = next(m for m in llm.last_messages if m["role"] == "user" and "rejected" in m["text"])
    assert "97.1" in rejection["text"]  # the model was told what to remove


def test_a_persistent_fabrication_ends_unverified_and_visible(tmp_path):
    signal = vibration()
    llm = Scripted([call("diagnose_bearing", signal), ticket_reply("Outer race fault, fails in 3 days."),
                    ticket_reply("Outer race fault, fails in 3 days.")])
    ticket = TicketAgent(llm, tmp_path).run(signal).ticket
    assert ticket.status == "UNVERIFIED"
    assert "UNVERIFIED" in ticket.to_markdown()
    assert ticket.urgency in ("high", "medium", "low")  # tool-set fields stay truthful


def test_a_ticket_with_no_tool_call_is_never_verified(tmp_path):
    llm = Scripted([ticket_reply("Looks fine."), ticket_reply("Looks fine.")])
    ticket = TicketAgent(llm, tmp_path).run(vibration()).ticket
    assert ticket.status == "UNVERIFIED"
    assert ticket.confidence == "none"


def test_the_finding_must_name_the_diagnosed_fault(tmp_path):
    signal = vibration()
    llm = Scripted([call("diagnose_bearing", signal), ticket_reply("All is well."), ticket_reply("All is well.")])
    assert TicketAgent(llm, tmp_path).run(signal).ticket.status == "UNVERIFIED"


@pytest.mark.parametrize("wording", ["outer race", "outer_race", "Outer-Race"])
def test_any_spelling_of_the_fault_name_is_accepted(tmp_path, wording):
    """A live model wrote the tool label outer_race, and an earlier check wrongly rejected it."""
    signal = vibration()
    llm = Scripted([call("diagnose_bearing", signal), ticket_reply(f"Likely {wording} fault.")])
    result = TicketAgent(llm, tmp_path).run(signal)
    assert result.ticket.status == "VERIFIED" and result.retries == 0


def test_llm_text_cannot_overwrite_tool_set_fields(tmp_path):
    signal = vibration()
    reply = LLMReply(json.dumps({"finding": "Outer race fault.", "recommended_action": "Inspect.",
                                 "urgency": "critical", "confidence": "certain"}))
    ticket = TicketAgent(Scripted([call("diagnose_bearing", signal), reply]), tmp_path).run(signal).ticket
    assert ticket.confidence == "high" and ticket.urgency == "high"


def test_the_step_limit_stops_a_looping_model(tmp_path):
    signal = vibration()
    llm = Scripted([call("plot_envelope_spectrum", signal)] * 10)
    result = TicketAgent(llm, tmp_path, max_steps=3).run(signal)
    assert result.llm_turns == 3
    assert result.ticket.status == "UNVERIFIED"


def test_model_never_receives_samples_or_paths(tmp_path):
    signal = vibration()
    llm = Scripted([call("plot_envelope_spectrum", signal), call("diagnose_bearing", signal),
                    ticket_reply("Outer race fault.")])
    TicketAgent(llm, tmp_path).run(signal)
    sent = json.dumps(llm.last_messages)
    assert str(tmp_path).replace("\\", "\\\\") not in sent and "envelope_spectrum.png" not in sent


# -------------------------------------------------------------------- client
@pytest.fixture
def stub_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def adapter(model, system, messages, tools):
        calls.append(messages)
        return LLMReply("answer", [])

    monkeypatch.setattr(llm_module, "_call_gemini", adapter)
    return calls


def test_responses_are_cached_to_disk(stub_provider, tmp_path):
    first = LLMClient(provider="gemini", cache_dir=tmp_path)
    assert first.chat("s", [{"role": "user", "text": "hi"}], []).cached is False
    second = LLMClient(provider="gemini", cache_dir=tmp_path)
    assert second.chat("s", [{"role": "user", "text": "hi"}], []).cached is True
    assert second.api_calls == 0 and len(stub_provider) == 1


def test_budget_stops_paid_calls_but_not_cached_ones(stub_provider, tmp_path):
    client = LLMClient(provider="gemini", cache_dir=tmp_path, max_calls=1)
    client.chat("s", [{"role": "user", "text": "one"}], [])
    assert client.chat("s", [{"role": "user", "text": "one"}], []).cached is True
    with pytest.raises(BudgetExceeded):
        client.chat("s", [{"role": "user", "text": "two"}], [])
    assert len(stub_provider) == 1


def test_mock_mode_never_touches_a_provider(stub_provider, tmp_path):
    TicketAgent(LLMClient(mock=True, cache_dir=tmp_path), tmp_path).run(vibration())
    assert stub_provider == []
