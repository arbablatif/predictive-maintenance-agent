"""Streamlit UI: pick or upload a signal, watch the tools run, read the ticket.

    streamlit run app.py

Mock mode is on by default, so nothing spends API credit unless you turn it off.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pm_agent.agent import AgentResult, TicketAgent
from pm_agent.bearing_physics import SKF_6205
from pm_agent.config import CMAPSS_DIR, CWRU_DIR, FIGURE_DIR, RECORDINGS, RESULTS_DIR, SAMPLE_DIR
from pm_agent.llm import DEFAULT_MODELS, LLMClient, LLMError, default_provider
from pm_agent.signals import BEARINGS, AssetSignal, SignalError, asset_label, load_upload

DEFAULT_REQUEST = "Write a maintenance ticket for this signal."
URGENCY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "routine": "🟢", "unknown": "⚪"}


def demo_files() -> dict[str, dict]:
    """Synthetic samples, plus real held-out recordings when data/raw is present."""
    truth = json.loads((SAMPLE_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    options = {f"Synthetic: {name}": {"path": SAMPLE_DIR / name, "fs": meta.get("fs"), "rpm": meta.get("rpm")}
               for name, meta in truth.items() if not name.startswith("_")}
    for rec in RECORDINGS:
        path = CWRU_DIR / f"{rec.file_id}.mat"
        if rec.load_hp == 3 and path.exists():
            options[f"Real, unseen load: {rec.name}"] = {"path": path}
    test_file = CMAPSS_DIR / "test_FD001.txt"
    if test_file.exists():
        for unit in (34, 81, 100):
            options[f"Real, NASA test engine {unit}"] = {"path": test_file, "unit": unit}
    return options


def choose_signal() -> AssetSignal | None:
    asset = asset_label(st.text_input("Asset name", "Line 3 drive motor"))
    source = st.radio("Signal", ("Demo file", "Upload"), horizontal=True)
    if source == "Demo file":
        options = demo_files()
        item = options[st.selectbox("Demo file", list(options), key="demo_file")]
        return load_upload(item["path"].read_bytes(), item["path"].name, asset, item.get("fs"), item.get("rpm"),
                           item.get("unit"))

    upload = st.file_uploader("Vibration (.mat, one-column .csv) or engine history (C-MAPSS .txt)",
                              type=["mat", "csv", "txt"])
    columns = st.columns(4)
    fs = columns[0].number_input("Sample rate (Hz)", 0.0, value=0.0, step=1000.0,
                                 help="0 means use the known rate of a CWRU file. Required for anything else.")
    rpm = columns[1].number_input("Shaft speed (rpm)", 0.0, value=0.0, step=10.0, help="0 means read it from the file.")
    unit = columns[2].number_input("Engine number", 0, value=0, help="Only for a file holding several engines.")
    bearing = BEARINGS[columns[3].selectbox("Bearing", list(BEARINGS))]
    if upload is None:
        return None
    return load_upload(upload.getvalue(), upload.name, asset, fs or None, rpm or None, int(unit) or None, bearing)


def show_ticket(result: AgentResult) -> None:
    ticket = result.ticket
    if ticket.status != "VERIFIED":
        st.error(f"{ticket.status}: do not act on this ticket without review. " + " ".join(ticket.issues))
    st.subheader(f"Ticket {ticket.ticket_id}")
    st.caption(f"{ticket.asset} ({ticket.asset_type})")
    cards = st.columns(4)
    cards[0].metric("Urgency", f"{URGENCY_ICON.get(ticket.urgency, '')} {ticket.urgency}")
    cards[1].metric("Act within", ticket.action_window)
    cards[2].metric("Confidence", ticket.confidence)
    cards[3].metric("Number check", ticket.status)

    st.markdown(f"**Finding.** {ticket.finding}")
    st.markdown(f"**Recommended action.** {ticket.recommended_action}")
    st.markdown(f"**Remaining useful life.** {ticket.remaining_useful_life}")
    for warning in ticket.warnings:
        st.warning(warning)
    with st.expander("Evidence", expanded=True):
        for item in ticket.evidence:
            st.markdown(f"- {item}")
        for plot in ticket.evidence_plots:
            st.image(plot)
    downloads = st.columns(2)
    downloads[0].download_button("Download Markdown", ticket.to_markdown(), f"{ticket.ticket_id}.md")
    downloads[1].download_button("Download JSON", json.dumps(ticket.to_dict(), indent=2), f"{ticket.ticket_id}.json")

    st.caption(f"{result.llm_turns} model turns, {result.retries} retries, {result.api_calls} paid API calls")
    for step in result.trace:
        with st.expander(f"Tool: {step.name}"):
            st.json(step.result)


def ticket_tab(mock: bool, provider: str) -> None:
    try:
        signal = choose_signal()
    except SignalError as exc:
        st.error(str(exc))
        signal = None
    request = st.text_input("Request", DEFAULT_REQUEST)
    # A ticket is shown only for the exact inputs that produced it, so an old one can't pass for the new asset.
    inputs = signal and (signal.signal_id, signal.asset, request, mock, provider)
    if st.button("Write ticket", type="primary", disabled=signal is None):
        st.session_state.pop("result", None)
        try:
            with st.spinner("Running tools"):
                llm = LLMClient(provider=provider, mock=mock)
                st.session_state["result"] = inputs, TicketAgent(llm).run(signal, request)
        except LLMError as exc:
            st.error(str(exc))
    saved_inputs, result = st.session_state.get("result", (None, None))
    if result is not None and saved_inputs == inputs:
        show_ticket(result)


def results_tab() -> None:
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        st.info("Run python train.py to produce results.")
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    part_a, part_b = metrics["part_a"], metrics["part_b"]

    st.subheader("Fault diagnosis")
    st.image(str(FIGURE_DIR / "split_comparison.png"))
    st.dataframe(pd.DataFrame([
        {"Test": "Random windows (leaky)", **part_a["random_split_leaky"]},
        {"Test": "Unseen load", **part_a["load_split"]["pooled"]},
        {"Test": "Unseen fault size", **part_a["size_split"]["pooled"]}]).round(3), hide_index=True)
    columns = st.columns(2)
    columns[0].image(str(FIGURE_DIR / "confusion_load_split.png"))
    columns[1].image(str(FIGURE_DIR / "confusion_size_split.png"))

    st.subheader("Remaining useful life")
    st.dataframe(pd.DataFrame(part_b["models"]).T.round(2))
    st.image(str(FIGURE_DIR / "rul_test_engines.png"))

    for mode in ("live", "mock"):
        report = RESULTS_DIR / f"eval_{mode}.md"
        if report.exists():
            st.subheader(f"Agent evaluation ({mode})")
            st.markdown(report.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Predictive Maintenance Agent", layout="wide")
    st.title("Predictive maintenance agent")
    st.caption("Signal in, maintenance ticket out. Every number in a ticket is checked against tool results.")
    with st.sidebar:
        mock = st.toggle("Mock mode (no API calls)", value=True)
        providers = list(DEFAULT_MODELS)
        configured = default_provider()
        provider = st.selectbox("LLM provider", providers, disabled=mock,
                                index=providers.index(configured) if configured in providers else 0)
        if not mock:
            st.warning("Live mode spends API credit. Replies are cached to disk.")
        st.caption(f"Default bearing: {SKF_6205.name}")
    tickets, results = st.tabs(["Ticket", "Results"])
    with tickets:
        ticket_tab(mock, provider)
    with results:
        results_tab()


main()
