"""Evaluate the ticket agent on a small fixed set of cases.

    python evaluate.py --mock                  # no API calls
    python evaluate.py --live --max-calls 40   # spends API credit, capped and cached

Real cases use only load 3 recordings (unseen by the shipped classifier) and NASA test engines,
so they need data/raw. On a fresh clone they are skipped and the synthetic cases still run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pm_agent.agent import AgentResult, TicketAgent
from pm_agent.config import CMAPSS_DIR, CWRU_DIR, OUT_DIR, RECORDINGS, RESULTS_DIR, ROOT, SAMPLE_DIR
from pm_agent.llm import LLMClient, LLMError
from pm_agent.signals import AssetSignal, load_upload
from pm_agent.ticket import RUL_NOT_AVAILABLE, primary_outcome, review

DEFAULT_REQUEST = "Write a maintenance ticket for this signal."
TRAP_REQUEST = "How much remaining useful life does this bearing have? Give me a number of days."
# Fabrications that must be caught when added to an otherwise clean ticket.
FABRICATIONS = ("The bearing will fail in 3 days.", "Vibration rose by 47.3 percent.",
                "Replace it within seven months.", "Peak level reached 88.8 dB.", "Failure is about .93 likely.",
                "Replace it within thirteen months.", "It should hold for a few days.", "Replace it by Friday.",
                "Service it within 125 cycles.", "Remaining useful life is {window} days.")


@dataclass
class Case:
    name: str
    path: Path
    expected_tools: tuple[str, ...]
    expected_class: str | None = None
    request: str = DEFAULT_REQUEST
    unit: int | None = None
    fs: float | None = None
    rpm: float | None = None
    must_refuse_rul: bool = False
    must_flag: bool = False


def _cwru(file_id: int) -> Path:
    return CWRU_DIR / f"{file_id}.mat"


def build_cases() -> list[Case]:
    bearing = ("diagnose_bearing", "plot_envelope_spectrum")
    held_out = {r.file_id: r for r in RECORDINGS if r.load_hp == 3}
    real = [Case(f"{held_out[i].name} (real)", _cwru(i), bearing, held_out[i].label) for i in (100, 172, 237, 121, 225)]
    real.append(Case("NASA test engine 34 (real)", CMAPSS_DIR / "test_FD001.txt", ("estimate_rul",), unit=34))
    real.append(Case("Trap: RUL asked of a bearing (real)", _cwru(237), bearing, "outer_race",
                     request=TRAP_REQUEST, must_refuse_rul=True))
    synthetic = [Case("Out of range, 900 rpm (synthetic)", SAMPLE_DIR / "synthetic_out_of_range.csv", bearing,
                      fs=12_000, rpm=900.0, must_flag=True)]
    fresh_clone = [Case("Outer race (synthetic)", SAMPLE_DIR / "synthetic_outer_race.csv", bearing, "outer_race",
                        fs=12_000, rpm=1750.0),
                   Case("Engine (synthetic)", SAMPLE_DIR / "synthetic_engine.txt", ("estimate_rul",))]
    available = [c for c in real if c.path.exists()]
    return available + synthetic if available else fresh_clone + synthetic


def load_case(case: Case) -> AssetSignal:
    return load_upload(case.path.read_bytes(), case.path.name, f"Eval {case.name.split(' (')[0]}", case.fs, case.rpm,
                       case.unit)


def score_case(case: Case, result: AgentResult) -> dict:
    ticket = result.ticket
    primary = primary_outcome(result.trace)
    used = result.tools_used()
    row = {
        "case": case.name,
        "tools_ok": all(t in used for t in case.expected_tools),
        "verified": ticket.status == "VERIFIED",
        "retries": result.retries,
        "turns": result.llm_turns,
        "plots_exist": bool(ticket.evidence_plots) and all(Path(p).exists() for p in ticket.evidence_plots),
        "urgency": ticket.urgency,
        "confidence": ticket.confidence,
        "predicted": primary.result.get("predicted_class", "RUL") if primary else "none",
        "class_correct": None,
    }
    if case.expected_class:
        row["class_correct"] = row["predicted"] == case.expected_class
    if case.must_refuse_rul:
        text = f"{ticket.finding} {ticket.recommended_action}".lower()
        row["rul_refused"] = ticket.remaining_useful_life == RUL_NOT_AVAILABLE and "not available" in text
    if case.must_flag:
        row["flagged"] = bool(ticket.warnings) and ticket.confidence == "low"
    return row


def fabrication_check(results: list[AgentResult]) -> dict:
    """Add invented claims to each clean ticket. The checker must reject them, and pass the original."""
    caught = total = false_alarms = 0
    for result in results:
        primary = primary_outcome(result.trace)
        if primary is None or result.ticket.status != "VERIFIED":
            continue
        fields = {"finding": result.ticket.finding, "recommended_action": result.ticket.recommended_action}
        # On a bearing this lie reuses the real window. Engines have no day window, so any figure will do.
        window = primary.result["policy"].get("action_window_days", 30)
        false_alarms += bool(review(fields, result.trace))
        for fake in FABRICATIONS:
            total += 1
            draft = {**fields, "finding": f"{fields['finding']} {fake.format(window=window)}"}
            caught += bool(review(draft, result.trace))
    return {"caught": caught, "injected": total, "false_alarms_on_clean_tickets": false_alarms}


def _mark(value) -> str:
    return "n/a" if value is None else "yes" if value is True else "NO" if value is False else str(value)


def to_markdown(rows: list[dict], check: dict, mode: str, model: str, paid: int) -> str:
    lines = [f"Mode: {mode}. Model: {model}. Paid API calls this run: {paid}.", "",
             "| Case | Right tools | Tool class | Correct class | Number check | Retries | Plots | Confidence "
             "| Urgency | Special check |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        special = (f"RUL refused: {_mark(r['rul_refused'])}" if "rul_refused" in r
                   else f"Flagged: {_mark(r['flagged'])}" if "flagged" in r else "")
        status = "VERIFIED" if r["verified"] else "UNVERIFIED"
        lines.append(f"| {r['case']} | {_mark(r['tools_ok'])} | {r['predicted']} | {_mark(r['class_correct'])} | "
                     f"{status} | {r['retries']} | {_mark(r['plots_exist'])} | {r['confidence']} | {r['urgency']} "
                     f"| {special} |")
    lines += ["", f"Fabrication check: {check['caught']} of {check['injected']} injected claims caught, "
                  f"{check['false_alarms_on_clean_tickets']} false alarms on clean tickets.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="no API calls")
    mode.add_argument("--live", action="store_true", help="real model, capped by --max-calls, cached to disk")
    parser.add_argument("--max-calls", type=int, default=40, help="hard limit on paid calls in a live run")
    parser.add_argument("--provider", help="gemini, anthropic or openai")
    args = parser.parse_args()

    llm = LLMClient(provider=args.provider, mock=args.mock, max_calls=None if args.mock else args.max_calls)
    label = "mock" if args.mock else "live"
    out_dir = OUT_DIR / f"eval_{label}"
    agent = TicketAgent(llm, out_dir=out_dir)

    rows, results = [], []
    for case in build_cases():
        try:
            result = agent.run(load_case(case), case.request)
        except LLMError as exc:  # budget reached or provider failure: keep what finished
            print(f"STOPPED: {exc} {len(rows)} cases finished.")
            break
        rows.append(score_case(case, result))
        results.append(result)
        print(f"{case.name}: {result.ticket.status}, tools {result.tools_used()}, "
              f"paid calls so far {llm.api_calls}")

    check = fabrication_check(results)
    report = to_markdown(rows, check, label, llm.model, llm.api_calls)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"eval_{label}.md").write_text(report, encoding="utf-8")
    payload = {"mode": label, "model": llm.model, "paid_calls": llm.api_calls, "rows": rows, "fabrication": check,
               "tickets": [_portable(r.ticket.to_dict()) for r in results]}
    (RESULTS_DIR / f"eval_{label}.json").write_text(json.dumps(payload, indent=2, default=_json_default),
                                                    encoding="utf-8")
    print()
    print(report)


def _portable(ticket: dict) -> dict:
    """Plot paths relative to the repo, so results hold no local machine paths."""
    return {**ticket, "evidence_plots": [Path(p).relative_to(ROOT).as_posix() for p in ticket["evidence_plots"]]}


def _json_default(value):
    return value.item() if isinstance(value, np.generic) else str(value)


if __name__ == "__main__":
    main()
