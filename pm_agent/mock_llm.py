"""A rule-based stand-in for the LLM, used by --mock.

It emits tool calls one turn at a time in the format a real provider returns, then writes
the ticket from a template. The agent loop, tools, plots and number check all run for real
with no API. It proves the plumbing, not language understanding: live-model behaviour is
measured separately (see README).
"""
from __future__ import annotations

import json
import re

CONTEXT_MARKER = "Signal: "
REQUEST_MARKER = "Request: "


def build_user_message(signal: dict, request: str) -> str:
    """The text sent to any model (real or mock) to start a ticket."""
    return f"{CONTEXT_MARKER}{json.dumps(signal)}\n\n{REQUEST_MARKER}{request}"


def _parse_user_message(text: str) -> tuple[dict, str]:
    head, _, request = text.partition(f"\n\n{REQUEST_MARKER}")
    return json.loads(head.removeprefix(CONTEXT_MARKER)), request


def _plan(signal: dict, request: str) -> list[str]:
    if signal["kind"] == "engine_history":
        return ["estimate_rul"]
    plan = ["diagnose_bearing", "plot_envelope_spectrum"]
    if re.search(r"\brul\b|remaining|how long", request, re.IGNORECASE):
        plan.append("estimate_rul")  # refused by the tool, which the ticket then has to respect
    return plan


def _bearing_ticket(r: dict, rul_requested: bool) -> dict:
    name = r["predicted_class"].replace("_", " ")
    policy = r["policy"]
    subject = "No bearing fault was detected" if r["predicted_class"] == "normal" else f"Likely {name} fault"
    finding = (f"{subject}, with class probability {r['class_probabilities'][r['predicted_class']]}. "
               f"The envelope check result is {r['envelope_check']}.")
    if r["warnings"]:
        finding += " Warning: " + " ".join(r["warnings"])
    if rul_requested:
        finding += " Remaining useful life is not available for this asset type."
    action = f"Follow {policy['action_code'].replace('_', ' ')} within {policy['action_window_days']} days."
    return {"finding": finding, "recommended_action": action}


def _engine_ticket(r: dict) -> dict:
    finding = (f"Estimated remaining life is {r['rul_cycles']} cycles, with an interval of "
               f"{r['interval_low_cycles']} to {r['interval_high_cycles']} cycles.")
    if r["warnings"]:
        finding += " Warning: " + " ".join(r["warnings"])
    action = (f"Follow {r['policy']['action_code'].replace('_', ' ')} before "
              f"{r['policy']['action_window_cycles']} cycles have passed.")
    return {"finding": finding, "recommended_action": action}


def _final(results: list[dict]) -> str:
    by_name = {r["name"]: r["result"] for r in results}
    diagnosis, engine = by_name.get("diagnose_bearing"), by_name.get("estimate_rul")
    if diagnosis and "error" not in diagnosis:
        return json.dumps(_bearing_ticket(diagnosis, "estimate_rul" in by_name))
    if engine and "error" not in engine:
        return json.dumps(_engine_ticket(engine))
    reason = next((r["result"]["error"] for r in results if "error" in r["result"]), "No tool succeeded.")
    return json.dumps({"finding": f"The signal could not be analysed: {reason}",
                       "recommended_action": "Check the upload and measure again."})


def mock_reply(messages: list[dict]):
    """Next turn of the plan. Stateless: progress is read from the message history."""
    from .llm import LLMReply  # imported here because llm.py imports this module

    signal, request = _parse_user_message(messages[0]["text"])
    results = [r for m in messages if m["role"] == "tool" for r in m["results"]]
    plan = _plan(signal, request)
    if len(results) >= len(plan):
        return LLMReply(_final(results), [], cached=True)
    call = {"id": f"mock_{len(results) + 1}", "name": plan[len(results)], "args": {"signal_id": signal["signal_id"]}}
    return LLMReply("", [call], cached=True)
