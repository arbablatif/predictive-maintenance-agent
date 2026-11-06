"""The agent loop: pick tools, run them, draft a ticket, check it, retry once if needed.

    signal + request
      -> model calls tools -> tools run -> results go back to the model
      -> model replies with the two text fields of the ticket
      -> code checks every number against the tool results
           pass: VERIFIED ticket
           fail: one retry, naming the offending numbers, then UNVERIFIED if still wrong
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import OUT_DIR
from .llm import LLMClient
from .mock_llm import build_user_message
from .signals import AssetSignal
from .ticket import Ticket, build_ticket, parse_reply, review
from .tools import TOOLS, ToolOutcome, execute

MAX_STEPS = 6
MAX_RETRIES = 1

SYSTEM_PROMPT = """You are a maintenance analyst. You write the text of a maintenance ticket from tool
results. Each user message lists one signal with its kind.

Rules:
1. Every number you write must be copied from a tool result. Never estimate, convert or invent a number.
2. For a vibration signal, call diagnose_bearing and plot_envelope_spectrum.
3. For an engine_history signal, call estimate_rul.
4. estimate_rul does not work on vibration. If someone asks for remaining life on a vibration signal, say
   in the finding that remaining useful life is not available for this asset type. Give no estimate.
5. The confidence label, urgency and deadline are set by the tools. Do not state your own. If you mention a
   deadline, use the policy action window exactly, written as digits.
6. Report warnings and an inconclusive or contradicting envelope_check plainly in the finding.
7. Match the recommended action to the policy action_code.
8. Write numbers as digits, not words.

When finished, reply with only this JSON and no tool call:
{"finding": "two or three plain sentences", "recommended_action": "one or two plain sentences"}"""


@dataclass
class AgentResult:
    ticket: Ticket
    trace: list[ToolOutcome] = field(default_factory=list)
    llm_turns: int = 0
    api_calls: int = 0     # paid calls. Cached and mock turns are not counted
    retries: int = 0

    def tools_used(self) -> list[str]:
        return [t.name for t in self.trace]


def _for_llm(result: dict) -> dict:
    """What the model sees. File paths are hidden: it cannot open them, and leaving them out
    keeps the cache key the same between sessions."""
    view = dict(result)
    if view.pop("figure", None):
        view["plot_saved_for_ticket"] = True
    return view


class TicketAgent:
    def __init__(self, llm: LLMClient, out_dir: Path = OUT_DIR, max_steps: int = MAX_STEPS):
        self.llm, self.out_dir, self.max_steps = llm, out_dir, max_steps

    def run(self, signal: AssetSignal, request: str = "Write a maintenance ticket for this signal.",
            on_tool: Callable[[ToolOutcome], None] | None = None) -> AgentResult:
        signals = {signal.signal_id: signal}
        messages: list[dict] = [{"role": "user", "text": build_user_message(signal.describe(), request)}]
        trace: list[ToolOutcome] = []
        calls_before = self.llm.api_calls
        turns = retries = 0
        fields, issues = None, ["The agent stopped before writing a ticket."]

        for _ in range(self.max_steps):
            reply = self.llm.chat(SYSTEM_PROMPT, messages, TOOLS)
            turns += 1
            messages.append({"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls})

            if reply.tool_calls:
                outputs = []
                for call in reply.tool_calls:
                    outcome = execute(signals, call["name"], call["args"], self.out_dir)
                    trace.append(outcome)
                    outputs.append({"id": call["id"], "name": call["name"], "result": _for_llm(outcome.result)})
                    if on_tool:
                        on_tool(outcome)
                messages.append({"role": "tool", "results": outputs})
                continue

            fields = parse_reply(reply.text)
            issues = review(fields, trace)
            if not issues or retries >= MAX_RETRIES:
                break
            retries += 1
            messages.append({"role": "user", "text": "Your ticket was rejected. Fix this and reply again:\n"
                                                     + "\n".join(issues)})

        return AgentResult(build_ticket(signal, fields, trace, issues), trace, turns,
                           self.llm.api_calls - calls_before, retries)
