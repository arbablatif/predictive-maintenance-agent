"""Build the maintenance ticket and enforce "every number must come from a tool".

A prompt can only ask for that. This module checks it. Division of labour:

    tools write    asset, confidence, urgency, action window, RUL line, evidence figures
    the LLM writes finding and recommended_action, in plain words

Then the LLM's text is reviewed: every number must match a tool result, deadlines must be the
policy window, a bearing's remaining life can't be stated, and the text can't contradict the
diagnosis. A ticket that fails stays visible but is stamped UNVERIFIED. These are pattern checks,
so they cover common phrasings, not every possible one.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .signals import AssetSignal
from .tools import ToolOutcome

# Digits, unless they are part of a word such as "h2" or "s11".
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s*(?:%|percent\b))?", re.IGNORECASE)
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"dozen": 12, "hundred": 100, "thousand": 1000, "million": 1_000_000}
_WORDS = {**_ONES, **_TENS, **_SCALES}
_WORD = "|".join(sorted(_WORDS, key=len, reverse=True))
# One spelled-out number, which may run over several words: "twenty-one", "two hundred and five".
_SPELLED = rf"(?:{_WORD})(?:[\s-]+(?:and[\s-]+)?(?:{_WORD}))*"
_WORD_NUMBER = re.compile(rf"\b({_SPELLED})\b(\s*(?:%|percent\b))?", re.IGNORECASE)
_DIGITS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# Deadlines are held to a stricter rule than other numbers: "3 days" must not pass just
# because the digit 3 appears somewhere in a tool result.
_TIME_UNITS = r"(minutes?|hours?|shifts?|days?|weeks?|months?|years?)"
_TIME_SPAN = re.compile(rf"(?<![\w.])({_DIGITS}|an?\b|{_SPELLED})[\s-]*{_TIME_UNITS}\b", re.IGNORECASE)
# Cycles are also an ordinary measurement ("6 cycles left"), so only deadline wording is held to the window.
_CYCLE_DEADLINE = re.compile(rf"\b(?:within|in|by|before|after|every)\s+(?:the\s+)?(?:next\s+)?"
                             rf"(?:about\s+|around\s+|roughly\s+)?({_DIGITS}|an?\b|{_SPELLED})[\s-]*cycles?\b",
                             re.IGNORECASE)
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
# Deadlines with no number to check. These are invented dates just the same.
_RELATIVE_DEADLINE = re.compile(
    r"\b(?:(?:next|this|coming)\s+(?:week|month|quarter|year)|tomorrow|today|tonight|fortnight|soon|shortly|promptly|"
    r"immediately|right away|as soon as possible|asap|"
    r"(?:a\s+)?(?:few|several|couple(?:\s+of)?|some|many)\s+(?:more\s+)?(?:minutes?|hours?|shifts?|days?|weeks?|months?|"
    r"years?|cycles?)|"
    r"end\s+of\s+(?:the\s+|this\s+|next\s+)?(?:shift|day|week|month|quarter|year)|"
    r"next\s+(?:planned\s+|scheduled\s+)?(?:shift|outage|shutdown|stop|service)|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)days?|"
    rf"(?:by|before|until|in|on|after|mid|early|late|end\s+of)[\s-]+(?:{_MONTHS}))\b", re.IGNORECASE)

_CLASS_WORDS = {"inner_race": "inner race", "outer_race": "outer race", "ball": "ball"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n")
_NEGATION = r"\b(?:no|not|none|without|free of|absence of|ruled out)\b"
# Any statement about how long a bearing has left. Bearings have no run-to-failure data behind them.
_LIFE_CLAIM = re.compile(r"\b(?:(?:remaining|useful|service)\s+(?:useful\s+)?life|rul|life\s*(?:time|span)?\s+"
                         r"(?:left|remaining|expectancy)|(?:time|hours?|days?|weeks?|months?|years?|cycles?)\s+"
                         r"(?:to|until|before)\s+fail\w*|fail(?:s|ure)?\s+(?:in|within|after|by)|"
                         r"(?:will|should|expected\s+to|likely\s+to)\s+(?:last|fail|survive))\b", re.IGNORECASE)
_LIFE_REFUSAL = re.compile(r"\bnot\s+(?:available|estimated)\b|\bunavailable\b|\bcannot\s+be\s+(?:estimated|given)\b",
                           re.IGNORECASE)
_FAULT_TERM = re.compile(r"\b(?:fault|defect|damage|spall|crack)\w*\b(?!\s+frequenc)", re.IGNORECASE)
_NO_FAULT = re.compile(r"\b(?:normal|healthy|no\s+(?:\w+\s+){0,2}(?:fault|defect|damage))\b", re.IGNORECASE)
_ALL_CLEAR = re.compile(r"\b(?:healthy|all clear|good condition|no\s+(?:\w+\s+)?(?:fault|defect|damage))\b", re.IGNORECASE)
_INTERVENTION = re.compile(r"\b(?:replac\w*|overhaul\w*|shut\s*down|repair\w*)\b", re.IGNORECASE)
_SKIP_KEYS = {"signal_id", "figure", "code"}

RUL_NOT_AVAILABLE = "Not available for this asset type (bearing recordings have no run-to-failure history)."


def tool_numbers(results: list[dict]) -> set[float]:
    """Every number found in the tool results, including those inside warning strings."""
    found: set[float] = set()

    def walk(value, key=None):
        if key in _SKIP_KEYS or isinstance(value, bool):
            return
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif isinstance(value, (int, float)):
            found.add(abs(float(value)))
        elif isinstance(value, str):
            for match in _NUMBER.finditer(value):
                found.add(float(match.group(1).replace(",", "") + (match.group(2) or "")))

    for result in results:
        walk(result)
    return found


def _is_grounded(value: float, decimals: int, percent: bool, allowed: set[float]) -> bool:
    """True if a tool number, rounded to the precision written, gives this value."""
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    for number in allowed:
        candidates = (number, number * 100) if percent else (number,)
        if any(abs(c - value) <= tolerance for c in candidates):
            return True
    return False


def _spelled_value(text: str) -> float:
    """Read a spelled-out number such as "two hundred and five" as 205."""
    total = current = 0
    for word in re.split(r"[\s-]+", text.lower()):
        if word in _SCALES:
            current = max(current, 1) * _SCALES[word]
            if _SCALES[word] >= 1000:
                total, current = total + current, 0
        elif word != "and":
            current += _WORDS[word]
    return float(total + current)


def _amount(text: str) -> float:
    """The value of a number written as digits, as words, or as "a" or "an"."""
    text = text.lower()
    if text in ("a", "an"):
        return 1.0
    return float(text.replace(",", "")) if text[0].isdigit() else _spelled_value(text)


def ungrounded_numbers(text: str, results: list[dict], allowed_days: int | None = None,
                       allowed_cycles: int | None = None) -> list[str]:
    """Return every number in `text` that no tool result supports."""
    allowed = tool_numbers(results)
    problems: list[str] = []
    text = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", text)  # read ".93" as "0.93" so it cannot slip through

    for match in _TIME_SPAN.finditer(text):
        is_day = match.group(2).lower().startswith("day")
        if not (is_day and allowed_days is not None and _amount(match.group(1)) == allowed_days):
            problems.append(match.group(0))
    for match in _CYCLE_DEADLINE.finditer(text):
        if allowed_cycles is None or _amount(match.group(1)) != allowed_cycles:
            problems.append(match.group(0))
    problems += [match.group(0) for match in _RELATIVE_DEADLINE.finditer(text)]

    # Spans already judged above are removed so they are not counted twice.
    remainder = _CYCLE_DEADLINE.sub(" ", _TIME_SPAN.sub(" ", text))
    for match in _NUMBER.finditer(remainder):
        fraction = match.group(2) or ""
        value = float(match.group(1).replace(",", "") + fraction)
        if not _is_grounded(value, max(len(fraction) - 1, 0), bool(match.group(3)), allowed):
            problems.append(match.group(0).strip())
    for match in _WORD_NUMBER.finditer(remainder):
        if match.group(1).lower() == "one":
            continue  # "inspect one bearing" is ordinary English, not a measurement
        if not _is_grounded(_spelled_value(match.group(1)), 0, bool(match.group(2)), allowed):
            problems.append(match.group(0).strip())
    return problems


def _has_number(text: str) -> bool:
    words = (m.group(1).lower() for m in _WORD_NUMBER.finditer(text))
    return bool(_NUMBER.search(text)) or any(word != "one" for word in words)


def bearing_life_claims(text: str) -> list[str]:
    """Sentences that state a bearing's remaining life. Saying it is not available is the only
    allowed form, because no number or date here could have data behind it."""
    claims = []
    for sentence in _SENTENCE_END.split(text):
        match = _LIFE_CLAIM.search(sentence)
        if match is None:
            continue
        # Start at the clause holding the claim: a speed quoted earlier in the sentence is not part of it.
        clause = re.split(r",\s|;\s|\sand\s", sentence[:match.start()])[-1] + sentence[match.start():]
        if _has_number(clause) or not _LIFE_REFUSAL.search(clause):
            claims.append(sentence.strip())
    return claims


def contradictions(result: dict, finding: str, action: str) -> list[str]:
    """Places where the text says the opposite of the diagnosis tool."""
    predicted = result["predicted_class"]
    if result["envelope_check"] == "contradicts":
        return []  # the tools disagree with each other, and the model is told to report that plainly
    sentences = _SENTENCE_END.split(re.sub(r"[-_]", " ", finding))
    if predicted == "normal":
        issues = []
        if any(_FAULT_TERM.search(s) and not re.search(_NEGATION, s, re.IGNORECASE) for s in sentences):
            issues.append("The diagnosis is normal, so the finding must not report a fault.")
        if not _NO_FAULT.search(finding):
            issues.append("The finding must say that the bearing was diagnosed as normal.")
        if _INTERVENTION.search(action):
            issues.append("The diagnosis is normal, so the action must not call for repair or replacement.")
        return issues

    issues = []
    fault = _CLASS_WORDS[predicted]
    denial = (rf"{_NEGATION}[^.,;]{{0,40}}\b{fault}\b|\b{fault}\b[^.,;]{{0,40}}"
              r"\b(?:(?:not|never)\s+(?:found|detected|present|seen)|ruled out|absent)\b")
    if re.search(denial, " ".join(sentences), re.IGNORECASE) or _ALL_CLEAR.search(finding):
        issues.append(f"The finding must not deny the diagnosed fault: {fault}.")
    for name in set(_CLASS_WORDS.values()) - {fault}:
        pattern = rf"\b{name}\s+(?:fault|defect|damage)"
        if any(re.search(pattern, s, re.IGNORECASE) and not re.search(_NEGATION, s, re.IGNORECASE) for s in sentences):
            issues.append(f"The finding names a {name} fault, but the diagnosis is {fault}.")
    return issues


def parse_reply(text: str) -> dict | None:
    """Pull the JSON ticket fields out of the model's reply. None if they are not usable."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    fields = {k: data.get(k) for k in ("finding", "recommended_action")}
    return fields if all(isinstance(v, str) and v.strip() for v in fields.values()) else None


def primary_outcome(trace: list[ToolOutcome]) -> ToolOutcome | None:
    """The successful tool call that the ticket's confidence and urgency come from."""
    return next((t for t in trace if t.ok and t.name in ("diagnose_bearing", "estimate_rul")), None)


def review(fields: dict | None, trace: list[ToolOutcome]) -> list[str]:
    """Everything wrong with a draft, as sentences the model can act on. Empty means it passed."""
    primary = primary_outcome(trace)
    if primary is None:
        return ["No diagnosis or RUL tool has succeeded. Call the tool that fits the signal kind."]
    if fields is None:
        return ['Reply with only a JSON object holding the string fields "finding" and "recommended_action".']
    issues = []
    policy = primary.result["policy"]
    text = f"{fields['finding']}\n{fields['recommended_action']}"
    bad = ungrounded_numbers(text, [t.result for t in trace], policy.get("action_window_days"),
                             policy.get("action_window_cycles"))
    if bad:
        issues.append(f"These numbers are not in any tool result: {', '.join(bad)}. Remove them or use the "
                      "exact tool value. Deadlines must be the policy action window.")
    if primary.name == "diagnose_bearing":
        claims = bearing_life_claims(text)
        if claims:
            issues.append(f"Remaining life cannot be stated for a bearing: {' '.join(claims)} Say only that "
                          "remaining useful life is not available for this asset type.")
        issues += contradictions(primary.result, fields["finding"], fields["recommended_action"])
    predicted = primary.result.get("predicted_class")
    if predicted in _CLASS_WORDS:
        # Models often echo the tool's own label, such as outer_race, which names the fault just as well.
        if _CLASS_WORDS[predicted] not in re.sub(r"[-_]", " ", fields["finding"].lower()):
            issues.append(f"The finding must name the diagnosed fault: {_CLASS_WORDS[predicted]}.")
    return issues


@dataclass
class Ticket:
    ticket_id: str
    asset: str
    asset_type: str
    finding: str
    evidence: list[str]
    evidence_plots: list[str]
    confidence: str
    recommended_action: str
    urgency: str
    action_window: str
    remaining_useful_life: str
    status: str                       # VERIFIED or UNVERIFIED
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_note: str = ("Tool outputs are model estimates from public benchmark data. "
                       "An engineer should confirm before any intervention.")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# Maintenance ticket {self.ticket_id}", ""]
        if self.status != "VERIFIED":
            lines += [f"> **{self.status}: do not act on this ticket without review.**", ""]
            lines += [f"> {issue}" for issue in self.issues] + [""]
        lines += ["| Field | Value |", "|---|---|",
                  f"| Asset | {self.asset} ({self.asset_type}) |",
                  f"| Urgency | **{self.urgency}** |",
                  f"| Act within | {self.action_window} |",
                  f"| Confidence | {self.confidence} |",
                  f"| Remaining useful life | {self.remaining_useful_life} |",
                  f"| Number check | {self.status} |", "",
                  "## Finding", "", self.finding, "",
                  "## Recommended action", "", self.recommended_action, "",
                  "## Evidence", ""]
        lines += [f"- {item}" for item in self.evidence]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        if self.evidence_plots:
            lines += [""] + [f"![Evidence plot]({p})" for p in self.evidence_plots]
        lines += ["", f"*{self.model_note}*", ""]
        return "\n".join(lines)


def _bearing_evidence(result: dict) -> list[str]:
    time_domain = result["time_domain"]
    items = [f"Classifier: {result['predicted_class']}, mean probability "
             f"{result['class_probabilities'][result['predicted_class']]}, agreement across "
             f"{result['windows_analysed']} windows {result['window_vote_share']}.",
             f"Time domain: RMS {time_domain['rms_g']} g, kurtosis {time_domain['kurtosis']}, "
             f"crest factor {time_domain['crest_factor']}.",
             f"Envelope check against the classifier: {result['envelope_check']} "
             f"(a line counts as real at {result['evidence_threshold_db']} dB above the floor)."]
    for key, values in result["envelope_evidence"].items():
        items.append(f"{key.upper()} at {values['expected_hz']} Hz: mean {values['mean_db']} dB above the floor.")
    return items


def _engine_evidence(result: dict) -> list[str]:
    return [f"{result['cycles_observed']} cycles of sensor history analysed.",
            f"Point estimate {result['rul_cycles']} cycles, {result['interval_level_percent']} % interval "
            f"{result['interval_low_cycles']} to {result['interval_high_cycles']} cycles."]


def _rul_line(result: dict) -> str:
    if result["saturated"]:
        return (f"At least about {result['interval_low_cycles']} cycles. The model cannot see further than about "
                f"{result['label_cap_cycles']} cycles, so this means healthy, not a precise figure.")
    return (f"{result['rul_cycles']} cycles ({result['interval_level_percent']} % interval "
            f"{result['interval_low_cycles']} to {result['interval_high_cycles']}).")


def build_ticket(signal: AssetSignal, fields: dict | None, trace: list[ToolOutcome], issues: list[str]) -> Ticket:
    primary = primary_outcome(trace)
    verified = primary is not None and fields is not None and not issues
    fields = fields or {"finding": "No usable finding was produced.",
                        "recommended_action": "Engineer review required."}
    common = dict(ticket_id=f"PM-{signal.signal_id[:8].upper()}", asset=signal.asset,
                  finding=fields["finding"], recommended_action=fields["recommended_action"],
                  evidence_plots=[str(p) for t in trace for p in t.figures],
                  status="VERIFIED" if verified else "UNVERIFIED", issues=issues)
    if primary is None:
        return Ticket(asset_type=signal.kind, evidence=[], confidence="none", urgency="unknown",
                      action_window="unknown", remaining_useful_life="Not estimated.", **common)
    result, policy = primary.result, primary.result["policy"]
    if primary.name == "diagnose_bearing":
        return Ticket(asset_type=f"bearing, {result['bearing']}", evidence=_bearing_evidence(result),
                      confidence=result["confidence"], urgency=policy["urgency"],
                      action_window=f"{policy['action_window_days']} days", warnings=result["warnings"],
                      remaining_useful_life=RUL_NOT_AVAILABLE, **common)
    return Ticket(asset_type="turbofan engine", evidence=_engine_evidence(result), confidence=result["confidence"],
                  urgency=policy["urgency"], action_window=f"{policy['action_window_cycles']} cycles",
                  warnings=result["warnings"], remaining_useful_life=_rul_line(result), **common)
