"""Maintenance policy: turns tool results into urgency and an action window.

These values are PLACEHOLDERS chosen for this demo, not an industry standard. A real site's
reliability engineer would set them from failure costs and spares lead times. They live in code,
in one place, so that the LLM cannot invent a deadline.
"""
from __future__ import annotations

# (is a fault predicted, confidence) -> urgency, days to act, what to do
BEARING_POLICY = {
    (True, "high"): ("high", 7, "plan_bearing_inspection"),
    (True, "medium"): ("medium", 14, "confirm_with_second_measurement"),
    (True, "low"): ("low", 30, "engineer_review_and_remeasure"),
    (False, "high"): ("routine", 90, "continue_routine_monitoring"),
    (False, "medium"): ("routine", 90, "continue_routine_monitoring"),
    (False, "low"): ("low", 30, "engineer_review_and_remeasure"),
}

# Lower bound of the RUL interval, in cycles -> urgency. Checked in order.
RUL_POLICY = ((20, "critical"), (50, "high"), (100, "medium"))


def bearing_policy(predicted_class: str, confidence: str) -> dict:
    urgency, days, action = BEARING_POLICY[(predicted_class != "normal", confidence)]
    return {"urgency": urgency, "action_window_days": days, "action_code": action}


def rul_policy(interval_low_cycles: int, confidence: str) -> dict:
    """Act before the pessimistic end of the interval, not the point estimate."""
    urgency = next((u for limit, u in RUL_POLICY if interval_low_cycles <= limit), "routine")
    if confidence == "low" and urgency == "routine":
        urgency = "low"  # an unreliable "healthy" should not read as all clear
    return {"urgency": urgency, "action_window_cycles": interval_low_cycles,
            "action_code": "plan_engine_maintenance" if urgency != "routine" else "continue_routine_monitoring"}
