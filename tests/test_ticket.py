"""The number check is the project's key safety claim, so it gets adversarial tests."""
import pytest

from pm_agent.ticket import parse_reply, tool_numbers, ungrounded_numbers

RESULTS = [{"class_probabilities": {"outer_race": 0.934}, "window_vote_share": 1.0, "windows_analysed": 19,
            "envelope_evidence": {"bpfo": {"expected_hz": 103.1, "mean_db": 23.3}},
            "bearing": "SKF 6205-2RS JEM", "shaft_speed_rpm": 1725.0, "rul_cycles": 1250,
            "warnings": ["Shaft speed 900 rpm is outside the training rig range 1720 to 1800 rpm."],
            "signal_id": "77777abc", "policy": {"action_window_days": 14}}]


def bad(text, days=14):
    return ungrounded_numbers(text, RESULTS, days)


@pytest.mark.parametrize("text", [
    "Class probability 0.934 across 19 windows.",
    "The BPFO line at 103.1 Hz is 23.3 dB above the floor.",
    "Probability about 0.93.",            # rounded down in precision
    "Probability of 93.4%.",              # as a percentage
    "Probability of 93 percent.",
    "Speed is 1725 rpm.",
    "Speed is 1,725 rpm with a 1,250 cycle estimate.",
    "Bearing SKF 6205-2RS JEM.",
    "Speed 900 rpm is outside 1720 to 1800 rpm.",  # numbers inside a warning string
    "Act within 14 days.",
    "Act within fourteen-free wording and no digits.",
    "Sensor s11 and harmonic h2 are named, not counted.",
    "Inspect one bearing as the first step.",
    "Vote share 1.0.",
])
def test_grounded_text_passes(text):
    assert bad(text) == []


@pytest.mark.parametrize("text", [
    "Probability 0.95.",                # close but not the tool's value
    "Probability of 95%.",
    "Line at 103.4 Hz.",                # more precise than the tool, and wrong
    "Level of 88.8 dB.",
    "Failure is about .95 likely.",     # no leading zero must not slip through
    "Speed 1,999 rpm.",
    "Seventy windows were read.",       # spelled out
    "There are forty sensors.",
])
def test_invented_numbers_are_caught(text):
    assert bad(text)


@pytest.mark.parametrize("text", [
    "Act within 3 days.",        # 3 is not the policy window
    "Act within 19 days.",       # 19 IS a tool number, but not a deadline
    "Act within three days.",
    "Replace in 2 weeks.",
    "Replace within a month.",
    "Failure in one year.",
    "Fix within next month.",    # no digits, but still an invented date
    "Replace it tomorrow.",
    "Shut down immediately.",
    "Within 48 hours.",
])
def test_deadlines_must_be_the_policy_window(text):
    assert bad(text)


def test_the_policy_window_is_allowed_in_digits_or_words():
    assert bad("Within 7 days.", days=7) == []
    assert bad("Within seven days.", days=7) == []
    assert bad("Within 14-day limits.") == []


def test_any_deadline_fails_when_the_policy_has_no_day_window():
    assert bad("Within 14 days.", days=None)


def test_ids_cannot_supply_numbers():
    assert 77777.0 not in tool_numbers(RESULTS)


def test_bools_are_not_numbers():
    assert tool_numbers([{"saturated": True}]) == set()


@pytest.mark.parametrize("reply", ['{"finding": "a", "recommended_action": "b"}',
                                   '```json\n{"finding": "a", "recommended_action": "b"}\n```'])
def test_parse_reply_accepts_plain_and_fenced_json(reply):
    assert parse_reply(reply) == {"finding": "a", "recommended_action": "b"}


@pytest.mark.parametrize("reply", ["", "not json", '{"finding": "a"}', '{"finding": "", "recommended_action": "b"}',
                                   '{"finding": 3, "recommended_action": "b"}', "{broken"])
def test_parse_reply_rejects_unusable_replies(reply):
    assert parse_reply(reply) is None
