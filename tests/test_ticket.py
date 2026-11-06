"""The number check is the project's key safety claim, so it gets adversarial tests."""
import pytest

from pm_agent.ticket import bearing_life_claims, contradictions, parse_reply, tool_numbers, ungrounded_numbers

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
    "Act within fourteen days.",          # the policy window, spelled out
    "Across nineteen windows.",           # a real tool number, spelled out
    "Plan a second measurement.",         # an ordinal, not a quantity
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
    "Thirteen sensors moved.",          # every word from thirteen to nineteen is covered
    "Sixteen lines were found.",
    "Seventeen sensors moved.",
    "Eighteen lines were found.",
    "Twenty-four sensors moved.",       # compound words are one number, not 20 and 4
    "Two hundred and five sensors.",
    "A dozen sensors moved.",
    "Ninety-nine percent likely.",
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
    "Replace within thirteen months.",
    "Act within nineteen days.",  # nineteen IS a tool number, but not a deadline
    "Act within twenty-one days.",
    "Fails in a few days.",       # vague deadlines have no number to check
    "Good for several more weeks.",
    "Replace by Friday.",
    "Replace by the end of the month.",
    "Replace before March.",
    "Replace at the next planned outage.",
    "Replace soon.",
    "Within 30 minutes.",
])
def test_deadlines_must_be_the_policy_window(text):
    assert bad(text)


def test_the_policy_window_is_allowed_in_digits_or_words():
    assert bad("Within 7 days.", days=7) == []
    assert bad("Within seven days.", days=7) == []
    assert bad("Within 14-day limits.") == []


def test_any_deadline_fails_when_the_policy_has_no_day_window():
    assert bad("Within 14 days.", days=None)


CYCLES = [{"rul_cycles": 6, "interval_low_cycles": 4, "interval_high_cycles": 7, "label_cap_cycles": 125,
           "cycles_observed": 203, "policy": {"action_window_cycles": 4}}]


@pytest.mark.parametrize("text", ["Service within 125 cycles.", "Service in the next 6 cycles.", "Before 203 cycles.",
                                  "Service within seven cycles."])
def test_cycle_deadlines_must_be_the_policy_window_even_if_the_number_is_real(text):
    assert ungrounded_numbers(text, CYCLES, None, 4)


@pytest.mark.parametrize("text", ["Plan maintenance within 4 cycles.", "Act before 4 cycles have passed.",
                                  "Estimated life of 6 cycles, interval 4 to 7 cycles.", "203 cycles observed."])
def test_real_cycle_counts_and_the_window_pass(text):
    assert ungrounded_numbers(text, CYCLES, None, 4) == []


def test_a_cycle_deadline_fails_when_the_policy_has_no_cycle_window():
    assert ungrounded_numbers("Service within 4 cycles.", CYCLES, 7, None)


@pytest.mark.parametrize("text", ["Remaining useful life is 7 days.", "RUL: 7.", "Useful life is short.",
                                  "The bearing will last 7 days.", "It will fail within the policy window.",
                                  "Remaining useful life is not available, though about seven weeks is likely.",
                                  "Roughly 90 days to failure."])
def test_any_stated_bearing_life_is_a_claim(text):
    assert bearing_life_claims(text)


@pytest.mark.parametrize("text", [
    "Remaining useful life is not available for this asset type.",
    "The shaft speed of 900 rpm is outside 1720 to 1800 rpm, and remaining useful life is not available.",
    "Plan a bearing inspection within 7 days.",
    "The fault is at 103.1 Hz.",
])
def test_a_refusal_and_unrelated_numbers_are_not_life_claims(text):
    assert bearing_life_claims(text) == []


def outcome(predicted, check="supports"):
    return {"predicted_class": predicted, "envelope_check": check}


@pytest.mark.parametrize("predicted,finding,action", [
    ("normal", "Severe inner race fault found.", "Inspect."),
    ("normal", "Damage is visible.", "Inspect."),
    ("normal", "The result is unclear.", "Inspect."),         # never says the diagnosis
    ("normal", "The bearing is normal.", "Replace the bearing."),
    ("outer_race", "No outer race fault was found.", "Inspect."),
    ("outer_race", "An outer race fault was not detected.", "Inspect."),
    ("outer_race", "Outer race label, but the bearing is healthy.", "Inspect."),
    ("outer_race", "Outer race result. An inner race fault was found.", "Inspect."),
    ("ball", "Ball result, no fault here.", "Inspect."),
])
def test_text_cannot_contradict_the_diagnosis(predicted, finding, action):
    assert contradictions(outcome(predicted), finding, action)


@pytest.mark.parametrize("predicted,finding", [
    ("normal", "The bearing is in normal condition with a high confidence level."),
    ("normal", "No bearing fault was detected."),
    ("normal", "Normal. No peak stands out at the fault frequencies."),
    ("outer_race", "The analysis indicates an outer_race fault with high confidence."),
    ("outer_race", "Likely outer race fault, not an inner race fault."),
    ("ball", "A ball fault is indicated, although the envelope check is inconclusive."),
    ("ball", "A ball fault. The ball spin frequency shows little."),
])
def test_honest_findings_are_not_contradictions(predicted, finding):
    assert contradictions(outcome(predicted), finding, "Continue routine monitoring.") == []


def test_a_normal_result_may_report_the_envelope_disagreeing():
    finding = "Classified as normal, but the envelope shows a possible outer race fault line."
    assert contradictions(outcome("normal", "contradicts"), finding, "Engineer review.") == []


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
