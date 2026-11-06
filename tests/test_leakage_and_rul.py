"""No split may leak, and the RUL scoring must match hand-worked values."""
import numpy as np
import pandas as pd
import pytest

from pm_agent import data_cmapss, diagnosis, rul
from pm_agent.config import CMAPSS_DIR, RECORDINGS
from pm_agent.data_cmapss import COLUMNS, RUL_CAP, WINDOW, build_features, feature_columns
from pm_agent.features import ALL_FEATURES, FAULT_KEYS
from pm_agent.policy import bearing_policy, rul_policy

needs_data = pytest.mark.skipif(not (CMAPSS_DIR / "train_FD001.txt").exists(), reason="C-MAPSS not downloaded")


def fake_table(seed=0):
    """Two windows per recording, with the real recording structure and random features."""
    rng = np.random.default_rng(seed)
    rows = [{"recording": r.name, "file_id": r.file_id, "label": r.label, "load_hp": r.load_hp,
             "fault_size_in": r.fault_size_in, "window": w, **dict(zip(ALL_FEATURES, rng.random(len(ALL_FEATURES))))}
            for r in RECORDINGS for w in range(2)]
    return pd.DataFrame(rows)


@pytest.fixture
def spy(monkeypatch):
    """Record what each fit and predict sees, without training anything."""
    calls = {"train": [], "test": []}

    class Dummy:
        classes_ = np.array(["ball", "inner_race", "normal", "outer_race"])

        def predict_proba(self, X):
            return np.tile([0.1, 0.2, 0.3, 0.4], (len(X), 1))

    def fit(df, features=ALL_FEATURES):
        calls["train"].append(df)
        return Dummy()

    real_predict = diagnosis.predict

    def predict(model, df, features=ALL_FEATURES):
        calls["test"].append(df)
        return real_predict(model, df, features)

    monkeypatch.setattr(diagnosis, "fit_classifier", fit)
    monkeypatch.setattr(diagnosis, "predict", predict)
    return calls


def test_load_split_never_shares_a_load_or_recording(spy):
    table = fake_table()
    diagnosis.leave_one_load_out({(2000, 4000): table, (3000, 5000): table})
    assert spy["train"]
    for train, test in zip(spy["train"], spy["test"]):
        assert not set(train["load_hp"]) & set(test["load_hp"])
        assert not set(train["recording"]) & set(test["recording"])


def test_band_selection_never_sees_the_held_out_load(spy):
    table = fake_table()
    diagnosis.select_band({(2000, 4000): table}, [0, 1, 2])
    seen = pd.concat(spy["train"] + spy["test"])
    assert 3 not in set(seen["load_hp"])


def test_size_split_holds_out_every_load_of_a_fault_size(spy):
    table = fake_table()
    diagnosis.leave_one_size_out({(2000, 4000): table})
    for train, test in zip(spy["train"], spy["test"]):
        assert not set(train["recording"]) & set(test["recording"])
        faults = test[test["label"] != "normal"]
        assert faults["fault_size_in"].nunique() == 1
        assert faults["fault_size_in"].iloc[0] not in set(train["fault_size_in"])


def test_every_test_side_of_the_size_split_includes_normal():
    table = fake_table()
    for size in diagnosis.SIZES:
        assert "normal" in set(table[diagnosis._size_test_mask(table, size)]["label"])


@pytest.mark.parametrize("prob,vote,check,warnings,expected", [
    (0.99, 0.99, "supports", [], "high"),
    (0.99, 0.99, "inconclusive", [], "medium"),    # high needs the physics to agree
    (0.99, 0.99, "supports", ["odd rpm"], "low"),  # any warning caps it
    (0.99, 0.99, "contradicts", [], "low"),
    (0.70, 0.70, "supports", [], "medium"),
    (0.50, 0.50, "supports", [], "low"),
])
def test_confidence_rule(prob, vote, check, warnings, expected):
    assert diagnosis.confidence_label(prob, vote, check, warnings) == expected


def test_envelope_check():
    quiet = {k: 5.0 for k in FAULT_KEYS}
    assert diagnosis.envelope_check("normal", quiet) == "supports"
    assert diagnosis.envelope_check("normal", {**quiet, "bpfo": 20.0}) == "contradicts"
    assert diagnosis.envelope_check("outer_race", {**quiet, "bpfo": 20.0}) == "supports"
    assert diagnosis.envelope_check("ball", {**quiet, "bpfo": 20.0}) == "inconclusive"


def test_nasa_score_hand_worked():
    assert rul.nasa_score([10], [23]) == pytest.approx(np.e - 1)   # 13 early: exp(13/13) - 1
    assert rul.nasa_score([33], [23]) == pytest.approx(np.e - 1)   # 10 late: exp(10/10) - 1
    assert rul.nasa_score([50, 50], [50, 50]) == 0


def test_nasa_score_punishes_late_more_than_early():
    assert rul.nasa_score([60], [50]) > rul.nasa_score([40], [50])


def test_rmse_hand_worked():
    assert rul.rmse([3, 4], [0, 0]) == pytest.approx(np.sqrt(12.5))


def test_baseline_uses_only_engines_that_outlived_the_age():
    baseline = rul.FleetLifeBaseline().fit(np.array([100, 200, 300]))
    assert baseline.predict([150])[0] == pytest.approx(100)  # mean(200, 300) - 150
    assert baseline.predict([400])[0] == 0


def synthetic_engine(rows=60, unit=1):
    rng = np.random.default_rng(unit)
    table = pd.DataFrame(rng.normal(size=(rows, len(COLUMNS))), columns=COLUMNS)
    table["unit"], table["cycle"] = unit, np.arange(1, rows + 1)
    return table


def test_features_at_a_row_never_use_later_cycles():
    engine = synthetic_engine()
    sensors = ["s2", "s3"]
    before = build_features(engine, sensors).iloc[39]
    changed = engine.copy()
    changed.loc[40:, sensors] += 1000  # rewrite the future
    after = build_features(changed, sensors).iloc[39]
    pd.testing.assert_series_equal(before, after)


def test_cycle_count_is_not_a_model_feature():
    assert "cycle" not in feature_columns(build_features(synthetic_engine(), ["s2"]))


def test_rolling_slope_finds_a_known_trend():
    engine = synthetic_engine()
    engine["s2"] = 3.0 * engine["cycle"]
    assert build_features(engine, ["s2"])["s2_slope"].iloc[-1] == pytest.approx(3.0)


def test_policy_tiers():
    assert bearing_policy("outer_race", "high")["urgency"] == "high"
    assert bearing_policy("outer_race", "medium")["action_code"] == "confirm_with_second_measurement"
    assert bearing_policy("normal", "high")["urgency"] == "routine"
    assert bearing_policy("normal", "low")["urgency"] == "low"
    assert rul_policy(10, "high")["urgency"] == "critical"
    assert rul_policy(200, "high")["urgency"] == "routine"
    assert rul_policy(200, "low")["urgency"] == "low"


@needs_data
def test_real_dataset_shape_and_engine_separation():
    train = data_cmapss.load_train()
    test, truth = data_cmapss.load_test()
    assert (train["unit"].nunique(), test["unit"].nunique(), truth.size) == (100, 100, 100)
    assert test.groupby("unit")["cycle"].max().min() >= WINDOW  # so test rows always have a full window
    sensors = data_cmapss.informative_sensors(train)
    train_rows = set(map(tuple, train[sensors].round(4).to_numpy()))
    assert not any(tuple(r) in train_rows for r in test[sensors].round(4).to_numpy())


@needs_data
def test_training_target_is_capped_and_starts_after_a_full_window():
    train = data_cmapss.load_train()
    features, target, groups = rul.training_rows(train, data_cmapss.informative_sensors(train))
    assert target.max() == RUL_CAP
    assert features["cycle"].min() >= WINDOW
    assert len(features) == len(target) == len(groups)
