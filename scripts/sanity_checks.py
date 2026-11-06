"""Checks that try to break the headline results. Writes results/sanity_checks.json.

    python scripts/sanity_checks.py

Needs data/raw and the trained models, and makes no API calls. It answers four questions:
    1. Is the RUL score real, or does something leak? (duplicates, shuffled labels)
    2. Does the RUL score hold beyond NASA's test distribution? (random truncation)
    3. Why was the healthy load 0 recording called a fault? (one-feature counterfactual)
    4. How much does the unseen-fault-size score depend on the demodulation band?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pm_agent import data_cmapss, diagnosis, rul  # noqa: E402
from pm_agent.config import RESULTS_DIR  # noqa: E402
from pm_agent.features import ALL_FEATURES, CANDIDATE_BANDS  # noqa: E402


def rul_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    sensors = data_cmapss.informative_sensors(train)
    train_rows = set(map(tuple, train[sensors].round(4).to_numpy()))
    duplicates = sum(tuple(row) in train_rows for row in test[sensors].round(4).to_numpy())

    features, target, groups = rul.training_rows(train, sensors)
    rng = np.random.default_rng(0)
    units = np.unique(groups)
    donors = dict(zip(units, rng.permutation(units)))
    shuffled = target.copy()
    for unit in units:  # each engine gets another engine's labels: sensors no longer explain them
        shuffled[groups == unit] = np.resize(target[groups == donors[unit]], (groups == unit).sum())
    return {"test_rows_identical_to_a_train_row": int(duplicates),
            "group_cv_rmse_real_labels": rul.cross_validate(features, target, groups)["gbm"],
            "group_cv_rmse_shuffled_labels": rul.cross_validate(features, shuffled, groups)["gbm"]}


def rul_truncation(train: pd.DataFrame, truth: np.ndarray) -> dict:
    """Cut held-out training engines at a uniformly random age, unlike NASA's test set,
    which has few engines far from failure."""
    units = train["unit"].unique()
    predictions, targets = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for fit_index, held_index in GroupKFold(5).split(units, groups=units):
            bundle, _ = rul.train_bundle(train[train["unit"].isin(units[fit_index])])
            held = train[train["unit"].isin(units[held_index])]
            cut = pd.concat([e.iloc[:int(rng.integers(31, len(e)))] for _, e in held.groupby("unit")])
            predictions += list(rul.predict_interval(bundle, rul.last_rows(cut, bundle.sensors))["rul"])
            targets += list(cut.groupby("unit")["rul"].last())
    p, t = np.array(predictions), np.array(targets)
    bands = []
    for low, high in ((0, 50), (50, 125), (125, 145), (145, 1000)):
        chosen = (t >= low) & (t < high)
        bands.append({"true_rul": f"{low}-{high}", "cuts": int(chosen.sum()), "rmse": rul.rmse(p[chosen], t[chosen]),
                      "mean_prediction": float(p[chosen].mean())})
    return {"cuts": int(t.size), "share_above_125": float(np.mean(t > 125)),
            "official_test_share_above_125": float(np.mean(truth > 125)),
            "official_test_max_rul": float(truth.max()), "rmse_true": rul.rmse(p, t),
            "rmse_capped": rul.rmse(p, np.minimum(t, data_cmapss.RUL_CAP)), "by_true_rul": bands}


def false_alarm_counterfactual(tables: dict) -> dict:
    """Train as in the load 0 fold, then change one feature of the healthy load 0 recording."""
    df = tables[(3000, 5000)]
    model = diagnosis.fit_classifier(df[df["load_hp"] != 0])
    windows = df[(df["load_hp"] == 0) & (df["label"] == "normal")]

    def votes(frame: pd.DataFrame) -> dict:
        return {k: int(v) for k, v in pd.Series(model.predict(frame[list(ALL_FEATURES)])).value_counts().items()}

    other_normals = df[(df["load_hp"] != 0) & (df["label"] == "normal")]
    changed = windows.copy()
    changed["rms"] = float(other_normals["rms"].mean())
    return {"recording_rms": float(windows["rms"].mean()),
            "other_normal_rms_range": [float(other_normals["rms"].min()), float(other_normals["rms"].max())],
            "votes_as_measured": votes(windows), "votes_with_rms_set_to_typical_normal": votes(changed)}


def band_sensitivity(tables: dict) -> dict:
    """Unseen fault size scored with each band forced on every fold. Diagnostic only: looking at
    these and picking the best would be choosing on test data."""
    out = {}
    for band in CANDIDATE_BANDS:
        df = tables[band]
        frames, recordings = [], []
        for size in diagnosis.SIZES:
            mask = diagnosis._size_test_mask(df, size)
            pred = diagnosis.predict(diagnosis.fit_classifier(df[~mask]), df[mask])
            frames.append(pred.frame)
            recordings.append(pred.recordings)
        pooled = diagnosis.Prediction(pd.concat(frames), pd.concat(recordings))
        out[f"{band[0]}-{band[1]} Hz"] = diagnosis.score(pooled)["recording_accuracy"]
    return out


def main() -> None:
    train = data_cmapss.load_train()
    test, truth = data_cmapss.load_test()
    tables = diagnosis.build_dataset()
    results = {"rul_leakage": rul_leakage(train, test), "rul_truncation": rul_truncation(train, truth),
               "false_alarm_file_97": false_alarm_counterfactual(tables),
               "size_split_by_fixed_band": band_sensitivity(tables)}
    (RESULTS_DIR / "sanity_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
