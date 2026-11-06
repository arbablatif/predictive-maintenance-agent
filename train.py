"""Train both models and write the Part A and Part B results.

    python train.py

Needs the raw data (scripts/download_data.py). Writes models/*.joblib, results/metrics.json
and results/figures/. Takes a few minutes on a laptop CPU and makes no API calls.
"""
from __future__ import annotations

import json

import joblib
import numpy as np

from pm_agent import data_cmapss, diagnosis, plots, rul
from pm_agent.config import CLASSES, FIGURE_DIR, MODEL_DIR, RESULTS_DIR

SHIPPED_TRAIN_LOADS = [0, 1, 2]  # load 3 stays unseen, for the demo and the agent evaluation
BANDS_FOR_RUL = ((0, 50), (50, 125), (125, 1000))


def part_a() -> dict:
    print("Part A: extracting features")
    tables = diagnosis.build_dataset()

    print("Part A: leave-one-load-out (bands chosen inside each fold)")
    load_out = diagnosis.leave_one_load_out(tables)
    fold_bands = {f["held_out_load"]: tuple(f["band_hz"]) for f in load_out["folds"]}

    print("Part A: leave-one-fault-size-out")
    size_out = diagnosis.leave_one_size_out(tables)

    print("Part A: ablation")
    ablation = {name: diagnosis.leave_one_load_out(tables, features, bands=fold_bands)["pooled"]
                for name, features in diagnosis.FEATURE_SETS.items()}

    print("Part A: shipped model")
    bundle = diagnosis.train_bundle(tables, SHIPPED_TRAIN_LOADS)
    df = tables[bundle.band]
    shipped = diagnosis.predict(bundle.model, df[df["load_hp"] == 3])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "bearing_classifier.joblib", compress=3)

    leaky = diagnosis.random_window_split(tables[bundle.band])

    labels = list(CLASSES)
    matrix = diagnosis.confusion(load_out["prediction"])
    plots.confusion_figure(matrix, labels, "Unseen load: 40 recordings", FIGURE_DIR / "confusion_load_split.png")
    plots.confusion_figure(diagnosis.confusion(size_out["prediction"]), labels, "Unseen fault size: 39 recordings",
                           FIGURE_DIR / "confusion_size_split.png")
    plots.split_comparison_figure(
        [("Random windows (leaky)", leaky["recording_accuracy"]),
         ("Unseen load", load_out["pooled"]["recording_accuracy"]),
         ("Unseen fault size", size_out["pooled"]["recording_accuracy"])],
        "Same features and model, three ways of testing", FIGURE_DIR / "split_comparison.png")

    wrong = load_out["prediction"].recordings
    wrong = wrong[wrong["label"] != wrong["pred"]]
    return {
        "load_split": {"folds": load_out["folds"], "pooled": load_out["pooled"],
                       "confusion_recordings": matrix.tolist(),
                       "confusion_windows": diagnosis.confusion(load_out["prediction"], "frame").tolist(),
                       "reliability": diagnosis.reliability(load_out["prediction"]),
                       "ece_windows": diagnosis.expected_calibration_error(load_out["prediction"]),
                       "misclassified": wrong[["recording", "label", "pred", "top_prob", "envelope_check",
                                               "confidence"]].to_dict("records")},
        "size_split": {"folds": size_out["folds"], "pooled": size_out["pooled"],
                       "confusion_recordings": diagnosis.confusion(size_out["prediction"]).tolist(),
                       "reliability": diagnosis.reliability(size_out["prediction"]),
                       "ece_windows": diagnosis.expected_calibration_error(size_out["prediction"])},
        "random_split_leaky": leaky,
        "ablation": ablation,
        "shipped_model": {"band_hz": list(bundle.band), "train_loads": SHIPPED_TRAIN_LOADS,
                          "test_load_3": diagnosis.score(shipped), "reliability": diagnosis.reliability(shipped)},
        "class_order": labels,
    }


def part_b() -> dict:
    print("Part B: RUL models")
    train = data_cmapss.load_train()
    test, truth = data_cmapss.load_test()
    sensors = data_cmapss.informative_sensors(train)
    features, target, groups = rul.training_rows(train, sensors)
    cv = rul.cross_validate(features, target, groups)

    bundle, ridge = rul.train_bundle(train)
    joblib.dump(bundle, MODEL_DIR / "rul_models.joblib", compress=3)

    rows = rul.last_rows(test, sensors)
    interval = rul.predict_interval(bundle, rows)
    point = interval["rul"].to_numpy()
    baseline = rul.FleetLifeBaseline().fit(train.groupby("unit")["cycle"].max().to_numpy()).predict(rows["cycle"])
    models = {
        "Fleet-life baseline (age only)": baseline,
        "Fleet-life baseline, capped at 125": np.minimum(baseline, data_cmapss.RUL_CAP),
        "Ridge (sensors)": np.clip(ridge.predict(rows[bundle.columns]), 0, data_cmapss.RUL_CAP),
        "Gradient boosting (sensors)": point,
    }

    rng = np.random.default_rng(rul.SEED)
    samples = [rul.rmse(point[i], truth[i]) for i in (rng.integers(0, truth.size, truth.size) for _ in range(2000))]
    capped = np.minimum(truth, data_cmapss.RUL_CAP)
    low, high = interval["low"].to_numpy(), interval["high"].to_numpy()

    by_band = []
    for start, stop in BANDS_FOR_RUL:
        chosen = (truth >= start) & (truth < stop)
        by_band.append({"true_rul": f"{start}-{stop}", "engines": int(chosen.sum()),
                        "rmse": rul.rmse(point[chosen], truth[chosen])})
    labels = np.array([rul.rul_confidence(a, b, []) for a, b in zip(low, high)])
    confidence = {label: {"engines": int((labels == label).sum()),
                          "rmse": rul.rmse(point[labels == label], capped[labels == label]) if (labels == label).any()
                          else None} for label in ("high", "medium", "low")}

    plots.rul_scatter_figure(truth, point, low, high, "100 test engines: predicted and true RUL",
                             FIGURE_DIR / "rul_test_engines.png")
    return {
        "models": {name: rul.metrics(pred, truth) for name, pred in models.items()},
        "group_cv_rmse_capped": cv,
        "gbm_rmse_true_ci95": [float(v) for v in np.percentile(samples, [2.5, 97.5])],
        "interval_coverage_capped": float(np.mean((low <= capped) & (capped <= high))),
        "interval_mean_width": float(np.mean(high - low)),
        "by_true_rul": by_band,
        "by_confidence": confidence,
        "sensors_used": sensors,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {"part_a": part_a(), "part_b": part_b()}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("wrote", RESULTS_DIR / "metrics.json")


if __name__ == "__main__":
    main()
