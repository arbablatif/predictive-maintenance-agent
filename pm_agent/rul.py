"""Part B: remaining useful life models and scoring on C-MAPSS FD001.

Compared on the same 100 test engines:
    fleet-life baseline   uses only the engine's age, and no sensors at all
    ridge                 linear model on the sensor features
    gradient boosting     non-linear model on the same features
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_cmapss import RUL_CAP, WINDOW, build_features, feature_columns, informative_sensors

SEED = 42
GBM_PARAMS = dict(learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=30,
                  l2_regularization=1.0, early_stopping=False, random_state=SEED)
RIDGE_ALPHA = 10.0
INTERVAL = (0.1, 0.9)  # an 80 % prediction interval
# The target is capped at 125, so predictions flatten near it: above this the model can only
# say "healthy", and the lower bound of the interval is the number to rely on.
SATURATED_ABOVE = 110.0
HIGH_WIDTH, MEDIUM_WIDTH = 25.0, 50.0   # interval width in cycles, for the confidence label
MAX_OUT_OF_RANGE_SHARE = 0.3


def nasa_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """PHM08 score, where lower is better. With d = pred - truth, a late prediction (d > 0)
    means maintenance after the true failure point, so it is penalised more steeply."""
    d = np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)) ** 2)))


class FleetLifeBaseline:
    """Expected remaining life given only age: mean life of training engines that outlived
    this age, minus the age. This is the benchmark any sensor-based model has to beat."""

    def fit(self, lives: np.ndarray) -> "FleetLifeBaseline":
        self.lives = np.asarray(lives, dtype=float)
        return self

    def predict(self, ages: np.ndarray) -> np.ndarray:
        out = []
        for age in np.asarray(ages, dtype=float):
            survivors = self.lives[self.lives > age]
            out.append(survivors.mean() - age if survivors.size else 0.0)
        return np.array(out)


@dataclass
class RulBundle:
    """Everything needed to score an engine later."""
    sensors: list[str]
    columns: list[str]
    point: HistGradientBoostingRegressor
    lower: HistGradientBoostingRegressor
    upper: HistGradientBoostingRegressor
    feature_low: dict
    feature_high: dict
    profile: dict   # per-sensor start and end levels: used only to build the synthetic sample


def training_rows(train: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features, capped target and engine IDs. Rows without a full window are dropped because
    every test engine has at least that much history, so they would not match test conditions."""
    features = build_features(train, sensors)
    keep = (features["cycle"] >= WINDOW).to_numpy()
    target = np.minimum(train["rul"].to_numpy(dtype=float), RUL_CAP)
    return features[keep], target[keep], features["unit"].to_numpy()[keep]


def _gbm(**extra) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**GBM_PARAMS, **extra)


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def cross_validate(features: pd.DataFrame, target: np.ndarray, groups: np.ndarray, folds: int = 5) -> dict:
    """GroupKFold by engine: rows of one engine are never on both sides of a fold."""
    columns = feature_columns(features)
    X = features[columns]
    predictions = {"ridge": np.zeros(target.size), "gbm": np.zeros(target.size)}
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(X, target, groups):
        for name, model in (("ridge", _ridge()), ("gbm", _gbm())):
            model.fit(X.iloc[train_idx], target[train_idx])
            predictions[name][test_idx] = model.predict(X.iloc[test_idx])
    return {name: rmse(np.clip(p, 0, RUL_CAP), target) for name, p in predictions.items()}


def train_bundle(train: pd.DataFrame) -> tuple[RulBundle, object]:
    sensors = informative_sensors(train)
    features, target, _ = training_rows(train, sensors)
    columns = feature_columns(features)
    X = features[columns]
    last = train.groupby("unit").tail(5)
    first = train.groupby("unit").head(30)
    sensor_columns = [c for c in train.columns if c.startswith(("s", "op"))]
    bundle = RulBundle(sensors, columns,
                       point=_gbm().fit(X, target),
                       lower=_gbm(loss="quantile", quantile=INTERVAL[0]).fit(X, target),
                       upper=_gbm(loss="quantile", quantile=INTERVAL[1]).fit(X, target),
                       feature_low=X.quantile(0.005).to_dict(), feature_high=X.quantile(0.995).to_dict(),
                       profile={"start": first[sensor_columns].mean().to_dict(),
                                "end": last[sensor_columns].mean().to_dict(),
                                "noise": first[sensor_columns].std().to_dict()})
    return bundle, _ridge().fit(X, target)


def last_rows(test: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """The final row of each engine: the moment a prediction is actually asked for."""
    return build_features(test, sensors).groupby("unit", sort=True).tail(1)


def predict_interval(bundle: RulBundle, rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[bundle.columns]
    point = np.clip(bundle.point.predict(X), 0, RUL_CAP)
    low = np.clip(bundle.lower.predict(X), 0, RUL_CAP)
    high = np.clip(bundle.upper.predict(X), 0, RUL_CAP)
    # Separately fitted quantile models can cross, so the interval is forced to contain the point.
    return pd.DataFrame({"rul": point, "low": np.minimum(low, point), "high": np.maximum(high, point)},
                        index=rows.index)


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """RMSE against both truths, because published papers differ on which one they use."""
    return {"rmse_true": rmse(pred, truth),
            "rmse_capped": rmse(pred, np.minimum(truth, RUL_CAP)),
            "nasa_score": nasa_score(pred, truth),
            "late_fraction": float(np.mean(np.asarray(pred) > truth))}


def rul_confidence(low: float, high: float, warnings: list[str]) -> str:
    if warnings:
        return "low"
    width = high - low
    return "high" if width <= HIGH_WIDTH else "medium" if width <= MEDIUM_WIDTH else "low"


def estimate(history: pd.DataFrame, bundle: RulBundle) -> dict:
    """RUL for one engine's history, in the C-MAPSS column layout."""
    row = last_rows(history, bundle.sensors)
    result = predict_interval(bundle, row).iloc[0]
    point, low, high = float(result["rul"]), float(result["low"]), float(result["high"])

    warnings = []
    cycles = int(history["cycle"].max())
    if cycles < WINDOW:
        warnings.append(f"Only {cycles} cycles of history. The model expects at least {WINDOW}.")
    values = row[bundle.columns].iloc[0]
    outside = float(((values < pd.Series(bundle.feature_low)[bundle.columns])
                     | (values > pd.Series(bundle.feature_high)[bundle.columns])).mean())
    if outside > MAX_OUT_OF_RANGE_SHARE:
        warnings.append(f"{outside:.0%} of features fall outside the training range.")

    return {"rul_cycles": round(point), "interval_low_cycles": round(low), "interval_high_cycles": round(high),
            "interval_level_percent": round((INTERVAL[1] - INTERVAL[0]) * 100),
            "saturated": point >= SATURATED_ABOVE, "cycles_observed": cycles, "label_cap_cycles": RUL_CAP,
            "confidence": rul_confidence(low, high, warnings), "warnings": warnings}
