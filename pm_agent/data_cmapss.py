"""Load NASA C-MAPSS FD001 (simulated turbofan run-to-failure) and build features.

Training engines run all the way to failure. Test engines stop at an unknown point before
failure, and RUL_FD001.txt gives the true remaining cycles at that last row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CMAPSS_DIR

SETTING_COLUMNS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS
RUL_CAP = 125      # the usual piecewise-linear label: early life is treated as "healthy"
WINDOW = 30        # cycles of history behind each feature row


def read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+", header=None)
    table = table.dropna(axis=1, how="all")
    if table.shape[1] != len(COLUMNS):
        raise ValueError(f"{Path(path).name}: expected {len(COLUMNS)} columns, found {table.shape[1]}")
    table.columns = COLUMNS
    return table


def load_train(directory: Path = CMAPSS_DIR) -> pd.DataFrame:
    table = read_table(directory / "train_FD001.txt")
    table["rul"] = table.groupby("unit")["cycle"].transform("max") - table["cycle"]
    return table


def load_test(directory: Path = CMAPSS_DIR) -> tuple[pd.DataFrame, np.ndarray]:
    """Test sensor rows, plus the true RUL at each engine's last row (ordered by unit)."""
    table = read_table(directory / "test_FD001.txt")
    truth = pd.read_csv(directory / "RUL_FD001.txt", header=None).iloc[:, 0].to_numpy(dtype=float)
    return table, truth


def informative_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors that actually vary. Chosen from training data only."""
    return [c for c in SENSOR_COLUMNS if train[c].nunique() > 2]


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope over the trailing window, for every row of one engine."""
    out = np.zeros(values.size)
    for end in range(1, values.size):
        start = max(0, end + 1 - window)
        t = np.arange(end + 1 - start, dtype=float)
        t -= t.mean()
        segment = values[start:end + 1]
        out[end] = float(t @ (segment - segment.mean()) / (t @ t))
    return out


def build_features(table: pd.DataFrame, sensors: list[str], window: int = WINDOW) -> pd.DataFrame:
    """Trailing-window statistics per engine. One output row per input row.

    The cycle count is deliberately not a feature: a model that sees age can score well by
    learning the fleet's average life, without reading the engine's condition.
    """
    pieces = []
    for unit, engine in table.groupby("unit", sort=True):
        columns: dict[str, np.ndarray] = {}
        for sensor in sensors:
            series = engine[sensor]
            rolling = series.rolling(window, min_periods=1)
            columns[f"{sensor}_last"] = series.to_numpy()
            columns[f"{sensor}_mean"] = rolling.mean().to_numpy()
            columns[f"{sensor}_std"] = rolling.std().fillna(0.0).to_numpy()
            columns[f"{sensor}_slope"] = _rolling_slope(series.to_numpy(dtype=float), window)
        piece = pd.DataFrame(columns, index=engine.index)
        piece.insert(0, "unit", unit)
        piece.insert(1, "cycle", engine["cycle"].to_numpy())
        pieces.append(piece)
    return pd.concat(pieces)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ("unit", "cycle")]
