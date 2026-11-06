"""Read uploads into one AssetSignal.

Two kinds of input are accepted, and they are kept apart on purpose:
    vibration       one accelerometer channel (.mat in CWRU layout, or a one-column .csv)
    engine_history  a C-MAPSS style sensor table (26 columns), one row per cycle

Sample rate and shaft speed come from the file or the person uploading it. They are never
guessed, because a wrong value moves every fault frequency without any visible error.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from .bearing_physics import SKF_6205, BearingGeometry
from .config import RECORDINGS
from .data_cmapss import COLUMNS

# CWRU's fan-end bearing is included as a known geometry outside the training conditions.
SKF_6203 = BearingGeometry("SKF 6203-2RS JEM", 8, 0.2656, 1.122, 0.0)
BEARINGS = {b.name: b for b in (SKF_6205, SKF_6203)}


class SignalError(ValueError):
    """The upload cannot be used. The message is shown to the person who uploaded it."""


@dataclass
class AssetSignal:
    signal_id: str               # content hash, so the LLM cache hits across sessions
    kind: str                    # "vibration" or "engine_history"
    asset: str
    source: str
    samples: np.ndarray | None = None
    history: pd.DataFrame | None = None
    fs: float | None = None
    rpm: float | None = None
    bearing: BearingGeometry = SKF_6205

    def describe(self) -> dict:
        """What the LLM is told. It never receives the samples."""
        return {"signal_id": self.signal_id, "kind": self.kind, "asset": self.asset}


def _signal_id(raw: bytes, *parts) -> str:
    digest = hashlib.sha256(raw)
    digest.update(repr(parts).encode())
    return digest.hexdigest()[:10]


def _read_mat(raw: bytes, filename: str) -> tuple[np.ndarray, float | None, float | None]:
    """Returns samples, RPM and sample rate. The variable is chosen by file number where
    possible, because 99.mat also carries a copy of the signal from 98.mat."""
    try:
        mat = sio.loadmat(io.BytesIO(raw))
    except Exception as exc:
        raise SignalError(f"Could not read {filename} as a MATLAB file.") from exc
    stem = Path(filename).stem
    wanted = f"X{int(stem):03d}_DE_time" if stem.isdigit() else None
    names = [k for k in mat if k.endswith("_DE_time")]
    if wanted in mat:
        name = wanted
    elif len(names) == 1:
        name = names[0]
    else:
        raise SignalError(f"{filename}: expected one drive-end channel, found {len(names)}.")
    prefix = name.removesuffix("_DE_time")
    rpm = float(np.ravel(mat[f"{prefix}RPM"])[0]) if f"{prefix}RPM" in mat else None
    # A CWRU file does not state its sample rate, but its number identifies it.
    known = next((r for r in RECORDINGS if stem.isdigit() and r.file_id == int(stem)), None)
    return np.ravel(mat[name]).astype(float), rpm, float(known.source_fs) if known else None


def _read_table(raw: bytes, filename: str) -> pd.DataFrame:
    try:
        table = pd.read_csv(io.BytesIO(raw), sep=r"[,\s]+", engine="python", header=None)
    except Exception as exc:
        raise SignalError(f"Could not read {filename} as a table.") from exc
    table = table.dropna(axis=1, how="all")
    if table.empty:
        raise SignalError(f"{filename} is empty.")
    if pd.to_numeric(table.iloc[0], errors="coerce").isna().any():
        table = table.iloc[1:]  # a header row
    table = table.apply(pd.to_numeric, errors="coerce")
    if table.isna().any().any():
        raise SignalError(f"{filename} contains values that are not numbers.")
    return table.reset_index(drop=True)


def load_upload(raw: bytes, filename: str, asset: str | None = None, fs: float | None = None,
                rpm: float | None = None, unit: int | None = None,
                bearing: BearingGeometry = SKF_6205) -> AssetSignal:
    suffix = Path(filename).suffix.lower()
    if suffix == ".mat":
        samples, file_rpm, file_fs = _read_mat(raw, filename)
        fs, rpm = fs or file_fs, rpm or file_rpm
    elif suffix in (".csv", ".txt"):
        table = _read_table(raw, filename)
        if table.shape[1] == len(COLUMNS):
            return _engine_signal(raw, filename, table, asset, unit)
        if table.shape[1] != 1:
            raise SignalError(f"{filename} has {table.shape[1]} columns. Expected 1 (vibration) "
                              f"or {len(COLUMNS)} (C-MAPSS engine history).")
        samples = table.iloc[:, 0].to_numpy(dtype=float)
    else:
        raise SignalError(f"Unsupported file type '{suffix}'. Use .mat, .csv or .txt.")

    if not fs:
        raise SignalError("Sample rate is required for this file.")
    if not rpm:
        raise SignalError("Shaft speed (rpm) is required for this file.")
    signal_id = _signal_id(raw, fs, rpm, bearing.name)
    return AssetSignal(signal_id, "vibration", asset or f"BEARING-{signal_id[:6].upper()}", filename,
                       samples=samples, fs=float(fs), rpm=float(rpm), bearing=bearing)


def _engine_signal(raw: bytes, filename: str, table: pd.DataFrame, asset: str | None,
                   unit: int | None) -> AssetSignal:
    table.columns = COLUMNS
    units = sorted(int(u) for u in table["unit"].unique())
    if unit is None:
        if len(units) > 1:
            raise SignalError(f"{filename} holds {len(units)} engines. Choose one (units {units[0]} to {units[-1]}).")
        unit = units[0]
    if unit not in units:
        raise SignalError(f"Engine {unit} is not in {filename}.")
    history = table[table["unit"] == unit].reset_index(drop=True)
    signal_id = _signal_id(raw, unit)
    return AssetSignal(signal_id, "engine_history", asset or f"ENGINE-{signal_id[:6].upper()}", filename,
                       history=history)


def asset_label(text: str) -> str:
    """Keep asset names short and printable, since they appear in prompts and tickets."""
    return re.sub(r"[^\w .#/-]", "", text).strip()[:60]
