"""Load CWRU recordings, handling the dataset's known quirks.

Three things in the raw files would silently corrupt results if ignored. Each was
confirmed against the real data (see README and tests/test_data_cwru.py):

1. Normal files are sampled at 48 kHz, not 12 kHz. Read at 48 kHz, their shaft line tracks
   speed across loads to within 0.03 Hz. Read at 12 kHz, it sticks at 30.00 Hz, which is
   really the fixed 120 Hz line from 60 Hz mains.
2. 99.mat also contains X098_DE_time, an exact copy of the signal in 98.mat. Taking the
   first drive-end variable would copy a load 1 recording into load 2 and leak across a
   load split, so the variable is always chosen by file number.
3. 98.mat and 99.mat have no RPM variable, so nominal speed is used and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal as sg

from .config import CWRU_DIR, NOMINAL_RPM, TARGET_FS, Recording


@dataclass
class LoadedRecording:
    meta: Recording
    x: np.ndarray       # drive-end acceleration in g, at TARGET_FS
    rpm: float
    rpm_source: str     # "file" or "nominal"


def to_target_rate(x: np.ndarray, fs: int) -> np.ndarray:
    """Resample to TARGET_FS. resample_poly applies an anti-alias filter first."""
    if fs == TARGET_FS:
        return x
    return sg.resample_poly(x, TARGET_FS, fs)


def load_recording(rec: Recording, directory: Path = CWRU_DIR) -> LoadedRecording:
    mat = sio.loadmat(directory / f"{rec.file_id}.mat")
    variable = f"X{rec.file_id:03d}_DE_time"
    if variable not in mat:
        raise KeyError(f"{variable} not found in {rec.file_id}.mat")
    x = np.ravel(mat[variable]).astype(np.float64)

    rpm_variable = f"X{rec.file_id:03d}RPM"
    if rpm_variable in mat:
        rpm, source = float(np.ravel(mat[rpm_variable])[0]), "file"
    else:
        rpm, source = NOMINAL_RPM[rec.load_hp], "nominal"
    return LoadedRecording(rec, to_target_rate(x, rec.source_fs), rpm, source)


def windows(x: np.ndarray, size: int, overlap: float = 0.5) -> np.ndarray:
    """Cut a signal into equal windows, shape (n_windows, size). A partial tail is dropped."""
    step = int(size * (1 - overlap))
    if x.size < size:
        return np.empty((0, size))
    starts = range(0, x.size - size + 1, step)
    return np.stack([x[s:s + size] for s in starts])
