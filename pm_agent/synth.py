"""Synthetic signals with a known answer, for tests and the fresh-clone demo.

No CWRU or NASA data is redistributed in this repo, so these stand in when data/ is absent.
They are simple physics sketches, not real machines, and every use of them says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_cmapss import COLUMNS


def bearing_signal(fault_hz: float | None, rpm: float = 1750.0, fs: int = 12_000, seconds: float = 5.0,
                   resonance_hz: float = 3000.0, decay_per_s: float = 900.0, impact_g: float = 2.0,
                   noise_g: float = 0.05, seed: int = 0) -> np.ndarray:
    """Noise and a shaft tone, plus decaying resonance bursts at fault_hz if one is given."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = noise_g * rng.standard_normal(n) + 0.03 * np.sin(2 * np.pi * (rpm / 60) * t)
    if fault_hz:
        burst_t = np.arange(int(0.008 * fs)) / fs
        burst = impact_g * np.exp(-decay_per_s * burst_t) * np.sin(2 * np.pi * resonance_hz * burst_t)
        moment = 0.0
        while moment < seconds:
            start = int(moment * fs)
            stop = min(start + burst.size, n)
            x[start:stop] += burst[:stop - start] * (0.8 + 0.4 * rng.random())
            moment += (1 / fault_hz) * (1 + 0.005 * rng.standard_normal())  # slight slip
    return x


def engine_history(profile: dict, cycles: int = 170, life: int = 200, seed: int = 0) -> pd.DataFrame:
    """A made-up engine in C-MAPSS layout: sensors drift from typical start levels towards
    typical end-of-life levels, following a curve that steepens late in life."""
    rng = np.random.default_rng(seed)
    cycle = np.arange(1, cycles + 1)
    progress = (cycle / life) ** 3
    table = {"unit": np.ones(cycles, dtype=int), "cycle": cycle}
    for column in COLUMNS[2:]:
        start, end = profile["start"][column], profile["end"][column]
        table[column] = start + (end - start) * progress + profile["noise"][column] * rng.standard_normal(cycles)
    return pd.DataFrame(table)[COLUMNS]
