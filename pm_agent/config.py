"""Paths, dataset tables and constants shared across the project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CWRU_DIR = DATA_DIR / "cwru"
CMAPSS_DIR = DATA_DIR / "cmapss"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
SAMPLE_DIR = ROOT / "samples"
OUT_DIR = ROOT / "out"

# --------------------------------------------------------------------- CWRU
CWRU_URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"
CMAPSS_URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
              "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
CMAPSS_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")

CLASSES = ("normal", "inner_race", "outer_race", "ball")
TARGET_FS = 12_000  # every recording is analysed at this rate


@dataclass(frozen=True)
class Recording:
    """One CWRU file: the unit of independence when splitting data."""
    file_id: int
    label: str
    load_hp: int
    fault_size_in: float  # 0.0 for normal
    source_fs: int

    @property
    def name(self) -> str:
        size = f"{int(round(self.fault_size_in * 1000)):03d}" if self.fault_size_in else "000"
        return f"{self.label}_{size}_load{self.load_hp}"


def _series(first_id: int, label: str, size: float, fs: int = 12_000) -> list[Recording]:
    """CWRU numbers each fault's four loads consecutively."""
    return [Recording(first_id + load, label, load, size, fs) for load in range(4)]


# 12 kHz drive-end set. Outer race is the @6:00 position (centred in the load zone).
RECORDINGS: tuple[Recording, ...] = tuple(
    _series(97, "normal", 0.0, fs=48_000)
    + _series(105, "inner_race", 0.007) + _series(169, "inner_race", 0.014) + _series(209, "inner_race", 0.021)
    + _series(118, "ball", 0.007) + _series(185, "ball", 0.014) + _series(222, "ball", 0.021)
    + _series(130, "outer_race", 0.007) + _series(197, "outer_race", 0.014) + _series(234, "outer_race", 0.021)
)

# Nominal shaft speed per motor load, used only if a file has no RPM variable.
NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
