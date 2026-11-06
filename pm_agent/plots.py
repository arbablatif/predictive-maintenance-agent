"""All figures. One neutral ink for data, and colours reserved for identity."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: figures are only written to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .features import EnvelopeSpectrum, N_HARMONICS  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
FAULT_COLOURS = {"bpfo": "#2a78d6", "bpfi": "#eb6834", "bsf": "#1baf7a"}  # validated categorical order
BLUE_RAMP = ["#f2f6fc", "#2a78d6", "#123a6b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def envelope_figure(spec: EnvelopeSpectrum, freqs: dict[str, float], title: str, path: Path) -> Path:
    """Envelope spectrum with a marker at each expected fault harmonic."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(spec.freqs, spec.amplitude, color=INK, linewidth=0.9)
    limit = 1.15 * N_HARMONICS * freqs["bpfi"]
    for key, colour in FAULT_COLOURS.items():
        for h in range(1, N_HARMONICS + 1):
            ax.axvline(h * freqs[key], color=colour, linewidth=1.3, linestyle=(0, (4, 3)),
                       label=f"{key.upper()} {freqs[key]:.1f} Hz and harmonics" if h == 1 else None)
    ax.set_xlim(0, limit)
    visible = spec.amplitude[spec.freqs <= limit]
    ax.set_ylim(0, float(visible[3:].max()) * 1.15)  # skip the lowest bins, which hold leftover DC
    ax.set_xlabel("Envelope frequency (Hz)")
    ax.set_ylabel("Amplitude (g)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return _save(fig, path)


def confusion_figure(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(int(matrix.max()), 1))
    names = [label.replace("_", " ") for label in labels]
    ax.set_xticks(range(len(labels)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            dark = matrix[i, j] > 0.55 * matrix.max()
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=SURFACE if dark else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, path)


def split_comparison_figure(rows: list[tuple[str, float]], title: str, path: Path) -> Path:
    """Same model, different splits: horizontal bars, labelled directly."""
    fig, ax = plt.subplots(figsize=(7, 2.4))
    names = [r[0] for r in rows][::-1]
    values = [r[1] * 100 for r in rows][::-1]
    ax.barh(names, values, color=FAULT_COLOURS["bpfo"], height=0.5)
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", color=INK)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Recording-level accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    return _save(fig, path)


def rul_scatter_figure(truth: np.ndarray, pred: np.ndarray, low: np.ndarray, high: np.ndarray, title: str,
                       path: Path) -> Path:
    """Each test engine, ordered by true RUL, with its prediction and 80 % interval."""
    order = np.argsort(truth)
    x = np.arange(truth.size)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.vlines(x, low[order], high[order], color=GRID, linewidth=2.4, label="80 % interval")
    ax.plot(x, pred[order], "o", color=FAULT_COLOURS["bpfo"], markersize=3.5, label="Predicted RUL")
    ax.plot(x, truth[order], color=INK, linewidth=1.5, label="True RUL")
    ax.set_xlabel("Test engines, sorted by true RUL")
    ax.set_ylabel("Cycles")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return _save(fig, path)


def sensor_trend_figure(cycles: np.ndarray, trends: dict[str, np.ndarray], title: str, path: Path) -> Path:
    """Small multiples, each on its own scale, rather than one chart with several axes."""
    fig, axes = plt.subplots(1, len(trends), figsize=(9, 2.8), sharex=True)
    for ax, (name, values) in zip(np.atleast_1d(axes), trends.items()):
        ax.plot(cycles, values, color=GRID, linewidth=1.0)
        window = min(15, values.size)
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(cycles[window - 1:], smooth, color=FAULT_COLOURS["bpfo"], linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel("Cycle")
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    return _save(fig, path)
