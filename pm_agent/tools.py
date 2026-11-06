"""The three tools the agent can call. Every number in a ticket starts here.

Tools refer to signals by ID. A failed call returns {"error": ...} to the model rather than
raising, so the model can recover or report the gap instead of inventing an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib

from . import diagnosis, plots, rul
from .bearing_physics import fault_frequencies
from .config import MODEL_DIR, OUT_DIR
from .policy import bearing_policy, rul_policy
from .signals import AssetSignal

_SIGNAL_ARG = {"type": "object", "required": ["signal_id"],
               "properties": {"signal_id": {"type": "string", "description": "ID from the signal list."}}}

TOOLS = [
    {"name": "diagnose_bearing",
     "description": "Classify a vibration signal as normal, inner_race, outer_race or ball. Returns class "
                    "probabilities, envelope evidence at bearing fault frequencies, a confidence label, "
                    "warnings, and the maintenance policy (urgency and action window).",
     "parameters": _SIGNAL_ARG},
    {"name": "plot_envelope_spectrum",
     "description": "Save an envelope spectrum plot of a vibration signal, marked with BPFO, BPFI and BSF "
                    "harmonics. Use it as evidence for the ticket.",
     "parameters": _SIGNAL_ARG},
    {"name": "estimate_rul",
     "description": "Estimate remaining useful life in cycles from a turbofan engine_history signal, with an "
                    "80 percent interval, a sensor trend plot and the maintenance policy. Works only on "
                    "engine_history signals, never on vibration.",
     "parameters": _SIGNAL_ARG},
]


@dataclass
class ToolOutcome:
    name: str
    args: dict
    result: dict
    figures: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return "error" not in self.result


@lru_cache(maxsize=1)
def bearing_model() -> diagnosis.DiagnosisBundle:
    return joblib.load(MODEL_DIR / "bearing_classifier.joblib")


@lru_cache(maxsize=1)
def rul_model() -> rul.RulBundle:
    return joblib.load(MODEL_DIR / "rul_models.joblib")


def _figure_dir(signal: AssetSignal, out_dir: Path) -> Path:
    return out_dir / signal.signal_id


def _diagnose_bearing(signal: AssetSignal, out_dir: Path) -> tuple[dict, list[Path]]:
    if signal.kind != "vibration":
        return _wrong_kind("diagnose_bearing", signal, "vibration"), []
    result = diagnosis.diagnose(signal.samples, signal.fs, signal.rpm, signal.bearing, bearing_model())
    return {"signal_id": signal.signal_id, "bearing": signal.bearing.name, **result,
            "policy": bearing_policy(result["predicted_class"], result["confidence"])}, []


def _plot_envelope_spectrum(signal: AssetSignal, out_dir: Path) -> tuple[dict, list[Path]]:
    if signal.kind != "vibration":
        return _wrong_kind("plot_envelope_spectrum", signal, "vibration"), []
    bundle = bearing_model()
    x = diagnosis.prepare_signal(signal.samples, signal.fs)
    freqs = fault_frequencies(signal.bearing, signal.rpm)
    path = plots.envelope_figure(diagnosis.mean_envelope(x, bundle.band), freqs,
                                 f"Envelope spectrum: {signal.asset} ({signal.rpm:g} rpm)",
                                 _figure_dir(signal, out_dir) / "envelope_spectrum.png")
    return {"signal_id": signal.signal_id, "figure": str(path), "band_hz": list(bundle.band),
            "marked_frequencies_hz": {k: round(freqs[k], 1) for k in ("bpfo", "bpfi", "bsf")}}, [path]


def _estimate_rul(signal: AssetSignal, out_dir: Path) -> tuple[dict, list[Path]]:
    if signal.kind != "engine_history":
        return _wrong_kind("estimate_rul", signal, "engine_history"), []
    result = rul.estimate(signal.history, rul_model())
    trends = {name: signal.history[name].to_numpy(dtype=float) for name in ("s4", "s11", "s15")}
    path = plots.sensor_trend_figure(signal.history["cycle"].to_numpy(), trends,
                                     f"Sensor trends: {signal.asset}",
                                     _figure_dir(signal, out_dir) / "sensor_trends.png")
    return {"signal_id": signal.signal_id, **result, "figure": str(path),
            "policy": rul_policy(result["interval_low_cycles"], result["confidence"])}, [path]


def _wrong_kind(tool: str, signal: AssetSignal, needed: str) -> dict:
    return {"error": f"{tool} needs a {needed} signal, but {signal.signal_id} is {signal.kind}.",
            "code": "wrong_signal_type"}


_IMPLEMENTATIONS = {"diagnose_bearing": _diagnose_bearing, "plot_envelope_spectrum": _plot_envelope_spectrum,
                    "estimate_rul": _estimate_rul}


def execute(signals: dict[str, AssetSignal], name: str, args: dict | None, out_dir: Path = OUT_DIR) -> ToolOutcome:
    """Run one tool call. Bad requests come back as an error result, never as an exception."""
    args = dict(args or {})
    if name not in _IMPLEMENTATIONS:
        return ToolOutcome(name, args, {"error": f"Unknown tool {name}.", "code": "unknown_tool"})
    signal = signals.get(str(args.get("signal_id")))
    if signal is None:
        return ToolOutcome(name, args, {"error": f"Unknown signal_id. Available: {sorted(signals)}.",
                                        "code": "unknown_signal"})
    try:
        result, figures = _IMPLEMENTATIONS[name](signal, out_dir)
    except ValueError as exc:  # raised by the input checks, such as a signal that is too short
        return ToolOutcome(name, args, {"error": str(exc), "code": "invalid_signal"})
    return ToolOutcome(name, args, result, figures)
