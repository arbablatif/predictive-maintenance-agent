"""Features for one vibration window: time domain plus envelope spectrum.

Why the envelope? A bearing defect produces short impacts that ring a structural resonance,
usually at several kHz. The fault rate (around 100 Hz) is hidden as the *repetition rate* of
those bursts, not as a frequency of the raw signal. So:

    band-pass around the resonance -> Hilbert envelope -> spectrum of the envelope

leaves a spectrum with peaks at the repetition rate: BPFO, BPFI or BSF.

Peaks are measured in dB above the local noise floor rather than in absolute units, so the
features depend on "does a peak stand out here?" and not on how hard the machine is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sg
from scipy import stats

from .bearing_physics import BearingGeometry, fault_frequencies

# Candidate demodulation bands in Hz. The band is picked using training data only.
# All stop at 5 kHz: the anti-alias filter on the resampled normal files acts near 6 kHz.
CANDIDATE_BANDS: tuple[tuple[int, int], ...] = ((1000, 3000), (2000, 4000), (3000, 5000), (2000, 5000))
FAULT_KEYS = ("bpfo", "bpfi", "bsf")
N_HARMONICS = 3
ENVELOPE_MAX_HZ = 1000.0  # all fault harmonics of interest sit below this
TOLERANCE = 0.02          # search +/-2 % around each target: real bearings slip a little
MIN_TOLERANCE_HZ = 2.0
FLOOR_HALF_WIDTH_HZ = 20.0


@dataclass
class EnvelopeSpectrum:
    freqs: np.ndarray
    amplitude: np.ndarray


def time_features(x: np.ndarray) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {
        "rms": rms,
        "kurtosis": float(stats.kurtosis(x)),          # excess: about 0 for Gaussian noise
        "crest_factor": float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0,
        "skewness": float(stats.skew(x)),
        "peak_to_peak": float(np.ptp(x)),
    }


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float]) -> EnvelopeSpectrum:
    """Band-pass, take the Hilbert envelope, and return its one-sided amplitude spectrum."""
    low, high = band
    if not 0 < low < high < fs / 2:
        raise ValueError(f"band {band} must lie between 0 and Nyquist ({fs / 2} Hz)")
    sos = sg.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sg.sosfiltfilt(sos, x)
    envelope = np.abs(sg.hilbert(filtered))
    envelope = envelope - envelope.mean()  # remove DC, or it swamps the low-frequency lines
    window = np.hanning(envelope.size)
    amplitude = 2 * np.abs(np.fft.rfft(envelope * window)) / window.sum()
    freqs = np.fft.rfftfreq(envelope.size, 1 / fs)
    keep = freqs <= ENVELOPE_MAX_HZ
    return EnvelopeSpectrum(freqs[keep], amplitude[keep])


def peak_above_floor(spec: EnvelopeSpectrum, target_hz: float) -> tuple[float, float]:
    """Strongest line near target_hz, in dB above the local median. Returns (dB, frequency)."""
    tolerance = max(TOLERANCE * target_hz, MIN_TOLERANCE_HZ)
    distance = np.abs(spec.freqs - target_hz)
    inside = distance <= tolerance
    floor_zone = (distance <= FLOOR_HALF_WIDTH_HZ + tolerance) & ~inside
    if not inside.any() or not floor_zone.any():
        return 0.0, float(target_hz)
    floor = max(float(np.median(spec.amplitude[floor_zone])), 1e-12)
    index = int(np.argmax(np.where(inside, spec.amplitude, -np.inf)))
    return 20 * np.log10(max(float(spec.amplitude[index]), 1e-12) / floor), float(spec.freqs[index])


def envelope_features(spec: EnvelopeSpectrum, freqs: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in FAULT_KEYS:
        levels = [peak_above_floor(spec, k * freqs[key])[0] for k in range(1, N_HARMONICS + 1)]
        for k, level in enumerate(levels, 1):
            out[f"{key}_h{k}_db"] = level
        out[f"{key}_mean_db"] = float(np.mean(levels))
    out["shaft_db"] = peak_above_floor(spec, freqs["shaft"])[0]
    # Inner race defects pass in and out of the load zone once per turn, adding shaft sidebands.
    out["bpfi_sideband_db"] = float(np.mean([peak_above_floor(spec, freqs["bpfi"] + s * freqs["shaft"])[0]
                                             for s in (-1, 1)]))
    return out


def extract_features(x: np.ndarray, fs: float, rpm: float, geometry: BearingGeometry,
                     band: tuple[float, float]) -> dict[str, float]:
    return {**time_features(x),
            **envelope_features(envelope_spectrum(x, fs, band), fault_frequencies(geometry, rpm))}


TIME_FEATURES = ("rms", "kurtosis", "crest_factor", "skewness", "peak_to_peak")
ENVELOPE_FEATURES = tuple(
    [f"{k}_h{h}_db" for k in FAULT_KEYS for h in range(1, N_HARMONICS + 1)]
    + [f"{k}_mean_db" for k in FAULT_KEYS] + ["shaft_db", "bpfi_sideband_db"])
ALL_FEATURES = TIME_FEATURES + ENVELOPE_FEATURES
