"""Part A: train and evaluate the bearing fault classifier.

Rows are 1 s windows, but the unit of independence is the recording: windows from one file
share a bearing, a sensor mounting and a moment in time. Every honest split below therefore
divides whole recordings, and scores are given per window and per recording.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
import sklearn
from scipy import signal as sg
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .bearing_physics import SKF_6205, BearingGeometry, fault_frequencies
from .config import CLASSES, RECORDINGS, TARGET_FS
from .data_cwru import load_recording, windows
from .features import (ALL_FEATURES, CANDIDATE_BANDS, ENVELOPE_FEATURES, FAULT_KEYS, N_HARMONICS, TIME_FEATURES,
                       EnvelopeSpectrum, envelope_spectrum, extract_features)

WINDOW_S = 1.0
SEED = 42
# Fixed in advance and never tuned on a test split.
MODEL_PARAMS = dict(learning_rate=0.1, max_iter=150, max_depth=3, min_samples_leaf=10,
                    l2_regularization=1.0, early_stopping=False, random_state=SEED)
# Which envelope line should confirm each fault class.
FAULT_KEY = {"inner_race": "bpfi", "outer_race": "bpfo", "ball": "bsf"}
# Peak level that counts as real. Taking the maximum over a search band reads several dB above
# the floor even on noise: on training loads the highest normal recording reached 7.6 dB.
EVIDENCE_DB = 10.0
HIGH_PROB, MEDIUM_PROB = 0.9, 0.6
RIG_RPM_RANGE = (1720.0, 1800.0)   # what the CWRU rig covers across its four loads
MAX_OUT_OF_RANGE_SHARE = 0.3       # above this share of features outside training, flag the input
FEATURE_SETS = {"time only": TIME_FEATURES, "envelope only": ENVELOPE_FEATURES, "time + envelope": ALL_FEATURES}


def build_dataset(bands=CANDIDATE_BANDS) -> dict[tuple[int, int], pd.DataFrame]:
    """One feature table per candidate band. Rows are windows with their recording's labels."""
    tables: dict[tuple[int, int], list[dict]] = {b: [] for b in bands}
    for rec in RECORDINGS:
        loaded = load_recording(rec)
        for i, w in enumerate(windows(loaded.x, int(WINDOW_S * TARGET_FS))):
            base = {"recording": rec.name, "file_id": rec.file_id, "label": rec.label, "load_hp": rec.load_hp,
                    "fault_size_in": rec.fault_size_in, "window": i}
            for band in bands:
                tables[band].append({**base, **extract_features(w, TARGET_FS, loaded.rpm, SKF_6205, band)})
    return {b: pd.DataFrame(rows) for b, rows in tables.items()}


def fit_classifier(df: pd.DataFrame, features=ALL_FEATURES) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    # Normal has few recordings, so classes are balanced by weight.
    return model.fit(df[list(features)], df["label"], sample_weight=compute_sample_weight("balanced", df["label"]))


@dataclass
class Prediction:
    frame: pd.DataFrame       # one row per test window: truth, prediction, probabilities
    recordings: pd.DataFrame  # one row per test recording, using its mean window probability


def predict(model, df: pd.DataFrame, features=ALL_FEATURES) -> Prediction:
    proba = model.predict_proba(df[list(features)])
    classes = list(model.classes_)
    level_columns = [f"{k}_mean_db" for k in FAULT_KEYS]
    frame = df[["recording", "label", "load_hp", "fault_size_in"] + level_columns].copy()
    frame["pred"] = [classes[i] for i in proba.argmax(axis=1)]
    frame["top_prob"] = proba.max(axis=1)
    for j, c in enumerate(classes):
        frame[f"p_{c}"] = proba[:, j]

    prob_columns = [f"p_{c}" for c in classes]
    grouped = frame.groupby("recording", sort=False)
    recordings = grouped[prob_columns].mean()
    recordings["label"] = grouped["label"].first()
    recordings["pred"] = [classes[i] for i in recordings[prob_columns].to_numpy().argmax(axis=1)]
    recordings["top_prob"] = recordings[prob_columns].max(axis=1)
    recordings[level_columns] = grouped[level_columns].mean()
    recordings["vote_share"] = [float((frame.loc[frame["recording"] == name, "pred"] == row["pred"]).mean())
                                for name, row in recordings.iterrows()]
    recordings["envelope_check"] = [envelope_check(row["pred"], {k: row[f"{k}_mean_db"] for k in FAULT_KEYS})
                                    for _, row in recordings.iterrows()]
    recordings["confidence"] = [confidence_label(row["top_prob"], row["vote_share"], row["envelope_check"], [])
                                for _, row in recordings.iterrows()]
    return Prediction(frame, recordings.reset_index())


def envelope_check(predicted: str, levels: dict[str, float]) -> str:
    """Does the physics back the classifier? Returns supports, inconclusive or contradicts."""
    if predicted == "normal":
        return "contradicts" if max(levels.values()) >= EVIDENCE_DB else "supports"
    return "supports" if levels[FAULT_KEY[predicted]] >= EVIDENCE_DB else "inconclusive"


def confidence_label(mean_prob: float, vote_share: float, check: str, warnings: list[str]) -> str:
    """A fixed rule, not a model output. High needs the classifier and the physics to agree."""
    if warnings or check == "contradicts":
        return "low"
    if mean_prob >= HIGH_PROB and vote_share >= HIGH_PROB and check == "supports":
        return "high"
    return "medium" if mean_prob >= MEDIUM_PROB and vote_share >= MEDIUM_PROB else "low"


def score(pred: Prediction) -> dict:
    labels = list(CLASSES)
    w, r = pred.frame, pred.recordings
    return {
        "n_windows": int(len(w)), "n_recordings": int(len(r)),
        "window_accuracy": float(accuracy_score(w["label"], w["pred"])),
        "window_macro_f1": float(f1_score(w["label"], w["pred"], labels=labels, average="macro", zero_division=0)),
        "recording_accuracy": float(accuracy_score(r["label"], r["pred"])),
        "recording_macro_f1": float(f1_score(r["label"], r["pred"], labels=labels, average="macro", zero_division=0)),
    }


def select_band(tables: dict, train_loads: list[int]) -> tuple[tuple[int, int], dict]:
    """Pick a demodulation band by leave-one-load-out inside the training loads only."""
    scores = {}
    for band, df in tables.items():
        fold_scores = []
        for held in train_loads:
            inner_train = df[df["load_hp"].isin(set(train_loads) - {held})]
            inner_test = df[df["load_hp"] == held]
            fold_scores.append(score(predict(fit_classifier(inner_train), inner_test))["window_macro_f1"])
        scores[band] = float(np.mean(fold_scores))
    return max(scores, key=scores.get), scores


def leave_one_load_out(tables: dict, features=ALL_FEATURES, bands: dict | None = None) -> dict:
    """Headline test. The held-out load plays no part in training or in choosing the band.

    `bands` maps held-out load to band, so an ablation can reuse the same per-fold choices.
    """
    folds, frames, recordings = [], [], []
    for held in range(4):
        train_loads = [load for load in range(4) if load != held]
        band = bands[held] if bands else select_band(tables, train_loads)[0]
        df = tables[band]
        train, test = df[df["load_hp"] != held], df[df["load_hp"] == held]
        pred = predict(fit_classifier(train, features), test, features)
        folds.append({"held_out_load": held, "band_hz": list(band), **score(pred)})
        frames.append(pred.frame)
        recordings.append(pred.recordings)
    pooled = Prediction(pd.concat(frames, ignore_index=True), pd.concat(recordings, ignore_index=True))
    return {"folds": folds, "pooled": score(pooled), "prediction": pooled}


SIZES = (0.007, 0.014, 0.021)


def _size_test_mask(df: pd.DataFrame, size: float) -> pd.Series:
    """Rows held out for one fault size. Normal has no size, so one normal load rotates in."""
    normal_load = SIZES.index(size) + 1
    return (df["fault_size_in"] == size) | ((df["label"] == "normal") & (df["load_hp"] == normal_load))


def _select_band_for_size(tables: dict, size: float) -> tuple[int, int]:
    """Pick a band using only the two fault sizes left on this fold's training side."""
    scores = {}
    for band, df in tables.items():
        remaining = df[~_size_test_mask(df, size)]
        fold_scores = []
        for inner in (s for s in SIZES if s != size):
            mask = _size_test_mask(remaining, inner)
            pred = predict(fit_classifier(remaining[~mask]), remaining[mask])
            fold_scores.append(score(pred)["window_macro_f1"])
        scores[band] = float(np.mean(fold_scores))
    return max(scores, key=scores.get)


def leave_one_size_out(tables: dict, features=ALL_FEATURES) -> dict:
    """Stress test: the fault specimen itself is unseen.

    CWRU reuses one seeded bearing across all four loads, so a load split still shows the model
    the same defect. Here every recording of one fault size is held out.
    """
    folds, frames, recordings = [], [], []
    for size in SIZES:
        band = _select_band_for_size(tables, size)
        df = tables[band]
        mask = _size_test_mask(df, size)
        pred = predict(fit_classifier(df[~mask], features), df[mask], features)
        folds.append({"held_out_size_in": size, "band_hz": list(band), **score(pred)})
        frames.append(pred.frame)
        recordings.append(pred.recordings)
    pooled = Prediction(pd.concat(frames, ignore_index=True), pd.concat(recordings, ignore_index=True))
    return {"folds": folds, "pooled": score(pooled), "prediction": pooled}


def random_window_split(df: pd.DataFrame) -> dict:
    """The WRONG way, kept only to show how much it flatters the model: overlapping windows
    of the same recording land on both sides of the split."""
    train, test = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
    return score(predict(fit_classifier(train), test))


def confusion(pred: Prediction, level: str = "recordings") -> np.ndarray:
    data = pred.recordings if level == "recordings" else pred.frame
    return confusion_matrix(data["label"], data["pred"], labels=list(CLASSES))


def expected_calibration_error(pred: Prediction, bins: int = 10) -> float:
    """Average gap between stated probability and actual accuracy, over windows."""
    frame = pred.frame
    correct = (frame["label"] == frame["pred"]).to_numpy(dtype=float)
    confidence = frame["top_prob"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    return float(sum(abs(correct[index == b].mean() - confidence[index == b].mean()) * np.mean(index == b)
                     for b in range(bins) if (index == b).any()))


def reliability(pred: Prediction) -> dict:
    """How often each confidence label was right, per recording."""
    table = {}
    for label in ("high", "medium", "low"):
        rows = pred.recordings[pred.recordings["confidence"] == label]
        table[label] = {"recordings": int(len(rows)), "correct": int((rows["label"] == rows["pred"]).sum())}
    return table


# ------------------------------------------------------------------ inference
@dataclass
class DiagnosisBundle:
    """The shipped classifier and what it needs to judge new input."""
    model: HistGradientBoostingClassifier
    band: tuple[int, int]
    train_loads: list[int]
    feature_low: dict[str, float]    # 0.5th percentile of each training feature
    feature_high: dict[str, float]   # 99.5th percentile
    sklearn_version: str


def train_bundle(tables: dict, train_loads: list[int]) -> DiagnosisBundle:
    band, _ = select_band(tables, train_loads)
    df = tables[band]
    train = df[df["load_hp"].isin(train_loads)]
    columns = list(ALL_FEATURES)
    return DiagnosisBundle(fit_classifier(train), band, train_loads,
                           train[columns].quantile(0.005).to_dict(), train[columns].quantile(0.995).to_dict(),
                           sklearn.__version__)


def prepare_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """Bring an uploaded signal to the analysis rate, or refuse it."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or x.std() == 0:
        raise ValueError("signal is constant or contains invalid values")
    if fs < TARGET_FS:
        raise ValueError(f"sample rate {fs:g} Hz is below the required {TARGET_FS} Hz")
    if fs != TARGET_FS:
        ratio = Fraction(TARGET_FS / fs).limit_denominator(1000)
        x = sg.resample_poly(x, ratio.numerator, ratio.denominator)
    if x.size < int(WINDOW_S * TARGET_FS):
        raise ValueError(f"signal is shorter than the {WINDOW_S:g} s analysis window")
    return x


def mean_envelope(x: np.ndarray, band: tuple[int, int]) -> EnvelopeSpectrum:
    """Envelope spectrum averaged over windows: steadier than any single window, for plotting."""
    specs = [envelope_spectrum(w, TARGET_FS, band) for w in windows(x, int(WINDOW_S * TARGET_FS))]
    return EnvelopeSpectrum(specs[0].freqs, np.mean([s.amplitude for s in specs], axis=0))


def diagnose(x: np.ndarray, fs: float, rpm: float, geometry: BearingGeometry, bundle: DiagnosisBundle) -> dict:
    """Classify one signal and report the evidence for and against that call."""
    signal = prepare_signal(x, fs)
    table = pd.DataFrame([extract_features(w, TARGET_FS, rpm, geometry, bundle.band)
                          for w in windows(signal, int(WINDOW_S * TARGET_FS))])
    columns = list(ALL_FEATURES)
    probabilities = bundle.model.predict_proba(table[columns])
    classes = list(bundle.model.classes_)
    mean_probs = probabilities.mean(axis=0)
    predicted = classes[int(mean_probs.argmax())]
    vote_share = float(np.mean(probabilities.argmax(axis=1) == int(mean_probs.argmax())))

    levels = {k: float(table[f"{k}_mean_db"].mean()) for k in FAULT_KEYS}
    check = envelope_check(predicted, levels)

    warnings = []
    if not RIG_RPM_RANGE[0] <= rpm <= RIG_RPM_RANGE[1]:
        warnings.append(f"Shaft speed {rpm:g} rpm is outside the training rig range "
                        f"{RIG_RPM_RANGE[0]:g} to {RIG_RPM_RANGE[1]:g} rpm.")
    if geometry != SKF_6205:
        warnings.append(f"Bearing {geometry.name} differs from the training bearing {SKF_6205.name}.")
    low, high = pd.Series(bundle.feature_low), pd.Series(bundle.feature_high)
    outside = float(((table[columns] < low[columns]) | (table[columns] > high[columns])).to_numpy().mean())
    if outside > MAX_OUT_OF_RANGE_SHARE:
        warnings.append(f"{outside:.0%} of feature values fall outside the training range.")

    freqs = fault_frequencies(geometry, rpm)
    evidence = {k: {"expected_hz": round(freqs[k], 1),
                    **{f"harmonic_{h}_db": round(float(table[f"{k}_h{h}_db"].mean()), 1)
                       for h in range(1, N_HARMONICS + 1)},
                    "mean_db": round(levels[k], 1)} for k in FAULT_KEYS}
    return {
        "predicted_class": predicted,
        "class_probabilities": {c: round(float(p), 3) for c, p in zip(classes, mean_probs)},
        "window_vote_share": round(vote_share, 3),
        "windows_analysed": int(len(table)),
        "shaft_speed_rpm": rpm,
        "shaft_frequency_hz": round(freqs["shaft"], 2),
        "time_domain": {"rms_g": round(float(table["rms"].mean()), 3),
                        "kurtosis": round(float(table["kurtosis"].mean()), 2),
                        "crest_factor": round(float(table["crest_factor"].mean()), 2)},
        "envelope_band_hz": list(bundle.band),
        "envelope_evidence": evidence,
        "evidence_threshold_db": EVIDENCE_DB,
        "envelope_check": check,
        "confidence": confidence_label(float(mean_probs.max()), vote_share, check, warnings),
        "warnings": warnings,
    }
