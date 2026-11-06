"""Bearing physics and signal features, checked against known answers."""
import numpy as np
import pytest

from pm_agent.bearing_physics import SKF_6205, fault_frequencies, fault_orders
from pm_agent.config import RECORDINGS
from pm_agent.features import (ALL_FEATURES, CANDIDATE_BANDS, envelope_spectrum, extract_features, peak_above_floor,
                               time_features)
from pm_agent.synth import bearing_signal

FS = 12_000


def test_fault_orders_match_the_published_cwru_multiples():
    orders = fault_orders(SKF_6205)
    assert orders["bpfo"] == pytest.approx(3.5848, abs=1e-3)
    assert orders["bpfi"] == pytest.approx(5.4152, abs=1e-3)
    assert orders["ftf"] == pytest.approx(0.39828, abs=1e-3)
    assert 2 * orders["bsf"] == pytest.approx(4.7135, abs=1e-3)  # CWRU lists 2 x BSF


def test_fault_frequencies_scale_with_speed():
    slow, fast = fault_frequencies(SKF_6205, 900), fault_frequencies(SKF_6205, 1800)
    assert fast["shaft"] == pytest.approx(30.0)
    assert fast["bpfo"] == pytest.approx(2 * slow["bpfo"])


def test_rpm_must_be_positive():
    with pytest.raises(ValueError):
        fault_frequencies(SKF_6205, 0)


def test_time_features_on_a_sine():
    t = np.arange(FS) / FS
    features = time_features(np.sin(2 * np.pi * 50 * t))
    assert features["rms"] == pytest.approx(1 / np.sqrt(2), abs=1e-3)
    assert features["crest_factor"] == pytest.approx(np.sqrt(2), abs=1e-2)
    assert features["kurtosis"] == pytest.approx(-1.5, abs=1e-2)  # a sine is not impulsive


def test_impacts_raise_kurtosis():
    healthy = time_features(bearing_signal(None))["kurtosis"]
    faulty = time_features(bearing_signal(100.0))["kurtosis"]
    assert faulty > healthy + 5


@pytest.mark.parametrize("key", ["bpfo", "bpfi"])
def test_envelope_spectrum_peaks_at_the_injected_fault_frequency(key):
    target = fault_frequencies(SKF_6205, 1750)[key]
    spec = envelope_spectrum(bearing_signal(target, seconds=2.0), FS, (2000, 4000))
    low_cut = spec.freqs > 20
    strongest = spec.freqs[low_cut][np.argmax(spec.amplitude[low_cut])]
    assert strongest == pytest.approx(target, abs=2.0)
    assert peak_above_floor(spec, target)[0] > 15


def test_a_clean_signal_has_no_fault_line():
    spec = envelope_spectrum(bearing_signal(None, seconds=2.0), FS, (2000, 4000))
    target = fault_frequencies(SKF_6205, 1750)["bpfo"]
    assert peak_above_floor(spec, target)[0] < 10


def test_band_must_fit_under_nyquist():
    with pytest.raises(ValueError):
        envelope_spectrum(np.ones(100), 8000, (3000, 5000))


def test_candidate_bands_stay_clear_of_the_resampling_filter():
    assert all(high <= 5000 for _, high in CANDIDATE_BANDS)


def test_feature_names_match_what_is_extracted():
    features = extract_features(bearing_signal(100.0, seconds=1.0), FS, 1750, SKF_6205, (2000, 4000))
    assert set(features) == set(ALL_FEATURES)
    assert all(np.isfinite(v) for v in features.values())


def test_recording_table_covers_the_expected_design():
    assert len(RECORDINGS) == 40
    assert len({r.file_id for r in RECORDINGS}) == 40
    assert sum(r.label == "normal" for r in RECORDINGS) == 4
    assert all(r.source_fs == 48_000 for r in RECORDINGS if r.label == "normal")
    assert all(r.source_fs == 12_000 for r in RECORDINGS if r.label != "normal")
