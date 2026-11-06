"""The dataset quirks, confirmed on the real files. Skipped when data/raw is absent."""
import hashlib

import numpy as np
import pytest
import scipy.io as sio

from pm_agent.config import CWRU_DIR, RECORDINGS, TARGET_FS
from pm_agent.data_cwru import load_recording, to_target_rate, windows

needs_data = pytest.mark.skipif(not (CWRU_DIR / "97.mat").exists(), reason="CWRU data not downloaded")
NORMAL = [r for r in RECORDINGS if r.label == "normal"]


def strongest_near(x, fs, low, high):
    x = (x - x.mean()) * np.hanning(x.size)
    freqs = np.fft.rfftfreq(x.size, 1 / fs)
    amplitude = np.abs(np.fft.rfft(x))
    band = (freqs > low) & (freqs < high)
    return freqs[band][np.argmax(amplitude[band])]


@needs_data
@pytest.mark.parametrize("rec", NORMAL, ids=lambda r: r.name)
def test_normal_files_are_48k_because_their_shaft_line_tracks_speed(rec):
    raw = np.ravel(sio.loadmat(CWRU_DIR / f"{rec.file_id}.mat")[f"X{rec.file_id:03d}_DE_time"]).astype(float)
    shaft_hz = load_recording(rec).rpm / 60
    assert strongest_near(raw, 48_000, 27.5, 31.5) == pytest.approx(shaft_hz, abs=0.1)


@needs_data
def test_read_at_12k_the_line_sticks_at_30_hz_whatever_the_speed():
    """It is really the fixed 120 Hz mains line, seen at a quarter of its frequency."""
    for rec in NORMAL[1:]:
        raw = np.ravel(sio.loadmat(CWRU_DIR / f"{rec.file_id}.mat")[f"X{rec.file_id:03d}_DE_time"]).astype(float)
        assert strongest_near(raw, 12_000, 27.5, 31.5) == pytest.approx(30.0, abs=0.1)


@needs_data
def test_file_99_loads_its_own_signal_and_not_the_copy_of_98():
    copy = np.ravel(sio.loadmat(CWRU_DIR / "99.mat")["X098_DE_time"]).astype(float)
    original = np.ravel(sio.loadmat(CWRU_DIR / "98.mat")["X098_DE_time"]).astype(float)
    assert np.array_equal(copy, original)  # the trap really exists in the file
    load2 = load_recording(next(r for r in RECORDINGS if r.file_id == 99))
    load1 = load_recording(next(r for r in RECORDINGS if r.file_id == 98))
    length = min(load1.x.size, load2.x.size)
    assert not np.allclose(load1.x[:length], load2.x[:length])


@needs_data
def test_no_two_recordings_share_a_signal():
    digests = {hashlib.sha256(load_recording(r).x.tobytes()).hexdigest() for r in RECORDINGS}
    assert len(digests) == len(RECORDINGS)


@needs_data
def test_missing_rpm_falls_back_to_nominal_and_says_so():
    loaded = load_recording(next(r for r in RECORDINGS if r.file_id == 98))
    assert (loaded.rpm, loaded.rpm_source) == (1772.0, "nominal")
    assert load_recording(RECORDINGS[0]).rpm_source == "file"


def test_resampling_keeps_duration_and_a_tone():
    t = np.arange(48_000) / 48_000
    out = to_target_rate(np.sin(2 * np.pi * 100 * t), 48_000)
    assert out.size == TARGET_FS
    assert strongest_near(out, TARGET_FS, 50, 150) == pytest.approx(100, abs=1)


def test_windows_overlap_and_drop_the_tail():
    assert windows(np.arange(25.0), 10, 0.5).shape == (4, 10)
    assert windows(np.arange(5.0), 10).shape[0] == 0
