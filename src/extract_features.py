"""
Feature Extraction Pipeline for Animal Sound MMDB
===================================================
Extracts 10 feature groups from each audio file and stores
them in SQLite + JSON for retrieval.

Feature Groups (matching reference system design):
  1.  Frequency        — mean_freq, peak_freq
  2.  Amplitude        — mean_amp, max_amp, min_amp, var_amp
  3.  Temporal         — duration, sample_rate, zero_crossing_rate
  4.  Spectral         — mfcc (13 coeffs), centroid, bandwidth, contrast, rolloff
  5.  Waveform         — variance, pct_above_threshold
  6.  Complexity       — num_peaks, num_valleys
  7.  Timbre           — max_spectrum, mean_spectrum
  8.  Brightness       — brightness_ratio, spectral_flatness
  9.  Attack           — attack_time, attack_max_magnitude
  10. Decay            — decay_ratio, vol_decay_500ms
"""

import os
import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import find_peaks
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DB = BASE_DIR / "features_db"
DB_PATH = FEATURES_DB / "animal_sounds.db"
JSON_PATH = FEATURES_DB / "features.json"

SAMPLE_RATE = 22050   # librosa default; enough for animal sounds
N_MFCC = 13
BRIGHTNESS_THRESHOLD_HZ = 1500  # frequencies above this count as "bright"
AMPLITUDE_THRESHOLD = 0.1       # waveform threshold for % above


# ── Individual Feature Extractors ─────────────────────────────────────────────

def extract_frequency_features(fft_magnitudes: np.ndarray, freqs: np.ndarray) -> dict:
    """
    Feature Group 1 — Frequency
    Uses the FFT magnitude spectrum to compute overall frequency characteristics.
    """
    mean_freq = float(np.average(freqs, weights=np.abs(fft_magnitudes) + 1e-9))
    peak_freq = float(freqs[np.argmax(np.abs(fft_magnitudes))])
    return {
        "mean_freq": mean_freq,
        "peak_freq": peak_freq,
    }


def extract_amplitude_features(y: np.ndarray) -> dict:
    """
    Feature Group 2 — Amplitude
    Works directly on the raw waveform samples.
    """
    return {
        "mean_amp": float(np.mean(np.abs(y))),
        "max_amp": float(np.max(np.abs(y))),
        "min_amp": float(np.min(np.abs(y))),
        "var_amp": float(np.var(y)),
    }


def extract_temporal_features(y: np.ndarray, sr: int) -> dict:
    """
    Feature Group 3 — Temporal
    Duration, sample rate, and zero-crossing rate.
    """
    duration = float(len(y) / sr)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    return {
        "duration": duration,
        "sample_rate": float(sr),
        "zero_crossing_rate": zcr,
    }


def extract_spectral_features(y: np.ndarray, sr: int) -> dict:
    """
    Feature Group 4 — Spectral
    MFCCs (13 coefficients mean), Centroid, Bandwidth, Contrast, Rolloff.
    All returned as 1-D vectors/scalars for cosine similarity computation.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfccs.mean(axis=1).tolist()

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).tolist()
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    return {
        "mfcc": mfcc_mean,         # list[13]
        "centroid": centroid,
        "bandwidth": bandwidth,
        "contrast": contrast,      # list[7]
        "rolloff": rolloff,
    }


def extract_waveform_features(y: np.ndarray, threshold: float = AMPLITUDE_THRESHOLD) -> dict:
    """
    Feature Group 5 — Waveform
    Variance and percentage of samples above a given amplitude threshold.
    """
    variance = float(np.var(y))
    pct_above = float(np.mean(np.abs(y) > threshold))
    return {
        "variance": variance,
        "pct_above_threshold": pct_above,
    }


def extract_complexity_features(y: np.ndarray) -> dict:
    """
    Feature Group 6 — Complexity
    Number of local maxima (peaks) and minima (valleys) in the waveform.
    Down-sampled to 1000 points for efficiency.
    """
    # Down-sample to speed up peak finding
    step = max(1, len(y) // 1000)
    y_ds = y[::step]
    peaks, _ = find_peaks(y_ds)
    valleys, _ = find_peaks(-y_ds)
    return {
        "num_peaks": len(peaks),
        "num_valleys": len(valleys),
    }


def extract_timbre_features(fft_magnitudes: np.ndarray) -> dict:
    """
    Feature Group 7 — Timbre
    Statistics of the FFT magnitude spectrum (max and mean).
    """
    mag = np.abs(fft_magnitudes)
    return {
        "max_spectrum": float(np.max(mag)),
        "mean_spectrum": float(np.mean(mag)),
    }


def extract_brightness_features(
    fft_magnitudes: np.ndarray, freqs: np.ndarray, threshold_hz: float = BRIGHTNESS_THRESHOLD_HZ
) -> dict:
    """
    Feature Group 8 — Brightness
    Fraction of spectral energy above threshold_hz, and spectral flatness.
    """
    mag = np.abs(fft_magnitudes)
    total_energy = np.sum(mag ** 2) + 1e-9
    bright_energy = np.sum((mag[freqs >= threshold_hz]) ** 2)
    brightness = float(bright_energy / total_energy)

    # Spectral flatness: geometric mean / arithmetic mean of power spectrum
    power = mag ** 2 + 1e-9
    geo_mean = float(np.exp(np.mean(np.log(power))))
    arith_mean = float(np.mean(power))
    flatness = float(geo_mean / arith_mean)

    return {
        "brightness_ratio": brightness,
        "spectral_flatness": flatness,
    }


def extract_attack_features(y: np.ndarray, sr: int) -> dict:
    """
    Feature Group 9 — Attack
    Attack time = time to reach global amplitude maximum.
    Also stores the FFT magnitude at the peak time.
    """
    env = np.abs(y)
    peak_idx = int(np.argmax(env))
    attack_time = float(peak_idx / sr)
    attack_max_magnitude = float(env[peak_idx])
    return {
        "attack_time": attack_time,
        "attack_max_magnitude": attack_max_magnitude,
    }


def extract_decay_features(y: np.ndarray, sr: int, window_ms: int = 500) -> dict:
    """
    Feature Group 10 — Decay
    Decay ratio: ratio of RMS energy in the last 500 ms vs peak 500 ms window.
    Also computes the raw RMS drop.
    """
    window = int(sr * window_ms / 1000)
    env = np.abs(y)
    peak_idx = int(np.argmax(env))

    # RMS in peak window
    peak_start = max(0, peak_idx - window // 2)
    peak_end = min(len(y), peak_idx + window // 2)
    rms_peak = float(np.sqrt(np.mean(y[peak_start:peak_end] ** 2)) + 1e-9)

    # RMS in last 500 ms
    tail = y[-window:] if len(y) >= window else y
    rms_tail = float(np.sqrt(np.mean(tail ** 2)) + 1e-9)

    decay_ratio = float(rms_tail / rms_peak)
    vol_decay = float(rms_peak - rms_tail)

    return {
        "decay_ratio": decay_ratio,
        "vol_decay_500ms": vol_decay,
    }


# ── Unified Extractor ─────────────────────────────────────────────────────────

def extract_all_features(filepath: str | Path) -> dict | None:
    """
    Load audio and extract all 10 feature groups.
    Returns a dict of feature_group → feature_dict, or None on error.
    """
    try:
        y, sr = librosa.load(str(filepath), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"[ERROR] Could not load {filepath}: {e}")
        return None

    # Compute FFT once for all frequency-domain features
    fft_result = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    features = {
        "frequency": extract_frequency_features(fft_result, freqs),
        "amplitude": extract_amplitude_features(y),
        "temporal": extract_temporal_features(y, sr),
        "spectral": extract_spectral_features(y, sr),
        "waveform": extract_waveform_features(y),
        "complexity": extract_complexity_features(y),
        "timbre": extract_timbre_features(fft_result),
        "brightness": extract_brightness_features(fft_result, freqs),
        "attack": extract_attack_features(y, sr),
        "decay": extract_decay_features(y, sr),
    }
    return features


# ── Flat Feature Vector Builder ────────────────────────────────────────────────

def flatten_features(features: dict) -> dict[str, list[float]]:
    """
    Convert each feature group into a flat numeric vector for cosine similarity.
    Returns {group_name: [float, ...]}
    """
    vectors: dict[str, list[float]] = {}

    for group, vals in features.items():
        vec = []
        for v in vals.values():
            if isinstance(v, list):
                vec.extend([float(x) for x in v])
            else:
                vec.append(float(v))
        vectors[group] = vec

    return vectors


# ── Database Storage ───────────────────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with features table."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_features (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT NOT NULL,
            species       TEXT NOT NULL,
            filepath      TEXT NOT NULL,
            -- Raw feature group JSON blobs
            frequency     TEXT,
            amplitude     TEXT,
            temporal      TEXT,
            spectral      TEXT,
            waveform      TEXT,
            complexity    TEXT,
            timbre        TEXT,
            brightness    TEXT,
            attack        TEXT,
            decay         TEXT,
            -- Flat vectors for fast cosine similarity (JSON arrays)
            vec_frequency TEXT,
            vec_amplitude TEXT,
            vec_temporal  TEXT,
            vec_spectral  TEXT,
            vec_waveform  TEXT,
            vec_complexity TEXT,
            vec_timbre    TEXT,
            vec_brightness TEXT,
            vec_attack    TEXT,
            vec_decay     TEXT
        )
        """
    )
    conn.commit()
    return conn


def insert_record(conn: sqlite3.Connection, filename: str, species: str, filepath: str, features: dict):
    """Insert one audio file's features into the DB."""
    vectors = flatten_features(features)
    conn.execute(
        """
        INSERT INTO audio_features
          (filename, species, filepath,
           frequency, amplitude, temporal, spectral, waveform,
           complexity, timbre, brightness, attack, decay,
           vec_frequency, vec_amplitude, vec_temporal, vec_spectral, vec_waveform,
           vec_complexity, vec_timbre, vec_brightness, vec_attack, vec_decay)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            filename, species, str(filepath),
            json.dumps(features["frequency"]),
            json.dumps(features["amplitude"]),
            json.dumps(features["temporal"]),
            json.dumps(features["spectral"]),
            json.dumps(features["waveform"]),
            json.dumps(features["complexity"]),
            json.dumps(features["timbre"]),
            json.dumps(features["brightness"]),
            json.dumps(features["attack"]),
            json.dumps(features["decay"]),
            json.dumps(vectors["frequency"]),
            json.dumps(vectors["amplitude"]),
            json.dumps(vectors["temporal"]),
            json.dumps(vectors["spectral"]),
            json.dumps(vectors["waveform"]),
            json.dumps(vectors["complexity"]),
            json.dumps(vectors["timbre"]),
            json.dumps(vectors["brightness"]),
            json.dumps(vectors["attack"]),
            json.dumps(vectors["decay"]),
        ),
    )


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def build_database():
    FEATURES_DB.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        print("[ERROR] Dataset directory is empty. Run collect_dataset.py first.")
        return

    conn = init_db(DB_PATH)

    # Clear existing data so we can re-run idempotently
    conn.execute("DELETE FROM audio_features")
    conn.commit()

    extensions = {".wav", ".mp3", ".ogg", ".flac"}
    audio_files = [
        p for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in extensions
    ]

    print(f"[INFO] Found {len(audio_files)} audio files to process.")
    all_records = []

    for filepath in tqdm(audio_files, desc="Extracting features"):
        species = filepath.parent.name  # folder name = species
        features = extract_all_features(filepath)
        if features is None:
            continue

        insert_record(conn, filepath.name, species, filepath, features)

        # Also collect for JSON dump
        all_records.append({
            "filename": filepath.name,
            "species": species,
            "filepath": str(filepath),
            "features": features,
            "vectors": flatten_features(features),
        })

    conn.commit()
    conn.close()

    # Save JSON backup
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Database built:")
    print(f"       SQLite  -> {DB_PATH}")
    print(f"       JSON    -> {JSON_PATH}")
    print(f"       Records -> {len(all_records)}")


if __name__ == "__main__":
    build_database()
