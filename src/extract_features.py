"""
Feature Extraction Pipeline — Animal Sound MMDB (v2)
=====================================================
Pre-processing pipeline (theo tài liệu tham khảo):
  Raw WAV -> Downmix Mono -> Resample (22050 Hz)
  -> Butterworth Bandpass (50-15000 Hz)
  -> Silence Trimming (top_db=30)
  -> Pre-emphasis (coef=0.97)
  -> Amplitude Normalization
  -> Framing & Windowing (inside librosa)
  -> FFT -> Feature Extraction

Feature Groups (10 nhóm, mỗi nhóm lưu mean + std):
  1.  MFCC             — 13 hệ số × {mean, std} = 26 dim
  2.  Spectral Centroid — {mean, std}
  3.  Spectral Bandwidth — {mean, std}
  4.  Spectral Rolloff  — {mean, std}
  5.  Spectral Contrast — 7 bands × {mean, std} = 14 dim
  6.  ZCR              — {mean, std}
  7.  RMS Energy       — {mean, std}
  8.  Chroma           — 12 bins × {mean, std} = 24 dim
  9.  Attack           — attack_time, attack_rms
  10. Decay            — decay_ratio, vol_decay_500ms

Storage:
  SQLite (features_db/animal_sounds.db) — structured, fast query
  JSON   (features_db/features.json)   — human-readable backup
"""

import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DB = BASE_DIR / "features_db"
DB_PATH     = FEATURES_DB / "animal_sounds.db"
JSON_PATH   = FEATURES_DB / "features.json"

SR          = 22050      # target sample rate (Hz)
N_MFCC      = 13         # số hệ số MFCC
N_CHROMA    = 12         # bins chroma
BP_LOW      = 50         # bandpass low  (Hz)
BP_HIGH     = 15000      # bandpass high (Hz)
BP_ORDER    = 4          # Butterworth order
PRE_EMPH    = 0.97       # pre-emphasis coefficient
TOP_DB      = 30         # silence trim threshold (dB)
DECAY_MS    = 500        # window for decay measurement (ms)


# ═══════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1: PRE-PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(y: np.ndarray, sr: int,
                    low: float = BP_LOW, high: float = BP_HIGH,
                    order: int = BP_ORDER) -> np.ndarray:
    """
    Lọc dải thông Butterworth: giữ lại [low, high] Hz.
    Loại bỏ tiếng gió thấp tần (<50 Hz) và nhiễu điện từ cao tần (>15kHz).
    """
    nyq  = sr / 2.0
    low_n  = low  / nyq
    high_n = min(high / nyq, 0.9999)          # tránh vượt Nyquist
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, y)                   # zero-phase filtering


def trim_silence(y: np.ndarray, sr: int, top_db: int = TOP_DB) -> np.ndarray:
    """
    Tỉa khoảng lặng: loại bỏ đoạn im lặng ở đầu và cuối file.
    Tránh khoảng lặng làm sai lệch mean của các đặc trưng.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    # Nếu trim quá mạnh (file quá ngắn), giữ nguyên
    if len(y_trimmed) < sr * 0.5:
        return y
    return y_trimmed


def pre_emphasis(y: np.ndarray, coef: float = PRE_EMPH) -> np.ndarray:
    """
    Pre-emphasis: y[n] = y[n] - coef * y[n-1]
    Tăng cường tần số cao để bù trừ hiện tượng tắt dần tự nhiên.
    Giúp MFCC và Spectral features ổn định hơn.
    """
    return np.append(y[0], y[1:] - coef * y[:-1])


def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """Chuẩn hóa biên độ về [-1, 1] (peak normalization)."""
    peak = np.max(np.abs(y))
    if peak > 1e-9:
        return y / peak
    return y


def preprocess(filepath: str | Path) -> tuple[np.ndarray, int] | None:
    """
    Full pre-processing pipeline:
      Raw -> Mono -> Resample -> Bandpass -> Trim silence
      -> Pre-emphasis -> Normalize
    """
    try:
        y, sr = librosa.load(str(filepath), sr=SR, mono=True)
    except Exception as e:
        print(f"[ERROR] Load failed: {filepath} — {e}")
        return None

    y = bandpass_filter(y, sr)       # Step 1: Butterworth bandpass
    y = trim_silence(y, sr)          # Step 2: Silence trimming
    y = pre_emphasis(y)              # Step 3: Pre-emphasis filter
    y = normalize_amplitude(y)       # Step 4: Peak normalization

    if len(y) < sr * 0.3:           # Bỏ qua file quá ngắn (<0.3s)
        print(f"[WARN] Too short after trim: {filepath}")
        return None

    return y, sr


# ═══════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2: FEATURE EXTRACTION
# Mỗi đặc trưng time-varying → mean + std (theo khuyến nghị tài liệu)
# ═══════════════════════════════════════════════════════════════════════════════

def feat_mfcc(y: np.ndarray, sr: int) -> dict:
    """
    Group 1 — MFCC (Mel-Frequency Cepstral Coefficients)
    13 hệ số, mỗi hệ số → mean + std theo thời gian = 26 chiều.
    Mô phỏng cách tai người cảm nhận âm thanh (thang Mel phi tuyến).
    Nắm bắt "hình bao phổ" — dấu vân tay âm học của mỗi loài.
    """
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)   # (13, T)
    return {
        "mfcc_mean": M.mean(axis=1).tolist(),               # list[13]
        "mfcc_std":  M.std(axis=1).tolist(),                # list[13]
    }


def feat_spectral_centroid(y: np.ndarray, sr: int) -> dict:
    """
    Group 2 — Spectral Centroid (Trọng tâm phổ)
    Tần số "trung tâm khối lượng" của phổ.
    Âm sáng (chim) → centroid cao; âm trầm (bò, voi) → centroid thấp.
    """
    C = librosa.feature.spectral_centroid(y=y, sr=sr)[0]   # (T,)
    return {
        "centroid_mean": float(C.mean()),
        "centroid_std":  float(C.std()),
    }


def feat_spectral_bandwidth(y: np.ndarray, sr: int) -> dict:
    """
    Group 3 — Spectral Bandwidth (Băng thông phổ)
    Độ rộng phân tán quanh centroid.
    Kết hợp centroid + bandwidth → mô tả hình dáng phổ chi tiết.
    """
    B = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    return {
        "bandwidth_mean": float(B.mean()),
        "bandwidth_std":  float(B.std()),
    }


def feat_spectral_rolloff(y: np.ndarray, sr: int) -> dict:
    """
    Group 4 — Spectral Rolloff
    Tần số tại đó 85% năng lượng phổ đã được tích luỹ.
    Phân biệt âm dải hẹp (dế kêu) vs dải rộng (hổ gầm).
    """
    R = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    return {
        "rolloff_mean": float(R.mean()),
        "rolloff_std":  float(R.std()),
    }


def feat_spectral_contrast(y: np.ndarray, sr: int) -> dict:
    """
    Group 5 — Spectral Contrast (Tương phản phổ)
    Chênh lệch giữa đỉnh và đáy trong 7 dải octave.
    Phân biệt âm thanh có cấu trúc hài âm vs nhiễu ngẫu nhiên.
    """
    SC = librosa.feature.spectral_contrast(y=y, sr=sr)    # (7, T)
    return {
        "contrast_mean": SC.mean(axis=1).tolist(),         # list[7]
        "contrast_std":  SC.std(axis=1).tolist(),          # list[7]
    }


def feat_zcr(y: np.ndarray) -> dict:
    """
    Group 6 — Zero-Crossing Rate (Tỷ lệ cắt điểm không)
    Đo số lần tín hiệu đổi dấu trong mỗi frame.
    ZCR cao = âm tạp (côn trùng, rắn khè).
    ZCR thấp = âm tông thuần (bò, sói hú).
    """
    Z = librosa.feature.zero_crossing_rate(y)[0]          # (T,)
    return {
        "zcr_mean": float(Z.mean()),
        "zcr_std":  float(Z.std()),
    }


def feat_rms(y: np.ndarray) -> dict:
    """
    Group 7 — RMS Energy (Năng lượng hiệu dụng)
    Root Mean Square của từng frame — đường bao năng lượng.
    Cơ sở để tính Attack và Decay chính xác.
    """
    E = librosa.feature.rms(y=y)[0]                       # (T,)
    return {
        "rms_mean": float(E.mean()),
        "rms_std":  float(E.std()),
    }


def feat_chroma(y: np.ndarray, sr: int) -> dict:
    """
    Group 8 — Chroma Features (Đặc trưng hòa âm)
    12 bins tương ứng 12 nốt nhạc trong 1 octave.
    Phân biệt chim hót (có cấu trúc nốt) vs tiếng ồn (chroma đều).
    """
    CH = librosa.feature.chroma_stft(y=y, sr=sr)          # (12, T)
    return {
        "chroma_mean": CH.mean(axis=1).tolist(),           # list[12]
        "chroma_std":  CH.std(axis=1).tolist(),            # list[12]
    }


def feat_attack(y: np.ndarray, sr: int) -> dict:
    """
    Group 9 — Attack (Độ chớp nhoáng)
    Thời gian tín hiệu đạt RMS cực đại từ khi bắt đầu.
    Attack nhanh = chó sủa; chậm = bò rống từ từ.
    Dựa trên RMS Envelope (chính xác hơn biên độ thô).
    """
    hop   = 512
    rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
    peak  = int(np.argmax(rms))
    attack_time = float(peak * hop / sr)
    attack_rms  = float(rms[peak])
    return {
        "attack_time": attack_time,
        "attack_rms":  attack_rms,
    }


def feat_decay(y: np.ndarray, sr: int, window_ms: int = DECAY_MS) -> dict:
    """
    Group 10 — Decay (Độ suy giảm)
    Tỷ lệ RMS cuối file / RMS đỉnh. decay_ratio gần 1 = duy trì lâu.
    vol_decay_500ms = lượng RMS giảm trong 500ms sau đỉnh.
    """
    hop    = 512
    rms    = librosa.feature.rms(y=y, hop_length=hop)[0]
    win_f  = max(1, int(sr * window_ms / 1000 / hop))  # frames

    peak   = int(np.argmax(rms))
    rms_pk = float(rms[peak]) + 1e-9

    # RMS trung bình trong win_f frames cuối
    rms_tail = float(rms[-win_f:].mean()) + 1e-9

    return {
        "decay_ratio":    float(rms_tail / rms_pk),
        "vol_decay_500ms": float(rms_pk - rms_tail),
    }


# ── Unified extractor ─────────────────────────────────────────────────────────

def extract_all(filepath: str | Path) -> dict | None:
    """
    Chạy full pre-processing + trích xuất 10 nhóm đặc trưng.
    Trả về dict hoặc None nếu lỗi.
    """
    result = preprocess(filepath)
    if result is None:
        return None
    y, sr = result

    return {
        "mfcc":      feat_mfcc(y, sr),
        "centroid":  feat_spectral_centroid(y, sr),
        "bandwidth": feat_spectral_bandwidth(y, sr),
        "rolloff":   feat_spectral_rolloff(y, sr),
        "contrast":  feat_spectral_contrast(y, sr),
        "zcr":       feat_zcr(y),
        "rms":       feat_rms(y),
        "chroma":    feat_chroma(y, sr),
        "attack":    feat_attack(y, sr),
        "decay":     feat_decay(y, sr),
    }


def flatten(features: dict) -> dict[str, list[float]]:
    """
    Chuyển mỗi nhóm đặc trưng thành vector phẳng 1 chiều.
    Dùng cho tính Cosine Similarity.

    Kích thước vector mỗi nhóm:
      mfcc      : 13 mean + 13 std = 26
      centroid  : 2
      bandwidth : 2
      rolloff   : 2
      contrast  : 7 mean + 7 std = 14
      zcr       : 2
      rms       : 2
      chroma    : 12 mean + 12 std = 24
      attack    : 2
      decay     : 2
      ─────────────────────
      Tổng      : 78 chiều
    """
    vectors: dict[str, list[float]] = {}
    for group, vals in features.items():
        vec = []
        for v in vals.values():
            if isinstance(v, list):
                vec.extend(float(x) for x in v)
            else:
                vec.append(float(v))
        vectors[group] = vec
    return vectors


# ═══════════════════════════════════════════════════════════════════════════════
# STORAGE — SQLite (primary) + JSON (backup)
#
# Tại sao SQLite chứ không phải MongoDB?
#  - Dataset 1000–5000 file: SQLite hoàn toàn đủ tốt (đọc ~1ms/record)
#  - Không cần cài server riêng, file single .db dễ di chuyển
#  - MongoDB phù hợp hơn khi: scale triệu record, cần sharding, nhiều node
#  - Với bài này: SQLite = lựa chọn tối ưu
# ═══════════════════════════════════════════════════════════════════════════════

GROUPS = ["mfcc","centroid","bandwidth","rolloff","contrast",
          "zcr","rms","chroma","attack","decay"]


def init_db(conn: sqlite3.Connection):
    raw_cols  = ", ".join(f"{g} TEXT" for g in GROUPS)
    vec_cols  = ", ".join(f"vec_{g} TEXT" for g in GROUPS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS audio_features (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            species     TEXT NOT NULL,
            filepath    TEXT NOT NULL,
            duration_s  REAL,
            {raw_cols},
            {vec_cols}
        )
    """)
    conn.commit()


def insert(conn: sqlite3.Connection, filename: str, species: str,
           filepath: str, features: dict):
    vecs    = flatten(features)
    raw_q   = ", ".join("?" for _ in GROUPS)
    vec_q   = ", ".join("?" for _ in GROUPS)
    raw_col = ", ".join(GROUPS)
    vec_col = ", ".join(f"vec_{g}" for g in GROUPS)
    raw_v   = [json.dumps(features[g]) for g in GROUPS]
    vec_v   = [json.dumps(vecs[g])     for g in GROUPS]
    dur     = features["attack"]["attack_time"]   # proxy for loaded duration

    conn.execute(
        f"INSERT INTO audio_features (filename,species,filepath,duration_s,{raw_col},{vec_col}) "
        f"VALUES (?,?,?,?,{raw_q},{vec_q})",
        [filename, species, filepath, dur] + raw_v + vec_v
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def build_database():
    FEATURES_DB.mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    files = [p for p in DATASET_DIR.rglob("*") if p.suffix.lower() in exts]
    print(f"[INFO] Found {len(files)} audio files across "
          f"{len(set(p.parent.name for p in files))} species.")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    conn.execute("DELETE FROM audio_features")
    conn.commit()

    records  = []
    errors   = 0

    for fp in tqdm(files, desc="Extracting"):
        species  = fp.parent.name
        features = extract_all(fp)
        if features is None:
            errors += 1
            continue

        # Store path relative to BASE_DIR for portability across machines
        rel_path = fp.relative_to(BASE_DIR).as_posix()  # e.g. "dataset/cat/file.wav"
        insert(conn, fp.name, species, rel_path, features)
        records.append({
            "filename": fp.name,
            "species":  species,
            "filepath": str(fp),
            "features": features,
            "vectors":  flatten(features),
        })

    conn.commit()
    conn.close()

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Stored {len(records)} records | {errors} errors")
    print(f"       SQLite -> {DB_PATH}  ({DB_PATH.stat().st_size//1024} KB)")
    print(f"       JSON   -> {JSON_PATH} ({JSON_PATH.stat().st_size//1024} KB)")
    print(f"       Vector size per file: 78 dimensions")


if __name__ == "__main__":
    build_database()
