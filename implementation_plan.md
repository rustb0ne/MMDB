# Animal Sound MMDB System — Implementation Plan

## Project Overview
Build an animal sound database and retrieval system (MMDB Assignment).

## Architecture
```
d:\SourceCode\MMDB\
├── dataset/                  # 500+ animal sound audio files (.wav/.mp3)
│   ├── cat/
│   ├── dog/
│   ├── bird/
│   └── ...
├── features_db/              # Extracted features stored as JSON/CSV
│   └── features.json
├── src/
│   ├── collect_dataset.py    # Script to download/verify dataset
│   ├── extract_features.py   # Feature extraction pipeline
│   ├── build_database.py     # Store features in DB (SQLite or JSON)
│   ├── retrieval.py          # Query & cosine similarity engine
│   └── app.py                # Web UI (Flask or Streamlit)
├── query/                    # Folder to place query audio files
├── requirements.txt
└── README.md
```

## Dataset Plan
- Source: ESC-50 (Environmental Sound Classification, 2000 clips × 5s)
  - 50 classes, many animal categories
  - Supplemented with UrbanSound8K animal classes
- Target: ≥500 animal-related audio clips
- Format: WAV, 44100 Hz, mono

## Features (10 feature groups, adapted for animals)
1. **Frequency** — Mean frequency, peak frequency (via FFT)
2. **Amplitude** — Mean, max, min, variance of amplitude
3. **Temporal** — Duration, sample rate, zero-crossing rate
4. **Spectral** — MFCC (13 coefficients), Spectral Centroid, Bandwidth, Contrast, Rolloff
5. **Waveform** — Variance, % time above threshold
6. **Complexity** — Number of peaks, number of valleys
7. **Timbre** — Max spectrum, mean spectrum (from FFT magnitudes)
8. **Brightness** — Brightness ratio, Spectral Flatness
9. **Attack** — Attack time, max frequency magnitude
10. **Decay** — Decay ratio, volume decay 500ms

## Similarity Metric
- Cosine Similarity per feature group → Average → Rank top-5

## Tech Stack
- Python 3.10+
- librosa (audio processing + feature extraction)
- numpy, scipy (numerical)
- pandas (tabular data)
- sqlite3 (feature storage)
- Flask + HTML/CSS/JS (Web UI)
- ESC-50 dataset (free, CC BY license)

## Deliverables
1. Dataset (500+ files)
2. Feature extraction script + stored DB
3. Retrieval engine (top-5 most similar)
4. Web UI demo
5. Report documentation
