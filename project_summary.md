# Animal Sound Retrieval System — Project Summary

## System is Live at http://127.0.0.1:5000

## What Was Built

### Dataset (Requirement 1 ✓)
- **520 audio files** across **13 animal species/categories**
- Source: [ESC-50 dataset](https://github.com/karoldvl/ESC-50) (CC BY license)
- Format: WAV, 5 seconds each, 44100 Hz stereo
- Species: `cat`, `dog`, `cow`, `pig`, `sheep`, `frog`, `hen`, `rooster`, `crow`, `insects`, `crickets`, `chirping_birds`, `crying_baby`

### Features (Requirement 2 ✓)
| # | Group | Features | Rationale |
|---|-------|----------|-----------|
| 1 | **Frequency** | mean_freq, peak_freq | Distinguishes high-pitched birds from low-frequency mammals |
| 2 | **Amplitude** | mean, max, min, variance | Loud barks vs soft chirps differ strongly in amplitude stats |
| 3 | **Temporal** | duration, sample_rate, ZCR | ZCR separates noisy hisses from tonal howls |
| 4 | **Spectral** | MFCC×13, centroid, bandwidth, contrast×7, rolloff | Richest discriminative features — widely used in species ID |
| 5 | **Waveform** | variance, pct_above_threshold | Energy distribution and signal activity density |
| 6 | **Complexity** | num_peaks, num_valleys | Cricket sounds have 10× more peaks than cow moos |
| 7 | **Timbre** | max_spectrum, mean_spectrum | Perceptual "color" of sound — different for each species |
| 8 | **Brightness** | brightness_ratio, spectral_flatness | Birds are brighter; frogs are flatter |
| 9 | **Attack** | attack_time, attack_max_magnitude | Sharp bark vs gradual moo |
| 10 | **Decay** | decay_ratio, vol_decay_500ms | Sustained howl vs short click |

### Retrieval Engine (Requirement 3 ✓)
- **Input:** Any WAV/MP3/OGG/FLAC file
- **Processing:** Extract 10 feature group vectors via FFT + librosa
- **Similarity:** Cosine similarity per group → average across all 10 groups
- **Output:** Top-5 ranked results with per-group similarity breakdown

## System Block Diagram

```
[Query Audio] → [Signal Digitization] → [FFT Transform] → [Feature Extraction]
                                                                    ↓
                                                          [10 Feature Vectors]
                                                                    ↓
                                            [SQLite DB] ← Cosine Similarity → Top-5 Results
```

## App Screenshots

![Hero section](C:\Users\osw21\.gemini\antigravity\brain\ed61a55e-cf12-4043-916c-3a6e702249af\artifacts\screenshot_hero.png)
![System Architecture](C:\Users\osw21\.gemini\antigravity\brain\ed61a55e-cf12-4043-916c-3a6e702249af\artifacts\screenshot_architecture.png)
![10 Feature Groups](C:\Users\osw21\.gemini\antigravity\brain\ed61a55e-cf12-4043-916c-3a6e702249af\artifacts\screenshot_features.png)

## File Structure
```
d:\SourceCode\MMDB\
├── dataset/              # 520 WAV files (13 species)
├── features_db/
│   ├── animal_sounds.db  # SQLite with 520 records + feature vectors
│   └── features.json     # JSON backup
├── src/
│   ├── collect_dataset.py   # ESC-50 download & organize
│   ├── extract_features.py  # 10-group feature extraction pipeline
│   ├── retrieval.py         # Cosine similarity query engine
│   └── app.py              # Flask web application
├── templates/index.html     # Web UI
├── static/css/style.css     # Dark-mode premium CSS
├── static/js/main.js        # Frontend JavaScript
└── requirements.txt
```

## How to Run
```bash
# (Already running) Start app:
cd d:\SourceCode\MMDB
python src/app.py
# → http://127.0.0.1:5000

# Re-build features database:
python src/extract_features.py

# CLI retrieval (no UI):
python src/retrieval.py path/to/audio.wav 5
```
