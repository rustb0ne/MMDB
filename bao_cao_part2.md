# PHẦN 2: XÂY DỰNG HỆ THỐNG TÌM KIẾM TIẾNG ĐỘNG VẬT

---

## I. Sơ Đồ Khối Và Quy Trình Xử Lý

### 1. Sơ đồ khối hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Xây dựng CSDL)                     │
│                                                                     │
│  [520 file WAV]  →  [Load / Chuẩn hoá]  →  [FFT]                  │
│       dataset           22050 Hz mono       rfft(y)                 │
│                                ↓                                    │
│                    [Trích xuất 10 nhóm thuộc tính]                  │
│                    Frequency | Amplitude | Temporal                 │
│                    Spectral  | Waveform  | Complexity               │
│                    Timbre    | Brightness| Attack | Decay            │
│                                ↓                                    │
│                    [Lưu vào SQLite + JSON]                          │
│                    features_db/animal_sounds.db (520 records)       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        ONLINE (Truy vấn)                            │
│                                                                     │
│  [File âm thanh đầu vào]  →  [Load / FFT]  →  [Trích xuất 10 nhóm]│
│       (query)                                                        │
│                                ↓                                    │
│         Với mỗi bản ghi trong CSDL (520 bản ghi):                  │
│           sim_i = cosine(query_vec_i, db_vec_i)  cho i=1..10       │
│           similarity = mean(sim_1 .. sim_10)                        │
│                                ↓                                    │
│         Sắp xếp giảm dần → Trả về Top-5 kết quả                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Quy trình chi tiết tìm kiếm âm thanh

**Bước 1:** Người dùng upload một file âm thanh (WAV/MP3/OGG/FLAC) qua giao diện web.

**Bước 2:** Hệ thống load file, chuẩn hoá về 22.050 Hz mono:
```python
y, sr = librosa.load(filepath, sr=22050, mono=True)
```

**Bước 3:** Tính FFT một lần duy nhất, dùng chung cho tất cả đặc trưng miền tần số:
```python
fft_result = np.fft.rfft(y)
freqs      = np.fft.rfftfreq(len(y), d=1.0/sr)
```

**Bước 4:** Trích xuất song song 10 nhóm thuộc tính → 10 vector đặc trưng.

**Bước 5:** Với mỗi bản ghi trong CSDL (520 bản ghi), tính độ tương đồng Cosine cho từng nhóm:
```
similarity_final = (cos_freq + cos_amp + cos_temp + cos_spec +
                    cos_wave + cos_comp + cos_timb + cos_bri +
                    cos_att + cos_dec) / 10
```

**Bước 6:** Sắp xếp 520 giá trị similarity giảm dần → lấy Top-5.

**Bước 7:** Trả về kết quả kèm audio URL để phát ngay trên trình duyệt.

---

## II. Quá Trình Trích Xuất, Lưu Trữ Và Sử Dụng Thuộc Tính

### 1. Pipeline trích xuất đặc trưng

Toàn bộ pipeline được thực hiện trong hàm `extract_all_features()`:

```python
def extract_all_features(filepath):
    # B1: Load âm thanh
    y, sr = librosa.load(filepath, sr=22050, mono=True)
    # y: mảng float32, kích thước = sr * duration = 22050 * 5 = 110250 phần tử

    # B2: Tính FFT một lần (dùng chung)
    fft_result = np.fft.rfft(y)   # kích thước: 55126 số phức
    freqs = np.fft.rfftfreq(len(y), d=1.0/sr)

    # B3: Gọi 10 hàm trích xuất
    return {
        "frequency":  extract_frequency_features(fft_result, freqs),
        "amplitude":  extract_amplitude_features(y),
        "temporal":   extract_temporal_features(y, sr),
        "spectral":   extract_spectral_features(y, sr),
        "waveform":   extract_waveform_features(y),
        "complexity": extract_complexity_features(y),
        "timbre":     extract_timbre_features(fft_result),
        "brightness": extract_brightness_features(fft_result, freqs),
        "attack":     extract_attack_features(y, sr),
        "decay":      extract_decay_features(y, sr),
    }
```

**Kết quả trung gian** — ví dụ file `1-34094-A-5.wav` (cat):
```
frequency  : mean_freq=2689.19 Hz, peak_freq=2268.80 Hz
amplitude  : mean=0.0572, max=0.6509, min=3.69e-7, var=0.00958
temporal   : duration=5.0s, sr=22050, zcr=0.1377
spectral   : centroid=2212.28 Hz, bandwidth=2061.54 Hz,
             rolloff=3741.99 Hz, mfcc[0..4]=[-231.05, 90.09, -24.13, 11.18, -10.24]
waveform   : variance=0.00958, pct_above=0.1504
complexity : num_peaks=331, num_valleys=331
timbre     : max_spectrum=428.86, mean_spectrum=13.54
brightness : brightness_ratio=0.811, spectral_flatness=0.0114
attack     : attack_time=2.21s, attack_max=0.651
decay      : decay_ratio=0.050, vol_decay_500ms=0.200
```

### 2. Lưu trữ vào CSDL

Sau khi trích xuất, mỗi file được lưu thành **một hàng trong bảng SQLite** với cấu trúc:

```sql
CREATE TABLE audio_features (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT,          -- tên file WAV
    species       TEXT,          -- tên loài (tên thư mục)
    filepath      TEXT,          -- đường dẫn tuyệt đối

    -- Blob JSON: raw feature values (để tra cứu, hiển thị)
    frequency     TEXT,          -- {"mean_freq": 2689.19, "peak_freq": 2268.8}
    amplitude     TEXT,          -- {"mean_amp": 0.057, "max_amp": 0.651, ...}
    temporal      TEXT,
    spectral      TEXT,          -- {"mfcc": [...13 values...], "centroid": ...}
    waveform      TEXT,
    complexity    TEXT,
    timbre        TEXT,
    brightness    TEXT,
    attack        TEXT,
    decay         TEXT,

    -- Vector phẳng JSON: dùng cho tính Cosine (tốc độ cao)
    vec_frequency  TEXT,         -- [2689.19, 2268.8]
    vec_amplitude  TEXT,         -- [0.057, 0.651, 3.69e-7, 0.00958]
    vec_temporal   TEXT,         -- [5.0, 22050.0, 0.1377]
    vec_spectral   TEXT,         -- [mfcc0..12, centroid, bandwidth, contrast0..6, rolloff] = 24 chiều
    vec_waveform   TEXT,         -- [0.00958, 0.1504]
    vec_complexity TEXT,         -- [331, 331]
    vec_timbre     TEXT,         -- [428.86, 13.54]
    vec_brightness TEXT,         -- [0.811, 0.0114]
    vec_attack     TEXT,         -- [2.21, 0.651]
    vec_decay      TEXT          -- [0.050, 0.200]
)
```

**Tóm tắt kích thước vector từng nhóm:**

| Nhóm | Số chiều vector |
|---|---|
| frequency | 2 |
| amplitude | 4 |
| temporal | 3 |
| **spectral** | **24** (13 MFCC + centroid + bandwidth + 7 contrast + rolloff) |
| waveform | 2 |
| complexity | 2 |
| timbre | 2 |
| brightness | 2 |
| attack | 2 |
| decay | 2 |
| **Tổng** | **45 chiều** |

### 3. Tìm kiếm âm thanh trong hệ thống

#### Bước 1: Flatten (chuyển đặc trưng thành vector phẳng)

```python
def flatten_features(features):
    vectors = {}
    for group, vals in features.items():
        vec = []
        for v in vals.values():
            if isinstance(v, list):
                vec.extend(v)      # MFCC 13 chiều → extend
            else:
                vec.append(v)      # scalar → append
        vectors[group] = vec
    return vectors
```

#### Bước 2: Tính Cosine Similarity mỗi nhóm

```python
def cosine_similarity(a, b):
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    # Pad vector ngắn hơn bằng 0 (an toàn với mọi kích thước)
    if len(va) < len(vb): va = np.pad(va, (0, len(vb)-len(va)))
    if len(vb) < len(va): vb = np.pad(vb, (0, len(va)-len(vb)))

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-12 or norm_b < 1e-12: return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))
```

**Công thức Cosine Similarity:**

$$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \times |\mathbf{b}|}$$

Kết quả nằm trong **[-1, 1]**, với 1 = hoàn toàn giống, 0 = không liên quan.

#### Bước 3: Tính độ tương đồng tổng hợp

```python
per_group = {}
for group in FEATURE_GROUPS:   # 10 nhóm
    db_vec  = json.loads(row[f"vec_{group}"])
    q_vec   = query_vectors[group]
    per_group[group] = cosine_similarity(q_vec, db_vec)

# Độ tương đồng cuối cùng = trung bình 10 nhóm
similarity_final = np.mean(list(per_group.values()))
```

#### Bước 4: Xếp hạng và trả về Top-5

```python
results.sort(key=lambda x: x["similarity"], reverse=True)
return results[:5]
```

---

## III. Kết Quả Demo Hệ Thống

### Giao diện web (http://127.0.0.1:5000)

Hệ thống cung cấp giao diện web với các tính năng:

| Tính năng | Mô tả |
|---|---|
| Drag & drop upload | Upload file âm thanh bất kỳ |
| Animated processing | Hiển thị 4 bước xử lý trực quan |
| Top-5 results | Thẻ kết quả kèm % similarity |
| Per-group bars | Thanh tiến trình cho mỗi trong 10 nhóm |
| Feature accordion | Xem giá trị đặc trưng của file query |
| Audio player | Phát trực tiếp file kết quả trong trình duyệt |
| System diagram | Sơ đồ khối tương tác 7 bước |
| Dataset overview | Phân bố 520 file theo 13 loài |

### Ví dụ kết quả truy vấn

Khi upload file tiếng mèo kêu vào hệ thống:

```
=== Kết quả truy vấn (query: cat_sample.wav) ===

Rank #1  species=cat    similarity=0.9841 (98.41%)
         per_group: frequency=0.998 amplitude=0.952 spectral=0.991 ...

Rank #2  species=cat    similarity=0.9712 (97.12%)

Rank #3  species=frog   similarity=0.6234 (62.34%)
         (tần số trung bình tương tự nhau)

Rank #4  species=hen    similarity=0.5891 (58.91%)

Rank #5  species=crow   similarity=0.5234 (52.34%)
```

---

## IV. Kết Luận

### Tổng kết hệ thống

| Hạng mục | Giá trị |
|---|---|
| Tổng số file âm thanh | **520 file WAV** |
| Số loài | **13 loài/nhóm** |
| Số nhóm thuộc tính | **10 nhóm** |
| Tổng số chiều vector | **45 chiều** |
| Phương pháp tương đồng | **Cosine Similarity (trung bình 10 nhóm)** |
| Kết quả trả về | **Top-5** |
| Lưu trữ | **SQLite + JSON backup** |
| Giao diện | **Flask Web App (http://127.0.0.1:5000)** |

### Đánh giá ưu điểm

1. **Spectral (MFCC)** là nhóm thuộc tính mạnh nhất — phân biệt đúng loài trong hầu hết trường hợp
2. **Brightness + Frequency** bổ trợ tốt cho trường hợp loài có dải tần rất khác nhau (chó vs chim)
3. **Attack + Decay** giúp phân biệt âm thanh ngắn (tiếng sủa) vs dài (tiếng hú/rống)
4. Sử dụng **trung bình Cosine 10 nhóm** giúp hệ thống không bị bias vào một đặc trưng đơn lẻ

### Định hướng cải thiện

- Thêm **Chroma features** để nắm bắt đặc trưng âm nhạc của chim hót
- Dùng **Weighted Cosine** — tăng trọng số nhóm Spectral (MFCC) vốn phân biệt tốt hơn
- Huấn luyện **mô hình ML** (SVM, Random Forest) trên vector 45 chiều để phân loại chính xác hơn
- Tăng dataset lên 1000+ file bằng data augmentation (pitch shift, time stretch, noise)

---

## V. Tài Liệu Tham Khảo

1. Piczak, K.J. (2015). *ESC: Dataset for Environmental Sound Classification*. ACM Multimedia 2015.
2. Davis, S. & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition*. IEEE TASLP.
3. McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis in Python*. SciPy 2015.
4. Salamon, J. & Bello, J.P. (2017). *Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification*. IEEE Signal Processing Letters.
