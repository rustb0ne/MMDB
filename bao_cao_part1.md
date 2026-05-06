# BÁO CÁO HỆ THỐNG LƯU TRỮ VÀ TÌM KIẾM TIẾNG ĐỘNG VẬT
## Môn: Hệ Cơ Sở Dữ Liệu Đa Phương Tiện (MMDB)

---

# PHẦN 1: GIỚI THIỆU HỆ THỐNG

## I. Bộ Dữ Liệu

### 1. Mô tả tổng quan

Bộ dữ liệu sử dụng trong hệ thống là tập con của **ESC-50** (*Environmental Sound Classification*), một bộ dữ liệu âm thanh môi trường mã nguồn mở được xuất bản trên GitHub, cấp phép **CC BY (Creative Commons Attribution)**.

| Thuộc tính | Giá trị |
|---|---|
| Tổng số file | **520 file WAV** |
| Số loài động vật | **13 loài/nhóm** |
| Số file mỗi loài | **40 file** |
| Nguồn | ESC-50 (GitHub) |
| Giấy phép | CC BY |

### 2. Danh sách 13 nhóm động vật

| STT | Tên (tiếng Anh) | Mô tả |
|---|---|---|
| 1 | `cat` | Tiếng mèo kêu |
| 2 | `dog` | Tiếng chó sủa |
| 3 | `cow` | Tiếng bò rống |
| 4 | `pig` | Tiếng lợn kêu |
| 5 | `sheep` | Tiếng cừu kêu |
| 6 | `frog` | Tiếng ếch kêu |
| 7 | `hen` | Tiếng gà mái |
| 8 | `rooster` | Tiếng gà trống |
| 9 | `crow` | Tiếng quạ kêu |
| 10 | `insects` | Tiếng côn trùng tổng hợp |
| 11 | `crickets` | Tiếng dế kêu |
| 12 | `chirping_birds` | Tiếng chim hót |
| 13 | `crying_baby` | Tiếng em bé khóc |

### 3. Định dạng và cấu trúc

- **Định dạng file:** WAV (PCM 16-bit)
- **Thời lượng:** 5 giây mỗi file
- **Tần số lấy mẫu gốc:** 44.100 Hz (stereo)
- **Tần số lấy mẫu sau load:** 22.050 Hz (mono) — do `librosa.load()` chuẩn hoá
- **Cấu trúc thư mục:**

```
dataset/
├── cat/          (40 files)
├── dog/          (40 files)
├── cow/          (40 files)
├── ...
└── chirping_birds/ (40 files)
```

### 4. Lý do chọn ESC-50

- Mỗi file chứa **đúng một loại tiếng động vật** — đáp ứng yêu cầu đề bài
- Được sử dụng rộng rãi trong nghiên cứu nhận dạng âm thanh môi trường
- Thời lượng 5 giây — đủ dài để trích xuất các đặc trưng phổ, MFCC, attack, decay
- Miễn phí, có gán nhãn chính xác, dễ tái hiện kết quả

---

## II. Bộ Thuộc Tính Nhận Diện Tiếng Động Vật

Hệ thống sử dụng **10 nhóm thuộc tính (feature groups)**, mỗi nhóm được biểu diễn dưới dạng một **vector đặc trưng số thực**. Độ tương đồng được tính riêng cho từng nhóm rồi lấy trung bình.

---

### Thuộc tính 1: Tần Số (Frequency)

#### A. Tổng quan

Tần số biểu thị tốc độ dao động của sóng âm thanh, đo bằng Hz. Sau khi áp dụng biến đổi Fourier (FFT), ta thu được phổ tần số — phân bố năng lượng của tín hiệu theo từng tần số. Từ phổ này có thể tính được **tần số trung bình** và **tần số đỉnh**.

#### B. Lý do chọn

Các loài động vật khác nhau phát ra âm thanh ở dải tần số rất khác nhau:
- **Chó sủa:** tập trung ở 300–900 Hz (thấp)
- **Mèo kêu:** 800–2.500 Hz (trung bình)
- **Chim hót:** 2.000–8.000 Hz (cao)
- **Ếch kêu:** 1.000–3.000 Hz (trung bình–cao)

Tần số do đó là **đặc trưng phân biệt loài** rất hiệu quả.

#### C. Công thức trích xuất

**Bước 1 — Biến đổi FFT:**
```
FFT_result = np.fft.rfft(y)          # y: mảng mẫu âm thanh
freqs      = np.fft.rfftfreq(N, 1/sr) # N mẫu, sr=22050 Hz
```

**Bước 2 — Tính đặc trưng:**
```
mean_freq = Σ(freqs[i] × |FFT[i]|) / Σ|FFT[i]|   # trung bình có trọng số
peak_freq = freqs[ argmax(|FFT|) ]                 # tần số có năng lượng cao nhất
```

#### D. Ví dụ thực tế (từ CSDL)

| Loài | mean_freq (Hz) | peak_freq (Hz) |
|---|---|---|
| `dog` | 2.144 | **852** |
| `cow` | 1.421 | **383** |
| `cat` | 2.689 | **2.269** |
| `frog` | 2.566 | **2.061** |
| `chirping_birds` | 4.253 | **3.312** |

> **Nhận xét:** Bò (cow) có peak_freq thấp nhất (383 Hz), chim hót có peak_freq cao nhất (3.312 Hz) — phân biệt rõ ràng.

---

### Thuộc tính 2: Biên Độ (Amplitude)

#### A. Tổng quan

Biên độ đo lường độ lớn dao động của tín hiệu âm thanh tại mỗi mẫu thời gian, biểu thị cường độ âm thanh. Tính trực tiếp trên mảng mẫu y (miền thời gian), không cần FFT.

#### B. Lý do chọn

- **Chó sủa:** biên độ đột biến cao, rồi về gần 0 (khoảng lặng giữa các tiếng sủa)  
- **Dế kêu:** biên độ nhỏ nhưng đều đặn liên tục  
- **Bò rống:** biên độ trung bình cao, duy trì lâu

Phương sai biên độ (`var_amp`) đặc biệt hữu ích để phân biệt **âm thanh liên tục** (côn trùng) vs **âm thanh ngắt quãng** (chó, mèo).

#### C. Công thức trích xuất

```python
mean_amp = np.mean(np.abs(y))   # Trung bình biên độ tuyệt đối
max_amp  = np.max(np.abs(y))    # Biên độ cực đại
min_amp  = np.min(np.abs(y))    # Biên độ cực tiểu
var_amp  = np.var(y)            # Phương sai biên độ
```

#### D. Ví dụ thực tế

| Loài | mean_amp | max_amp | var_amp |
|---|---|---|---|
| `dog` | **0.0057** | 0.986 | 0.00172 |
| `cow` | **0.1051** | 1.014 | 0.05960 |
| `cat` | 0.0572 | 0.651 | 0.00958 |
| `chirping_birds` | 0.0374 | 0.551 | 0.00377 |

> **Nhận xét:** Bò có mean_amp cao gấp ~18× so với chó — phản ánh tiếng rống liên tục vs tiếng sủa ngắt quãng với nhiều khoảng lặng.

---

### Thuộc tính 3: Thời Gian (Temporal)

#### A. Tổng quan

Nhóm thuộc tính thời gian mô tả các đặc điểm cấu trúc tín hiệu theo trục thời gian. Ngoài `duration` và `sample_rate`, **Zero-Crossing Rate (ZCR)** là đặc trưng quan trọng nhất.

#### B. Lý do chọn

**ZCR (Tỉ lệ vượt ngưỡng 0)** — số lần tín hiệu đổi dấu mỗi giây:
- **Âm thanh tạp (hiss, côn trùng):** ZCR rất cao (nhiều dao động ngẫu nhiên)
- **Âm thanh tông (hú, rống):** ZCR thấp (dao động đều đặn)
- **Chim hót:** ZCR trung bình–cao (nốt nhạc nhanh)

ZCR là đặc trưng đơn giản nhưng rất hiệu quả để phân biệt **âm thanh nhiễu vs tông thuần**.

#### C. Công thức trích xuất

```python
duration          = len(y) / sr           # Độ dài tín hiệu (giây)
sample_rate       = sr                    # Tần số lấy mẫu (Hz)
zero_crossing_rate = np.mean(
    librosa.feature.zero_crossing_rate(y) # Số lần đổi dấu / tổng mẫu
)
```

#### D. Ví dụ thực tế

| Loài | duration (s) | ZCR |
|---|---|---|
| `dog` | 5.0 | **0.0132** (rất thấp — tông rõ) |
| `cow` | 5.0 | **0.0482** (thấp — tông thấp) |
| `cat` | 5.0 | **0.1377** (trung bình) |
| `frog` | 5.0 | **0.1015** (trung bình) |
| `chirping_birds` | 5.0 | **0.3035** (cao — chim hót nhanh) |

> **Nhận xét:** ZCR của chim hót cao gấp 23× chó — phản ánh tốc độ thay đổi âm thanh rất khác nhau.

---

### Thuộc tính 4: Phổ Âm (Spectral)

#### A. Tổng quan

Đây là **nhóm thuộc tính quan trọng nhất**, bao gồm 5 đặc trưng phổ tần số được tính từ STFT (Short-Time Fourier Transform). Tổng cộng tạo ra vector **24 chiều** (13 MFCC + centroid + bandwidth + 7 contrast + rolloff).

#### B. Các đặc trưng và lý do chọn

**4.1 — MFCC (Mel-Frequency Cepstral Coefficients)**

MFCCs mô phỏng cách tai người cảm nhận âm thanh theo thang tần số Mel (phi tuyến). 13 hệ số MFCC nắm bắt **hình dạng phổ âm** — "dấu vân tay" âm học của mỗi loài.

```python
mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = mfccs.mean(axis=1)   # vector 13 chiều (trung bình theo thời gian)
```

**4.2 — Spectral Centroid (Trọng tâm phổ)**

Tần số "trung tâm khối lượng" của phổ — âm thanh sáng có centroid cao.

```python
centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
```

**4.3 — Spectral Bandwidth (Băng thông phổ)**

Độ rộng phân tán của phổ quanh centroid.

```python
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
```

**4.4 — Spectral Contrast (Tương phản phổ)**

Chênh lệch giữa đỉnh và hố trong từng dải tần — 7 giá trị cho 7 dải octave.

```python
contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)  # vector 7 chiều
```

**4.5 — Spectral Rolloff (Điểm suy giảm phổ)**

Tần số mà tại đó 85% năng lượng phổ đã được tích luỹ.

```python
rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
```

#### C. Lý do chọn nhóm Spectral

MFCCs là backbone của hầu hết hệ thống nhận dạng âm thanh hiện đại (ASR, SER). Kết hợp 5 đặc trưng phổ tạo ra vector đặc trưng **phân biệt loài hiệu quả nhất** trong 10 nhóm.

#### D. Ví dụ thực tế

| Loài | centroid (Hz) | bandwidth (Hz) | rolloff (Hz) | MFCC[0] |
|---|---|---|---|---|
| `dog` | **217** | **177** | **329** | -600.94 |
| `cow` | 2.046 | 2.578 | 5.133 | -343.76 |
| `cat` | 2.212 | 2.062 | 3.742 | -231.05 |
| `frog` | 3.196 | 2.792 | 6.486 | -390.90 |
| `chirping_birds` | 3.859 | 2.255 | 6.368 | -193.98 |

> **Nhận xét:** Chó (dog) có centroid cực kỳ thấp (217 Hz) trong khi chim hót gần 3.900 Hz — tương phản rõ nét.

---

### Thuộc tính 5: Hình Dạng Sóng (Waveform)

#### A. Tổng quan

Mô tả hình dạng tổng thể của sóng âm theo trục thời gian: mức độ biến thiên và tỷ lệ thời gian tín hiệu "hoạt động".

#### B. Lý do chọn

- `variance`: Đo mức phân tán biên độ — tiếng bò (sustained loud) có variance cao; tiếng chó sủa thưa có variance thấp do nhiều khoảng yên lặng.
- `pct_above_threshold`: Phần trăm mẫu có |biên độ| > 0.1. Tiếng côn trùng liên tục có giá trị cao; tiếng chó sủa có giá trị thấp.

#### C. Công thức trích xuất

```python
variance          = np.var(y)
pct_above_threshold = np.mean(np.abs(y) > 0.1)  # ngưỡng = 0.1
```

#### D. Ví dụ thực tế

| Loài | variance | pct_above_threshold |
|---|---|---|
| `cow` | **0.0596** | **0.2195** |
| `cat` | 0.0096 | 0.1504 |
| `frog` | 0.0074 | 0.0715 |
| `chirping_birds` | 0.0038 | 0.0716 |
| `dog` | **0.0017** | **0.0174** |

---

### Thuộc tính 6: Độ Phức Tạp (Complexity)

#### A. Tổng quan

Đo số lượng **cực đại cục bộ (peaks)** và **cực tiểu cục bộ (valleys)** trong dạng sóng. Phản ánh mức độ "gồ ghề" hay "mượt mà" của tín hiệu.

#### B. Lý do chọn

- **Tiếng dế / côn trùng:** rất nhiều peaks do dao động nhanh liên tục
- **Tiếng bò / cừu:** ít peaks hơn do âm thanh dài, chậm
- **Tiếng chó:** num_peaks rất thấp vì phần lớn là khoảng yên lặng giữa tiếng sủa

#### C. Công thức trích xuất

```python
from scipy.signal import find_peaks

# Down-sample về 1000 điểm để tăng tốc
step  = max(1, len(y) // 1000)
y_ds  = y[::step]

peaks,   _ = find_peaks(y_ds)    # cực đại cục bộ
valleys, _ = find_peaks(-y_ds)   # cực tiểu cục bộ (đảo dấu)

num_peaks   = len(peaks)
num_valleys = len(valleys)
```

#### D. Ví dụ thực tế

| Loài | num_peaks | num_valleys |
|---|---|---|
| `cat` | 331 | 331 |
| `chirping_birds` | 332 | 331 |
| `frog` | 268 | 268 |
| `cow` | 247 | 247 |
| `dog` | **24** | **24** |

> **Nhận xét:** Chó chỉ có 24 peaks — phản ánh tiếng sủa ngắn với nhiều khoảng im lặng dài.

---

### Thuộc tính 7: Âm Sắc / Biên Độ Màu (Timbre)

#### A. Tổng quan

Timbre (âm sắc) là đặc tính làm cho các loại âm thanh khác nhau dù cùng tần số và âm lượng. Trong hệ thống, timbre được đặc trưng bởi **cường độ phổ FFT** — max và mean của độ lớn FFT.

#### B. Lý do chọn

`max_spectrum` phản ánh đỉnh năng lượng phổ (một thành phần tần số chiếm ưu thế). `mean_spectrum` phản ánh năng lượng phổ trung bình — âm thanh phong phú về tần số có mean_spectrum cao hơn.

#### C. Công thức trích xuất

```python
mag          = np.abs(fft_result)       # độ lớn FFT
max_spectrum = np.max(mag)
mean_spectrum = np.mean(mag)
```

#### D. Ví dụ thực tế

| Loài | max_spectrum | mean_spectrum |
|---|---|---|
| `cow` | **2657.65** | **18.06** |
| `frog` | 576.99 | 7.84 |
| `cat` | 428.86 | 13.54 |
| `chirping_birds` | 245.66 | 10.83 |
| `dog` | **120.59** | **5.42** |

---

### Thuộc tính 8: Độ Chói (Brightness)

#### A. Tổng quan

Brightness đo mức độ "sáng" của âm thanh — tỷ lệ năng lượng phổ ở **tần số cao (≥ 1.500 Hz)** so với tổng năng lượng. Spectral Flatness đo mức độ "phẳng" của phổ (noise vs tonal).

#### B. Lý do chọn

- **Chim hót, ếch:** brightness_ratio gần 1.0 — hầu hết năng lượng ở tần cao
- **Bò, lợn:** brightness_ratio thấp — năng lượng tập trung ở tần thấp
- **Spectral Flatness:** gần 1.0 = nhiễu trắng; gần 0 = tông thuần

#### C. Công thức trích xuất

```python
threshold_hz = 1500  # Hz

mag          = np.abs(fft_result)
total_energy = np.sum(mag**2) + 1e-9
bright_energy = np.sum(mag[freqs >= threshold_hz]**2)
brightness_ratio = bright_energy / total_energy

# Spectral Flatness = geometric_mean(power) / arithmetic_mean(power)
power        = mag**2 + 1e-9
geo_mean     = np.exp(np.mean(np.log(power)))
arith_mean   = np.mean(power)
spectral_flatness = geo_mean / arith_mean
```

#### D. Ví dụ thực tế

| Loài | brightness_ratio | spectral_flatness |
|---|---|---|
| `frog` | **0.989** | 0.00226 |
| `chirping_birds` | **0.963** | 0.06220 |
| `cat` | 0.811 | 0.01144 |
| `dog` | 0.532 | 0.00530 |
| `cow` | **0.062** | **0.00104** |

> **Nhận xét:** Bò có brightness chỉ 6.2% (năng lượng thấp ở tần cao), ếch gần 99% — phân biệt rất tốt.

---

### Thuộc tính 9: Độ Chạy (Attack)

#### A. Tổng quan

Attack là thời gian tín hiệu âm thanh cần để đạt tới **biên độ cực đại** sau khi bắt đầu. Phản ánh mức độ "đột ngột" hay "từ từ" của âm thanh.

#### B. Lý do chọn

- **Chó sủa:** attack ngắn — tiếng sủa xuất hiện đột ngột
- **Bò rống:** attack dài hơn — âm thanh phát triển từ từ
- **Chim hót:** attack rất ngắn — nốt nhạc bắt đầu ngay lập tức

Attack time cung cấp thông tin về **cấu trúc thời gian** của âm thanh — bổ sung cho các đặc trưng phổ.

#### C. Công thức trích xuất

```python
env       = np.abs(y)              # đường bao biên độ
peak_idx  = np.argmax(env)         # vị trí mẫu có biên độ cực đại
attack_time          = peak_idx / sr   # đổi sang giây
attack_max_magnitude = env[peak_idx]   # biên độ tại điểm đỉnh
```

#### D. Ví dụ thực tế

| Loài | attack_time (s) | attack_max_magnitude |
|---|---|---|
| `chirping_birds` | **0.229** | 0.551 |
| `cow` | **0.859** | 1.014 |
| `dog` | 2.352 | 0.986 |
| `cat` | 2.212 | 0.651 |
| `frog` | 4.409 | 0.893 |

---

### Thuộc tính 10: Độ Suy Giảm (Decay)

#### A. Tổng quan

Decay mô tả tốc độ âm thanh **giảm dần** sau khi đạt đỉnh. Được đo qua tỷ lệ RMS ở phần cuối file (500ms cuối) so với vùng đỉnh.

#### B. Lý do chọn

- **Tiếng hú dài (bò, sói):** decay_ratio cao — âm thanh duy trì lâu
- **Tiếng sủa ngắn (chó):** decay_ratio gần 0 — tắt rất nhanh sau đỉnh
- **Tiếng dế liên tục:** decay_ratio trung bình — ổn định

#### C. Công thức trích xuất

```python
window    = int(sr * 500 / 1000)    # 500ms = 11025 mẫu

# RMS vùng đỉnh (±250ms quanh peak)
peak_idx  = np.argmax(np.abs(y))
peak_start = max(0, peak_idx - window//2)
peak_end   = min(len(y), peak_idx + window//2)
rms_peak  = np.sqrt(np.mean(y[peak_start:peak_end]**2)) + 1e-9

# RMS 500ms cuối file
tail      = y[-window:]
rms_tail  = np.sqrt(np.mean(tail**2)) + 1e-9

decay_ratio    = rms_tail / rms_peak        # tỷ lệ suy giảm
vol_decay_500ms = rms_peak - rms_tail       # lượng giảm tuyệt đối
```

#### D. Ví dụ thực tế

| Loài | decay_ratio | vol_decay_500ms |
|---|---|---|
| `frog` | **0.301** | 0.116 (duy trì tốt) |
| `chirping_birds` | **0.233** | 0.101 |
| `cat` | 0.050 | 0.200 |
| `cow` | 0.003 | 0.519 (suy giảm mạnh) |
| `dog` | **~0.000** | 0.131 (tắt gần hoàn toàn) |
