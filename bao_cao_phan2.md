# PHẦN 4: 10 NHÓM ĐẶC TRƯNG — CHI TIẾT KỸ THUẬT

> Mỗi đặc trưng time-varying đều trích xuất cả **Mean (trung bình)** và **Std (độ lệch chuẩn)** theo thời gian để nắm bắt cả giá trị trung tâm lẫn mức độ biến thiên. Tổng cộng: **78 chiều vector** mỗi file.

---

## Nhóm 1 — MFCC (Mel-Frequency Cepstral Coefficients)
**26 chiều: 13 mean + 13 std**

### A. Lý do chọn

MFCC là tiêu chuẩn vàng trong xử lý âm thanh và nhận dạng tiếng nói. Nó mô phỏng cách tai người cảm nhận âm thanh (thính giác nhạy hơn ở tần số thấp, ít nhạy ở tần số cao). MFCC mã hóa **hình bao phổ** (spectral envelope) — tức là "dấu vân tay" âm học phản ánh cấu tạo giải phẫu (kích thước họng, cổ, khoang miệng) của từng loài động vật.

### B. Quy trình tính MFCC (13 bước):

```
1. Tín hiệu y[n] → Pre-emphasis → Framing (frame 25ms, hop 10ms)
2. Windowing: nhân mỗi frame với cửa sổ Hann
3. FFT từng frame → Power Spectrum P[k]
4. Mel Filter Bank: 26 bộ lọc tam giác trên thang Mel
   - Công thức đổi Hz → Mel: Mel(f) = 2595 × log10(1 + f/700)
5. Log năng lượng từng bộ lọc: log(E_m)
6. DCT (Discrete Cosine Transform) lấy 13 hệ số đầu tiên → MFCC[1..13]
```

### C. Code trích xuất:

```python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# mfccs.shape = (13, T)  — T: số frames
mfcc_mean = mfccs.mean(axis=1).tolist()   # [m1, m2, ..., m13]
mfcc_std  = mfccs.std(axis=1).tolist()    # [s1, s2, ..., s13]
```

### D. Giá trị thực từ CSDL (MFCC[0] — hệ số năng lượng tổng):

| Loài | MFCC[0] mean | Ý nghĩa |
|---|---|---|
| dog | -600.94 | Âm trầm, ít năng lượng tần cao |
| cow | -343.76 | Âm vừa, rống to nhưng thấp |
| cat | -231.05 | Âm trung bình |
| chirping_birds | -193.98 | Âm cao, nhiều năng lượng tần cao |

---

## Nhóm 2 — Spectral Centroid (Trọng Tâm Phổ)
**2 chiều: mean + std**

### A. Lý do chọn

Centroid là "trung tâm khối lượng" của phổ tần số — cho biết âm thanh thiên về tần số cao hay thấp. Âm thanh **sáng** (chim hót, dế) có centroid cao; âm thanh **tối** (bò rống, voi) có centroid thấp.

### B. Công thức:

$$\text{Centroid}(t) = \frac{\sum_{k=0}^{N/2} f_k \cdot |X_t[k]|}{\sum_{k=0}^{N/2} |X_t[k]|}$$

Trong đó: $f_k$ = tần số của bin $k$, $|X_t[k]|$ = magnitude tại frame $t$ bin $k$.

### C. Code:

```python
C = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # shape (T,)
centroid_mean = float(C.mean())
centroid_std  = float(C.std())
```

### D. Ví dụ thực tế:

| Loài | centroid_mean (Hz) |
|---|---|
| dog | **217 Hz** (cực thấp — tiếng sủa trầm) |
| cow | 2.046 Hz |
| cat | 2.212 Hz |
| chirping_birds | **3.859 Hz** (cực cao — chim hót) |

---

## Nhóm 3 — Spectral Bandwidth (Băng Thông Phổ)
**2 chiều: mean + std**

### A. Lý do chọn

Bandwidth đo độ **rộng phân tán** của phổ quanh centroid. Kết hợp centroid + bandwidth mô tả đầy đủ hình dạng phổ: centroid cho biết vị trí trung tâm, bandwidth cho biết phổ hẹp hay rộng.

### B. Công thức:

$$\text{Bandwidth}(t) = \sqrt{\frac{\sum_{k} (f_k - \text{Centroid}(t))^2 \cdot |X_t[k]|}{\sum_{k} |X_t[k]|}}$$

### C. Code:

```python
B = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
bandwidth_mean = float(B.mean())
bandwidth_std  = float(B.std())
```

---

## Nhóm 4 — Spectral Rolloff (Điểm Suy Giảm Phổ)
**2 chiều: mean + std**

### A. Lý do chọn

Rolloff là tần số mà **85% năng lượng phổ** nằm phía dưới. Phân biệt âm dải hẹp (dế kêu ở tần số cố định) với âm dải rộng (hổ gầm trải rộng nhiều tần số).

### B. Công thức:

$$\sum_{k=0}^{k_{\text{rolloff}}} P_t[k] = 0.85 \times \sum_{k=0}^{N/2} P_t[k]$$

Tìm $k_{\text{rolloff}}$ thỏa mãn điều kiện trên, sau đó: $\text{Rolloff}(t) = f_{k_{\text{rolloff}}}$

### C. Code:

```python
R = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
rolloff_mean = float(R.mean())
rolloff_std  = float(R.std())
```

---

## Nhóm 5 — Spectral Contrast (Tương Phản Phổ)
**14 chiều: 7 mean + 7 std**

### A. Lý do chọn

Contrast đo chênh lệch giữa **đỉnh (peaks)** và **đáy (valleys)** trong 7 dải tần octave. Âm thanh có cấu trúc hài âm (harmonic) như tiếng chim hót có contrast cao; âm thanh nhiễu ngẫu nhiên như tiếng côn trùng có contrast thấp.

### B. 7 dải tần octave:

| Dải | Phạm vi Hz |
|---|---|
| 1 | 0 – 250 |
| 2 | 250 – 500 |
| 3 | 500 – 1000 |
| 4 | 1000 – 2000 |
| 5 | 2000 – 4000 |
| 6 | 4000 – 8000 |
| 7 | 8000 – 11025 |

### C. Code:

```python
SC = librosa.feature.spectral_contrast(y=y, sr=sr)  # shape (7, T)
contrast_mean = SC.mean(axis=1).tolist()   # list[7]
contrast_std  = SC.std(axis=1).tolist()    # list[7]
```

---

## Nhóm 6 — Zero-Crossing Rate (ZCR)
**2 chiều: mean + std**

### A. Lý do chọn

ZCR đếm số lần tín hiệu đổi dấu (cắt qua trục hoành) trong mỗi frame. Đây là đặc trưng phân biệt:
- **Âm tạp (noisy)** như tiếng rắn khè, côn trùng → ZCR rất cao
- **Âm tông thuần (tonal)** như bò rống, sói hú → ZCR thấp

### B. Công thức:

$$\text{ZCR}(t) = \frac{1}{2L} \sum_{n=tL}^{(t+1)L-1} |\text{sgn}(y[n]) - \text{sgn}(y[n-1])|$$

Trong đó $L$ = độ dài frame, $\text{sgn}$ = hàm dấu (+1 hoặc -1).

### C. Code:

```python
Z = librosa.feature.zero_crossing_rate(y)[0]  # shape (T,)
zcr_mean = float(Z.mean())
zcr_std  = float(Z.std())
```

### D. Ví dụ thực tế:

| Loài | ZCR mean | Ý nghĩa |
|---|---|---|
| dog | 0.0132 | Rất thấp — tông rõ ràng |
| cow | 0.0482 | Thấp — âm trầm |
| cat | 0.1377 | Trung bình |
| chirping_birds | 0.3035 | Cao — chim hót nhanh, nhiều biến đổi |

---

## Nhóm 7 — RMS Energy (Năng Lượng Hiệu Dụng)
**2 chiều: mean + std**

### A. Lý do chọn

RMS (Root Mean Square) là thước đo năng lượng tổng thể của tín hiệu theo thời gian — **đường bao năng lượng** (energy envelope). Quan trọng vì:
- Đường bao RMS là cơ sở để tính Attack và Decay chính xác hơn biên độ thô
- Phân biệt âm thanh liên tục (côn trùng) với âm ngắt quãng (chó sủa)

### B. Công thức:

$$\text{RMS}(t) = \sqrt{\frac{1}{L} \sum_{n=tL}^{(t+1)L-1} y[n]^2}$$

### C. Code:

```python
E = librosa.feature.rms(y=y)[0]   # shape (T,)
rms_mean = float(E.mean())
rms_std  = float(E.std())
```

---

## Nhóm 8 — Chroma Features (Đặc Trưng Hòa Âm)
**24 chiều: 12 mean + 12 std**

### A. Lý do chọn

Chroma biểu diễn năng lượng phân bố trên 12 nốt nhạc trong một octave (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Rất hiệu quả cho:
- **Chim hót**: có cấu trúc nốt nhạc rõ → chroma không đều, tập trung ở vài nốt
- **Tiếng ồn ngẫu nhiên**: chroma đều ở 12 nốt
- **Côn trùng**: chroma tập trung ở 1-2 nốt (tần số gần như cố định)

### B. Code:

```python
CH = librosa.feature.chroma_stft(y=y, sr=sr)  # shape (12, T)
chroma_mean = CH.mean(axis=1).tolist()   # list[12]
chroma_std  = CH.std(axis=1).tolist()    # list[12]
```

---

## Nhóm 9 — Attack (Độ Chớp Nhoáng)
**2 chiều: attack_time, attack_rms**

### A. Lý do chọn

Attack là thời gian tín hiệu đạt tới **đỉnh năng lượng RMS** từ lúc bắt đầu. Phản ánh mức độ "đột ngột" của âm thanh:
- Tiếng chó sủa, chim → attack rất nhanh (< 0.5 giây)
- Tiếng bò rống, voi → attack chậm (1–3 giây)
- Tiếng côn trùng liên tục → attack gần như 0

### B. Công thức:

```
hop = 512 mẫu
RMS_frame[t] = sqrt(mean(y[t*hop : (t+1)*hop]²))
peak_frame = argmax(RMS_frame)
attack_time = peak_frame × hop / sr    (giây)
attack_rms  = RMS_frame[peak_frame]    (biên độ tại đỉnh)
```

### C. Code:

```python
hop      = 512
rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
peak     = int(np.argmax(rms))
attack_time = float(peak * hop / sr)
attack_rms  = float(rms[peak])
```

### D. Ví dụ thực tế:

| Loài | attack_time (s) |
|---|---|
| chirping_birds | **0.229** (rất nhanh) |
| cow | 0.859 |
| dog | 2.352 |
| frog | **4.409** (chậm — âm thanh tích lũy dần) |

---

## Nhóm 10 — Decay (Độ Suy Giảm)
**2 chiều: decay_ratio, vol_decay_500ms**

### A. Lý do chọn

Decay mô tả tốc độ âm thanh tắt dần sau đỉnh. Phân biệt:
- Âm thanh duy trì (sustained): bò hú, sói → decay_ratio cao (gần 1.0)
- Âm thanh ngắn (staccato): chó sủa → decay_ratio gần 0

### B. Công thức:

```
hop = 512 mẫu, win = sr × 0.5 / hop  frames (≈ 500ms)

rms_peak = mean(RMS[peak - win/2 : peak + win/2])
rms_tail = mean(RMS[-win:])     (500ms cuối file)

decay_ratio    = rms_tail / rms_peak
vol_decay_500ms = rms_peak - rms_tail
```

### C. Code:

```python
hop   = 512
rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
win_f = max(1, int(sr * 500 / 1000 / hop))   # frames
peak  = int(np.argmax(rms))
rms_pk   = float(rms[peak]) + 1e-9
rms_tail = float(rms[-win_f:].mean()) + 1e-9
decay_ratio    = rms_tail / rms_pk
vol_decay_500ms = rms_pk - rms_tail
```

### D. Ví dụ thực tế:

| Loài | decay_ratio | Ý nghĩa |
|---|---|---|
| frog | 0.301 | Duy trì 30% năng lượng |
| chirping_birds | 0.233 | |
| cow | 0.003 | Tắt nhanh sau đỉnh |
| dog | ~0.000 | **Tắt gần hoàn toàn** — sủa rồi im |

---

## BẢNG TỔNG HỢP 10 NHÓM ĐẶC TRƯNG

| # | Nhóm | Chiều | Thư viện | Miền tính |
|---|---|---|---|---|
| 1 | MFCC | 26 | `librosa.feature.mfcc` | Tần số (Mel) |
| 2 | Spectral Centroid | 2 | `librosa.feature.spectral_centroid` | Tần số |
| 3 | Spectral Bandwidth | 2 | `librosa.feature.spectral_bandwidth` | Tần số |
| 4 | Spectral Rolloff | 2 | `librosa.feature.spectral_rolloff` | Tần số |
| 5 | Spectral Contrast | 14 | `librosa.feature.spectral_contrast` | Tần số |
| 6 | Zero-Crossing Rate | 2 | `librosa.feature.zero_crossing_rate` | Thời gian |
| 7 | RMS Energy | 2 | `librosa.feature.rms` | Thời gian |
| 8 | Chroma | 24 | `librosa.feature.chroma_stft` | Tần số (Mel) |
| 9 | Attack | 2 | Tính từ RMS envelope | Thời gian |
| 10 | Decay | 2 | Tính từ RMS envelope | Thời gian |
| **Tổng** | | **78** | | |
