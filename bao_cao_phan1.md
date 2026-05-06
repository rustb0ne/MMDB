# BÁO CÁO HỆ THỐNG LƯU TRỮ VÀ TÌM KIẾM TIẾNG ĐỘNG VẬT
### Môn: Hệ Cơ Sở Dữ Liệu Đa Phương Tiện (MMDB)

---

## PHẦN 1: GIỚI THIỆU HỆ THỐNG

### I. Lý Do Ra Đời

Việc nhận diện và phân loại âm thanh động vật đặt ra nhiều thách thức:
- **Khối lượng dữ liệu lớn**: 1121 file âm thanh từ 20 loài khác nhau cần được quản lý và truy xuất hiệu quả.
- **Phân tích phức tạp**: Mỗi âm thanh mang thông tin đặc trưng về tần số, biên độ, phổ âm cần được trích xuất chính xác.
- **Tìm kiếm nội dung**: Cần đo độ tương đồng giữa âm thanh truy vấn với toàn bộ CSDL và trả về kết quả chính xác.

Hệ thống được xây dựng để giải quyết ba thách thức trên thông qua: pipeline tiền xử lý tín hiệu, trích xuất 10 nhóm đặc trưng âm học, lưu trữ vào SQLite, và tìm kiếm bằng độ tương đồng Cosine.

---

### II. Bộ Dữ Liệu

#### 1. Mô tả tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Tổng số file | **1121 file WAV** (sau khi lọc chất lượng) |
| Số loài | **20 loài/nhóm** |
| Nguồn gốc | ESC-50 + bổ sung thêm (Kaggle Animal Sound) |
| Giấy phép | CC BY (Creative Commons) |

#### 2. Danh sách 20 loài

| STT | Tên | STT | Tên |
|---|---|---|---|
| 1 | Aslan (sư tử) | 11 | Horse (ngựa) |
| 2 | Bear (gấu) | 12 | insects (côn trùng) |
| 3 | cat (mèo) | 13 | Monkey (khỉ) |
| 4 | Chicken (gà) | 14 | pig (lợn) |
| 5 | chirping_birds (chim hót) | 15 | rooster (gà trống) |
| 6 | cow (bò) | 16 | sheep (cừu) |
| 7 | crickets (dế) | 17 | Dolphin (cá heo) |
| 8 | crow (quạ) | 18 | Donkey (lừa) |
| 9 | dog (chó) | 19 | Elephant (voi) |
| 10 | frog (ếch) | 20 | hen (gà mái) |

#### 3. Cấu trúc lưu trữ

```
dataset/
├── cat/          (90 files)
├── dog/          (90 files)
├── cow/          (90 files)
├── frog/         (90 files)
├── sheep/        (90 files)
├── Bear/         (50 files)
├── Elephant/     (50 files)
...
└── crickets/     (40 files)
```

Mỗi thư mục con = 1 loài. Tên thư mục = nhãn loài được gán trực tiếp khi lưu vào CSDL.

---

## PHẦN 2: TIỀN XỬ LÝ TÍN HIỆU ÂM THANH

Trước khi trích xuất đặc trưng, mỗi file âm thanh trải qua pipeline tiền xử lý gồm 5 bước:

### Pipeline tiền xử lý (Pre-processing)

```
File âm thanh thô
      ↓
  [1] Downmix Mono + Resample → 22050 Hz
      ↓
  [2] Butterworth Bandpass Filter [50 Hz – 15000 Hz]
      ↓
  [3] Silence Trimming (cắt khoảng lặng)
      ↓
  [4] Pre-emphasis Filter (coef = 0.97)
      ↓
  [5] Peak Normalization [-1.0, +1.0]
      ↓
  Tín hiệu sạch → Trích xuất đặc trưng
```

#### Bước 1 — Downmix Mono & Resample

```python
y, sr = librosa.load(filepath, sr=22050, mono=True)
# y: mảng float32, kích thước = 22050 × duration (giây)
# Ví dụ file 5 giây: 22050 × 5 = 110250 phần tử
```

Lý do chọn 22050 Hz: đủ bắt được dải âm động vật (50–15000 Hz) theo định lý Nyquist (cần sr > 2 × fmax = 30000 Hz), đồng thời tiết kiệm bộ nhớ hơn 44100 Hz.

#### Bước 2 — Butterworth Bandpass Filter

Loại bỏ tiếng gió thấp tần (<50 Hz) và nhiễu điện từ cao tần (>15000 Hz):

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(y, sr, low=50, high=15000, order=4):
    nyq   = sr / 2.0               # tần số Nyquist = 11025 Hz
    low_n = low  / nyq             # chuẩn hóa: 50/11025 ≈ 0.00454
    high_n = min(high / nyq, 0.9999)  # 15000/11025 → cắt tại 0.9999
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, y)       # lọc zero-phase (không lệch pha)
```

Bộ lọc Butterworth bậc 4 đảm bảo độ suy giảm -80 dB/decade ngoài dải thông mà không gây méo pha.

#### Bước 3 — Silence Trimming

Cắt bỏ khoảng yên lặng đầu/cuối để tránh sai lệch giá trị mean:

```python
y_trimmed, _ = librosa.effects.trim(y, top_db=30)
# top_db=30: cắt các đoạn có năng lượng < (peak - 30 dB)
# Nếu file quá ngắn sau trim (<0.3s) → giữ nguyên
if len(y_trimmed) < sr * 0.5:
    y_trimmed = y   # an toàn, không bỏ file
```

Trong quá trình xây dựng CSDL, 9 file bị loại do quá ngắn sau trim (< 0.3 giây), còn lại 1121 file hợp lệ.

#### Bước 4 — Pre-emphasis

Tăng cường tần số cao để bù trừ hiện tượng suy giảm tự nhiên, ổn định MFCC:

```
y_preemph[n] = y[n] - 0.97 × y[n-1]
```

```python
y = np.append(y[0], y[1:] - 0.97 * y[:-1])
```

#### Bước 5 — Peak Normalization

Đưa biên độ về khoảng [-1, +1] để so sánh công bằng giữa các file:

```python
peak = np.max(np.abs(y))
if peak > 1e-9:
    y = y / peak
```

---

## PHẦN 3: BIẾN ĐỔI FOURIER (FFT)

Sau tiền xử lý, toàn bộ tín hiệu thời gian được chuyển sang miền tần số bằng **Real FFT**:

```python
fft_result = np.fft.rfft(y)
# y: N mẫu → rfft cho N/2+1 số phức
# Ví dụ N=110250 → 55126 bin tần số

freqs = np.fft.rfftfreq(len(y), d=1.0/sr)
# freqs[k] = k × sr/N (Hz)
# freqs[0]=0 Hz, freqs[55125]≈11025 Hz
```

**FFT chỉ tính một lần** và chia sẻ kết quả cho tất cả các đặc trưng miền tần số (Frequency, Timbre, Brightness) → tối ưu hiệu năng.

Độ lớn phổ (magnitude spectrum):
```
|X[k]| = sqrt(Re(X[k])² + Im(X[k])²)
```

Năng lượng phổ (power spectrum):
```
P[k] = |X[k]|² = Re(X[k])² + Im(X[k])²
```

---

*[Tiếp theo: Phần 4 — 10 Nhóm Đặc Trưng chi tiết]*
