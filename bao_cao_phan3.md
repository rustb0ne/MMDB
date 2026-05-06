# PHẦN 5: LƯU TRỮ VÀ PHẦN 6: HỆ THỐNG TÌM KIẾM

---

## PHẦN 5: LƯU TRỮ — CƠ SỞ DỮ LIỆU

### I. Lựa Chọn Hệ Quản Trị CSDL

Hệ thống sử dụng **SQLite** thay vì MySQL (theo tài liệu tham khảo) hoặc MongoDB.

| Tiêu chí | SQLite ✅ | MySQL | MongoDB |
|---|---|---|---|
| Cài đặt | Không cần, tích hợp Python | Cần server | Cần server |
| Tốc độ query (1121 record) | < 2ms | ~5ms | ~10ms (network) |
| Di chuyển | 1 file `.db` | dump/restore | export/import |
| Phù hợp khi | < 100k record | > 100k record | Dữ liệu phi cấu trúc |

**Kết luận:** SQLite là tối ưu cho bài toán này. MongoDB phù hợp hơn khi hệ thống mở rộng lên hàng triệu file.

Ngoài SQLite, hệ thống còn xuất **JSON backup** (`features.json`) để dễ kiểm tra bằng mắt thường.

### II. Schema Bảng `audio_features`

```sql
CREATE TABLE audio_features (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT NOT NULL,   -- tên file WAV
    species      TEXT NOT NULL,   -- tên loài (= tên thư mục)
    filepath     TEXT NOT NULL,   -- đường dẫn tuyệt đối
    duration_s   REAL,            -- thời lượng sau trim (giây)

    -- Blob JSON: raw feature values (hiển thị, debug)
    mfcc         TEXT,   -- {"mfcc_mean": [...13], "mfcc_std": [...13]}
    centroid     TEXT,   -- {"centroid_mean": 2212.3, "centroid_std": 451.2}
    bandwidth    TEXT,
    rolloff      TEXT,
    contrast     TEXT,   -- {"contrast_mean": [...7], "contrast_std": [...7]}
    zcr          TEXT,
    rms          TEXT,
    chroma       TEXT,   -- {"chroma_mean": [...12], "chroma_std": [...12]}
    attack       TEXT,
    decay        TEXT,

    -- Flat vectors JSON: dùng cho Cosine Similarity (tốc độ cao)
    vec_mfcc      TEXT,   -- [m1..m13, s1..s13] = 26 số
    vec_centroid  TEXT,   -- [mean, std] = 2 số
    vec_bandwidth TEXT,   -- [mean, std] = 2 số
    vec_rolloff   TEXT,   -- [mean, std] = 2 số
    vec_contrast  TEXT,   -- [7 mean, 7 std] = 14 số
    vec_zcr       TEXT,   -- [mean, std] = 2 số
    vec_rms       TEXT,   -- [mean, std] = 2 số
    vec_chroma    TEXT,   -- [12 mean, 12 std] = 24 số
    vec_attack    TEXT,   -- [time, rms] = 2 số
    vec_decay     TEXT    -- [ratio, vol_drop] = 2 số
)
```

**Lý do lưu 2 dạng:** Raw JSON để con người đọc được và debug; Flat vector để truy xuất nhanh và tính Cosine không cần parse lồng nhau.

### III. Hàm Chuyển Đổi Feature → Vector Phẳng

```python
def flatten(features: dict) -> dict[str, list[float]]:
    """Chuyển mỗi nhóm đặc trưng thành list[float] phẳng."""
    vectors = {}
    for group, vals in features.items():
        vec = []
        for v in vals.values():
            if isinstance(v, list):
                vec.extend(float(x) for x in v)  # MFCC 13 chiều → extend
            else:
                vec.append(float(v))              # scalar → append
        vectors[group] = vec
    return vectors
```

**Ví dụ với MFCC:**
```
features["mfcc"] = {
    "mfcc_mean": [-231.05, 90.09, -24.13, ..., -3.21],  # list[13]
    "mfcc_std":  [  45.12, 12.33,   8.91, ...,  2.45],  # list[13]
}
→ vectors["mfcc"] = [-231.05, 90.09, ..., -3.21, 45.12, 12.33, ..., 2.45]
                     ←──────── 13 mean ──────────→←──────── 13 std ────────→
```

---

## PHẦN 6: HỆ THỐNG TÌM KIẾM — CƠ CHẾ COSINE SIMILARITY

### I. Quy Trình Tổng Quan (6 Bước)

```
Bước 1: Upload file âm thanh truy vấn (query)
         ↓
Bước 2: Pre-processing pipeline (Bandpass → Trim → Pre-emph → Norm)
         ↓
Bước 3: Trích xuất 10 nhóm đặc trưng → 10 vector đặc trưng
         ↓
Bước 4: Với MỖI bản ghi trong CSDL (1121 bản ghi):
         - Tính Cosine Similarity cho từng trong 10 cặp vector
         - Lấy trung bình → similarity_score ∈ [0, 1]
         ↓
Bước 5: Sắp xếp 1121 similarity_score giảm dần
         ↓
Bước 6: Trả về Top-5 kết quả kèm per-group breakdown
```

### II. Công Thức Cosine Similarity

Cho hai vector đặc trưng của cùng một nhóm $g$:
- $\mathbf{q}_g$ = vector của âm thanh **query**
- $\mathbf{d}_g$ = vector của âm thanh **trong CSDL**

$$\text{cosine}(\mathbf{q}_g, \mathbf{d}_g) = \frac{\mathbf{q}_g \cdot \mathbf{d}_g}{|\mathbf{q}_g| \times |\mathbf{d}_g|} = \frac{\sum_{i=1}^{n} q_i \cdot d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \times \sqrt{\sum_{i=1}^{n} d_i^2}}$$

**Kết quả nằm trong [-1, 1]:**
- `1.0` = hoàn toàn giống nhau (cùng hướng)
- `0.0` = không liên quan (vuông góc)
- `-1.0` = đối lập nhau (ngược hướng)

**Tại sao Cosine chứ không phải Euclidean Distance?**
- Cosine bất biến với **độ lớn** của vector — chỉ quan tâm đến **hướng/hình dạng**
- Hai file cùng loài nhưng khác âm lượng vẫn có Cosine cao
- Euclidean bị ảnh hưởng bởi độ lớn vector → phụ thuộc vào âm lượng tuyệt đối

### III. Tính Độ Tương Đồng Tổng Hợp

Sau khi có Cosine của 10 nhóm, lấy **trung bình số học**:

$$\text{Similarity}_{\text{final}} = \frac{1}{10} \sum_{g=1}^{10} \text{cosine}(\mathbf{q}_g, \mathbf{d}_g)$$

$$= \frac{\text{cos}_{mfcc} + \text{cos}_{centroid} + \text{cos}_{bandwidth} + \text{cos}_{rolloff} + \text{cos}_{contrast} + \text{cos}_{zcr} + \text{cos}_{rms} + \text{cos}_{chroma} + \text{cos}_{attack} + \text{cos}_{decay}}{10}$$

**Hiển thị cho người dùng:** Nhân với 100 → phần trăm (%)

### IV. Code Triển Khai Chi Tiết

#### Hàm tính Cosine (`retrieval.py`):

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)

    # Xử lý trường hợp 2 vector khác chiều (padding bằng 0)
    if len(va) < len(vb):
        va = np.pad(va, (0, len(vb) - len(va)))
    elif len(vb) < len(va):
        vb = np.pad(vb, (0, len(va) - len(vb)))

    na = np.linalg.norm(va)   # = sqrt(Σ va_i²)
    nb = np.linalg.norm(vb)   # = sqrt(Σ vb_i²)

    if na < 1e-12 or nb < 1e-12:  # tránh chia cho 0
        return 0.0

    return float(np.dot(va, vb) / (na * nb))   # Σ(va_i × vb_i) / (|va| × |vb|)
```

#### Hàm truy vấn CSDL (`retrieval.py`):

```python
FEATURE_GROUPS = [
    "mfcc","centroid","bandwidth","rolloff","contrast",
    "zcr","rms","chroma","attack","decay",
]

def query_database(query_vectors, top_k=5):
    conn = sqlite3.connect(DB_PATH)
    # Lấy toàn bộ 1121 bản ghi (chỉ lấy cột cần thiết)
    cols = ["id","filename","species","filepath"] + \
           [f"vec_{g}" for g in FEATURE_GROUPS]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM audio_features").fetchall()
    conn.close()

    results = []
    for row in rows:
        per_group = {}
        for g in FEATURE_GROUPS:
            db_vec = json.loads(row[f"vec_{g}"])   # vector từ CSDL
            q_vec  = query_vectors[g]               # vector từ query
            per_group[g] = cosine_similarity(q_vec, db_vec)

        # Trung bình 10 nhóm
        avg_similarity = np.mean(list(per_group.values()))
        results.append({
            "filename": row["filename"],
            "species":  row["species"],
            "similarity": avg_similarity,         # điểm tổng hợp
            "per_group_similarity": per_group,    # chi tiết từng nhóm
        })

    # Sắp xếp giảm dần → lấy Top-5
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
```

### V. API Web (Flask)

Endpoint `POST /api/retrieve` xử lý toàn bộ luồng:

```python
@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    # 1. Nhận file từ user
    f = request.files["file"]
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    f.save(upload_path)

    # 2. Trích xuất đặc trưng (pre-processing + feature extraction)
    features = extract_all(upload_path)

    # 3. Chuyển thành flat vectors
    vectors = flatten(features)

    # 4. Tính similarity vs toàn bộ CSDL, lấy Top-5
    results = query_database(vectors, top_k=5)

    # 5. Trả về JSON cho Frontend
    return jsonify({
        "query_filename": f.filename,
        "results": [
            {
                "rank": idx + 1,
                "filename": r["filename"],
                "species": r["species"],
                "similarity_pct": round(r["similarity"] * 100, 2),
                "per_group": r["per_group_similarity"],
                "audio_url": url_for("stream_audio", record_id=r["id"]),
            }
            for idx, r in enumerate(results)
        ],
        "query_features": features,   # hiển thị accordion
    })
```

---

## PHẦN 7: SƠ ĐỒ KHỐI HOÀN CHỈNH

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         GIAI ĐOẠN OFFLINE                                │
│                         (Xây dựng CSDL)                                  │
│                                                                          │
│  [1121 WAV files]                                                        │
│       ↓                                                                  │
│  librosa.load(sr=22050, mono=True)   ← Downmix + Resample               │
│       ↓                                                                  │
│  Butterworth Bandpass [50–15000 Hz]  ← Lọc nhiễu                        │
│       ↓                                                                  │
│  librosa.effects.trim(top_db=30)     ← Cắt khoảng lặng                  │
│       ↓                                                                  │
│  Pre-emphasis (coef=0.97)            ← Tăng tần cao                     │
│       ↓                                                                  │
│  Peak Normalization → [-1, +1]       ← Chuẩn hóa                        │
│       ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │  10 Nhóm Đặc Trưng (78 chiều)                              │         │
│  │  MFCC(26) Centroid(2) Bandwidth(2) Rolloff(2) Contrast(14) │         │
│  │  ZCR(2)   RMS(2)      Chroma(24)  Attack(2)  Decay(2)      │         │
│  └─────────────────────────────────────────────────────────────┘         │
│       ↓                                                                  │
│  SQLite DB + JSON Backup                                                 │
│  (1121 records × 10 vec_* columns)                                       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         GIAI ĐOẠN ONLINE                                 │
│                         (Truy vấn người dùng)                            │
│                                                                          │
│  [Upload file âm thanh] → Pre-processing (5 bước) → 10 vectors          │
│                                                           ↓              │
│                                         For each record in DB (1121):    │
│                                           cos_g = cosine(q_g, db_g)     │
│                                           avg   = mean(cos_1..cos_10)   │
│                                           ↓                              │
│                                         Sort giảm dần → Top-5           │
│                                           ↓                              │
│                                         Hiển thị Web UI + Audio Player  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## PHẦN 8: KẾT LUẬN

### I. Tổng Kết Hệ Thống

| Hạng mục | Giá trị |
|---|---|
| Tổng file âm thanh | **1121 file WAV** |
| Số loài | **20 loài** |
| Pipeline tiền xử lý | **5 bước** (Bandpass → Trim → Pre-emph → Norm) |
| Số nhóm đặc trưng | **10 nhóm** |
| Tổng chiều vector | **78 chiều** |
| Thuật toán tương đồng | **Cosine Similarity — trung bình 10 nhóm** |
| Kết quả trả về | **Top-5** |
| CSDL | **SQLite** (4.5 MB) + JSON backup (6.4 MB) |
| Giao diện | **Flask Web App** (http://127.0.0.1:5000) |

### II. Ưu Điểm Thiết Kế

1. **Pre-processing đầy đủ**: Bandpass filter loại nhiễu → kết quả đặc trưng sạch hơn
2. **Mean + Std**: Nắm bắt cả giá trị trung tâm và biến thiên theo thời gian → vector phong phú hơn chỉ lấy mean
3. **MFCC là backbone**: 26 chiều MFCC chiếm 33% tổng vector — nhóm quan trọng nhất trong phân loại âm thanh sinh học
4. **Cosine thay vì Euclidean**: Bất biến với âm lượng tuyệt đối, chỉ quan tâm hình dạng phổ
5. **FFT tính một lần**: Chia sẻ cho tất cả đặc trưng miền tần số → tối ưu tốc độ

### III. Định Hướng Cải Thiện

- **Weighted Cosine**: Tăng trọng số nhóm MFCC (quan trọng nhất) và Chroma, giảm Attack/Decay
- **Thêm đặc trưng**: Tonnetz, Mel Spectrogram, Gammatone filterbank
- **Deep Learning**: Dùng VGGish / YAMNet để trích xuất embedding tự động
- **Data augmentation**: Pitch shift ±2 semitones, time stretch ×0.9/1.1, thêm noise → tăng robustness

### IV. Tài Liệu Tham Khảo

1. Piczak, K.J. (2015). *ESC: Dataset for Environmental Sound Classification*. ACM Multimedia.
2. Davis & Mermelstein (1980). *Comparison of parametric representations for monosyllabic word recognition*. IEEE TASLP.
3. McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python*. SciPy.
4. Logan, B. (2000). *Mel Frequency Cepstral Coefficients for Music Modeling*. ISMIR.
5. Salamon & Bello (2017). *Deep CNNs and Data Augmentation for Environmental Sound Classification*. IEEE SPL.
