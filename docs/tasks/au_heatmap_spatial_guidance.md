# Task 9.1. Cài đặt và kiểm tra AU heatmap tool

## 1. Objective và trạng thái

Cài một tool offline để chạy AU heatmap regression trên ảnh khuôn mặt. Tool
trả 12 heatmap cùng intensity tương ứng, phục vụ kiểm tra trực quan và chuẩn bị
đầu vào cho phần tích hợp sau này.

Task 9.1 chỉ bao gồm model inference, checkpoint, môi trường uv, CLI và smoke
test trên ảnh RAF-DB thật. Tool chưa thay LibreFace, chưa được import từ
`src/cs_aualign_vit` và chưa tham gia training.

Trạng thái: hoàn thành.

## 2. Nguồn model và kiến trúc

Phương pháp lấy từ paper
[*A Transfer Learning approach to Heatmap Regression for Action Unit intensity
estimation*](https://arxiv.org/abs/2004.06657) của Ioanna Ntinou, Enrique
Sanchez, Adrian Bulat, Michel Valstar và Georgios Tzimiropoulos. Paper biểu
diễn mỗi AU bằng một heatmap để ước lượng đồng thời vùng tác động và intensity,
sau đó chuyển feature từ một mạng face alignment sang AU estimation.

Tool dùng biến thể adaptation layers. Luồng inference hiện tại là:

```text
RGB face crop [B,3,256,256]
  → QFAN
  → 68 landmark heatmap + feature [B,128,64,64]
  → hai projection 1×1 rồi cộng lại
  → lightweight Hourglass
  → 12 AU heatmap [B,12,64,64]
  → spatial maximum
  → intensity [B,12]
```

Code kiến trúc trong [`model.py`](../../tools/au_heatmap/au_heatmap/model.py)
được chép từ evaluation code của WarpedGANSpace và giữ tên tham số để nạp
checkpoint `disfa_adaptation_f0.pth`. Đây là implementation inference, không
phải training pipeline đầy đủ của paper.

## 3. Cấu trúc tool

```text
tools/au_heatmap/
├── README.md
├── pyproject.toml
├── uv.lock
├── download_checkpoint.py
├── infer.py
└── au_heatmap/
    ├── __init__.py
    ├── detector.py
    └── model.py
```

Vai trò các file:

- `download_checkpoint.py` tải archive từ Dropbox, kiểm tra SHA-256 rồi tách
  `disfa_adaptation_f0.pth` vào `checkpoints/`.
- `model.py` chứa QFAN, adaptation layers, Hourglass và output head.
- `detector.py` nạp checkpoint, preprocess ảnh và cung cấp
  `AUHeatmapDetector`.
- `infer.py` xử lý file hoặc thư mục ảnh, crop mặt nếu cần và ghi kết quả.
- `pyproject.toml` cùng `uv.lock` quản lý môi trường riêng của tool.

`checkpoints/` và `results/` là artifact cục bộ, được bỏ qua trong Git.

## 4. Input và output contract

Model nhận face crop RGB. Với input là NumPy `uint8`, detector:

1. đổi tensor sang layout channel-first;
2. resize về `256×256`;
3. min-max normalize về `[0,1]`;
4. chạy inference trong `torch.no_grad()`.

Tool không dùng ImageNet mean và standard deviation. QFAN chạy ở `eval()`
trong forward.

Thứ tự 12 channel được khóa theo checkpoint DISFA:

```text
AU 1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26
```

Mỗi ảnh thành công tạo:

```text
<stem>_heatmaps.npy   # float32 [12,64,64]
<stem>_heatmaps.png   # input và 12 heatmap overlay
intensities.csv       # một dòng cho mỗi ảnh
```

Intensity được tính bằng giá trị lớn nhất của mỗi heatmap:

```python
intensities = heatmaps.amax(dim=(2, 3))
```

CSV hiện dùng cột `image,AU1,...,AU26`. Đây là output để kiểm tra tool, chưa
phải schema `sample_id,status,au_<id>_intensity` mà `AUCache` của training yêu
cầu.

## 5. Cài đặt và sử dụng

Từ thư mục tool:

```bash
cd tools/au_heatmap
uv sync
uv run python download_checkpoint.py
```

Downloader đã xác minh archive bằng SHA-256:

```text
dbdf18bf541de3c46769d712866bef38496b7528072850c28207747b2b2c101e
```

Chạy một face crop đã chuẩn bị:

```bash
uv run python infer.py \
  --input anh.png \
  --detector none \
  --device cpu
```

Chạy một thư mục crop:

```bash
uv run python infer.py \
  --input thu_muc_anh \
  --out results/rafdb_test_50 \
  --detector none \
  --device cpu
```

RAF-DB và ExpW trong data pipeline đã là face crop, nên dùng
`--detector none` để tránh detect và crop lần hai. `--no-png` bỏ ảnh overlay
khi chỉ cần `.npy` và CSV.

CLI tự chọn CUDA nếu không truyền `--device`. GPU trên máy development từng
hết VRAM ngay lúc chuyển model sang CUDA, nên các smoke test dưới đây dùng
CPU. Cảnh báo active `VIRTUAL_ENV` không trùng `.venv` của tool không làm thay
đổi kết quả; uv dùng project environment tại `tools/au_heatmap/.venv`.

## 6. Kết quả đã kiểm tra

Checkpoint đã tải, qua SHA-256 và nạp vào model thành công. Các lần kiểm tra
trên RAF-DB test gồm:

- `raf_db:test_0003.jpg`, nhãn `happiness`, chạy đủ `.npy`, PNG và CSV;
- `raf_db:test_0002.jpg`, nhãn `surprise`, kích hoạt rõ AU1, AU2 và AU5;
- `raf_db:test_0013.jpg`, nhãn `happiness`, có 8/12 AU đạt intensity ít nhất
  `0.5` trong batch kiểm tra.

Batch 50 ảnh RAF-DB test tạo đủ 50 file `.npy`, 50 PNG overlay và 50 dòng
intensity. Các file `.npy` có shape `[12,64,64]`, dtype `float32` và toàn bộ
giá trị hữu hạn. Intensity quan sát được nằm trong khoảng `0.0111` đến
`5.4648`.

Các kết quả này xác nhận đường inference và output contract hoạt động. Chúng
không đo accuracy của AU detector trên RAF-DB hoặc khả năng tổng quát sang
ExpW. Checkpoint được train trên DISFA fold 0 nên domain shift vẫn là một giới
hạn cần giữ lại khi diễn giải heatmap.

## 7. Handoff cho Task 9.2

Task 9.2 có thể dùng tool này làm điểm bắt đầu cho integration, nhưng cần đọc
đúng các boundary hiện tại:

| Hiện trạng | Việc Task 9.2 cần quyết định |
|---|---|
| Heatmap nằm trên lưới `64×64` | Cách ánh xạ sang Stage 4 token grid `7×7` |
| Intensity là spatial maximum | Cách normalize hoặc calibration trước `AUCache` |
| Peak quan sát có thể lớn hơn 5 | Cách xử lý range `[0,5]` mà loader yêu cầu |
| CSV dùng tên file | Cách join bằng `sample_id` và ghi `status` |
| Mỗi ảnh lưu một `.npy` | Artifact format phù hợp cho toàn bộ dataset |
| Output bám theo crop `256×256` | Cách giữ heatmap đồng bộ với random horizontal flip |
| `infer.py` chạy offline từng ảnh | Batch extractor theo manifest và retry contract |
| Tool nằm ngoài package chính | Điểm nối với dataset và AU guidance hiện tại |

Task 9.1 không chọn sẵn phép downsample, cách normalize, artifact format hoặc
cách kết hợp FANAU với LibreFace. Nó chỉ giữ source model và contract đã chạy
được để Task 9.2 không phải dựng lại tool.

## 8. Acceptance criteria

Task 9.1 hoàn thành khi:

1. `uv sync` resolve được môi trường từ `uv.lock`.
2. Downloader lấy đúng checkpoint và xác minh SHA-256 trước khi giải nén.
3. Inference một face crop tạo heatmap `[12,64,64]`, overlay và intensity CSV.
4. Batch 50 ảnh RAF-DB test chạy đủ và output hữu hạn.
5. README và task ghi cách cài, cách chạy, nguồn paper, checkpoint cùng các
   boundary dành cho Task 9.2.

Các điều kiện trên đã đạt. Integration với training thuộc Task 9.2.
