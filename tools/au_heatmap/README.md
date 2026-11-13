# AU heatmap tool

Gói này chạy model heatmap regression cho 12 AU của DISFA (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26). Đầu ra cho mỗi ảnh là tensor `12 x 64 x 64`, cùng cường độ của từng AU lấy bằng giá trị lớn nhất trên mỗi heatmap.

Model là biến thể *Adaptation layers* trong paper của Ntinou và cộng sự, "A Transfer Learning approach to Heatmap Regression for Action Unit intensity estimation" (IEEE T-AFFC). Checkpoint `disfa_adaptation_f0.pth` được tác giả Christos Tzelepis đưa vào repo [WarpedGANSpace](https://github.com/chi0tzp/WarpedGANSpace) để làm công cụ đánh giá. Phần code model trong gói này chép lại từ `lib/evaluation/au_detector/hourglass.py` của repo đó, giữ nguyên tên tham số để checkpoint nạp được.

## Cài đặt

```bash
uv sync
uv run python download_checkpoint.py  # tải và kiểm tra sha256, lưu vào checkpoints/
```

Nếu script không tải được, bạn mở một trong hai link sau bằng trình duyệt, giải nén file tar, rồi đặt `disfa_adaptation_f0.pth` vào thư mục `checkpoints/`:

- https://www.dropbox.com/s/qzeugjqi4ss2w5m/au_detector.tar?dl=1
- https://www.dropbox.com/s/jkkf1gda9o8ed47/au_detector.tar?dl=1

## Chạy

```bash
uv run python infer.py --input anh.jpg
uv run python infer.py --input thu_muc_anh/ --out results/
uv run python infer.py --input mat_da_crop/ --detector none  # ảnh đã crop sẵn
```

Trong thư mục `results/` sẽ có:

- `<tên>_heatmaps.npy`: mảng `12 x 64 x 64`, thứ tự kênh theo AU 1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26.
- `<tên>_heatmaps.png`: ảnh mặt đã crop và 12 heatmap chồng lên, có ghi cường độ dự đoán.
- `intensities.csv`: mỗi dòng là một ảnh, mỗi cột là một AU.

Khi dùng trong code của bạn:

```python
from au_heatmap import AUHeatmapDetector, AU_IDS
det = AUHeatmapDetector("checkpoints/disfa_adaptation_f0.pth")
heatmaps, intensities = det([face_rgb_uint8])   # (1, 12, 64, 64), (1, 12)
```

## Lưu ý

Model được train trên ảnh mặt crop chặt cỡ 256 x 256, nên chất lượng crop ảnh hưởng nhiều đến kết quả. Detector `sfd` gần với code gốc nhất, còn `haar` chỉ để thử nhanh khi chưa cài `face-alignment`. Bạn có thể chỉnh `--margin` nếu thấy crop quá chặt hoặc quá rộng.

Ảnh đầu vào được chuẩn hoá min-max về [0, 1] giống code gốc, không trừ mean theo ImageNet.

Cường độ AU nằm trong thang 0 đến 5, và giá trị dự đoán có thể hơi âm khi AU không xuất hiện.

Checkpoint là model của fold 0 trong 3 fold cross-validation trên DISFA. Nếu bạn đánh giá trên DISFA thì một phần subject đã nằm trong tập train của model này.
