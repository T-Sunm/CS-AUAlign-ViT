# Tổng hợp kết quả thí nghiệm

## Phạm vi

Báo cáo này tổng hợp các kết quả đã có trên cùng tập ExpW test gồm 8.062 ảnh.
Tất cả metric được tính lại từ confusion matrix trong `expw_test.json` bằng
cùng một hàm đánh giá.

Mỗi run dùng checkpoint được chọn trước theo validation accuracy. Bảng này chưa
trả lời câu hỏi liệu E0 có checkpoint khác tốt hơn theo validation balanced
accuracy hoặc validation macro-F1. Phần đó cần một audit riêng bên trong run
E0.

## Kết quả chính

| Phương án | Checkpoint epoch | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| E0 — POST, không AU | 63 | 63,67% | 42,98% | 44,20% | 62,27% |
| E1 — AU attention | 7 | 64,35% | 42,77% | 43,67% | 62,43% |
| **E2 — AU alignment** | 12 | **65,86%** | 44,58% | 44,25% | 64,21% |
| E3 — AU attention + AU alignment | 21 | 65,65% | 44,01% | 43,83% | 63,80% |
| E3-no-STR | 34 | 65,78% | 44,12% | 43,39% | 63,81% |
| E4 — AU alignment shuffled | 16 | 65,64% | 44,82% | 44,07% | 64,50% |
| E5 — AU attention shuffled | — | Chưa có | Chưa có | Chưa có | Chưa có |
| G1 cũ — heatmap, peak threshold 0,8 | 2 | 64,17% | 44,61% | 42,92% | 63,11% |
| **G1 mới — heatmap, bỏ peak gate** | 60 | 65,60% | **45,37%** | **44,31%** | **64,61%** |

E2 có accuracy cao nhất. G1 bỏ peak gate đứng đầu balanced accuracy,
macro-F1 và weighted-F1.

## Chênh lệch so với E0 POST

Các giá trị dưới đây là điểm phần trăm.

| Phương án | Δ Accuracy | Δ Balanced accuracy | Δ Macro-F1 | Δ Weighted-F1 |
|---|---:|---:|---:|---:|
| E1 | +0,68 | −0,21 | −0,53 | +0,16 |
| **E2** | **+2,20** | +1,60 | +0,05 | +1,94 |
| E3 | +1,98 | +1,03 | −0,37 | +1,52 |
| E3-no-STR | +2,11 | +1,14 | −0,81 | +1,54 |
| E4 | +1,97 | +1,84 | −0,13 | +2,22 |
| G1 cũ | +0,50 | +1,63 | −1,28 | +0,84 |
| **G1 bỏ peak gate** | +1,94 | **+2,39** | **+0,11** | **+2,33** |

G1 bỏ peak gate cải thiện rõ nhất ở balanced accuracy và weighted-F1. Mức tăng
macro-F1 so với E0 chỉ là 0,11 điểm phần trăm, nên kết quả chưa cho thấy vấn đề
ở các lớp khó đã được giải quyết triệt để.

## So sánh trực tiếp hai bản G1

| Metric | G1 cũ | G1 bỏ peak gate | Thay đổi |
|---|---:|---:|---:|
| Accuracy | 64,17% | **65,60%** | **+1,44 điểm %** |
| Balanced accuracy | 44,61% | **45,37%** | **+0,76 điểm %** |
| Macro-F1 | 42,92% | **44,31%** | **+1,39 điểm %** |
| Weighted-F1 | 63,11% | **64,61%** | **+1,50 điểm %** |

Bỏ peak gate cải thiện cả bốn metric test chính so với G1 cũ.

### Recall theo lớp của hai bản G1

| Lớp | G1 cũ | G1 bỏ peak gate | Thay đổi |
|---|---:|---:|---:|
| anger | 52,34% | 52,05% | −0,29 điểm % |
| disgust | 13,02% | 11,54% | −1,48 điểm % |
| fear | 0,00% | 1,77% | +1,77 điểm % |
| happiness | 88,57% | 87,01% | −1,56 điểm % |
| neutral | 60,70% | 65,06% | **+4,36 điểm %** |
| sadness | 40,60% | 45,09% | **+4,49 điểm %** |
| surprise | 57,03% | 55,09% | −1,94 điểm % |

Phần lớn mức tăng của G1 mới đến từ neutral và sadness. Fear tăng từ 0% lên
1,77%, tương ứng 2/113 ảnh đúng, nên vẫn là bottleneck rõ nhất. Recall của
disgust giảm từ 13,02% xuống 11,54%.

## Xếp hạng theo từng metric

| Metric | Hạng 1 | Hạng 2 | Hạng 3 |
|---|---|---|---|
| Accuracy | **E2: 65,86%** | E3-no-STR: 65,78% | E3: 65,65% |
| Balanced accuracy | **G1 bỏ peak gate: 45,37%** | E4: 44,82% | G1 cũ: 44,61% |
| Macro-F1 | **G1 bỏ peak gate: 44,31%** | E2: 44,25% | E0: 44,20% |
| Weighted-F1 | **G1 bỏ peak gate: 64,61%** | E4: 64,50% | E2: 64,21% |

## Cách đọc kết quả

E2 vẫn là run có accuracy cao nhất, hơn G1 bỏ peak gate 0,26 điểm phần trăm.
G1 bỏ peak gate có bộ metric cân bằng hơn: accuracy gần E2, đồng thời đứng đầu
ba metric còn lại.

Kết quả E4 cần được giữ trong phần diễn giải. Alignment shuffled vẫn đạt 65,64%
accuracy, 44,82% balanced accuracy và 64,50% weighted-F1. Vì vậy, các run hiện
có chưa tách hoàn toàn lợi ích của quan hệ AU đúng khỏi ảnh hưởng của quy trình
training hoặc regularization.

E3-no-STR cao hơn E3 0,12 điểm phần trăm accuracy, nhưng thấp hơn 0,44 điểm
macro-F1. Checkpoint E3 được chọn ở epoch 21, trước đợt STR đầu tiên tại epoch
25, nên phép so sánh này không phải hai checkpoint cùng epoch chỉ khác STR.

## Khác biệt protocol cần ghi nhận

| Nhóm run | Batch size | Training horizon | Tiêu chí chọn checkpoint |
|---|---:|---:|---|
| E0 | 64 | 256 epoch | Validation accuracy |
| E1, E2 | 45 | 100 epoch | Validation accuracy |
| E3, E3-no-STR, E4 | 64 | 100 epoch | Validation accuracy |
| G1 cũ, G1 bỏ peak gate | 64 | 100 epoch | Validation accuracy |

Chênh lệch giữa các run không chỉ phản ánh thành phần AU. E0 dùng horizon dài
hơn; E1 và E2 dùng batch size khác E3, E4 và G1. Các kết quả hiện cũng mới có
một seed.

## Artifact nguồn

| Run | Test artifact |
|---|---|
| E0 | `runs/adaptation/run-001/expw_test.json` |
| E1 | `runs/adaptation/e1_attention-100epochs-seed42-bs45/expw_test.json` |
| E2 | `runs/adaptation/e2_alignment-100epochs-seed42-bs45/expw_test.json` |
| E3 | `runs/adaptation/e3-both-100epochs-seed42/expw_test.json` |
| E3-no-STR | `runs/adaptation/e3-both-no-str-100epochs-seed42/expw_test.json` |
| E4 | `runs/adaptation/e4_alignment_shuffled-100epochs-seed42-bs64/expw_test.json` |
| E5 | Chưa có test artifact |
| G1 cũ | `runs/adaptation/g1-heatmap-100epochs-exploratory-seed42/expw_test.json` |
| G1 bỏ peak gate | `runs/adaptation/g1-no-peak-gate-100epochs-seed42/expw_test.json` |

## Việc còn lại

1. Audit riêng run E0 để so sánh checkpoint tốt nhất theo validation accuracy,
   balanced accuracy và macro-F1.
2. Chạy E5 nếu vẫn cần control attention shuffled hoàn chỉnh.
3. Không dùng thêm ExpW test để chọn hyperparameter. Mọi quyết định tiếp theo
   cần dựa trên train diagnostics và ExpW validation.
