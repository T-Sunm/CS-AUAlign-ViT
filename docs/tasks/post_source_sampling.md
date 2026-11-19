# Task 6.1. Square-root source sampling cho adaptation

## 1. Mục tiêu và phạm vi

Đánh giá liệu giảm chênh lệch tần suất lấy mẫu giữa các lớp source có cải thiện
recall của các lớp yếu, đặc biệt là Fear, trong POST adaptation.

Đây là task con của [Task 6](post_adaptation_loop.md). Task 6 sở hữu source-target
iteration, trainer, checkpoint/resume và logging. Task [5](post_self_training_resampling.md)
sở hữu STR transaction và tạo lại source loader sau mỗi round.

Không đổi source pretraining, kiến trúc BCS, prototype losses, adversarial
alignment, AU guidance, target loader, target supervision hay thuật toán tạo
ảnh STR. Square-root sampling thay tần suất xuất hiện của source samples trong
training batches, không thay số lượng ảnh trong dataset.

## 2. Sampling contract

Thêm cấu hình source sampling cho adaptation:

```yaml
source_sampling:
  mode: random  # random | sqrt_balanced
  replacement: true  # chỉ áp dụng cho sqrt_balanced
```

- `random` giữ hành vi E0 hiện tại: lấy mẫu source thông thường với shuffle.
- `sqrt_balanced` dùng `WeightedRandomSampler` với replacement. Với mẫu $i$ có
  nhãn $y_i$, đặt $w_i=1/\sqrt{n_{y_i}}$. Xác suất lấy class $c$ do đó tỷ lệ
  với $\sqrt{n_c}$, không phải phân phối đồng đều.
- `num_samples` của sampler bằng `len(source_dataset)`. Một lượt source loader
  có số lượt lấy như vậy; mẫu có thể lặp lại hoặc không được lấy trong lượt đó.
- Khi dùng sampler, `DataLoader` đặt `shuffle=False`.
- Chỉ áp dụng policy này cho source adaptation loader. Source pretraining và
  target adaptation/feature/validation loaders giữ nguyên.

Đếm class từ metadata `sample["label"]`, không gọi `__getitem__()` chỉ để lấy
nhãn. Tính weights trên toàn bộ `ConcatDataset`, gồm RAF-DB gốc và các manifest
STR hiện có.

## 3. STR và vòng đời loader

Giữ nguyên thời điểm, lựa chọn class, confidence gate, ảnh ghép và transaction
của Task 5. Sau STR, dựng lại loader và sampler từ manifest source gốc cùng STR
manifest đã hoàn thành. Cách dựng này cũng phải được dùng khi resume từ
checkpoint có STR.

POST lấy số optimizer steps mỗi epoch từ target loader và tạo lại iterator
source khi source loader cạn. Vì vậy, `num_samples=len(source_dataset)` chỉ xác
định một lượt source loader. Logging phải tính mọi source batch thực sự được
tiêu thụ trong epoch.

## 4. RNG và checkpoint

Tạo một source generator và truyền cùng object đó cho `WeightedRandomSampler`
và `DataLoader`. Checkpoint lưu và khôi phục generator này, đồng thời giữ
sampling mode và replacement trong resolved config/checkpoint identity. Nhờ
vậy sampler không có một RNG ẩn nằm ngoài checkpoint.

Resume ở epoch boundary phải tiếp tục cùng chuỗi source indices như training
liên tục, kể cả khi source loader đã được cycle nhiều lần trong epoch. Tại STR
boundary, thứ tự phải là: hoàn tất STR, dựng lại loader/sampler theo dataset mới,
lưu trạng thái checkpoint nhất quán; resume từ boundary phải tái tạo được cùng
chuỗi lấy mẫu tiếp theo.

## 5. Ma trận thí nghiệm B1

| Thành phần | E0 | B1 |
|---|---|---|
| Source initialization | Source-pretraining checkpoint đã dùng để khởi tạo E0 | Cùng checkpoint và identity |
| Source sampling | Random/shuffle như baseline | Square-root, replacement bật |
| Source dataset | RAF-DB gốc và STR samples theo lịch | Cùng nguồn và lịch; manifest STR có thể khác do model trajectory |
| Target loader và supervision | E0 | Giữ nguyên protocol E0 |
| Backbone, losses, optimizer và scheduler | E0 | Giữ nguyên |
| AU guidance | Off | Off |
| Horizon, batch size, seed | 100 epochs, batch size 64, seed 42 | Giữ nguyên |
| STR | Lịch và thuật toán E0 | Giữ nguyên; rebuild sampler sau mỗi round |
| Best checkpoint | ExpW validation accuracy | Cùng tiêu chí |

Trong task này, E0 là control 100 epoch dùng cấu hình random sampling tại
[`adaptation.yaml`](../../configs/training/adaptation.yaml); B1 chỉ khác policy
source sampling. Cả hai phải khởi tạo adaptation từ cùng source-pretraining
checkpoint. E0 lịch sử tại `runs/adaptation/run-001` đã chạy 256 epoch; giữ nó
làm kết quả tham khảo, không dùng trực tiếp làm matched control cho B1-100.

Trước run, xác minh source-pretraining checkpoint, split/manifests, seed,
resolved config và identity của hai run 100 epoch. Không khởi tạo B1 từ best
adaptation checkpoint epoch 63. Đánh giá ExpW test một lần sau khi khóa
protocol; không dùng test để chọn sampling policy hoặc checkpoint.

## 6. Logging và phân tích

Ghi theo epoch và theo STR round khi phù hợp:

| Metric | Diễn giải |
|---|---|
| Dataset class counts | Phân phối source sau STR; ghi trước và sau mỗi round |
| Sampled class counts | Số lượt từng class trong mọi source batch thực sự đi vào `train_step` |
| Unique samples per class | Mức lặp/bao phủ của source samples trong các batch đã tiêu thụ |
| Trusted pseudo-label counts by predicted class | Chẩn đoán target supervision; không xem là ground-truth accuracy |
| Validation accuracy, balanced accuracy, macro-F1, per-class recall | Lợi ích và đánh đổi giữa các lớp; checkpoint vẫn chọn theo validation accuracy |
| STR selected class, gate count, created count, rejects | Phân biệt tác động của sampler và các thay đổi tiếp diễn qua STR |

Unique sample counts phải dựa trên sample identity/index, không suy ra từ class
counts. Vì source batch hiện không trả `sample_id`, cần ánh xạ sampler indices
về metadata. Tính sampled counts qua mọi lần source iterator được tạo lại trong
epoch.

## 7. Tiêu chí hoàn thành

1. `random` giữ sampling behavior của E0; `sqrt_balanced` chỉ tác động source
   adaptation loader và dùng `shuffle=False`.
2. Class weights dùng đủ metadata từ source gốc và STR manifests mà không đọc
   ảnh chỉ để lấy nhãn.
3. Loader rebuild sau STR và khi resume dùng đúng sampling policy và class
   counts hiện tại.
4. Kiểm thử xác nhận training liên tục và resume tại epoch boundary tạo cùng
   chuỗi source indices; bao gồm boundary có STR và trường hợp source loader bị
   cycle trong epoch.
5. E0 và B1 dùng cùng protocol ngoài source sampling; artifacts ghi đủ config,
   RNG/sampling diagnostics, STR metrics và validation metrics.
6. Báo cáo nêu rõ kết quả một seed chỉ là bằng chứng thăm dò. Không đặt tiêu chí
   thành công là bắt buộc tăng recall: kết quả không cải thiện vẫn hoàn thành
   mục tiêu nếu protocol hợp lệ và giới hạn được phân tích.

## 8. Giới hạn diễn giải

B1 là phép đo tác động end-to-end của policy sampling khi STR vẫn hoạt động.
Sampling có thể làm thay đổi model trajectory và do đó thay đổi số lượng/nội
dung STR được tạo; đây là hiệu ứng tiếp diễn cần log, không phải lỗi protocol.
Kết quả không tách riêng tác động trực tiếp của sampler trên một bộ STR cố
định. Nếu cần tách cơ chế đó, thiết kế thêm đối chứng no-STR hoặc replay cùng STR
manifests sau khi xem kết quả B1.
