# Task 8.2. Full AU-guided adaptation và ablation

## 1. Objective

Chạy full adaptation để đánh giá hai thành phần đã hoàn thành ở Task 7 và
Task 8.1:

- AU guidance trong cross-attention của BCS Block 1;
- AU-weighted batch-level alignment trên native Stage 4 tokens.

Thí nghiệm dùng RAF-DB train làm source, ExpW train làm target, batch size
64, seed 42 và 100 epochs. Giữ source checkpoint, POST losses, optimizer,
scheduler, pseudo-label refresh và STR theo baseline hiện tại. Không thêm
prototype bank, EMA, MMD, discriminator hoặc loss mới.

Task cần trả lời liệu từng thành phần có cải thiện E0, hai thành phần có bổ
sung cho nhau và kết quả có phụ thuộc vào AU intensity đúng theo từng ảnh hay
không. Một seed chỉ cung cấp kết quả thăm dò; không dùng kết quả này để kết
luận về độ ổn định thống kê.

## 2. Protocol và cấu hình cố định

E0 tái sử dụng `runs/adaptation/run-001/best.pt`, được chọn tại epoch 63 sau
khi run hoàn thành 256 epochs. E1-E5 dùng horizon 100 epochs và STR tại epoch
25, 50, 75. Trước khi chạy, đối chiếu source checkpoint, manifests, split,
lịch pseudo-label và quy tắc chọn checkpoint của E0. Báo rõ E0 có horizon dài
hơn; E1-E5 phải dùng cùng protocol 100 epochs với nhau.

E1-E5 phải dùng cùng source initialization và POST recipe. Chọn best
checkpoint bằng ExpW validation theo cách hiện tại, sau đó đánh giá checkpoint
đã chọn một lần trên toàn bộ ExpW test. Báo rõ việc model selection dùng nhãn
ExpW validation; adaptation training không dùng target labels. ExpW test không
được dùng để chỉnh cấu hình hoặc chọn checkpoint.

AU attention dùng `g_scale=3000000.0` và learning rate của `λ=0.000001`, đã
khóa từ pilot Task 7. E1, E3 và E5 dùng cùng hai giá trị này. AU alignment
dùng `tau_mask=1.0`, `n_eff_min=8`, `eta=0.0008634870173409581` và 12 giá trị
`q_c` trong
[`adaptation.yaml`](../../configs/training/adaptation.yaml). Cơ sở calibration
nằm trong [`au_alignment_calibration.md`](../experiments/au_alignment_calibration.md).
Không calibration lại các giá trị này giữa các experiment. AU attention pilot
không dùng ExpW validation hoặc test accuracy; kết quả được lưu tại
[`au-attention-pilot-matrix/report.md`](../../runs/adaptation/au-attention-pilot-matrix/report.md).

Kết quả và artifact của ablation được tổng hợp tại
[`au_alignment_ablation.md`](../experiments/au_alignment_ablation.md). E0 hiện
có accuracy `63.67%`, balanced accuracy `42.98%`, macro F1 `44.20%` và
weighted F1 `62.27%` trên 8.062 ảnh ExpW test.

## 3. Experiment matrix

| ID | AU attention | AU alignment | Mục đích |
|---|---|---|---|
| E0 | `off` | `off` | POST baseline đã có |
| E1 | `dynamic` | `off` | Tác động của AU attention |
| E2 | `off` | `normal` | Tác động của AU alignment |
| E3 | `dynamic` | `normal` | Tác động kết hợp |
| E4 | `off` | `shuffled` | Control cho sample-specific AU intensity của alignment |
| E5 | `dynamic_shuffled` | `off` | Control cho sample-specific AU intensity của attention |

`normal` dùng AU intensity đúng theo `sample_id`. Alignment `shuffled` hoán
vị nguyên vector `p:[12]` giữa các mẫu AU hợp lệ trong từng domain; source và
target shuffle độc lập. Ảnh, spatial mask và vị trí missing giữ nguyên. RNG
phải lấy từ trạng thái experiment để checkpoint/resume tái tạo được lần
shuffle tiếp theo. Shuffle của alignment không làm thay đổi metadata mà
attention đang dùng.

Giữ nguyên `q_c`, `tau_mask`, `n_eff_min` và `eta` trong E2-E4. Vì shuffle có
thể thay đổi effective support của từng AU, E4 phải báo `N_eff`, coverage và
tỷ lệ mẫu thực sự đổi AU bên cạnh kết quả chính.

E4 chỉ phá quan hệ giữa AU intensity và ảnh. Spatial mask vẫn thuộc ảnh gốc và
`q_c` vẫn được tính từ phân phối AU thật. Vì vậy, E2 tốt hơn E4 là bằng chứng
rằng AU intensity đúng theo từng sample hữu ích hơn intensity bị hoán vị; so
sánh này không tách riêng toàn bộ ảnh hưởng của spatial guidance và AU prior.

Vòng đầu chỉ dùng seed 42. E6 kết hợp attention thật với alignment shuffled và
các seed bổ sung nằm ngoài phạm vi task này.

Các experiment dùng config riêng trong `configs/training/ablation/`:

| ID | Config |
|---|---|
| E1 | `e1_attention.yaml` |
| E2 | `e2_alignment.yaml` |
| E3 | `e3_both.yaml` |
| E4 | `e4_alignment_shuffled.yaml` |
| E5 | `e5_attention_shuffled.yaml` |

Các file giữ cùng POST recipe và calibration values; chỉ `au_guidance.mode`,
`au_alignment.enabled` và `au_alignment.mode` thay đổi theo experiment matrix.

## 4. Tích hợp train và dữ liệu STR

Trước full training, bổ sung alignment mode `normal | shuffled`, các metric
đánh giá còn thiếu và contribution của từng AU. Với batch `b`, đặt:

```math
u_{b,c}=v_{b,c}q_cD_{b,c},\qquad
Q_b=\sum_c v_{b,c}q_c,
```

```math
a_{b,c}=\frac{u_{b,c}}{Q_b+\epsilon},\qquad
\operatorname{Contribution}^{epoch}_c=
\frac{\sum_b a_{b,c}}
{\sum_b\sum_j a_{b,j}+\epsilon}.
```

Khi các train step có trọng số bằng nhau, `a_{b,c}` phân rã đúng scalar
`L_local` của từng batch. Nếu training loop dùng trọng số step khác nhau, phép
tổng hợp phải dùng cùng trọng số. Chỉ số này mô tả contribution vào giá trị
local loss, không đại diện trực tiếp cho gradient contribution. Không lưu toàn
bộ tensor theo step và không chạy thêm backward chỉ để log.

Hai cache gốc `runs/au_cache/raf_db_train.csv` và
`runs/au_cache/expw_train.csv` là input bất biến. Mỗi experiment phải có run
directory riêng chứa resolved config, checkpoints, STR manifest, ảnh STR và AU
records của ảnh STR. Nếu loader chỉ nhận một source CSV, tạo working copy của
RAF-DB cache trong run directory rồi append STR records vào bản đó. Không ghi
STR records vào cache gốc và không đọc cache của run khác.

Sau mỗi STR round, extractor chỉ xử lý ảnh mới đã commit, cập nhật cache của
run hiện tại rồi rebuild source loader. Ảnh STR thiếu AU vẫn tham gia POST
training nhưng có AU validity bằng 0. `q_c` tiếp tục dùng giá trị đã tính từ
original train sets. Retry không tạo record trùng.

Checkpoint lưu AU modes, resolved AU config, optimizer, RNG, pseudo-label và
STR state. Resume phải dùng đúng manifest và cache của run. Cache hoặc config
không khớp phải báo lỗi rõ.

## 5. Monitoring và evaluation

Mỗi run tiếp tục ghi POST losses, pseudo-label statistics, STR rounds và
validation accuracy. Khi AU attention bật, ghi `lambda` theo head, valid-pair
rate, bias-to-logit ratio và attention probability change. Khi AU alignment
bật, ghi:

- `loss_local` và `eta * loss_local`;
- valid AU count và tỷ lệ batch có local loss;
- `N_eff`, effective weight và centroid distance theo AU;
- contribution theo AU và non-finite count.

Evaluation trên ExpW test xuất accuracy, balanced accuracy, macro F1, weighted
F1, per-class precision/recall/F1, confusion matrix, best epoch và completed
epoch. Metric utility tính các chỉ số này từ cùng confusion matrix, xử lý lớp
có mẫu số bằng 0 mà không sinh NaN.

So sánh chính gồm `E1-E0`, `E2-E0` và `E3-E0`. So sánh E1 với E5 và E2 với
E4 để kiểm tra vai trò của sample-specific AU intensity. Nếu mode thật không
tốt hơn control tương ứng, báo cáo không được cho rằng AU intensity đúng theo
từng ảnh tạo ra cải thiện. Per-class results và các nhầm lẫn lớn của E0 được
dùng cho error analysis sau khi các run hoàn thành, không dùng để điều chỉnh
hyperparameter.

## 6. Test scenarios

### Scenario 1: Experiment isolation

Khởi tạo E0-E5 từ resolved config và kiểm tra hai thành phần bật/tắt đúng theo
bảng. Alignment shuffle và attention shuffle không thay đổi input của nhau.
So sánh một chuỗi step chạy liên tục với cùng chuỗi có save/resume tại
checkpoint boundary mà trainer hỗ trợ. Các step tương ứng sau resume phải có
cùng batch IDs và alignment permutations. Nếu shuffle dùng generator riêng,
checkpoint phải lưu và khôi phục state của generator đó. Resume giữa epoch chỉ
cần khôi phục sampler và loader position khi trainer tuyên bố hỗ trợ boundary
này.

### Scenario 2: Real-data smoke

Chạy vài optimizer steps của E2 và E3 trên RAF-DB/ExpW batches thật. AU join
đúng `sample_id`; local loss, gradients và optimizer update hữu hạn; có AU hợp
lệ trong batch. Với cùng attention mode, `eta=0` khớp alignment disabled trong
sai số số học.

### Scenario 3: STR cache isolation

Dùng một STR fixture nhỏ để tạo record mới, rebuild loader, save và resume.
Cache gốc không đổi; record thuộc đúng run; retry không tạo duplicate; resume
từ chối manifest, AU cache hoặc config không khớp.

### Scenario 4: Evaluation metrics

Dùng confusion matrix E0 để kiểm tra utility tái tạo accuracy `63.67%`,
balanced accuracy `42.98%`, macro F1 `44.20%` và weighted F1 `62.27%`. Kiểm
tra class order và trường hợp một lớp có mẫu số bằng 0.

### Scenario 5: Training contract

Automated integration test chạy vài step, kích hoạt pseudo-label refresh và
STR trên fixture nhỏ, rồi kiểm tra checkpoint, resume và target-only
evaluation. Test suite thông thường không chạy full 100 epochs.

Sau khi chạy E1-E5, full-run verification đọc resolved config, logs và
artifacts để xác nhận từng run hoàn thành 100 epochs, thực hiện đủ STR rounds,
lưu best/final checkpoint và đánh giá checkpoint đã chọn.

## 7. Acceptance criteria

Task hoàn thành khi:

1. E0 được xác minh tương thích và không bị chạy lại nếu protocol đã khớp.
2. Alignment `shuffled`, metric utility, AU contribution logging và cache STR
   theo run được triển khai; năm test scenarios đạt.
3. E1, E2 và E3 hoàn thành 100 epochs với seed 42 và cùng protocol ablation.
4. E4 và E5 hoàn thành với cùng protocol. Nếu thiếu control tương ứng, báo cáo
   không được kết luận về tác dụng của nội dung AU cho thành phần đó.
5. Mỗi run có resolved config, best/final checkpoint, training log, validation
   result, ExpW test result và STR artifacts thuộc đúng run.
6. Bảng E0-E5 có accuracy, balanced accuracy, macro F1, weighted F1,
   per-class metrics và confusion matrix; báo cáo ghi rõ giới hạn của một seed
   và việc chọn checkpoint bằng ExpW validation.
7. Original AU caches không thay đổi và calibration values được giữ cố định.

Không yêu cầu một mức cải thiện tối thiểu. Kết quả âm hoặc bằng baseline vẫn
được ghi đầy đủ.
