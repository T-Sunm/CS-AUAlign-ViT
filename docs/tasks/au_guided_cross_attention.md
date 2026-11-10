# Task 7. AU-guided cross-domain token mixing ở Swin Stage 4

## 1. Objective

Thử một thay đổi duy nhất trên POST: thêm AU guidance vào hai đường
cross-attention của BCS Block 1 tại lưới `7×7`. Câu hỏi cần trả lời là AU
đúng của từng ảnh có giúp hơn AU bị xáo giữa các ảnh hay không. Kiểm thử
ba mode `off`, `dynamic`, `dynamic_shuffled` với **một seed 42**
và cùng RAF-DB → ExpW split hiện tại. Task này nghiệm thu phần cài đặt bằng
test và smoke test. So sánh kết quả sau train đầy đủ là bước thí nghiệm tiếp
theo.

| Loại | Gồm gì | Cách xử lý |
|---|---|---|
| Cần để thí nghiệm có nghĩa | LibreFace và AU cache; mask landmark và `G`; bias đủ mạnh để đổi attention; ba mode | Làm trong Task 7, kiểm tra từng bước trước khi train đầy đủ |
| Giữ theo POST hiện có | Swin-T, hai prototype banks, bốn prototype losses, `L_ad`, pseudo-label refresh, STR | Giữ kiến trúc, loss weights và lịch chạy |
| Để sau | `static`, `learned_bias`, nhiều seed, `L_local`, MMD, discriminator mới | Không triển khai hoặc diễn giải trong Task 7 |

Notebook [`au_guidance_sample.ipynb`](../../notebooks/au_guidance_sample.ipynb)
là mẫu kiểm tra schema, `M` và `G`; code train đặt trong module của repo.

## 2. Input và AU cache

Chạy LibreFace bằng `.venv-au` **ngoài vòng train** trên crop `usable=true`,
`split=train` của RAF-DB và ExpW. Cache theo `sample_id`, gồm `status` và
12 cột `au_<id>_intensity` cho AU `(1,2,4,5,6,9,12,15,17,20,25,26)`.
Loader trong `.venv` kiểm tra schema, giá trị hữu hạn thuộc `[0,5]`, rồi
chia `5` để được `p∈[0,1]`. Ghép bằng `sample_id`, không ghép theo thứ tự
CSV; record thiếu hoặc LibreFace thất bại được đếm và dùng `G=0`.

Trước khi train, báo tỉ lệ AU thiếu của từng domain và đối chiếu AU12 với
`happiness`, AU4 với `anger` trên RAF-DB train. Nếu tín hiệu AU không phân
biệt được những nhóm này hoặc ExpW thiếu AU nhiều, dừng ablation để kiểm tra
detector; không coi kết quả âm là bằng chứng AU vô ích.

Sau mỗi STR round, chạy extractor cho ảnh ghép mới đã commit rồi cập nhật
cache trước khi source loader được rebuild. Retry/resume không tạo record
trùng. STR tiếp tục dùng lịch và ảnh ghép của Task 5; landmark của ảnh ghép
hiện kế thừa donor, nên kiểm tra mask trên vài ảnh STR.

Extractor có thể retry có chọn lọc các record `failed` bằng
`--retry-failed --retry-error-type <ExceptionType>`. Record `success` và các
loại lỗi không được chọn giữ nguyên; kết quả retry thay record cũ theo
`sample_id` và CSV vẫn được ghi atomically, không thêm dòng trùng. LibreFace
được giới hạn một mặt mỗi crop và dùng `LANCZOS` thay API `ANTIALIAS` đã bị
gỡ để tương thích với dependency hiện tại.

## 3. Mask, `G` và attention

Dùng năm MTCNN `landmarks_crop` và kích thước crop thực tế để đặt Gaussian
theo ánh xạ AU→vùng trong notebook. Mỗi ảnh có `M:[49,12]`. Ảnh và mask dùng
cùng quyết định random horizontal flip. Ảnh thiếu landmark hợp lệ tạo `G=0`;
ghi riêng tỉ lệ thiếu landmark và thiếu AU.

```math
A_i=M_i\odot p_i[None,:],\qquad
G_{t\rightarrow s}[i]=A_t[i]A_s[i]^T/12\in\mathbb{R}^{49\times49}.
```

Hàng `G` là target query token, cột là source key token. Chiều `s→t` dùng
`G.transpose(-2,-1)`. Trước khi train, vẽ riêng `M_mouth`, `M_eye` trên ảnh
`224×224`, rồi mới vẽ một hàng `G`; test shape, hướng chuyển vị, flip và
fallback `G=0`.

Trong `BCSBlock`, giữ relative-position bias của POST. Chỉ cộng
`λ_h · g_scale · G` vào **hai cross-attention logits, trước softmax**.
`λ_t→s` và `λ_s→t` là hai vector trainable độc lập, mỗi vector 24 head,
khởi tạo 0. Native attention và target-only path giữ nguyên. `off` không
dùng AU và phải cho kết quả POST gốc; `λ=0` phải khớp bốn output BCS trong
`eval` mode.

`G` có thể rất nhỏ sau phép chia 12. Trên pilot, đo độ lệch chuẩn theo từng
hàng key của `λ · g_scale · G` so với logits gốc và đo thay đổi attention
probability sau softmax. Chọn **một** `g_scale` cố định và learning rate riêng
cho `λ` trên pilot, bỏ weight decay cho `λ`, rồi khóa các giá trị này cho cả
`dynamic` và `dynamic_shuffled`. Tiêu chí chọn là tỉ số độ lệch chuẩn của
bias so với logits và mức thay đổi attention, không dựa trên accuracy của
ExpW validation. Đo trên một batch train cố định gồm các cặp có AU hợp lệ,
bao gồm cặp biểu cảm mạnh, ở cùng checkpoint và `eval` mode. Gọi
`P_0=softmax(S)` và `P_G=softmax(S+λ·g_scale·G)`, rồi tính thay đổi mỗi hàng
attention là `D=0.5·sum_k |P_G-P_0|`. Sau smoke test, yêu cầu phân vị 90%
của `D` trên các hàng/head hợp lệ đạt ít nhất `0.01` ở mỗi chiều. Để tránh
softmax bão hòa, tối đa `5%` số hàng được chuyển từ
`max(P_0)<0.95` sang `max(P_G)≥0.95`, tính riêng trên các cặp biểu cảm mạnh.
Đây là ngưỡng kỹ thuật cho pilot, không phải kết luận về accuracy; cố định
chúng trước ablation. Nếu không đạt một trong hai cận, chỉnh pilot rồi đo
lại trước khi chạy ablation.

### 3.1. Pilot và cấu hình đã khóa

Ma trận calibration cuối dùng cấu hình E3, tức attention và alignment cùng
bật; alignment giữ nguyên các giá trị đã khóa ở Task 8.1. Pilot chạy từ cùng
source checkpoint, seed 42, batch size 16 và 100 optimizer step cho mỗi cấu
hình. Sau mỗi run, checkpoint được đo trên 768 batch train cố định. Một cặp
được tính là mạnh khi `max(p_s) >= 0.5` và `max(p_t) >= 0.5`. Một cấu hình đạt
khi P90 của `D` từ hai chiều đều không thấp hơn `0.01`, tỷ lệ saturation không
vượt `5%`, có ít nhất 20 cặp AU mạnh và AMP bỏ qua không quá `5%` số step.
ExpW validation và test không tham gia lựa chọn.

| Run | `g_scale` | Learning rate của `λ` | P90 t→s / s→t | Saturation t→s / s→t | AMP skip | Kết quả |
|---|---:|---:|---:|---:|---:|---|
| A | `3e6` | `1e-6` | `0.0232 / 0.0374` | `0.84% / 2.68%` | `3/100` | Đạt |
| B | `3e6` | `3e-6` | `0.0516 / 0.0885` | `2.81% / 5.58%` | `3/100` | Không đạt saturation s→t |
| C | `3e6` | `1e-5` | `0.1189 / 0.1468` | `6.82% / 6.77%` | `2/100` | Không đạt saturation |
| D | `1e7` | `9e-7` | `0.1460 / 0.1599` | `9.02% / 8.75%` | `2/100` | Không đạt saturation |

Mỗi phép đo có 5.459 cặp hợp lệ, trong đó 26 cặp đạt điều kiện AU mạnh. Run
A là cấu hình duy nhất đạt toàn bộ điều kiện, nên Task 7 khóa:

```yaml
au_guidance:
  g_scale: 3000000.0
  lambda_learning_rate: 0.000001
```

Smoke một epoch với cấu hình A tiếp tục đạt hai cận: P90 là `0.0551 / 0.0846`
và saturation là `1.72% / 3.74%`. Loss, feature và parameters hữu hạn trong
suốt run. E1, E3 và E5 phải dùng cùng hai giá trị trên; không calibration lại
theo validation accuracy của từng experiment.

Quy trình nằm trong
[`pilot_au_attention_training.py`](../../scripts/pilot_au_attention_training.py),
[`calibrate_au_attention.py`](../../scripts/calibrate_au_attention.py) và
[`run_au_attention_pilot_matrix.py`](../../scripts/run_au_attention_pilot_matrix.py).
Bảng kết quả đầy đủ được lưu tại
[`au-attention-pilot-matrix/report.md`](../../runs/adaptation/au-attention-pilot-matrix/report.md).

## 4. Ba mode và tích hợp train

| Mode | Hành vi |
|---|---|
| `off` | POST gốc, không đọc AU cache; mặc định và load được checkpoint cũ |
| `dynamic` | Dùng `p` đúng `sample_id` của mỗi ảnh |
| `dynamic_shuffled` | Xáo `p` độc lập trong source và target batch, giữ ảnh, `M` và vị trí thiếu AU; dùng seed 42 và báo tỉ lệ mẫu thực sự đổi `p` |

Với hai mode có `G`, source/target training loader mang `sample_id`, `p`,
`M` đi cùng ảnh trong batch; source vẫn mang label, target không mang nhãn
thật. Khi source loader cycle, metadata vẫn đi theo batch được lấy. Lưu mode,
`g_scale`, AU cache identity, `λ`, optimizer và RNG trong checkpoint; resume
không được đổi cấu hình hay xáo lại cache.

## 5. Monitoring và evaluation contract

Smoke test ba mode với **seed 42** trên cùng split, source checkpoint và
cấu hình train. Ghi tỉ số bias/logits, thay đổi attention và `λ` theo head.
CLI evaluation phải chạy được trên checkpoint của từng mode và giữ cách chọn
best checkpoint bằng ExpW validation hiện tại. Chạy train đầy đủ rồi báo cáo
ExpW test accuracy, per-class accuracy và confusion matrix của ba mode cạnh
nhau thuộc bước thí nghiệm sau Task 7; khi báo cáo phải nêu rõ checkpoint
được chọn bằng **nhãn ExpW validation**.

## 6. Test scenarios

### Scenario 1: AU cache và mask

Cache đủ một record cho mỗi ảnh train usable; báo tỉ lệ AU và landmark
thiếu theo domain. Kiểm tra AU theo class nguồn và mask trên ảnh thật,
gồm vài ảnh STR.

### Scenario 2: Guidance và baseline equivalence

Test `G:[B,49,49]`, hai chiều, `G=0` khi thiếu dữ liệu, flip đồng bộ và
`λ=0` khớp POST. `off` load được checkpoint cũ.

### Scenario 3: Smoke ba mode

Smoke vài epoch không NaN, `λ` nhận gradient và hai cận attention ở mục 3
đạt; báo tỉ số bias/logits và phân phối `D`. Cả ba mode chạy qua train step,
checkpoint/resume và evaluation trên một tập nhỏ; kiểm tra shuffle giữ vị trí
thiếu AU và báo tỉ lệ mẫu thực sự đổi `p`.

## 7. Acceptance criteria

Task cài đặt hoàn thành khi cả ba scenario đạt, ba mode
`off`, `dynamic`, `dynamic_shuffled` chạy được từ train step tới evaluation
và checkpoint/resume hoạt động. Không yêu cầu train đầy đủ hoặc chạy toàn bộ
ExpW test để nghiệm thu task này.
