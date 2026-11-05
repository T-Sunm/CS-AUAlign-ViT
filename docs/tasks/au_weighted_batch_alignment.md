# Task 8.1. AU-weighted batch-level alignment và calibration

## 1. Objective

Thêm một local alignment loss dùng AU vào adaptation loop. Loss tạo
representation riêng cho 12 AU từ native Stage 4 tokens, tính centroid theo
batch cho source và target, rồi căn chỉnh hai centroid cùng AU. Phép tính theo
batch nên không phụ thuộc vị trí ghép ngẫu nhiên giữa ảnh source và target.

Task này gồm phần cài đặt loss, calibration `tau_mask`, `n_eff_min`, `q_c` và
`eta`, unit test, smoke train, checkpoint và resume. Task chưa chạy full
adaptation experiment hoặc báo target test accuracy. EMA, class-conditional
alignment, MMD và discriminator mới nằm ngoài phạm vi.

Thiết lập dùng RAF-DB train làm source, ExpW train làm target, Swin-T Stage 4
với 49 token và 768 chiều, 12 AU của Task 7, batch size 64 và seed 42.
Discrepancy giữa hai centroid là squared Euclidean distance.

## 2. Input và output contract

### 2.1 AU metadata

Tái sử dụng trực tiếp AU cache và mask của Task 7:

- `p:[B,12]` theo thứ tự AU `(1,2,4,5,6,9,12,15,17,20,25,26)`, đã chuẩn hóa
  về `[0,1]`;
- `M:[B,49,12]`;
- `au_valid:[B]` và `landmark_valid:[B]`;
- cùng quyết định horizontal flip giữa ảnh và `M`.

Không định nghĩa lại AU schema, mask construction hoặc missing-data policy.
AU cache thiếu hay detector thất bại phải đi qua validity metadata hiện có.

### 2.2 Native tokens

Local loss dùng native source và target streams của paired forward:

```math
Z_s=Norm_s(Y_s),\qquad Z_t=Norm_t(Y_t),
```

với `Z_s,Z_t:[B,49,768]`. Đây là hai native streams sau Stage 4 Block 1 và
domain-specific norm, trước global average pooling. Không dùng cross-domain
streams hoặc pooled features.

Native tokens và AU spatial masks phải dùng cùng thứ tự flatten lưới `7×7`
theo row-major, với `i=7r+c` và `r,c∈{0,...,6}`. Unit test dùng token grid có
giá trị vị trí xác định để kiểm tra ánh xạ này trước khi region pooling.

Paired forward phải trả các native tokens này từ cùng lượt backbone đã dùng để
tính bốn BCS features. Không chạy backbone lần thứ hai. Mean theo chiều token
của `Z_s` và `Z_t` phải khớp `source_features` và `target_features` hiện có
trong sai số số học.

Mở rộng `BCSFeatures` bằng `source_native_tokens` và
`target_native_tokens` ở cuối dataclass, khai báo optional với default `None`.
Giữ nguyên tên và thứ tự của bốn fields hiện có để các caller positional cũ
tiếp tục hoạt động. Khi `au_alignment.enabled=true`, paired forward phải trả
đủ hai tensor; nếu một tensor là `None` thì fail rõ thay vì âm thầm bỏ local
loss. Target-only inference không đổi interface.

Hàm local loss hỗ trợ `B_s != B_t`. Adaptation loop hiện vẫn ghép source và
target batch cùng kích thước theo contract của Task 6.

### 2.3 Output

Module trả:

- `loss_local`, scalar float32 chưa nhân `eta`;
- `region_features_s:[B_s,12,768]` và
  `region_features_t:[B_t,12,768]` khi cần diagnostic;
- `source_centroids:[12,768]`, `target_centroids:[12,768]`;
- `n_eff_s:[12]`, `n_eff_t:[12]` và `valid_aus:[12]`;
- các thống kê coverage dùng cho logging.

## 3. AU-weighted local alignment

### 3.1 Region features và spatial validity

Với domain `d`, ảnh `k` và AU `c`, đặt:

```math
m_{d,k}^c=\sum_{i=1}^{49}M_{d,k,i}^c,
```

```math
r_{d,k}^c=
\frac{\sum_iM_{d,k,i}^cZ_{d,k,i}}
{m_{d,k}^c+\epsilon}.
```

`region_features` có shape `[B,12,768]`. Tính region pooling trong float32 và
giữ computation graph tới native tokens.

Một region hợp lệ khi AU cache, landmark và mask của ảnh hợp lệ, đồng thời:

```math
m_{d,k}^c\ge\tau_{\mathrm{mask}}.
```

Gọi validity này là `g_{d,k}^c` và effective image weight là:

```math
w_{d,k}^c=p_{d,k}^c g_{d,k}^c.
```

Region không hợp lệ có weight bằng 0. Cached AU values và masks không nhận
gradient.

### 3.2 Batch centroids và effective sample size

Tính một centroid cho mỗi AU ở từng domain:

```math
\bar z_d^c=
\frac{\sum_k w_{d,k}^c r_{d,k}^c}
{\sum_k w_{d,k}^c+\epsilon}.
```

Mười hai centroid được giữ riêng, kể cả khi nhiều AU dùng chung spatial mask.
AU intensity theo ảnh quyết định trọng số của từng centroid.

Kish effective sample size dùng cùng weights:

```math
N_{\mathrm{eff},d}^c=
\frac{(\sum_k w_{d,k}^c)^2}
{\sum_k(w_{d,k}^c)^2+\epsilon}.
```

AU `c` tham gia loss khi cả hai domain đạt `n_eff_min` và tổng weight của mỗi
domain lớn hơn `epsilon_weight`:

```math
S_d^c=\sum_k w_{d,k}^c,
```

```math
v_c=
\mathbb 1[N_{\mathrm{eff},s}^c\ge n_{\min}]\cdot
\mathbb 1[N_{\mathrm{eff},t}^c\ge n_{\min}]\cdot
\mathbb 1[S_s^c>\epsilon_{\mathrm{weight}}]\cdot
\mathbb 1[S_t^c>\epsilon_{\mathrm{weight}}].
```

### 3.3 Fixed AU weights và loss

Tính `q_c` một lần từ toàn bộ original usable train sets của RAF-DB và ExpW:

```math
q_c=\bar p_{s,\mathrm{train}}^c\bar p_{t,\mathrm{train}}^c.
```

Mẫu thiếu AU hoặc detector thất bại đóng góp intensity bằng 0 vào mean toàn
tập; báo missing rate riêng. `q_c` có đúng 12 giá trị, không cập nhật theo
mini-batch hoặc sau STR.

Với:

```math
D_c=\|\bar z_s^c-\bar z_t^c\|_2^2,
```

local loss là:

```math
L_{\mathrm{local}}=
\frac{\sum_c v_cq_cD_c}
{\sum_c v_cq_c+\epsilon}.
```

Nếu không có AU hợp lệ, trả `Z_s.sum()*0 + Z_t.sum()*0`. Giá trị bằng 0,
backward vẫn chạy và gradient local về native tokens bằng 0.

Adaptation loop cộng loss đúng một lần:

```math
L_{\mathrm{total}}=L_{\mathrm{POST}}+\eta L_{\mathrm{local}}.
```

Local loss không đọc source labels, target labels hoặc target pseudo-labels.

## 4. Calibration

Calibration dùng AU cache của Task 7 và seed 42. Không dùng ExpW validation
hoặc test accuracy để chọn tham số.

### 4.1 Mask và batch reliability

Trên original usable train sets của hai domain, báo mask mass theo AU với
min, P1, P5, P10, median và P95; báo thêm near-zero rate, AU missing rate và
landmark invalid rate. Chọn `tau_mask` từ mask construction và các thống kê
này. Không lấy một percentile làm ngưỡng tự động.

Mô phỏng batch theo sampling policy của adaptation loader, batch size 64 và
seed 42. Thử một danh sách `n_eff_min` cố định trước khi xem kết quả. Với mỗi
candidate, báo phân phối `N_eff`, tỷ lệ AU hợp lệ, tỷ lệ batch có local loss,
tỷ lệ batch không có AU hợp lệ và tần suất tham gia của từng AU.

Nếu các thống kê coverage chưa đủ để đánh giá centroid, có thể chạy thêm
frozen-feature centroid stability diagnostic từ source checkpoint. Diagnostic
này không phải tiêu chí bắt buộc.

Phase này cũng tính 12 giá trị `q_c` và ghi AU cache identity vào kết quả
calibration.

### 4.2 Gradient calibration

Load cùng adaptation initialization và dùng một danh sách train batch cố định.
Đo gradient trên cùng một tập encoder parameters nhận gradient từ cả
`L_POST` và `L_local`:

```math
R_{\mathrm{raw}}=
\frac{\|\nabla_\theta L_{\mathrm{local}}\|_2}
{\|\nabla_\theta L_{\mathrm{POST}}\|_2+\epsilon}.
```

Đặt `r_target` trước pilot; giá trị bắt đầu đề xuất là `0.1`. Với mỗi batch:

```math
\eta_0=
r_{\mathrm{target}}
\frac{\|\nabla_\theta L_{\mathrm{POST}}\|_2}
{\|\nabla_\theta L_{\mathrm{local}}\|_2+\epsilon}.
```

Chọn một `eta` cố định từ nhiều batch bằng median và P10/P90. Báo
`loss_post`, `loss_local`, raw gradient ratio, weighted gradient ratio, tỷ lệ
local gradient bằng 0, số batch không có AU hợp lệ và non-finite count.
Gradient cosine là diagnostic tùy chọn.

Calibration phải fail rõ nếu local gradient gần như luôn bằng 0. Không tăng
`eta` vô hạn để bù cho AU coverage không đủ.

## 5. Integration, config và checkpoint

Đặt cấu hình trong mục `au_alignment` của
`configs/training/adaptation.yaml`; không tạo config runtime thứ hai.

```yaml
au_alignment:
  enabled: false
  tau_mask: null
  n_eff_min: null
  q_c: null
  epsilon: 1.0e-8
  epsilon_weight: 1.0e-8
  r_target: 0.1
  eta: null
```

Khi `enabled=false`, paired forward, loss, optimizer và evaluation giữ
contract cũ. Khi `enabled=true`, config validation yêu cầu `tau_mask`,
`n_eff_min`, `eta` và đủ 12 giá trị hữu hạn không âm trong `q_c`.

Training loader trả AU metadata khi `au_guidance` hoặc `au_alignment` cần
dùng. Cấu hình attention off và alignment on phải chạy được; hai mode không
được dùng chung một cờ bật tắt. Shuffling của attention cũng không tự động
xáo AU dùng cho alignment.

Checkpoint lưu toàn bộ config `au_alignment` và AU cache identities. Resume
phải từ chối alignment config hoặc cache identity khác. Khi `eta=0`, output
và loss phải tương đương cấu hình có cùng `au_guidance.mode` nhưng
`au_alignment.enabled=false`. Chỉ khi AU attention cũng tắt thì kết quả mới
tương đương POST gốc.

Calibration nằm trong `notebooks/au_alignment_calibration.ipynb`. Notebook
ghi lựa chọn cuối cùng và số liệu hỗ trợ vào
`reports/au_alignment_calibration.md`; các giá trị đã chọn được chép vào
adaptation config trước smoke train.

Mỗi epoch log `loss_local`, `eta*loss_local`, số AU hợp lệ, tỷ lệ batch có
local loss, `N_eff` theo AU và hai domain, tổng weight theo AU, missing rates
và non-finite count.

## 6. Test scenarios

### Scenario 1: Region pooling và centroids

Dùng tensor nhỏ có kết quả tính tay được. Kiểm tra mask chọn đúng token,
intensity tạo đúng image weight, invalid region không đóng góp và hai AU dùng
chung mask vẫn có thể tạo centroid khác nhau. Chạy source và target với batch
size khác nhau ở unit test của local loss. Dùng token grid có giá trị theo vị
trí để xác nhận token và mask cùng flatten row-major trước khi region pooling.

### Scenario 2: Effective sample size và zero evidence

Kiểm tra `N_eff` dùng đúng `w=p*g`; zero weight không được tính là sample.
AU phải bị loại nếu một trong hai domain không đạt `n_eff_min` hoặc tổng
weight không vượt `epsilon_weight`. Khi mọi weight bằng 0, loss finite, bằng
0 và backward an toàn.

### Scenario 3: Pairing invariance

Giữ source batch cố định, hoán vị target images cùng toàn bộ metadata rồi chạy
paired forward trong `eval()` mode. Native token extraction của một domain
không được phụ thuộc domain còn lại. `L_local` trước và sau hoán vị phải khớp
trong sai số số học.

### Scenario 4: Gradient và baseline equivalence

Kiểm tra gradient từ `L_local` đi tới native encoder tokens và parameters.
AU values cùng masks không có gradient. Xác nhận `eta` chỉ nhân một lần;
`eta=0` khớp cấu hình có cùng AU attention mode nhưng tắt alignment trong sai
số số học. Kiểm tra riêng attention off để xác nhận tương đương POST gốc.
Caller cũ khởi tạo `BCSFeatures` bằng bốn positional fields vẫn chạy; khi
alignment bật, thiếu một trong hai native-token fields phải báo lỗi rõ.

### Scenario 5: Smoke, checkpoint và resume

Chạy vài train step với alignment bật. Loss và gradient phải finite, metrics
được ghi, checkpoint/resume khôi phục cùng config và cho bước tiếp theo nhất
quán. Checkpoint có alignment config hoặc AU cache identity khác phải bị từ
chối.

## 7. Acceptance criteria

Task hoàn thành khi:

1. Paired forward expose `Z_s,Z_t:[B,49,768]` từ cùng lượt backbone và mean
   của chúng khớp hai native pooled features hiện có; token và mask cùng dùng
   row-major, còn caller `BCSFeatures` cũ vẫn tương thích.
2. Region pooling, spatial validity, batch centroid, `N_eff`, fixed `q_c` và
   `L_local` đúng các công thức ở mục 3.
3. `tau_mask`, `n_eff_min`, `q_c` và `eta` có số liệu calibration đi kèm;
   calibration không dùng ExpW validation hoặc test accuracy.
4. Năm test scenarios đạt, gồm pairing invariance, zero-valid-AU, gradient,
   baseline equivalence theo cùng AU attention mode và checkpoint/resume.
5. Smoke train chạy được với alignment bật; tỷ lệ batch có local loss và
   các chỉ số `N_eff` được báo cáo.
6. Adaptation config và `reports/au_alignment_calibration.md` chứa các giá trị
   cuối cùng. Task không yêu cầu full adaptation training.

Full experiments thuộc Task 8.2. Vòng đầu so sánh bốn cấu hình: attention và
alignment cùng tắt; chỉ attention bật; chỉ alignment bật; cả hai bật. Các
shuffled controls phải tách riêng từng thành phần để có thể diễn giải hiệu ứng.
