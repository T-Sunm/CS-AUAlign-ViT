# Task 9.2. Learned AU heatmap spatial guidance

## 1. Objective

Thay Gaussian mask dựng từ năm landmark bằng spatial mask lấy từ pretrained AU
heatmap model của Task 9.1. Thay đổi chỉ nằm ở cách tạo `M:[49,12]`. LibreFace
tiếp tục cung cấp AU intensity `p:[12]`. Kiến trúc POST, hai đường
cross-attention, AU-weighted local alignment và lịch training giữ nguyên.

Thí nghiệm so sánh:

| ID | Spatial mask | AU intensity | Mục đích |
|---|---|---|---|
| G0 | Gaussian từ landmark | LibreFace | Baseline E3 hiện tại |
| G1 | FANAU heatmap | LibreFace | Đánh giá learned spatial guidance |

G1 gồm cả learned spatial distribution và peak gate mô tả ở Section 3. Vì
vậy, so sánh G0 với G1 đánh giá toàn bộ phương pháp tạo mask mới. Nó không
tách riêng đóng góp của heatmap shape và peak gate.

G1 giữ landmark validity làm sample-level eligibility gate giống G0. Landmark
không tham gia tạo learned mask. Gate này chỉ hạn chế thay đổi tập sample hợp
lệ trong so sánh đầu tiên. Khả năng chạy learned guidance không phụ thuộc
landmark thuộc một experiment riêng.

Task không yêu cầu G1 phải tốt hơn G0 để được nghiệm thu. Một seed chỉ cho kết
quả thăm dò, không đủ để kết luận về độ ổn định thống kê.

## 2. Scope và giả thuyết

Gaussian baseline gom 12 AU vào năm vùng `brow`, `eye`, `nose`, `mouth` và
`chin`. Các AU cùng vùng có cùng spatial mask trên một ảnh. FANAU trả một
heatmap riêng cho từng AU và từng face crop, nên có thể giữ khác biệt không
gian mà Gaussian mapping bỏ qua.

Giả thuyết của G1 là learned mask cung cấp spatial guidance hữu ích hơn cho
cross-domain token mixing và region pooling. Trước full training, task báo cáo
hai diagnostic để giải thích kết quả:

1. Heatmap còn phân biệt được các AU sau khi downsample từ `64×64` xuống
   `7×7`.
2. Peak gate không loại phần lớn AU evidence ở RAF-DB hoặc ExpW train.

Hai mục này không phải pass/fail gate. Điều kiện chặn full run chỉ nằm ở
Section 6.2, nơi kiểm tra guidance scale, attention saturation, loss và
gradient với hyperparameter của E3.

FANAU chỉ chạy offline. Model không được đưa vào adaptation process, optimizer
hoặc checkpoint của POST.

## 3. Offline heatmap artifact

### 3.1. Input và checkpoint

Extractor đọc manifest và chỉ xử lý record `usable=true` thuộc split được chỉ
định. Input là đúng processed face crop mà dataset sẽ dùng. Không chạy face
detector và không crop lần hai.

Checkpoint là `disfa_adaptation_f0.pth` đã cài trong Task 9.1. Extractor resize
ảnh RGB `uint8` về `256×256` bằng bilinear interpolation, chuyển sang tensor và
min-max normalize toàn ảnh về `[0,1]`. Model trả:

```math
H_i\in\mathbb R^{12\times64\times64}.
```

Channel order phải đúng:

```text
(1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
```

### 3.2. Peak gate, pooling và normalization

Với AU `c` của ảnh `i`, lấy phần dương và peak:

```math
H^+_{i,c}=\max(H_{i,c},0),\qquad
r_{i,c}=\max_{x,y}H^+_{i,c}(x,y).
```

G1 dùng peak threshold cố định:

```math
\tau_{\mathrm{peak}}=0.8.
```

`0.8` là một phần của G1, không phải AU selection hoặc pseudo-label filter.
Giá trị này được đăng ký trước audit Task 9.2 và không được điều chỉnh sau khi
xem ExpW validation/test. Peak audit ở Section 6.1 chỉ là diagnostic.

Heatmap đạt gate được downsample bằng adaptive average pooling:

```math
U_{i,c}=\operatorname{AdaptiveAvgPool}_{7\times7}(H^+_{i,c}).
```

Sau đó chuẩn hóa từng AU bằng maximum trên lưới đã downsample:

```math
M_{i,u,c}=
\begin{cases}
\dfrac{U_{i,u,c}}{\max_v U_{i,v,c}+\epsilon},
& r_{i,c}\ge\tau_{\mathrm{peak}},\\[8pt]
0,
& r_{i,c}<\tau_{\mathrm{peak}}.
\end{cases}
```

Nhánh đầu chỉ hợp lệ khi raw heatmap, peak, pooled heatmap và mẫu số đều hữu
hạn, mẫu số lớn hơn `epsilon`. Nếu không, mask của channel đó bằng 0. Sau khi
flatten theo row-major order:

```math
M_i\in[0,1]^{49\times12}.
```

Không dùng Gaussian fallback. Channel bị gate vẫn giữ vị trí trong tensor để
shape và AU order không đổi.

### 3.3. Artifact schema và identity

Mỗi export tạo một thư mục:

```text
heatmap_cache/
├── spatial_masks.npy   # [N,49,12]
├── peaks.npy           # [N,12]
├── index.csv           # sample_id,row,status
└── metadata.json
```

`index.csv` có đúng một record cho mỗi `sample_id` được yêu cầu. `status` là
`success` hoặc `failed`. Record failed trỏ tới một hàng mask và peak bằng 0,
nhưng vẫn phải có mặt để phân biệt inference failure với missing record.

`metadata.json` ghi:

- manifest path và SHA-256.
- checkpoint path và SHA-256.
- AU order.
- resize interpolation, input normalization, `tau_peak`, pooling, mask
  normalization, epsilon và dtype.
- shape, record count, success count và failure count.

Join luôn dùng `sample_id`, không dùng row order hoặc tên file. Export ghi file
tạm rồi rename atomically. Retry chỉ thay record failed được chọn, không đổi
hàng của record success và không tạo duplicate.

`spatial_masks.npy` có thể được mở read-only bằng NumPy memory mapping. Khi STR
thêm sample mới, extractor tạo bản kế tiếp trong file tạm rồi thay artifact cũ
atomically sau khi cả data, index và metadata đã hoàn tất.

## 4. Dataset integration

### 4.1. Spatial cache và metadata

Thêm loader cho heatmap artifact, tách khỏi `AUCache`. Dataset tiếp tục đọc
LibreFace intensity từ CSV hiện tại và đọc learned mask theo cùng `sample_id`.
Guidance metadata dùng contract:

```python
{
    "au": ...,              # [12], LibreFace intensity trong [0,1]
    "mask": ...,            # [49,12]
    "au_valid": ...,        # LibreFace record hợp lệ
    "landmark_valid": ...,  # landmark Gaussian có thể dựng được
    "heatmap_valid": ...,   # heatmap inference thành công
    "spatial_valid": ...,   # eligibility gate của spatial mode
}
```

G0 và G1 tính:

```python
# G0
spatial_valid = landmark_valid

# G1
spatial_valid = landmark_valid & heatmap_valid
```

G1 không dùng landmark để tạo `M`. Việc giữ landmark gate làm sample set gần
G0 hơn, nên khác biệt chính vẫn là mask construction. `landmark_valid` và
`heatmap_valid` được log riêng để biết sample bị loại vì lý do nào.

Sample thiếu record là lỗi artifact và phải dừng trước training. Record có
`status=failed` là fallback hợp lệ: mask bằng 0, `heatmap_valid=false` và
`spatial_valid=false`. Ảnh vẫn tham gia các loss POST không dùng AU guidance.

### 4.2. Horizontal flip

Training hiện dùng random horizontal flip với xác suất `0.5`. Heatmap được
extract từ crop gốc, chưa augmentation. `GuidanceTransform` phải áp dụng cùng
một quyết định flip cho ảnh và learned mask:

```python
mask.reshape(7, 7, 12).flip(1).reshape(49, 12)
```

Không extract lại heatmap sau mỗi online augmentation. Resize ảnh về `224×224`
và FANAU resize crop về `256×256` đều giữ toàn bộ crop, nên mask và Stage-4
tokens dùng cùng hệ tọa độ tương đối.

Horizontal flip chỉ đảo spatial dimension. Không hoán đổi channel vì 12
channel biểu diễn loại AU, không biểu diễn phía trái hoặc phải của khuôn mặt.

### 4.3. Config và checkpoint

Thêm spatial config độc lập với `au_guidance.mode` và `au_alignment.enabled`:

```yaml
au_spatial:
  mode: heatmap
  peak_threshold: 0.8
```

Hai cache là run input, truyền bằng `--source-heatmap-cache` và
`--target-heatmap-cache`; resolved checkpoint lưu identity của chúng thay vì
nhúng đường dẫn máy cục bộ vào config.

G0 dùng `mode: landmark` và không yêu cầu heatmap cache. Resolved config và
checkpoint lưu spatial mode, threshold cùng các artifact identity ở Section
3.3. Resume từ chối config hoặc identity khác run đã lưu.

## 5. Attention và alignment contract

### 5.1. Cross-attention guidance

Task 7 tiếp tục cung cấp công thức guidance và cách cộng AU bias vào hai
cross-attention logits. G1 chỉ thay `M`. LibreFace intensity `p:[12]`, AU order,
target-query/source-key direction và phép chuyển vị cho chiều ngược giữ nguyên.

Sample-level valid pair dùng:

```text
source.au_valid
and source.spatial_valid
and target.au_valid
and target.spatial_valid
```

Channel có mask bằng 0 tự không đóng góp vào `G`. Nếu cặp ảnh không còn channel
chung, guidance bằng 0 và BCS tiếp tục chạy như POST.

### 5.2. AU-weighted local alignment

Task 8.1 tiếp tục cung cấp region pooling, centroid, effective sample size và
local-loss contract. G1 chỉ thay mask và per-sample spatial validity:

```math
v_{d,i,c}=v_{d,i}^{\mathrm{AU}}
v_{d,i}^{\mathrm{spatial}}
\mathbb 1\!\left[\sum_u M_{d,i,u,c}\ge\tau_{\mathrm{mask}}\right].
```

Weight vẫn là `w_{d,i,c}=p_{d,i,c}v_{d,i,c}`.

Với G1, `v_spatial = v_landmark * v_heatmap`. Peak gate tiếp tục hoạt động theo
channel qua mask mass, không được gộp vào sample-level `heatmap_valid`.

G1 giữ `tau_mask=1.0` cho drop-in comparison. Vì MaxNorm làm maximum của mọi
mask không rỗng bằng 1, threshold này chủ yếu loại channel rỗng. Nó không còn
mang ý nghĩa reliability calibration của Gaussian mask. Audit phải báo mask
mass và coverage mới.

Giữ `n_eff_min=8` và `q_c` hiện tại. `q_c` chỉ phụ thuộc mean LibreFace
intensity trên original RAF-DB/ExpW train, nên không đổi khi chỉ thay `M`.

### 5.3. STR lifecycle

Ảnh STR phải được đưa qua FANAU sau khi ảnh ghép đã commit. Không tái sử dụng
heatmap của donor hoặc target gốc.

Tại mỗi STR boundary:

1. Commit ảnh và STR manifest theo transaction hiện tại.
2. Chạy LibreFace extractor cho sample mới.
3. Chạy heatmap extractor cho cùng sample mới.
4. Ghi hai cache atomically và xác nhận mọi sample mới có status.
5. Rebuild source loader.

Crash hoặc resume giữa các bước phải tiếp tục transaction hiện tại. Retry
không tạo duplicate và không thay artifact của round khác. Original RAF-DB và
ExpW heatmap caches là input bất biến. Mỗi run dùng working cache riêng cho STR.

## 6. Technical gate và experiment

### 6.1. Peak và mask audit

Trước training, chạy extractor trên original usable RAF-DB train và ExpW
train. Report theo AU và domain:

- success, failed và missing count.
- raw peak min, median, P90, P99 và maximum.
- tỷ lệ peak dưới `0.8`.
- tỷ lệ `p_c>=0.5` nhưng peak dưới `0.8`.
- mask mass min, P1, P5, median và P95.
- tỷ lệ zero-mask channel.

Thống kê peak chỉ dùng record có `status=success`. Record failed có peak bằng
0 trong artifact nhưng không được tính như một heatmap hợp lệ có tín hiệu thấp.

Báo thêm G0 valid rate, heatmap success rate, G1 valid rate và số sample G0
hợp lệ nhưng G1 bị loại do heatmap failure. Nếu failure làm thay đổi rõ
valid-pair rate hoặc `N_eff`, report phải coi sample selection là một confound
thay vì quy toàn bộ chênh lệch cho learned mask.

Đo cosine similarity sau downsampling cho các cặp dùng chung Gaussian region,
ít nhất gồm AU1/AU2, AU12/AU25 và AU25/AU26. G0 có cosine similarity bằng 1
cho các cặp dùng cùng mask. G1 report distribution thực tế, không đặt một
ngưỡng similarity tùy ý để pass.

Overlay một review subset cố định của RAF-DB, ExpW và ảnh STR. Checkpoint được
train trên DISFA, nên report phải giữ domain shift như một giới hạn khi diễn
giải kết quả.

### 6.2. Guidance và gradient scale

G1 bắt đầu với hyperparameter của E3:

```text
g_scale = 3000000.0
lambda_learning_rate = 0.000001
tau_mask = 1.0
n_eff_min = 8
eta = 0.0008634870173409581
q_c = 12 giá trị trong e3_both.yaml
```

Giữ các giá trị này chỉ khi G1 qua technical gate trên train batches cố định:

- P90 attention probability change đạt ít nhất `0.01` ở cả hai chiều.
- attention saturation không vượt `5%` ở mỗi chiều.
- có ít nhất 20 cặp AU mạnh theo contract Task 7.
- guidance, loss và gradient đều hữu hạn.
- local gradient khác 0 khi có AU hợp lệ.
- report raw và weighted local-gradient ratio trên cùng encoder parameters.

Nếu attention gate hoặc gradient scale thất bại, dừng full run. Việc chọn
`g_scale` hoặc `eta` mới thuộc một follow-up `G1-calibrated`. Không thay giá
trị giữa run rồi vẫn gọi kết quả là G1-fixed.

### 6.3. G0/G1 protocol

G0 là E3 trong [`e3_both.yaml`](../../configs/training/ablation/e3_both.yaml).
Có thể tái sử dụng run E3 nếu source checkpoint, manifests, AU caches, seed,
training horizon, STR schedule, model-selection rule và code contract đều
khớp. Nếu một identity khác, chạy lại G0.

G1 dùng cùng:

- source initialization.
- RAF-DB train và ExpW train manifests.
- batch size 64, seed 42 và 100 epochs.
- STR sau epoch 25, 50 và 75.
- POST losses, optimizer, scheduler và pseudo-label schedule.
- ExpW validation model-selection rule.

Chỉ `au_spatial.mode` và heatmap artifacts khác G0. Chọn best checkpoint bằng
ExpW validation theo rule hiện tại, sau đó đánh giá checkpoint đã chọn một lần
trên toàn bộ ExpW test. Không dùng test để chọn threshold, hyperparameter,
checkpoint hoặc implementation variant.

## 7. Monitoring và evaluation

Ngoài metric của E3, G1 ghi:

- valid-pair rate và số cặp không có channel chung.
- landmark-invalid, heatmap-failed và combined-spatial-invalid rate theo domain.
- mean, standard deviation và quantiles của `G`.
- bias-to-logit ratio, attention probability change và saturation.
- `N_eff`, AU contribution và local-gradient diagnostics.
- số STR sample success, failed và retried trong heatmap cache.

Spatial mode và artifact identities nằm trong resolved config. Peak, zero-mask
và mask-mass statistics được ghi một lần trong audit Section 6.1, không lặp lại
theo epoch.

Evaluation báo accuracy, balanced accuracy, macro F1, weighted F1, per-class
precision/recall/F1, confusion matrix, best epoch và completed epoch. So sánh
G1 với G0 trên cùng metric và diễn giải theo phạm vi đã khóa ở Section 1. Không
đặt mức cải thiện tối thiểu trong acceptance criteria.

## 8. Test scenarios

### Scenario 1: Heatmap conversion

Dùng tensor nhỏ tính tay để kiểm tra ReLU, peak, adaptive average pooling,
MaxNorm và row-major flatten. Kiểm tra peak đúng bằng `0.8`, peak ngay dưới
ngưỡng, heatmap âm hoàn toàn, mẫu số gần 0, `NaN` và `Inf`. Mọi failure trả
mask 0 mà không sinh giá trị không hữu hạn.

### Scenario 2: Artifact export và identity

Export một manifest fixture có success, failed và hai file trùng stem nhưng
khác `sample_id`. Kiểm tra schema, shape, AU order, row mapping và atomic write.
Duplicate ID bị từ chối. Retry không đổi hàng success. Artifact sai manifest,
checkpoint hoặc threshold identity bị từ chối.

### Scenario 3: Dataset join và flip

Đảo thứ tự `index.csv` nhưng giữ `sample_id`, rồi kiểm tra dataset vẫn lấy đúng
mask. Ép horizontal flip và xác nhận ảnh cùng mask đảo theo chiều ngang. G0
vẫn dựng Gaussian mask như hiện tại và không yêu cầu heatmap cache.

### Scenario 4: Spatial validity và guidance

Kiểm tra G0 dùng landmark validity và G1 dùng giao của landmark validity với
heatmap artifact status. Landmark lỗi hoặc heatmap failure đều làm
`spatial_valid=false` trong G1, nhưng ảnh vẫn tham gia các loss POST không dùng
AU. Channel dưới peak gate không đóng góp. Các channel còn lại vẫn tạo guidance
đúng hướng target-query/source-key.

### Scenario 5: Alignment

Kiểm tra learned mask pooling đúng native token row. Channel mask 0 không tham
gia centroid. `tau_mask=1.0`, `n_eff_min=8`, `q_c` và loss formula giữ contract
Task 8.1. Loss và gradient hữu hạn. Batch không có AU hợp lệ trả zero loss có
gradient 0.

### Scenario 6: G0 baseline equivalence

Với `au_spatial.mode=landmark`, cùng seed, batch và checkpoint phải cho `p`,
`M`, `G`, alignment result và bốn BCS outputs khớp implementation hiện tại
trong sai số số học.

### Scenario 7: STR cache lifecycle

Commit một STR fixture, chạy hai extractor rồi rebuild loader. Mọi sample mới
có LibreFace và heatmap status. Retry/resume không tạo duplicate. Original
caches không đổi. Working cache chỉ thuộc run hiện tại.

### Scenario 8: Checkpoint và resume

Save và resume G1 tại boundary được hỗ trợ. Batch tiếp theo dùng cùng mask và
cho cùng output. Resume từ chối spatial mode, threshold, manifest, heatmap
checkpoint hoặc artifact content khác.

### Scenario 9: Real-data technical gate

Chạy audit và vài optimizer step trên RAF-DB/ExpW batches thật. Kiểm tra
coverage, attention-change, saturation, local loss và gradient theo Section
6.2. Không chạy full 100 epochs nếu technical gate thất bại.

## 9. Acceptance criteria

Task hoàn thành khi:

1. Offline extractor tạo `spatial_masks.npy`, `peaks.npy`, `index.csv` và
   `metadata.json` đúng schema, identity và atomic-write contract.
2. Dataset join learned mask bằng `sample_id`, giữ riêng `landmark_valid` và
   `heatmap_valid`, đặt `spatial_valid` bằng giao của hai flag trong G1, rồi lật
   ảnh/mask bằng cùng random decision.
3. Attention và alignment dùng learned mask nhưng giữ LibreFace intensity,
   AU order, BCS architecture và Task 8.1 loss contract.
4. STR tạo heatmap từ ảnh đã ghép, update working cache đúng run và resume được
   mà không sửa original caches.
5. Chín test scenarios đạt. G0 baseline-equivalence test khớp implementation
   trước Task 9.2.
6. Train-only audit ghi peak coverage, mask mass, mask distinguishability,
   guidance scale và gradient scale. G1 qua technical gate trước full run.
7. G0 được xác minh hoặc chạy lại theo Section 6.3. G1 hoàn thành protocol 100
   epoch, hoặc report ghi rõ technical gate đã chặn full run. Mỗi run có
   resolved config, cache identities, checkpoints, logs và evaluation artifacts.
8. Báo cáo so sánh đầy đủ ExpW metrics, giới hạn của một seed và domain shift
   từ DISFA. Kết quả âm vẫn được giữ nguyên.

Task không calibration lại FANAU intensity và không thay LibreFace bằng peak
heatmap. Các hướng đó thuộc experiment khác.
