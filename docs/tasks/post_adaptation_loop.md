# Task 6. Adaptation Loop and Parameter Updates

## 0. Đọc nhanh

Các task trước đã cung cấp từng thành phần độc lập của POST:

- Task 1 cung cấp source/target datasets, `sample_id`, transforms và manifests.
- Task 2.1 tạo source-pretrained backbone và source prototype \(M_s^{src}\).
- Task 2.2 chịu trách nhiệm dựng hoặc restore hai prototype banks \(M_s,M_t\) cho adaptation.
- Task 3 cung cấp bốn feature:
  \[
  F_s,\quad F_t,\quad F_{t\rightarrow s},\quad F_{s\rightarrow t}.
  \]
- Task 4.1 quản lý target pseudo-label và trusted-target state.
- Task 4.2 trả bốn prototype losses.
- Task 4.3 cung cấp discriminator, GRL và adversarial loss.
- Task 5 cung cấp STR procedure cùng transaction và manifest state.

Task 6 không cài lại logic bên trong các module trên. Nó là **adaptation trainer**, chịu trách nhiệm nối chúng thành một training run hoàn chỉnh: tổ chức batches, gọi module đúng thứ tự, tổng hợp loss, cập nhật tham số, chạy các procedure định kỳ và quản lý checkpoint/resume.

POST coi BCS representation learning, prototype learning và adversarial alignment là các thành phần của cùng adaptation process; pseudo-label refresh và STR là các procedure định kỳ nằm ngoài mini-batch update thông thường.

---

## 1. Mục tiêu và ownership

Task 6 sở hữu:

- adaptation epoch loop;
- paired source-target iteration;
- loss weighting và total objective;
- optimizer;
- learning-rate scheduler;
- AMP/GradScaler;
- backward và parameter update;
- lịch gọi Task 4.1;
- lịch gọi Task 5;
- adaptation checkpoint;
- epoch-boundary state;
- phối hợp normal resume và STR resume;
- logging ở mức trainer.

Task 6 **không sở hữu**:

- cách tạo \(M_s,M_t\);
- kiến trúc prototype classifier;
- BCS internals;
- thuật toán k-means;
- công thức prototype losses;
- kiến trúc discriminator hoặc GRL;
- STR image construction;
- lifecycle nội bộ của STR transaction.

### Prototype ownership

Task 2.1 chỉ cung cấp source checkpoint chứa source prototype đã pretrain. Task 2.2 sở hữu fresh initialization và resume của hai banks.

Với fresh adaptation:

\[
M_s\leftarrow M_s^{src},
\qquad
M_t\leftarrow \operatorname{copy}(M_s).
\]

Với resume:

\[
M_s\leftarrow M_s^{ckpt},
\qquad
M_t\leftarrow M_t^{ckpt}.
\]

Task 6 chỉ gọi API của Task 2.2; không tự copy hoặc restore prototype rows.

Task 2.1 cũng đã khóa boundary này: nó chỉ xuất \(M_s^{src}\), còn việc tạo \(M_t\) thuộc Task 2.2.

### Adversarial-module ownership

Task 4.3 sở hữu discriminator architecture và GRL. Task 6 gọi constructor/factory của Task 4.3 theo adaptation config, rồi chịu trách nhiệm đưa parameters của module vào optimizer và save/restore state của nó.

---

## 2. Cấu hình reproduction

Cần phân biệt hai nhóm cấu hình.

### 2.1 Giá trị được paper báo cáo

POST báo cáo:

- adaptation training: 256 epochs;
- SGD;
- learning rate \(0.001\);
- momentum \(0.9\);
- weight decay \(0.001\);
- cosine learning-rate decay;
- batch size 64;
- STR threshold \(0.95\);
- number of resamples = 10.

Loss weights:

\[
\alpha=1,\qquad
\beta=0.5,\qquad
\gamma=0.1.
\]

Các giá trị này được paper chọn qua hyperparameter analysis.

### 2.2 Quyết định execution của repo

Paper không khóa các chi tiết sau, vì vậy reproduction baseline của repo chọn:

| Quyết định | Baseline của repo | Lý do |
|---|---|---|
| Epoch reference | Một lượt target training loader | Giữ số step không đổi khi STR làm source dataset lớn dần |
| Source loader hết trước | Cycle source iterator | Vẫn ghép đủ source batch cho mỗi target batch |
| Paired batch cuối | `drop_last=true` | Task 3 yêu cầu hai batch cùng \(B\) |
| Initial k-means refresh | Epoch 0 | Có supervision trước training step đầu tiên |
| K-means interval | 10 epochs | Repo default của Task 4.1; paper chỉ nói “every several epochs” |
| Optimizer organization | Một SGD optimizer cho toàn bộ trainable state | Baseline đơn giản với GRL |
| Scheduler granularity | Step một lần mỗi completed epoch | Repo choice |
| Discriminator update | Cùng một backward với các loss khác | Baseline GRL, không alternating optimization |

Task 4.1 đã xác định `kmeans_interval=10` là lựa chọn của repo, không phải giá trị được POST báo cáo.

---

## 3. Khởi tạo adaptation run

### 3.1 Fresh run

Fresh adaptation run thực hiện:

1. Load source-pretraining checkpoint.
2. Task 3 dựng shared trunk và hai domain branches từ source-pretrained backbone.
3. Task 2.2 dựng \(M_s,M_t\) theo fresh-initialization contract.
4. Task 4.3 dựng discriminator/GRL module.
5. Tạo optimizer.
6. Tạo scheduler.
7. Tạo GradScaler nếu AMP bật.
8. Dựng source và target training loaders.
9. Kiểm tra loaders hợp lệ.
10. Chạy Task 4.1 refresh tại epoch 0.
11. Bắt đầu epoch 1.

Không component trainable nào bị freeze trong baseline.

### 3.2 Resume

Resume phải restore adaptation state đã học thay vì chạy fresh initialization.

Không được:

- copy lại \(M_s\rightarrow M_t\);
- tạo discriminator mới thay cho state đã học;
- reset optimizer;
- reset scheduler;
- reset GradScaler;
- bỏ pseudo-label state hợp lệ đã lưu.

Task 2.2 đã quy định riêng rằng resume phải restore từng bank và không copy lại source bank sang target bank.

---

## 4. Source-target batching

### 4.1 Inputs

Source training batch:

\[
(x_s,y_s).
\]

Target adaptation batch:

\[
(x_t,\text{sample\_id}).
\]

Target ground-truth labels không được đưa vào trainer.

Task 3 baseline ghép source sample thứ \(i\) với target sample thứ \(i\), không dùng source label hoặc pseudo-label để chọn pair.

### 4.2 Paired loader requirement

Paired forward yêu cầu:

\[
B_s=B_t.
\]

Baseline dùng cùng configured batch size cho hai training loaders và `drop_last=true`.

Các full-dataset loaders dùng cho Task 4.1 hoặc Task 5 không được drop sample chỉ để phục vụ paired training.

### 4.3 Loader validation

Sau khi dựng paired training loaders, Task 6 phải kiểm tra:

\[
len(loader_s)>0,
\qquad
len(loader_t)>0.
\]

Nếu một loader rỗng, trainer fail trước epoch đầu tiên.

Thông báo lỗi tối thiểu phải chứa:

- domain bị lỗi;
- số training samples;
- configured batch size;
- giá trị `drop_last`.

Không bắt đầu source cycling hoặc epoch loop với loader rỗng.

### 4.4 Định nghĩa epoch

Một adaptation epoch là **một lượt hoàn chỉnh qua target training loader**:

\[
steps_{\text{epoch}}=|Loader_t|.
\]

Source iterator được tạo độc lập.

Nếu source iterator hết trước target iterator, tạo source iterator mới và tiếp tục.

Nếu source loader dài hơn target loader, epoch vẫn kết thúc khi target loader hoàn thành.

Điều này giữ training horizon độc lập với việc source dataset tăng dần sau STR.

---

## 5. Target-supervision lifecycle

Task 4.1 sở hữu:

- pseudo-label generation;
- trusted-mask calculation;
- target feature scan;
- cache structure;
- lookup theo `sample_id`.

Task 6 chỉ sở hữu lịch gọi và persistence của state đó.

### 5.1 Initial refresh

Trước epoch 1, gọi refresh với:

\[
epoch=0.
\]

Task 4.1 sử dụng target-only path:

\[
x_t\rightarrow F_t
\]

trong `eval()` và không gradient.

### 5.2 Periodic refresh

Với repo default:

\[
kmeans\_interval=10,
\]

refresh diễn ra sau các completed epochs:

\[
10,20,30,\ldots,250.
\]

Pseudo-label state giữ nguyên giữa hai lần refresh.

### 5.3 Lookup trong training

Với target `sample_id` của batch hiện tại, Task 6 gọi lookup API của Task 4.1 để lấy:

\[
\hat y_t,\qquad m_t.
\]

Trong đó:

- \(\hat y_t\): pseudo-label;
- \(m_t\): trusted mask.

Task 6 không tự duplicate mapping `sample_id → index`.

---

## 6. Canonical training step

Đây là hợp đồng duy nhất cho một mini-batch update.

### 6.1 Lookup target supervision

Từ target `sample_id`, lấy:

\[
\hat y_t,\qquad m_t.
\]

### 6.2 BCS forward

Gọi Task 3:

\[
(x_s,x_t)
\rightarrow
F_s,F_t,F_{t\rightarrow s},F_{s\rightarrow t}.
\]

Mỗi feature có shape:

\[
[B,768].
\]

### 6.3 Prototype losses

Gọi Task 4.2 với:

- bốn feature;
- \(M_s,M_t\);
- \(y_s\);
- \(\hat y_t\);
- \(m_t\).

Nhận:

\[
L_s,\quad
L_t,\quad
L_{t\rightarrow s},\quad
L_{s\rightarrow t}.
\]

Task 4.2 trả các loss chưa weighting và không gọi backward.

### 6.4 Adversarial loss

Gọi Task 4.3 với:

\[
F_s,\qquad F_t.
\]

Nhận:

\[
L_{ad}.
\]

Task 4.3 trả \(L_{ad}\) chưa nhân \(\gamma\).

### 6.5 Loss assembly

Task 6 là nơi duy nhất lắp:

\[
L_{in}=L_s+L_t,
\]

\[
L_{cross}
=
\alpha L_{t\rightarrow s}
+
\beta L_{s\rightarrow t}
+
\gamma L_{ad},
\]

và:

\[
L_{train}=L_{in}+L_{cross}.
\]

Baseline:

\[
\alpha=1,\qquad
\beta=0.5,\qquad
\gamma=0.1.
\]

Không có outer coefficient kiểu:

\[
L_{in}+\lambda L_{cross}.
\]

STR không phải scalar loss và không được cộng vào objective.

### 6.6 Update

Một mini-batch có đúng một optimization cycle:

1. zero gradients;
2. forward;
3. loss calculation;
4. backward trên \(L_{train}\);
5. optimizer step.

Baseline không có discriminator-only step riêng.

Task 4.3 tự thực hiện gradient reversal ở adversarial branch.

---

## 7. Optimizer, scheduler và AMP

### 7.1 Optimizer

Một SGD optimizer chứa toàn bộ trainable parameters của:

- Task 3 adaptation model;
- \(M_s\);
- \(M_t\);
- Task 4.3 discriminator.

Một parameter không được xuất hiện ở nhiều optimizer groups ngoài ý muốn.

Baseline dùng:

- learning rate `0.001`;
- momentum `0.9`;
- weight decay `0.001`.

### 7.2 Scheduler

Baseline repo dùng cosine schedule với scheduler step sau mỗi completed training epoch.

Scheduler state phải được save và restore.

Resume không được bắt đầu lại cosine schedule.

### 7.3 AMP

Khi `amp=true`:

- model forward chạy dưới autocast FP16 ở các operations phù hợp;
- `GradScaler` được dùng cho backward và optimizer step;
- model/prototype master parameters vẫn ở FP32.

Task 4.2 tự chịu trách nhiệm tính prototype CE trong FP32 theo contract riêng của nó.

Checkpoint lưu scaler state nếu AMP được bật.

---

## 8. Epoch boundary và periodic procedures

Sau optimizer step cuối của epoch \(e\):

1. epoch training được coi là hoàn thành;
2. scheduler step;
3. nếu đến lịch, refresh Task 4.1;
4. persist **epoch-boundary state**;
5. nếu epoch thuộc STR schedule, resolve STR transaction state;
6. chạy, resume hoặc finalize STR theo state đó;
7. nếu cần, rebuild source dataset/loader từ manifest;
8. mark boundary là **completed**;
9. bắt đầu epoch \(e+1\).

Hai trạng thái ở bước 4 và bước 8 có ý nghĩa khác nhau.

### 8.1 Epoch-boundary state

Epoch-boundary state chứa training state sau epoch \(e\) và trước epoch \(e+1\):

- model;
- prototype banks;
- discriminator;
- optimizer;
- scheduler;
- scaler;
- pseudo-label state;
- epoch/global step;
- RNG state.

Đây là snapshot mà STR round của epoch đó sử dụng.

### 8.2 Completed boundary

Boundary chỉ được coi là completed khi mọi periodic operation bắt buộc tại boundary đó đã hoàn thành và trainer đã đưa source-training state về trạng thái sẵn sàng cho epoch tiếp theo.

Nếu epoch không thuộc STR schedule:

\[
boundary\ state=completed\ boundary.
\]

Nếu epoch thuộc STR schedule:

- epoch-boundary state được persist trước khi STR chạy;
- STR transaction có lifecycle riêng;
- boundary chưa completed cho tới khi transaction đã hoàn thành, source dataset/loader đã được dựng lại và trainer đã finalization xong.

Hai state machines có liên quan nhưng không đồng nhất:

- Task 5 quyết định STR round đã hoàn tất hay chưa.
- Task 6 quyết định epoch boundary đã sẵn sàng để chuyển sang epoch tiếp theo hay chưa.

---

## 9. STR integration

STR chạy sau:

\[
25,50,75,100,125,150,175,200,225,250.
\]

Task 5 xác định lịch này là repo interpretation của `number of resamples = 10`; sau epoch 250 còn sáu adaptation epochs.

Tại một STR boundary:

1. epoch training hoàn thành;
2. scheduler và pseudo-label refresh cần thiết đã hoàn thành;
3. epoch-boundary state được persist;
4. Task 6 xác định expected STR `round_id`;
5. Task 6 đọc persistent transaction state hiện tại của round đó;
6. tùy trạng thái, tạo mới, resume hoặc chỉ finalize;
7. khi STR đã `completed`, source dataset và sampler được dựng lại từ source gốc + committed STR manifest;
8. boundary được mark completed;
9. epoch tiếp theo mới được bắt đầu.

Source records mới chỉ được đưa vào training loader sau khi STR round đã completed.

### 9.1 K-means và STR cùng boundary

Ví dụ epoch 50:

\[
\text{train epoch 50}
\]

\[
\rightarrow
\text{scheduler step}
\]

\[
\rightarrow
\text{k-means refresh}
\]

\[
\rightarrow
\text{persist epoch-50 boundary}
\]

\[
\rightarrow
\text{resolve STR transaction}
\]

\[
\rightarrow
\text{STR/finalization}
\]

\[
\rightarrow
\text{rebuild source loader}
\]

\[
\rightarrow
\text{mark boundary completed}.
\]

K-means không update model parameters, vì vậy k-means và STR đều nhìn cùng model parameters sau epoch 50.

STR tự chạy target prediction mới tại mỗi round và không dùng stale `predicted_labels`/`confidence` cache của Task 4.1 làm source of truth.

---

## 10. Checkpoint và resume

### 10.1 Checkpoint content

Epoch-boundary checkpoint tối thiểu lưu:

- schema/version;
- completed training epoch;
- global step;
- Task 3 state;
- \(M_s\);
- \(M_t\);
- discriminator state;
- optimizer state;
- scheduler state;
- GradScaler state nếu có;
- adaptation configuration;
- class mapping;
- feature dimension;
- Task 4.1 state;
- target `sample_ids`;
- RNG states;
- source-pretraining checkpoint provenance;
- expected STR `round_id` hoặc transaction reference nếu boundary thuộc STR schedule;
- boundary completion status.

Checkpoint có thể lưu STR metadata để đối chiếu, nhưng **không phải source of truth cho live STR transaction status**.

### 10.2 Nguồn xác định STR lifecycle

STR transaction và manifest tiếp tục thay đổi sau khi epoch-boundary checkpoint đã được ghi.

Vì vậy khi resume:

1. load epoch-boundary checkpoint để restore training state;
2. từ epoch và STR schedule xác định expected `round_id`;
3. đọc **persistent STR transaction state hiện tại** cho round đó;
4. phân biệt:
   - transaction chưa tồn tại;
   - `in_progress`;
   - `completed`;
5. dùng checkpoint metadata chỉ để kiểm tra consistency.

Không được chọn resume branch chỉ từ STR status được serialize trong một checkpoint cũ.

Manifest là source of truth cho augmented records đã commit; transaction store là source of truth cho lifecycle hiện tại của STR round.

### 10.3 Resume state machine

Với boundary thuộc STR schedule:

| Boundary state | Live STR transaction | Hành vi |
|---|---|---|
| `completed` | `completed` | Bắt đầu epoch \(e+1\) |
| `completed` | absent hoặc `in_progress` | Fail: boundary/STR state không nhất quán |
| incomplete | absent | Tạo transaction từ epoch-boundary snapshot → chạy STR → rebuild source loader → complete boundary |
| incomplete | `in_progress` | Restore epoch-boundary snapshot → resume STR → rebuild source loader → complete boundary |
| incomplete | `completed` | Không chạy lại STR → rebuild source loader từ manifest → complete boundary |

Nếu boundary không thuộc STR schedule, `boundary=completed` là đủ để sang epoch tiếp theo; incomplete boundary chỉ cần được finalization theo state của trainer và không tạo STR transaction.

Với epoch thuộc STR schedule, `boundary=completed` chỉ hợp lệ khi live transaction của expected `round_id` cũng là `completed`. Nếu transaction absent hoặc `in_progress`, trainer phải fail rõ ràng vì persistence không nhất quán; không tự tạo hoặc sửa transaction và không bắt đầu epoch tiếp theo. Lỗi phải nêu epoch, expected `round_id`, boundary state và live transaction state.

### 10.4 Normal completed-boundary resume

Nếu boundary đã `completed` và thỏa invariant ở mục 10.3, resume bắt đầu từ:

\[
epoch=e+1.
\]

Không chạy lại:

- epoch \(e\);
- scheduler step của epoch đó;
- k-means refresh của boundary đó;
- STR của boundary đó.

### 10.5 Resume khi STR transaction chưa tồn tại

Crash có thể xảy ra sau khi:

- training epoch hoàn thành;
- scheduler step xong;
- k-means refresh cần thiết xong;
- epoch-boundary checkpoint đã persist;

nhưng trước khi Task 6 tạo STR transaction.

Khi resume:

1. restore epoch-boundary state;
2. xác định rằng epoch thuộc STR schedule;
3. tính expected `round_id`;
4. kiểm tra persistent transaction store và xác nhận transaction chưa tồn tại;
5. tạo đúng một transaction mới tham chiếu epoch-boundary snapshot này;
6. chạy STR round theo Task 5;
7. khi transaction `completed`, rebuild source dataset/loader;
8. mark boundary completed;
9. bắt đầu epoch tiếp theo.

Không:

- train lại epoch vừa hoàn thành;
- step scheduler thêm lần nữa;
- refresh k-means thêm lần nữa;
- rollback về completed boundary trước đó.

Transaction creation phải có semantics tránh tạo hai round cùng `(epoch, round_id)` nếu restart xảy ra ngay tại thao tác tạo transaction.

### 10.6 Resume khi STR `in_progress`

Nếu Task 6 resume và thấy boundary chưa completed cùng live transaction `in_progress`:

1. không bắt đầu epoch tiếp theo;
2. load epoch-boundary checkpoint mà transaction tham chiếu;
3. restore model, optimizer, scheduler, scaler và pseudo-label state từ boundary đó;
4. giao Task 5 resume cùng transaction;
5. giữ nguyên `selected_class`;
6. để Task 5 dùng manifest làm source of truth cho records đã commit;
7. khi STR chuyển sang `completed`, rebuild source dataset/loader từ manifest;
8. mark boundary completed;
9. mới tiếp tục epoch tiếp theo.

Không tạo fresh STR round mới cho cùng epoch.

### 10.7 Resume khi STR đã `completed` nhưng boundary chưa completed

Crash có thể xảy ra sau khi Task 5 đã:

- commit đầy đủ STR manifest;
- chuyển transaction sang `status=completed`;

nhưng trước khi Task 6:

- rebuild source dataset/loader;
- mark epoch boundary là `completed`.

Khi resume:

1. đọc live transaction và thấy `completed`;
2. không chạy lại STR;
3. không chạy lại target inference;
4. không tính lại minority class;
5. không tạo transaction mới;
6. đọc committed augmented records từ STR manifest;
7. dựng lại source training dataset và sampler/loader từ source gốc + manifest;
8. kiểm tra các invariant cần thiết của manifest/files;
9. mark epoch boundary là `completed`;
10. bắt đầu epoch \(e+1\).

Bước finalization này phải **idempotent**.

Nếu trainer crash thêm một lần trong quá trình rebuild/finalize, chạy lại cùng procedure từ manifest phải tạo ra cùng source-training state và không duplicate augmented records.

### 10.8 Checkpoint stale so với transaction

Một case hợp lệ là:

- epoch-boundary checkpoint cho thấy STR chưa bắt đầu hoặc chỉ chứa metadata cũ;
- persistent transaction store thực tế đã `in_progress` hoặc `completed`.

Resume phải ưu tiên **live transaction state**.

Ví dụ nếu checkpoint cũ ngụ ý round chưa chạy nhưng transaction store cho thấy `completed`, trainer phải đi thẳng vào nhánh finalization ở Section 10.7, không tạo hoặc chạy lại STR round.

### 10.9 Checkpoint granularity

Baseline đảm bảo resume tại:

- epoch boundaries;
- khoảng sau boundary persist nhưng trước STR transaction creation;
- STR mid-round;
- khoảng sau STR `completed` nhưng trước trainer boundary finalization.

Exact resume giữa hai mini-batches của cùng epoch không phải requirement.

Crash giữa epoch quay về completed boundary gần nhất.

---

## 11. Logging

### Step/interval level

Task 6 log:

- `loss_total`;
- `loss_in`;
- `loss_cross`;
- `loss_s`;
- `loss_t`;
- `loss_t2s`;
- `loss_s2t`;
- `loss_ad`;
- `num_trusted`;
- `trusted_ratio`;
- learning rate.

Task 4.2 có thể trả các term chi tiết để diagnostic; Task 6 chỉ forward chúng sang logging nếu cần, không tái định nghĩa semantics.

### Epoch level

Log:

- mean losses;
- trusted ratio;
- optimizer-step count;
- learning rate;
- source training-set size;
- target training-set size;
- cumulative STR sample count.

### Periodic modules

Task 4.1 và Task 5 tự tính module-specific statistics.

Task 6 chỉ ghi structured outputs của chúng vào run log thay vì duplicate calculation.

---

## 12. Target-label isolation và evaluation boundary

Adaptation trainer không được đọc target ground-truth labels.

Target ground truth không được dùng cho:

- paired training;
- pseudo-label refresh;
- trusted-set selection;
- STR;
- optimizer decisions;
- stopping decisions.

ExpW validation chỉ dùng để đo accuracy sau mỗi epoch và chọn `best.pt`;
không đi vào optimizer, pseudo-label hoặc STR. ExpW test dành cho evaluation cuối.

Baseline reproduction chạy đủ 256 epochs, không early stop dựa trên target accuracy.

Sau STR round cuối ở epoch 250, epochs:

\[
251,\ldots,256
\]

tiếp tục train với cumulative source dataset hiện tại.

Final adaptation checkpoint sau epoch 256 và `best.pt` đều có thể được Task 7
đánh giá. Checkpoint `best.pt` chỉ được lưu sau khi STR boundary của epoch đó
hoàn tất.

---

## 13. Interface với các task

| Task | Contract Task 6 sử dụng |
|---|---|
| Task 1 | source/target datasets, loaders, `sample_id`, manifests |
| Task 2.1 | source-pretraining checkpoint |
| Task 2.2 | fresh/restore API cho \(M_s,M_t\) |
| Task 3 | paired forward và target-only path |
| Task 4.1 | refresh, lookup và supervision state |
| Task 4.2 | prototype-loss API |
| Task 4.3 | adversarial-module factory và \(L_{ad}\) |
| Task 5 | STR round + persistent transaction/manifest API |
| Task 7 | nhận final checkpoint để inference/evaluation |

Task 6 không được phụ thuộc vào internals ngoài các interface này.

---

## 14. Test scenarios

### Scenario 1: Fresh initialization

Từ source checkpoint, xác nhận:

- Task 3 được dựng đúng;
- Task 2.2 trả \(M_s,M_t\) đúng fresh contract;
- Task 4.3 factory tạo discriminator;
- optimizer chứa toàn bộ trainable parameters đúng một lần;
- scheduler/scaler được tạo;
- Task 4.1 refresh epoch 0 xảy ra trước optimizer step đầu tiên.

Không test lại internal copy logic của Task 2.2 hoặc architecture internals của Task 4.3.

### Scenario 2: Loader và epoch semantics

Kiểm tra:

- paired loaders rỗng bị fail trước training;
- lỗi nêu domain, sample count, batch size và `drop_last`;
- source và target batch luôn cùng \(B\);
- source loader ngắn hơn thì được cycle;
- source loader dài hơn không kéo dài epoch;
- optimizer steps mỗi epoch bằng `len(target_loader)`.

### Scenario 3: Một training step

Dùng controlled outputs để xác nhận execution order:

1. pseudo-label lookup;
2. Task 3 forward;
3. Task 4.2;
4. Task 4.3;
5. loss assembly;
6. backward;
7. optimizer step.

Với:

\[
L_s=1,\;
L_t=2,\;
L_{t\rightarrow s}=3,\;
L_{s\rightarrow t}=4,\;
L_{ad}=5,
\]

baseline phải cho:

\[
L_{in}=3,
\]

\[
L_{cross}=3+0.5(4)+0.1(5)=5.5,
\]

\[
L_{train}=8.5.
\]

Xác nhận coefficients chỉ được áp dụng ở Task 6.

Nếu trusted set của batch rỗng, training step vẫn finite và vẫn update được từ các loss còn lại.

### Scenario 4: Periodic scheduling

Với `kmeans_interval=10`, xác nhận:

- refresh tại epoch 0;
- refresh sau 10,20,...;
- không refresh giữa epoch.

Với STR schedule, xác nhận chỉ chạy sau:

\[
25,50,\ldots,250.
\]

Epoch có cả k-means và STR phải tuân:

\[
scheduler
\rightarrow
kmeans
\rightarrow
persist\ boundary
\rightarrow
resolve\ STR\ transaction.
\]

### Scenario 5: STR boundary

Mô phỏng một STR round thành công.

Xác nhận:

- epoch-boundary checkpoint tồn tại trước round;
- transaction tham chiếu checkpoint đó;
- source loader chưa thay đổi khi round còn `in_progress`;
- sau `completed`, source dataset/sampler được rebuild;
- target loader không đổi;
- epoch tiếp theo nhìn thấy augmented records.

### Scenario 6: Checkpoint và resume

#### A. Normal completed-boundary resume

Sau khi train để \(M_s\neq M_t\):

- model state khớp;
- hai prototype banks khớp;
- discriminator khớp;
- optimizer/scheduler/scaler khớp;
- pseudo-label state khớp;
- RNG state khớp;
- không fresh-init bất kỳ module nào.

#### B. Crash sau boundary persist nhưng trước transaction creation

Mô phỏng:

- epoch thuộc STR schedule;
- epoch-boundary checkpoint đã persist;
- boundary incomplete;
- persistent transaction chưa tồn tại.

Sau resume, xác nhận:

1. epoch vừa xong không chạy lại;
2. scheduler không step lần hai;
3. k-means không refresh lần hai;
4. trainer phát hiện transaction absent;
5. tạo đúng một transaction cho expected round;
6. transaction tham chiếu đúng epoch-boundary snapshot;
7. STR chạy đúng một lần;
8. boundary chỉ completed sau source-loader rebuild.

#### C. STR mid-round resume

Mô phỏng crash khi transaction `in_progress`.

Xác nhận:

- trainer load đúng epoch-boundary snapshot;
- không bắt đầu epoch tiếp;
- `selected_class` không được tính lại;
- Task 5 tiếp tục từ manifest;
- không tạo duplicate records;
- chỉ sau `completed` mới rebuild source loader và tiếp tục training.

#### D. STR completed nhưng boundary chưa finalized

Mô phỏng:

- STR đã commit manifest;
- STR transaction đã `completed`;
- trainer chưa rebuild source loader;
- boundary vẫn incomplete.

Sau resume, xác nhận:

1. STR procedure không được gọi lại;
2. không chạy lại target inference;
3. không tính lại minority class;
4. không tạo STR transaction mới;
5. không có augmented record duplicate;
6. source dataset được dựng lại từ manifest và chứa đúng committed records;
7. source sampler/loader nhìn thấy đúng dataset mới;
8. boundary được chuyển sang `completed`;
9. epoch tiếp theo bắt đầu đúng một lần.

Chạy finalization lần hai với cùng manifest phải cho cùng kết quả để xác nhận idempotency.

#### E. Checkpoint status stale hơn transaction

Tạo trạng thái:

- checkpoint metadata cho thấy STR chưa chạy;
- persistent transaction hiện tại là `completed`;
- manifest chứa committed records.

Resume phải:

1. đọc live transaction state;
2. chọn nhánh `completed`;
3. không tạo transaction mới;
4. không chạy STR;
5. rebuild source state từ manifest;
6. finalize boundary.

Test này xác nhận checkpoint STR metadata không phải source of truth.

#### F. Completed boundary nhưng STR state lệch

Với epoch thuộc STR schedule, mô phỏng `boundary=completed` nhưng live transaction của expected `round_id` absent hoặc `in_progress`.

Xác nhận trainer không bắt đầu epoch tiếp theo, không tạo transaction mới, không tự mark transaction `completed` và fail rõ ràng với epoch, expected `round_id`, boundary state và live transaction state.

### Scenario 7: Full lifecycle và target-label isolation

Chạy trainer giả lập 256 epochs với modules nhẹ.

Xác nhận:

- không target ground-truth field nào được adaptation trainer truy cập;
- đúng 10 STR rounds;
- round cuối sau epoch 250;
- epochs 251–256 vẫn chạy;
- không có round thứ 11;
- final checkpoint sau epoch 256 load được bởi Task 7 interface.

---

## 15. Acceptance criteria

Task 6 hoàn thành khi:

1. Fresh run có thể dựng toàn bộ adaptation trainer chỉ qua public interfaces của Tasks 1–5.
2. Prototype initialization/restore được giao hoàn toàn cho Task 2.2; Task 6 không duplicate copy logic.
3. Discriminator được dựng qua Task 4.3 interface; Task 6 chỉ sở hữu orchestration và state persistence.
4. Paired training loaders rỗng bị phát hiện trước training với lỗi rõ ràng.
5. Một epoch có đúng một lượt target loader và source iterator được cycle khi cần.
6. Mỗi training step thực hiện đúng một loss assembly, backward và optimizer update.
7. Total loss đúng:
   \[
   L_s+L_t
   +L_{t\rightarrow s}
   +0.5L_{s\rightarrow t}
   +0.1L_{ad}.
   \]
8. Empty trusted batch không làm trainer fail hoặc tạo NaN.
9. K-means refresh chạy đúng lifecycle của Task 4.1 và không nằm trong mini-batch loop.
10. STR chạy đúng 10 scheduled rounds và source loader chỉ nhìn thấy dữ liệu mới sau khi round completed.
11. Epoch-boundary checkpoint được persist trước STR và là snapshot của toàn bộ training state sau epoch đó.
12. Nếu boundary incomplete và transaction chưa tồn tại, resume tạo đúng STR transaction từ epoch-boundary snapshot mà không chạy lại epoch, scheduler hoặc k-means.
13. Nếu STR `in_progress`, resume cùng transaction và hoàn thành round trước khi bước sang epoch tiếp theo.
14. Nếu STR đã `completed` nhưng boundary chưa finalized, resume không chạy lại STR mà dựng lại source training state từ manifest rồi hoàn tất boundary.
15. Live transaction state quyết định nhánh STR resume; checkpoint STR metadata chỉ dùng để đối chiếu.
16. Boundary finalization sau STR là idempotent và không tạo duplicate augmented records.
17. Với epoch có STR, `boundary=completed` chỉ hợp lệ khi live transaction tương ứng cũng `completed`; trạng thái lệch phải fail rõ ràng, không tự sửa hoặc tiếp tục training.
18. Normal resume restore model, prototypes, discriminator, optimizer, scheduler, scaler và target-supervision state mà không fresh-initialize lại.
19. Adaptation trainer không đọc target ground-truth labels.
20. Baseline chạy đủ 256 epochs; epochs 251–256 chạy sau STR round cuối.
21. Final checkpoint có thể được Task 7 load trực tiếp cho target inference/evaluation.
