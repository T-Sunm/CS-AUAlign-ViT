# Task 5. Self-training resampling (STR)

## 1. Objective

STR tạo thêm ảnh source cho class đang ít mẫu bằng cách thay bốn vùng mặt của
một ảnh source bằng vùng tương ứng từ target. Ảnh hoàn chỉnh giữ nhãn source và
được đưa vào source training set. STR chạy theo lịch trong adaptation; nó không
phải một scalar loss.

POST mô tả STR khác nhau giữa đoạn văn và Algorithm 1: đoạn văn nói chọn class
theo target predictions và đưa ảnh vào target set, còn Algorithm 1 đếm source
labels và đưa ảnh vào source set. Baseline này ưu tiên Algorithm 1. Các quy tắc
về lịch chạy, vòng đời ảnh và hình học patch bên dưới là lựa chọn của repo vì
paper không chỉ rõ.

## 2. Input và prediction

STR dùng source train gốc $D_s^{orig}$ với ground-truth labels, target train
$D_t$ không expose ground-truth labels, face crops và `landmarks_crop` đã lưu
từ Task 1. Bốn landmark cần dùng là left eye, right eye, left mouth corner và
right mouth corner; không dùng nose.

Mỗi round chạy model hiện tại trên toàn bộ target train trong `eval` mode và
không tính gradient. Trước inference, lưu `was_training = model.training`; sau
inference, khôi phục đúng mode trước đó. Inference dùng evaluation transform
của repo: face crop, resize `224 × 224`, tensor, normalize; không có random
flip. Prediction đi qua target domain-specific feature và target prototype
head:

```math
Z_t=F_tM_t^T,\qquad
p_t=\operatorname{softmax}(Z_t),\qquad
pre(x_t)=\arg\max_k p_{t,k},\qquad
cnf(x_t)=\max_k p_{t,k}.
```

Chỉ dùng target `sample_id`, prediction và confidence; không đọc target label
trong manifest. Dùng cùng canonical class mapping của source và target.

## 3. Lịch và vòng đời dữ liệu

POST baseline chạy đủ 256 adaptation epochs, đánh số từ 1, và không bật early
stopping. Chạy STR sau các epoch:

```text
25, 50, 75, 100, 125, 150, 175, 200, 225, 250
```

Không chạy trước epoch 1. Sau round cuối, tiếp tục train 6 epochs với ảnh đã
tạo. Repo diễn giải `number of resamples = 10` của POST là 10 rounds, không
phải 10 ảnh mỗi round.

Framework có thể hỗ trợ early stopping như một option ngoài baseline. Khi
option này được bật, adaptation kết thúc ngay tại epoch dừng, STR không chạy
bù các rounds chưa đến lịch.

Ảnh augmented được tích lũy trong source training set:

```math
D_s^{(0)}=D_s^{orig},\qquad
D_s^{(r)}=D_s^{orig}\cup A_1\cup\cdots\cup A_r.
```

Mỗi target `sample_id` được dùng tối đa một lần trong cả adaptation run. Vì
vậy tổng ảnh tạo thành không vượt quá số target train samples. Nếu một round
không tạo được ảnh, training tiếp tục với source set hiện tại và round sau vẫn
refresh predictions. Không hạ confidence threshold để tạo thêm ảnh.

## 4. Chọn class và cặp ảnh trong một round

Ở đầu round $r$, đếm class trên $D_s^{(r-1)}$, gồm source gốc và toàn bộ ảnh
augmented đã giữ lại. Chọn class có count thấp nhất; nếu hòa, lấy class index
nhỏ nhất:

```math
y_r^*=\arg\min_k count(D_s^{(r-1)},k).
```

Giữ $y_r^*$ cố định cho hết round. Duyệt mỗi target train `sample_id` chưa
được dùng theo thứ tự ID ổn định. Một target là candidate khi:

```math
pre(x_t)=y_r^*,\qquad cnf(x_t)>0.95.
```

Kiểm tra landmark và bốn target ROI một lần trước khi tìm donor. Nếu target
geometry không hợp lệ thì bỏ candidate ngay, không duyệt donor pool.

Với mỗi candidate, tạo donor ordering bằng seed xác định từ
`global_seed + round_id + target_id`, rồi duyệt các donor có ground-truth
label $y_r^*$ trong $D_s^{orig}$. Seed theo candidate giúp retry giữ nguyên
thứ tự donor dù các target đã commit được bỏ qua. Chỉ copy face crop của
donor; không sửa ảnh source gốc. Nếu donor không hợp lệ về hình học, thử donor
tiếp theo, tối đa một lần mỗi donor. Ảnh augmented không bao giờ là donor. Khi
không có donor hợp lệ, bỏ qua candidate.

## 5. Bốn vùng patch

Ghép patch trên face crops ở kích thước gốc, trước online training transforms.
Landmark coordinates là `landmarks_crop` của từng ảnh. Với source và target,
tính riêng khoảng cách giữa hai mắt:

```math
d_{io}=\|p_{left\ eye}-p_{right\ eye}\|_2,\qquad
s=\operatorname{round}(0.4d_{io}).
```

`0.4` là hệ số của repo. Dùng quy tắc làm tròn của Python `round` cho $s$ và
từng tọa độ tâm $(x,y)$. Với tâm nguyên $(c_x,c_y)$ và cạnh nguyên $s$, đặt
$left=c_x-\lfloor s/2\rfloor$, $top=c_y-\lfloor s/2\rfloor$. ROI là bbox
half-open:

```math
[left,\ top,\ left+s,\ top+s).
```

Yêu cầu khoảng cách và tọa độ hữu hạn, $s\ge1$, cả bốn ROI nằm trọn trong
face crop, và không có hai ROI nào overlap trên cùng một ảnh. Kiểm tra riêng
cho source và target. Hai ROI chỉ chạm biên không tính là overlap. Nếu một
điều kiện sai, bỏ qua cặp ảnh đó; không clip ROI.

Resize từng target patch bằng bilinear interpolation về đúng kích thước source
ROI rồi hard paste lên bản copy của source crop. Hoàn tất cả bốn vùng trước khi
lưu. Mỗi target thành công tạo đúng một augmented image có label $y_r^*$;
không lưu ảnh trung gian sau từng patch. Đây là cách repo diễn giải vị trí
`put back` trong Algorithm 1.

## 6. Đưa ảnh vào training và resume

Mỗi ảnh STR có `sample_id` xác định chỉ từ `round_id + target_id`, chẳng hạn
hash của chuỗi canonical `str:{round_id}:{target_id}`. `donor_id` chỉ là
provenance và không tham gia định danh. Retry với donor khác vì thế không thể
tạo record thứ hai cho cùng target trong cùng round.

Sau khi ghép đủ bốn vùng, ghi ảnh vào file tạm rồi atomic rename sang đường
dẫn cuối. Tiếp theo, thêm record vào manifest STR bằng thao tác temp-file và
atomic replace. Record chỉ được xem là committed khi manifest đã chứa record
và file ảnh cuối tồn tại. Crash giữa hai bước có thể để lại orphan image;
resume bỏ qua file này vì manifest là source of truth. Nếu manifest trỏ tới
file thiếu hoặc hỏng, resume phải báo lỗi thay vì âm thầm bỏ record.

Record source train chứa label $y_r^*$ và provenance gồm round, donor ID và
target ID. Record dùng `dataset=raf_db`, `split=train`, `class_name` theo
canonical mapping và `group_id` của donor để tương thích source loader. Record
cũng có `usable=true`, `processed_path`, `landmarks_crop` và
`landmarks_original` kế thừa từ donor. Các tọa độ này giữ coordinate frame của
donor để downstream dùng lại; chúng không phải landmarks được MTCNN detect lại
trên ảnh augmented. Lưu provenance `landmark_source=donor_inherited` để tránh
nhầm hai trường hợp.

Chỉ sau khi record đã commit mới tăng class count trong memory và đánh dấu
target ID đã dùng. Lỗi tạo ảnh hoặc lỗi commit không thay đổi count và danh
sách ID đã dùng.

Source training loader phải đọc được source gốc cùng ảnh STR, áp dụng training
transform hiện tại lên cả hai. Target adaptation loader tiếp tục chỉ trả ảnh
và `sample_id`; test/validation splits không tham gia STR. Không thay đổi ảnh
hoặc manifest gốc của Task 1. Cách lưu record cần phù hợp với validation và
source loader hiện có.

Sau khi round commit hoàn tất, rebuild hoặc refresh source training dataset và
sampler trước epoch tiếp theo. Sampler không được giữ dataset length cũ.
Validation và test loaders không thay đổi. Training không chạy song song với
một STR round đang `in_progress`, nên các record commit giữa round chỉ được đưa
vào loader sau khi round chuyển sang `completed`.

Manifest STR là source of truth cho các ảnh đã commit. Khi resume, dựng lại:

```math
used\_target\_ids=\{target\_id\mid record\in manifest_{STR}\}.
```

Checkpoint lưu training state và dùng để đối chiếu, không phải nguồn duy nhất
của `used_target_ids` hay augmented records. Nếu checkpoint và manifest lệch
nhau, manifest quyết định record nào đã commit; các invariant hoặc file bị
thiếu vẫn phải làm resume fail rõ ràng.

Trước sample đầu tiên của round, ghi atomically round transaction state gồm
`round_id`, epoch, `selected_class`, `status=in_progress`, `global_seed` và
tham chiếu tới checkpoint model ở đầu round. Model snapshot này cố định
prediction cho cả round. Nếu crash, resume restore snapshot, tiếp tục cùng
`selected_class`, chạy lại deterministic target inference theo thứ tự ID ổn
định và bỏ qua target đã có record trong manifest. Baseline không cần lưu
traversal cursor hoặc toàn bộ prediction cache.

Chỉ sau khi quét hết target set mới đổi transaction thành
`status=completed` bằng atomic replace. Nếu resume gặp round `in_progress`,
không tính lại minority class. Round kế tiếp chỉ được chọn class mới sau khi
round hiện tại đã `completed`. Adaptation training task chịu trách nhiệm nối
transaction này vào epoch loop và kiểm thử save/resume end-to-end.

## 7. Monitoring

Mỗi round ghi class được chọn, số target đạt confidence gate, số ảnh tạo
thành công, source class counts trước/sau round và số cặp bị loại theo lý do
`invalid_landmark`, `out_of_bounds`, `overlap` hoặc không có donor hợp lệ.
Theo dõi các số này để biết ROI rule có loại quá nhiều candidates.

## 8. Test scenarios

### Scenario 1: Chọn sample và cập nhật state

Với source counts và target predictions cố định, xác nhận chọn đúng minority
class, xử lý hòa theo class index, dùng threshold nghiêm ngặt `>0.95`, giữ
class cố định trong round và chọn donor từ source gốc. Round sau đếm lại trên
source set tích lũy. Target đã tạo ảnh không được dùng lần nữa; round rỗng
không làm training dừng. Target geometry lỗi bị loại trước donor search.

### Scenario 2: Ghép ảnh

Với face crops và landmarks biết trước, xác nhận ROI theo $0.4d_{io}$, target
patch được resize và hard paste đủ bốn vùng lên bản copy của source. Ảnh gốc
không đổi. ROI vượt biên hoặc overlap bị loại. Một cặp thành công sinh đúng
một ảnh, một label và một lần tăng count; cặp thất bại không tiêu thụ target ID.

### Scenario 3: Lịch và integration

Với adaptation loop giả lập, xác nhận POST baseline chạy 256 epochs, STR chỉ
chạy sau các epoch đã liệt kê, refresh predictions mỗi round và khôi phục đúng
model mode sau inference. Sau mỗi round hoàn tất, source dataset và sampler
được refresh trước epoch tiếp theo; target loader không expose labels và
validation/test loaders không đổi. Với early stopping option ngoài baseline,
các rounds chưa đến lịch không được chạy bù.

Mô phỏng crash sau khi manifest đã commit nhưng trước khi checkpoint update.
Resume phải suy ra target đã dùng từ manifest và không tạo duplicate. Mô phỏng
crash giữa round sau khi minority count có thể đã đổi. Resume phải giữ
`selected_class` của transaction `in_progress`, restore model snapshot đầu
round, bỏ qua records đã commit và chỉ mark round `completed` sau khi quét hết
target. Manifest trỏ tới file thiếu hoặc hỏng phải làm resume fail rõ ràng.

## 9. Acceptance criteria

Task hoàn thành khi STR tạo được source training records có provenance từ
target train mà không đọc target ground truth, không sửa source gốc và không
thêm ảnh vào validation/test. Mỗi ảnh thành công có đủ bốn patch, giữ nhãn
source donor, và mỗi target ID được dùng tối đa một lần. Class selection,
confidence gate, lịch 10 rounds trong baseline 256 epochs và source set tích
lũy tuân theo các contract trên. Manifest STR quyết định các record đã commit;
resume giữa round giữ nguyên model snapshot và minority class. Source dataset
và sampler nhìn thấy ảnh mới từ epoch kế tiếp sau khi round hoàn tất. Ba test
scenarios xác nhận selection, patch geometry, transaction resume và
integration.
