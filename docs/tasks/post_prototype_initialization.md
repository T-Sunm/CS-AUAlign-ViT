# Task 2.2. Prototype Initialization for Adaptation

## 1. Objective

Đưa source prototype bank đã học ở Task 2 vào adaptation model và tạo target
prototype bank độc lập. Hai banks có dạng:

```math
M_s=\{\mu_1^s,\mu_2^s,\ldots,\mu_K^s\},
\qquad
M_t=\{\mu_1^t,\mu_2^t,\ldots,\mu_K^t\},
```

trong đó mỗi prototype có cùng dimension $m$ với final feature $F$. Khi bắt
đầu một adaptation run mới:

```math
M_t^{(0)} \leftarrow M_s^{(0)}.
```

Sau khi copy, $M_s$ và $M_t$ là hai learnable parameters độc lập, không dùng
chung storage.

### Đã có từ Task 2

- `SourcePrototypeClassifier.source_prototypes` được học bằng source CE loss;
- source checkpoint lưu `source_prototypes`, `feature_dim` và `class_mapping`;
- `copy_target_prototypes()` tạo một parameter mới có cùng giá trị với $M_s$.

### Cần làm trong Task 2.2

- tích hợp hai banks vào adaptation model và cung cấp logits cho từng bank;
- kiểm tra source checkpoint trước khi dùng cho adaptation;
- xác định đường khởi tạo mới và đường resume, để adaptation checkpoint có thể
  giữ nguyên hai banks đã học.

## 2. Input contract

Task nhận source-pretraining checkpoint và checkpoint phải chứa:

- source prototype state $M_s^{src}$;
- số expression classes $K$;
- final feature dimension $m$;
- source class mapping theo đúng thứ tự row của $M_s^{src}$.

Class index $k$ có cùng ý nghĩa ở source label, prototype row $k$ và target
pseudo-label của Task 4. Target data pipeline dùng cùng canonical class mapping
trong closed-set adaptation.

Feature đi vào prototype classifier có shape:

```math
F\in\mathbb{R}^{B\times m}.
```

Hai prototype banks có shape:

```math
M_s,M_t\in\mathbb{R}^{K\times m}.
```

Loader phải dừng với thông báo rõ ràng nếu checkpoint thiếu metadata hoặc nếu
$K$, $m$, class mapping hay prototype shape không khớp với configuration hiện
tại. Không tự reorder, truncate hoặc bổ sung prototype rows.

`source_prototypes` được lấy trực tiếp từ source checkpoint của Task 2. POST
yêu cầu khởi tạo target prototypes từ pretrained source prototypes.

## 3. Prototype banks

Mỗi row của $M_s$ và $M_t$ tương ứng với một expression class trong class
mapping của source checkpoint. Khi đưa vào adaptation model:

- $M_s$ có shape `[K, m]`;
- dtype khớp với model parameters;
- $M_s$ nằm cùng device với feature $F$;
- `requires_grad=True`;
- $M_s$ được register là learnable parameter.

Target prototype bank được tạo bằng exact value copy từ $M_s$:

```math
M_t^{(0)}=M_s^{(0)}.
```

Hai banks bằng nhau về giá trị ngay sau initialization nhưng khác parameter
identity và underlying storage. Việc cập nhật một bank không tự thay đổi bank
còn lại. Cả hai được register trong adaptation model để optimizer nhận được
parameters của chúng.

Paper mô tả các prototypes tiếp tục được học bằng backpropagation trong
adaptation, vì vậy task này không freeze $M_s$ hoặc $M_t$.

## 4. Prototype classifier contract

Với feature $F_i\in\mathbb{R}^{m}$, logits từ source và target prototype banks
được tính bằng dot product:

```math
z_{s,i,j}=\mu_j^s\cdot F_i,
\qquad
z_{t,i,j}=\mu_j^t\cdot F_i.
```

Dạng batch tương đương là:

```math
Z_s=FM_s^T,
\qquad
Z_t=FM_t^T,
```

với $Z_s,Z_t\in\mathbb{R}^{B\times K}$. Probability vectors dùng ký hiệu của
paper:

```math
h_s(F_i)=\operatorname{softmax}(z_{s,i}),
\qquad
h_t(F_i)=\operatorname{softmax}(z_{t,i}).
```

Prototype module cung cấp logits thông qua hai operation tương đương:

- `source_logits(F)` trả $Z_s$;
- `target_logits(F)` trả $Z_t$.

Trong paper, $h_s$ và $h_t$ là probability vectors. Implementation truyền raw
logits $Z_s,Z_t$ vào `CrossEntropyLoss`; truyền kết quả softmax vào loss này sẽ
áp dụng softmax hai lần. Chỉ tính probability tại nơi cần confidence hoặc
prediction. Baseline reproduction không thêm cosine normalization, L2
normalization hoặc temperature scaling vì POST không định nghĩa các thành phần
này.

## 5. Fresh initialization và resume

Khi bắt đầu adaptation run mới, load $M_s$ từ source checkpoint và tạo $M_t$
bằng `copy_target_prototypes(M_s)`:

```math
M_s\leftarrow M_s^{src},\qquad M_t\leftarrow\operatorname{copy}(M_s).
```

Khi resume adaptation, restore hai banks đã học từ adaptation checkpoint:

```math
M_s\leftarrow M_s^{ckpt},\qquad M_t\leftarrow M_t^{ckpt}.
```

Không copy lại $M_s$ sang $M_t$ sau khi restore. Adaptation checkpoint cần lưu
hai banks dưới các state keys riêng cùng metadata về class mapping, $K$ và $m$.
Adaptation training task chịu trách nhiệm lưu và restore optimizer state của cả
hai banks; test round-trip đầy đủ thuộc test tích hợp của task đó.

## 6. Task boundary

Task hoàn thành khi cung cấp được $M_s$, $M_t$, `source_logits` và
`target_logits` cho Task 4. Task này không implement:

- $L_s$, $L_t$, $L_{t\rightarrow s}$ hoặc $L_{s\rightarrow t}$;
- target pseudo-label hoặc trusted target selection;
- k-means;
- STR;
- BCS/cross-attention forward và domain discriminator.

Dependency của task:

```math
\text{Source Pretraining}
\rightarrow
\boxed{\text{Prototype Initialization}}
\rightarrow
\begin{cases}
L_s,L_t\\
L_{t\rightarrow s}\\
L_{s\rightarrow t}\\
\text{target prediction}
\end{cases}
```

## 7. Test scenarios

### Scenario 1: Fresh initialization

Load source-pretraining checkpoint và xác nhận:

- $M_s$ và $M_t$ có shape `[K, m]`;
- $M_t$ bằng $M_s$ về giá trị ngay sau initialization;
- hai banks có parameter identity và underlying storage khác nhau;
- class mapping của hai banks giữ đúng thứ tự từ source checkpoint;
- loader từ chối checkpoint có $K$, $m$, shape hoặc class mapping không khớp.

### Scenario 2: Classification and gradient contract

Với dummy feature `[B, m]` cố định, không tham gia computation graph, xác nhận:

- source và target logits có shape `[B, K]`;
- logits bằng kết quả của $FM^T$;
- optimizer chứa parameters của cả hai banks;
- source-only loss truyền gradient vào $M_s$ nhưng không truyền vào $M_t$;
- target-only loss truyền gradient vào $M_t$ nhưng không truyền vào $M_s$.

### Scenario 3: Resume contract (kiểm thử khi có adaptation training loop)

Sau khi cập nhật để $M_s$ và $M_t$ có giá trị khác nhau, save rồi resume và xác
nhận:

- từng bank được restore đúng giá trị;
- hai banks vẫn là parameters độc lập;
- optimizer state của cả hai banks được restore bởi adaptation training loop;
- resume không copy lại $M_s$ sang $M_t$.

## 8. Acceptance criteria

Task hoàn thành khi:

1. Task 2 checkpoint và `copy_target_prototypes()` được dùng lại; source
   checkpoint được validate trước khi khởi tạo adaptation model.
2. $M_s$ và $M_t$ được register độc lập; fresh initialization cho hai banks
   cùng giá trị và đúng class mapping.
3. Classifier trả raw logits `[B, K]` từ feature `[B, m]` bằng dot product;
   gradient đi vào đúng bank được sử dụng.
4. Baseline không tự thêm normalization hoặc temperature ngoài mô tả của paper.
5. API phục vụ Task 4 và nêu rõ dữ liệu cần restore cho resume. Kiểm thử
   optimizer checkpoint và resume end-to-end được thực hiện cùng adaptation
   training task.
