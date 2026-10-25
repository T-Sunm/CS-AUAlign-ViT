# Task 2. Source Pretraining

## 1. Objective

Sử dụng labeled RAF-DB từ Task 1 để tạo source initialization cho POST. Fine-tune Swin-Transformer-Tiny pretrained trên ImageNet bằng source self-attention path, đồng thời học source prototype bank \(M_s\).

\[
x_s \rightarrow \text{Swin-Tiny} \rightarrow F_s \rightarrow M_s \rightarrow L_s.
\]

Output của task gồm:

\[
\theta_{\mathrm{Swin}}^{src}, \qquad M_s^{src}.
\]

Sau source pretraining, target prototype bank được khởi tạo bằng cách copy:

\[
M_t \leftarrow M_s^{src}.
\]

Checkpoint của task này được dùng làm source initialization cho Task 3.

## 2. Source Data and Backbone

### Source data

Sử dụng trực tiếp RAF-DB train loader của Task 1. Mỗi sample trả về:

\[
(x_s, y_s).
\]

Task 2 chỉ sử dụng labeled source data. ExpW và các target input không tham gia source pretraining.

### Backbone

Sử dụng Swin-Transformer-Tiny pretrained trên ImageNet, với input size mặc định \(224 \times 224\). USTST, tiền thân trực tiếp của POST, sử dụng cấu hình backbone này để xây Cross-Swin Transformer.

Trong Task 2 chỉ chạy source self-attention path:

\[
x_s \rightarrow F_s, \qquad F_s \in \mathbb{R}^{B \times m}.
\]

Chưa xây target path hoặc cross-attention. Backbone được cập nhật trong quá trình training.

## 3. Source Prototype Learning

Với \(K=7\) expression classes, tạo source prototype bank:

\[
M_s = \{\mu_1^s,\ldots,\mu_K^s\}, \qquad
M_s \in \mathbb{R}^{K \times m}.
\]

Mỗi prototype là một learnable parameter có cùng dimension \(m\) với feature representation. POST không chỉ rõ cách khởi tạo ban đầu của \(M_s\); implementation sử dụng:

\[
M_s \sim \operatorname{TruncNormal}(0, 0.02).
\]

Logit của class \(j\) là dot product giữa feature và prototype:

\[
z_s^{i,j} = \mu_j^s \cdot F_i^s.
\]

Xác suất dự đoán được tính bằng softmax:

\[
p_s^{i,j}
=
\frac{\exp(z_s^{i,j})}
{\sum_{r=1}^{K}\exp(z_s^{i,r})},
\qquad
h_s(F_i^s) = [p_s^{i,1},\ldots,p_s^{i,K}].
\]

Reproduction baseline không thêm feature hoặc prototype normalization vì POST không định nghĩa bước này.

## 4. Source Pretraining

Training flow:

\[
x_s \rightarrow F_s \rightarrow z_s \rightarrow L_s.
\]

Source-domain prototype learning loss là:

\[
L_s
=
\frac{1}{N_s}
\sum_i
L_{CE}(z_s^i, y_s^i).
\]

Trong Task 2:

\[
L_{\mathrm{pretrain}} = L_s.
\]

Gradient cập nhật đồng thời backbone và source prototype bank:

\[
\theta_{\mathrm{Swin}}, \qquad M_s.
\]

Cấu hình training mặc định:

- optimizer: SGD;
- learning rate: \(0.001\);
- momentum: \(0.9\);
- weight decay: \(0.001\);
- learning-rate schedule: cosine;
- input size: \(224 \times 224\).

Các giá trị trên lấy từ POST training recipe và phải configurable vì paper không báo cáo schedule riêng cho source pretraining. `source_pretrain_epochs` cũng phải configurable; không mặc định gán 256 epochs.

## 5. Target Prototype Initialization

Sau khi hoàn tất source pretraining, tạo target prototype bank \(M_t\) và copy giá trị từ \(M_s^{src}\):

\[
M_t \leftarrow M_s^{src}.
\]

Ngay sau initialization, hai bank bằng nhau về giá trị:

\[
M_t = M_s.
\]

\(M_t\) và \(M_s\) là hai learnable parameter banks độc lập, không chia sẻ parameter storage. POST quy định target prototypes được khởi tạo từ pretrained source prototypes.

Task 3 quyết định cách đưa source-pretrained Swin weights vào source path và target path, cũng như cách tổ chức parameter sharing trong BCS.

## 6. Required Deliverables

### Source-pretraining pipeline

Pipeline chạy end-to-end trên RAF-DB train của Task 1 và tạo:

\[
(\theta_{\mathrm{Swin}}^{src}, M_s^{src}).
\]

### Checkpoint

Checkpoint phải đủ thông tin để Task 3 restore trực tiếp:

- Swin-Tiny weights;
- source prototype bank;
- feature dimension \(m\);
- class mapping;
- backbone configuration;
- preprocessing configuration;
- training configuration.

### Target prototype initialization

Cung cấp cách tạo \(M_t\) từ \(M_s^{src}\) sao cho hai bank có cùng giá trị ban đầu nhưng là các parameters độc lập.

## 7. Acceptance Criteria

Task hoàn thành khi:

- RAF-DB image được forward thành \(F_s \in \mathbb{R}^{B \times m}\).
- Source prototype bank có shape \([7,m]\).
- Prototype prediction có shape \([B,7]\).
- \(L_s\) backward được qua cả backbone và \(M_s\).
- Training cập nhật cả backbone và prototypes.
- Source pretraining không phụ thuộc target data.
- Checkpoint restore được source backbone và \(M_s\).
- \(M_t\) được copy từ \(M_s\) nhưng không chia sẻ parameter storage.
- Output checkpoint có thể được Task 3 sử dụng trực tiếp.
