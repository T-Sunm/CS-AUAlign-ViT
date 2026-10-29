# Task 4.3. Adversarial domain alignment

## 1. Objective

BCS cung cấp hai domain-specific features $F_s$ và $F_t$. Domain discriminator
$D$ học phân biệt source với target; feature extractor nhận gradient theo
hướng làm hai domain khó phân biệt hơn. POST mô tả hai mục tiêu đối nghịch và
định nghĩa adversarial loss ở Equation 10:

```math
L_{ad}
=
\frac{1}{N_s}\sum_i\log\frac{1}{D(F_i^s)}
+
\frac{1}{N_t}\sum_i\log\frac{1}{1-D(F_i^t)}.
```

Task cung cấp discriminator, gradient reversal và $L_{ad}$ chưa nhân hệ số cho
adaptation pipeline. GRL và kiến trúc discriminator dưới đây là lựa chọn của
repo; POST không chỉ rõ cách triển khai hai phần này.

## 2. Input và output

Module nhận:

```math
F_s\in\mathbb{R}^{B_s\times m},\qquad
F_t\in\mathbb{R}^{B_t\times m}.
```

$B_s$ và $B_t$ có thể khác nhau. Chỉ dùng domain-specific features $F_s,F_t$;
không đưa $F_{t\rightarrow s}$ hoặc $F_{s\rightarrow t}$ vào discriminator.
Module không cần expression labels, target pseudo-labels hay prototype banks.

Discriminator trả một raw logit cho mỗi feature, shape `[B]` hoặc `[B, 1]`.
Sigmoid chỉ dùng khi cần probability hoặc domain accuracy. Quy ước domain
label là source $=1$, target $=0$.

## 3. Discriminator và GRL baseline

Với run baseline có $m=768$, discriminator dùng:

```text
Linear(768, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 1)
```

Lớp cuối trả raw logits, không có sigmoid. `input_dim`, `hidden_dim` và
`dropout` có thể nằm trong config; giá trị baseline lần lượt là `768`, `256`
và `0.5`. Adaptation checkpoint cần lưu cấu hình để dựng lại discriminator.

GRL dùng một `torch.autograd.Function` nhỏ. Forward giữ nguyên feature;
backward đổi dấu gradient:

```math
GRL(F)=F,\qquad
\frac{\partial GRL}{\partial F}=-\lambda_{grl}I.
```

Baseline cố định $\lambda_{grl}=1.0$, không dùng schedule hoặc warm-start.
Pipeline của mỗi domain là:

```math
F\rightarrow GRL(1.0)\rightarrow D\rightarrow L_{ad}.
```

Implementation chỉ cần `torch.nn`, `BCEWithLogitsLoss` và
`torch.autograd.Function`; không thêm framework domain adaptation.

## 4. Loss và gradient contract

Tính BCE mean riêng cho từng domain rồi cộng lại, kể cả khi $B_s\ne B_t$:

```math
L_{ad}
=\operatorname{BCEWithLogits}(z_s^D,\mathbf{1})
+\operatorname{BCEWithLogits}(z_t^D,\mathbf{0}).
```

Đây là dạng ổn định số của Equation 10. Không áp dụng sigmoid trước
`BCEWithLogitsLoss`, và không lấy mean trên batch ghép source với target.
Adversarial module trả $L_{ad}$ chưa nhân $\gamma$.

Adaptation loop tổng hợp loss theo Equation 11:

```math
L_{cross}
=\alpha L_{t\rightarrow s}
+\beta L_{s\rightarrow t}
+\gamma L_{ad}.
```

Baseline dùng $\alpha=1$, $\beta=0.5$, $\gamma=0.1$. Các hệ số này thuộc
adaptation configuration; $\gamma$ chỉ nhân một lần ở total loss và không
được đưa vào GRL. Với $\lambda_{grl}=1$, một backward cho nhánh adversarial
tạo gradient $+\gamma\nabla_{\theta_D}L_{ad}$ lên discriminator và đóng góp
$-\gamma\nabla_F L_{ad}$ lên feature extractor. Gradient tổng lên feature
extractor còn bao gồm các loss khác của adaptation.

## 5. Integration và checkpoint

Adversarial module nhận $F_s,F_t$ từ BCS forward và trả $L_{ad}$ để adaptation
loop cộng vào $L_{cross}$. Discriminator là learnable module có state trong
adaptation model. Adaptation training task chịu trách nhiệm tạo optimizer, lưu
discriminator state và config, rồi kiểm thử end-to-end save/resume.

Task này không implement prototype banks, prototype losses, pseudo-label
generation, k-means, STR, BCS forward hay phép tổng hợp $L_{cross}$.

## 6. Monitoring

Khi tích hợp training, log `loss_ad`, source domain accuracy $Acc_s$, target
domain accuracy $Acc_t$ và balanced domain accuracy:

```math
Acc_{balanced}=\frac{Acc_s+Acc_t}{2}.
```

Với raw domain logits, threshold $0$ tương ứng probability $0.5$. Balanced
accuracy cho hai domain trọng số bằng nhau khi $B_s\ne B_t$. Các metrics này
chỉ dùng để theo dõi training, không tham gia loss hoặc chọn checkpoint target
FER. Accuracy gần $50\%$ chỉ là tín hiệu cần kiểm tra thêm; tự nó không chứng
minh feature extractor đã căn chỉnh domain.

## 7. Test scenarios

### Scenario 1: Forward và loss

Với $B_s\ne B_t$, xác nhận discriminator trả một raw logit cho mỗi feature.
Kiểm tra source label $1$, target label $0$, và $L_{ad}$ bằng tổng hai BCE mean
riêng theo domain. Loss phải finite và chưa nhân $\gamma$. Kiểm tra module chỉ
cần $F_s,F_t$ để chạy.

### Scenario 2: Gradient direction

Dùng cùng discriminator và cùng feature để so sánh gradient khi có và không có
GRL. Tắt dropout trong phép so sánh để hai forward dùng cùng hàm số. Với
$\lambda_{grl}=1$, gradient lên feature qua GRL phải đảo dấu; gradient lên
discriminator giữ nguyên hướng. Dùng feature đã detach để kiểm tra riêng
discriminator có thể nhận gradient và được optimizer cập nhật.

### Scenario 3: Integration boundary

Xác nhận module hoạt động khi không có prototype banks, expression labels,
pseudo-labels hoặc k-means. API trả $L_{ad}$ chưa nhân hệ số để adaptation loop
có thể áp dụng $\gamma=0.1$ đúng một lần. Kiểm thử checkpoint đầy đủ thuộc
adaptation training task.

## 8. Acceptance criteria

Task hoàn thành khi:

1. Discriminator baseline dùng kiến trúc `768 -> 256 -> 1`, ReLU, dropout
   `0.5` và trả raw logits; architecture config có thể lưu cùng checkpoint.
2. $L_{ad}$ là tổng BCE mean của source label $1$ và target label $0$, chấp
   nhận $B_s\ne B_t$ và chưa nhân $\gamma$.
3. GRL cố định $\lambda_{grl}=1.0$ đảo gradient về feature và giữ gradient
   thông thường cho discriminator.
4. Module chỉ phụ thuộc $F_s,F_t$ và có thể được adaptation loop dùng để tính
   $L_{cross}$ với $\gamma$ nhân một lần.
5. Ba test scenarios xác nhận loss, gradient direction và integration boundary.
