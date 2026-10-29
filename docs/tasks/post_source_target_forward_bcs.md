# Task 3. Source-target forward và BCS representation learning

## 1. Objective

Task 3 đưa source-pretrained Swin-T từ Task 2 vào adaptation model và tạo bốn
feature streams cho các task downstream:

```math
F_s,\quad F_t,\quad F_{t\rightarrow s},\quad F_{s\rightarrow t}
\in\mathbb{R}^{B\times768}.
```

$F_s,F_t$ là domain-specific features. $F_{t\rightarrow s}$ là source-anchored
feature được điều kiện hóa bởi target; $F_{s\rightarrow t}$ là target-anchored
feature được điều kiện hóa bởi source. POST định nghĩa chiều cross-attention
bằng $Q_t,K_s,V_s$ và $Q_s,K_t,V_t$ trong Equations 3-4. Vị trí chèn BCS,
parameter sharing và residual equations bên dưới là lựa chọn reproduction của
repo vì paper không xác định chúng ở mức implementation.

Task này chỉ tạo representations và forward interface. Prototype classifier,
losses, pseudo-labels, discriminator và STR được tích hợp ở các task khác.

## 2. Input và pairing

Forward nhận source images $x_s$ và target images $x_t$ đã qua cùng image
preprocessing của adaptation. Baseline dùng $224\times224$ và yêu cầu batch
sizes bằng nhau:

```math
x_s,x_t\in\mathbb{R}^{B\times3\times224\times224}.
```

BCS ghép ảnh theo batch position: source thứ $i$ đi với target thứ $i$. Hai
loaders được shuffle độc lập; không dùng source label hay target pseudo-label
để chọn pair. Mỗi paired forward yêu cầu hai batches có cùng $B$; model báo
lỗi nếu batch size hoặc input shape không đúng và không tự pad, truncate,
repeat hay cycle samples. Source label chỉ được adaptation loop giữ để tính
loss; target label không đi vào forward. Pairing theo vị trí là repo choice.

Task 3 không định nghĩa số steps mỗi epoch hay cách duyệt DataLoaders.
Adaptation training task quyết định epoch horizon, cách xử lý khi hai loaders
có độ dài khác nhau và batch cuối, rồi cấp các paired batches hợp lệ cho model.
Yêu cầu cùng $B$ chỉ áp dụng cho paired forward. Target-only path ở Section 6
nhận một target batch mà không cần source batch.

## 3. Backbone và checkpoint initialization

Với `torchvision` Swin-T của Task 2, shared trunk gồm patch embedding, Stages
1-3 và các patch-merging layers đến đầu Stage 4. Tại đó, token grids của hai
domain có shape `[B, 7, 7, 768]` với input $224\times224$. Stage 4 có hai
Swin blocks:

```text
shared trunk -> Stage4.Block0_s/t -> Stage4.Block1_BCS -> final norm -> pool
```

`Block0_s` và `Block0_t` là hai bản sao độc lập của pretrained Stage 4 Block 0.
Block 1 có source và target parameter sets độc lập, mỗi bộ được khởi tạo từ
pretrained Stage 4 Block 1. Cả hai domain ban đầu có cùng giá trị nhưng không
share parameter storage. Shared trunk dùng trực tiếp source-pretrained weights.
Final LayerNorm cũng được copy thành `Norm_s` và `Norm_t` độc lập. Global average
pooling không có learnable parameters và dùng chung quy tắc của source Swin.

Load pretrained parameters và attention buffers cần thiết từ
`backbone_state_dict` trong source checkpoint của Task 2. Fail rõ khi thiếu
key, shape không khớp hoặc pretrained state không được gán vào shared trunk,
cả hai Stage 4 copies hay hai final norms. Buffer
`relative_position_index` của mỗi attention block phải được restore hoặc
reconstruct đúng theo pretrained Swin, rồi kiểm tra giá trị và indexing.
BCS Block 1 reuse pretrained QKV, output projection, relative-position bias,
norms và MLP của từng branch; baseline không thêm trainable cross-only
projections hoặc FFN. Task 2.2 quản lý prototype banks riêng.

## 4. BCS Block 1

Gọi input sau native Block 0 là $X_s,X_t\in\mathbb{R}^{B\times7\times7\times768}$.
Block 1 áp dụng pre-norm và tạo QKV đúng một lần cho mỗi domain:

```math
\bar X_s=LN_{1,s}(X_s),\quad (Q_s,K_s,V_s)=QKV_s(\bar X_s),
```

```math
\bar X_t=LN_{1,t}(X_t),\quad (Q_t,K_t,V_t)=QKV_t(\bar X_t).
```

Từng tensor có shape `[B, 24, 49, 32]`: Stage 4 của pretrained Swin-T dùng
`num_heads=24`, `head_dim=32`, tức $h d_h=24\times32=768$. Block 0 có QKV riêng
của chính nó; quy tắc reuse này chỉ áp dụng trong Block 1. Native và cross
attention dùng cùng QKV tensors của Block 1:

```math
A_s=Attn_s(Q_s,K_s,V_s),\quad A_t=Attn_t(Q_t,K_t,V_t),
```

```math
A_{t\rightarrow s}=Attn_s(Q_t,K_s,V_s),\quad
A_{s\rightarrow t}=Attn_t(Q_s,K_t,V_t).
```

Native và cross attention là hai phép tính cùng lấy input từ pre-norm Block 1
QKV. Cross attention không nhận $U_s$, $U_t$ hoặc output native self-attention
làm input.

Ở đây `Attn` kết thúc sau phép nhân với $V$:

```math
Attn_j(Q,K,V)
=Dropout_{attn,j}\!\left(
\operatorname{softmax}\!\left(
\frac{QK^T}{\sqrt{d_h}}+RPB_j
\right)\right)V.
```

`Attn` chưa gồm output projection hoặc output dropout. Native path dùng bias
table của chính domain. Cross path dùng bias table của destination branch:
$RPB_s$ cho $t\rightarrow s$ và $RPB_t$ cho $s\rightarrow t$. Source và target grids phải
có cùng kích thước và thứ tự token để cùng relative-position index có nghĩa.
Đây là repo choice; POST không xác định cross-domain bias table.

Với grid $7\times7$ và window $7\times7$, Stage 4 có một window gồm 49 tokens.
Logic của `torchvision` đưa shift về 0 ở kích thước này. Sau `Attn`, mỗi stream
đi qua output projection, output dropout rồi stochastic depth trước khi cộng
residual. Attention dropout và output dropout dùng probabilities trong config
của destination block; chúng không có learned parameters. Native và cross
calls không cần dùng cùng random dropout mask.

## 5. Residual streams và feature aggregation

Đặt $DP_j$ là stochastic depth của destination Block 1 và
$Proj_j(A)=Dropout_{out,j}(Linear_{out,j}(A))$. `Proj` chỉ thực hiện output
projection và output dropout đúng một lần; `Attn` đã kết thúc trước bước này.
Native và cross streams dùng `Proj` của destination branch:

```math
\begin{aligned}
U_s &= X_s+DP_s(Proj_s(A_s)), &
U_{t\rightarrow s} &= X_s+DP_s(Proj_s(A_{t\rightarrow s})),\\
U_t &= X_t+DP_t(Proj_t(A_t)), &
U_{s\rightarrow t} &= X_t+DP_t(Proj_t(A_{s\rightarrow t})).
\end{aligned}
```

Mỗi lần gọi $DP_j$ cho attention hoặc MLP residual tạo một stochastic-depth
draw riêng khi training và drop probability lớn hơn 0, kể cả khi native và
cross streams dùng cùng destination block. Không đồng bộ DropPath mask giữa
các streams; hai draws vẫn có thể tình cờ cho cùng mask.

Mỗi stream tiếp tục qua pretrained-compatible $LN_2$ và MLP của destination
branch, với residual và stochastic depth như Swin:

```math
\begin{aligned}
Y_s &= U_s+DP_s(MLP_s(LN_{2,s}(U_s))),\\
Y_{t\rightarrow s} &= U_{t\rightarrow s}
  +DP_s(MLP_s(LN_{2,s}(U_{t\rightarrow s}))),\\
Y_t &= U_t+DP_t(MLP_t(LN_{2,t}(U_t))),\\
Y_{s\rightarrow t} &= U_{s\rightarrow t}
  +DP_t(MLP_t(LN_{2,t}(U_{s\rightarrow t}))).
\end{aligned}
```

`Proj_s`, `LN2_s` và `MLP_s` được reuse giữa hai source-anchored streams;
target-anchored streams tương tự. Source residual giúp $F_{t\rightarrow s}$
giữ source anchor, nhưng không bảo đảm target query không làm thay đổi expression
semantics. Eq. 8 dùng source label cho stream này theo POST; pairing và residual
ở đây là giả định của baseline cần kiểm tra trong adaptation.

Final LayerNorm và global average pooling áp dụng theo destination domain:

```math
\begin{aligned}
F_s &= Pool(Norm_s(Y_s)), &
F_{t\rightarrow s} &= Pool(Norm_s(Y_{t\rightarrow s})),\\
F_t &= Pool(Norm_t(Y_t)), &
F_{s\rightarrow t} &= Pool(Norm_t(Y_{s\rightarrow t})).
\end{aligned}
```

Pooling lấy trung bình trên 49 spatial tokens, cho mỗi feature shape `[B, 768]`.
Forward fail rõ nếu feature dimension không khớp `feature_dim` của adaptation
model và prototype banks từ Task 2.2.

## 6. Output và downstream boundary

`forward_features(source_images, target_images)` trả structured result với
bốn tên cố định:

- `source_features` = $F_s$;
- `target_features` = $F_t$;
- `target_to_source_features` = $F_{t\rightarrow s}$;
- `source_to_target_features` = $F_{s\rightarrow t}$.

Các outputs giữ computation graph. Cross stream giữ đường gradient tới cả
source-side và target-side computation; không yêu cầu mọi parameter của hai
branches đều có gradient khác 0. Native source và target streams chỉ phụ
thuộc input của chính domain, ngoài shared parameters. Task 3 cũng cung cấp
target-only native feature path $x_t\rightarrow F_t$ cho STR inference và
target evaluation; caller này không cần một source batch hoặc chạy
cross-attention.

Task 4 dùng Equations 8-9 làm canonical loss mapping:

| Prototype bank | Features dùng trong cross-domain loss |
| --- | --- |
| $M_s$ | $F_{t\rightarrow s}$ với source label; $F_t$ với target pseudo-label |
| $M_t$ | $F_{s\rightarrow t}$ với target pseudo-label; $F_s$ với source label |

Native classification vẫn dùng $F_s\rightarrow M_s$ và $F_t\rightarrow M_t$.
Figure 1(b) trình bày ý tưởng hai classifiers và cross-domain features nhưng
không được dùng để thay mapping trong Equations 8-9. Task 3 không tính logits,
probabilities hay losses.

## 7. Test scenarios

### Scenario 1: Checkpoint mapping và native parity

Load checkpoint Task 2 vào adaptation model. Trước optimizer step, đưa cùng
input tensor qua source-pretrained backbone và native paths của hai branches.
Trong `eval()` mode, cùng dtype/device và không AMP, so token tensors sau
Stage 4 Block 1, trước final norm/pooling, với output Block 1 của pretrained
Swin. Cả $Y_s$ và $Y_t$ phải khớp trong numerical tolerance; kiểm tra cả
`relative_position_index` và cách nó index learned bias table. Sau đó so
$F_s^{adapt}(x)$ và $F_t^{adapt}(x)$ với $F_s^{pretrained}(x)$; cả hai phải
khớp. Kiểm tra shared trunk, Stage 4 copies và final norms được gán đúng
weights; hai Stage 4 branches và final norms có cùng initial values nhưng
không shared storage. Optimizer thấy toàn bộ trainable BCS parameters. Source
checkpoint thiếu hoặc sai shape phải fail rõ.

### Scenario 2: Attention direction và feature dependencies

Với source/target tensors biết trước, xác nhận bốn outputs có shape `[B, 768]`
và attention-level test bắt được việc swap $Q_t,K_s,V_s$ với $Q_s,K_t,V_t$.
Trước routing test, đặt deterministic distinct values cho `RPB_s/RPB_t`,
`Proj_s/Proj_t` và `Norm_s/Norm_t`; không dùng hai copies còn giống nhau ngay
sau checkpoint initialization. Trong `eval()` mode, kiểm tra từng cross path
dùng đúng destination residual, bias, output projection, final norm và
pooling. Test riêng thứ tự toán tử bằng hooks hoặc operators quan sát được:
`Attn -> Linear_out -> Dropout_out -> DropPath -> residual`. Routing test
trong `eval()` không chứng minh được dropout order. Đổi target input phải giữ
$F_s$ ở `eval()` mode nhưng thay đổi cross features; đổi source input phải
giữ $F_t$ nhưng thay đổi cross features. Với input không suy biến, dummy
objective trên từng cross feature phải cho gradient khác 0 ở cả $X_s$ và
$X_t$; kiểm tra thêm một số parameter groups đúng routing, không yêu cầu mọi
parameter tensor có gradient khác 0. Native và cross calls không cần dùng
cùng dropout hay DropPath mask trong training mode. Không assert hai mask phải
khác nhau, vì các draws có thể tình cờ trùng.

### Scenario 3: Forward boundary

Forward từ hai batches cùng size trả đúng bốn tên, giữ computation graph và
không cần prototype banks, expression labels, pseudo-labels hay discriminator.
Unequal batch sizes hoặc token grids không tương thích phải fail rõ; model
không sửa batch để ép pairing. Target-only path trả cùng $F_t$ như paired
forward trong `eval()` mode với cùng target input; nó hoạt động khi không có
source input. Task 4 có thể nhận trực tiếp bốn feature tensors mà không phụ
thuộc BCS internals.

## 8. Acceptance criteria

Task hoàn thành khi source checkpoint tạo được shared trunk, hai Stage 4
branches, attention buffers và final norms đúng contract; token-level và
final-feature native parity tests pass cho cả source và target trước
adaptation. BCS Block 1 dùng 24 heads và tạo bốn `[B, 768]` feature streams
theo đúng QKV, residual và destination components đã khóa. Routing test phân
biệt được hai branch sau khi đặt giá trị khác nhau; operator-order test kiểm
tra riêng đường dropout. Forward API hỗ trợ target-only inference và giữ
gradient cho downstream losses. Adaptation training task chịu trách nhiệm
paired steps mỗi epoch, xử lý loaders và batch cuối, optimizer update, losses
và save/resume end-to-end.
