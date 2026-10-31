# Task 4.2. Prototype learning: $\mathcal{L}_s$, $\mathcal{L}_t$, $\mathcal{L}_{t\rightarrow s}$, $\mathcal{L}_{s\rightarrow t}$

## 0. Đọc nhanh

Mỗi lớp cảm xúc có một vector đại diện gọi là prototype. Source có bộ $M_s$, target có bộ $M_t$, mỗi bộ có shape `[7, 768]`. Một prototype đóng vai trò như một hàng của lớp phân loại: nhân feature với cả bảy hàng thì ra bảy điểm số, softmax rồi cross entropy với nhãn thì ra loss.

Task 4.2 chỉ làm đúng việc đó cho bốn feature của Task 3. Nó nhận tensor và trả loss, không giữ trạng thái, không chạy k-means, không chạy optimizer và không nhân trọng số.

Bốn loss, mỗi loss gồm một hoặc hai term cross entropy:

| Loss | Term | Feature | Prototype | Nhãn | Có mask không |
| --- | --- | --- | --- | --- | --- |
| $\mathcal{L}_s$ | 1 | $F_s$ | $M_s$ | $y_s$ | Không |
| $\mathcal{L}_t$ | 1 | $F_t$ | $M_t$ | $\hat y_t$ | Có |
| $\mathcal{L}_{t\rightarrow s}$ | 1 | $F_{t\rightarrow s}$ | $M_s$ | $y_s$ | Không |
| $\mathcal{L}_{t\rightarrow s}$ | 2 | $F_t$ | $M_s$ | $\hat y_t$ | Có |
| $\mathcal{L}_{s\rightarrow t}$ | 1 | $F_{s\rightarrow t}$ | $M_t$ | $\hat y_t$ | Có |
| $\mathcal{L}_{s\rightarrow t}$ | 2 | $F_s$ | $M_t$ | $y_s$ | Không |

Trong đó $y_s$ là nhãn thật của ảnh nguồn, $\hat y_t$ là pseudo label từ Task 4.1, và "có mask" nghĩa là chỉ ảnh thuộc tập tin cậy của 4.1 mới đóng góp.

Ba điều dễ hiểu nhầm:

* Term thứ hai của hai loss chéo dùng feature native ($F_t$ hoặc $F_s$), không dùng feature cross. Ví dụ $\mathcal{L}_{t\rightarrow s}$ có một term đưa $F_t$ vào prototype của source $M_s$ với pseudo label.
* Nhãn đi theo nhánh giữ K, V của Task 3, nên $F_{t\rightarrow s}$ dùng nhãn nguồn và $F_{s\rightarrow t}$ dùng pseudo label, dù query của chúng đến từ miền ngược lại.
* Mỗi term được trung bình riêng. Hai term của cùng một loss không được gộp lại rồi chia chung một số.

## 1. Mục tiêu

Task 4.2 tính bốn loss dựa trên prototype từ Eq. 5 đến Eq. 9 của POST, và trả chúng dưới dạng riêng biệt. Việc nhân $\alpha, \beta, \gamma$, cộng với $\mathcal{L}_{ad}$ và gọi `backward()` thuộc task khác.

POST xác định bốn điều: dạng softmax trên tích vô hướng, cross entropy trung bình, cặp (feature, prototype, nhãn) của từng term, và việc chỉ dùng pseudo label cho mẫu tin cậy. POST không xác định cách chuẩn hóa theo mini-batch, việc có chặn gradient hay không, temperature, hay xử lý tập tin cậy rỗng. Các chi tiết đó là lựa chọn của repo, được liệt kê ở Section 6.

Task này không tạo pseudo label (4.1), không chọn ảnh cho STR (Spec 5), không tính $\mathcal{L}_{ad}$ và không lên lịch huấn luyện (Spec 6).

## 2. Đầu vào và đầu ra

Hàm tính loss nhận:

* Bốn feature từ Task 3, mỗi feature có shape `[B, 768]`: $F_s$, $F_t$, $F_{t\rightarrow s}$, $F_{s\rightarrow t}$.
* Hai ma trận prototype từ Task 2.2: $M_s$ và $M_t$, mỗi ma trận có shape `[7, 768]`. Hàm nhận chính tham số `nn.Parameter` để gradient chảy về, không nhận bản sao.
* `source_labels`: shape `[B]`, kiểu `long`, giá trị từ 0 đến 6.
* `pseudo_labels`: shape `[B]`, kiểu `long`, lấy từ 4.1 cho đúng các ảnh đích trong batch.
* `trusted_mask`: shape `[B]`, kiểu `bool`, cũng từ 4.1.

Chỉ số $i$ trong batch là một cặp (ảnh nguồn $i$, ảnh đích $i$). Nhãn nguồn $i$, pseudo label $i$ và mask $i$ đều đi theo cặp này.

Đầu ra là một kết quả có cấu trúc:

| Trường | Shape | Ghi chú |
| --- | --- | --- |
| `l_s`, `l_t`, `l_t2s`, `l_s2t` | Vô hướng | Bốn loss, có gradient |
| `terms` | Sáu vô hướng | Chính các tensor có đồ thị mà bốn loss được cộng từ đó, chưa `detach`. Caller tự `detach()` khi log. Tên: `s`, `t`, `t2s_cross`, `t2s_native`, `s2t_cross`, `s2t_native` |
| `num_trusted` | Số nguyên | Số ảnh có `trusted_mask` bằng true trong batch |

Loss tổng của hai loss chéo bằng tổng hai term tương ứng. $\mathcal{L}_{in}$ và $\mathcal{L}_{cross}$ không do task này lắp ráp.

## 3. Hợp đồng chi tiết

### 3.1 Logit và cross entropy

Với một feature $F^i \in \mathbb{R}^{768}$ và prototype $M_j = \{\mu_j^1, \dots, \mu_j^7\}$, xác suất theo POST là:

```math
p_j^{i,c} = \frac{\exp(\mu_j^c \cdot F^i)}{\sum_{r=1}^{7} \exp(\mu_j^r \cdot F^i)}
```

* Logit là tích vô hướng thô `features @ prototypes.T`, shape `[B, 7]`. Không chuẩn hóa L2, không temperature, không scale.
* Cross entropy tính từ log-softmax của logit, dùng một hàm ổn định số (kiểu `log_softmax` hoặc `F.cross_entropy` với `reduction='none'`). Không tự tính softmax rồi lấy log.
* Phép tính chạy trong `float32` kể cả khi mô hình chạy mixed precision.
* Cách tính này giống logit $z = F_t M_t^\top$ mà Task 4.1 dùng để lấy độ tin cậy, nên hai task nhìn cùng một không gian điểm số.

### 3.2 Term có mask và term không mask

Gọi $\ell_i$ là cross entropy của mẫu $i$ cho một term. Term không mask là trung bình trên cả batch:

```math
\frac{1}{B}\sum_{i=1}^{B} \ell_i
```

Term có mask chỉ lấy các mẫu tin cậy và chia cho số mẫu tin cậy trong batch:

```math
\frac{1}{B^*}\sum_{i=1}^{B} m_i\,\ell_i,\qquad B^* = \sum_{i=1}^{B} m_i
```

Trong đó $m_i \in \{0, 1\}$ là `trusted_mask`. Đây là phiên bản mini-batch của $\frac{1}{N_t^*}\sum_{i=1}^{N_t^*}$ trong POST.

Quy tắc cho tập tin cậy rỗng ($B^* = 0$):

* Term có mask bằng 0 và vẫn nằm trong đồ thị tính toán, nên `backward()` chạy được và cho gradient bằng 0 cho term đó. Không trả NaN, không raise lỗi.
* Cách làm là nhân loss từng mẫu với mask rồi chia cho `max(B*, 1)`. Không dùng chỉ số boolean để cắt tensor nếu việc đó làm đứt đồ thị khi batch rỗng.
* Loss của mẫu không tin cậy không được ảnh hưởng đến giá trị hay gradient, kể cả khi pseudo label của mẫu đó nằm ngoài khoảng hợp lệ. Vì `F.cross_entropy` báo lỗi ngay khi gặp nhãn ngoài khoảng, trước bước nhân mask, quy trình bắt buộc là: (1) kiểm tra phạm vi nhãn chỉ ở vị trí có mask bằng true (xem 3.5), (2) thay nhãn ở vị trí có mask bằng false bằng chỉ số hợp lệ tạm thời, ví dụ 0, (3) tính cross entropy từng mẫu, (4) nhân mask rồi chia cho `max(B*, 1)`. Nhãn tạm chỉ tồn tại bên trong hàm, không ghi ngược vào tensor của caller.

### 3.3 Bốn loss

Theo bảng ở Section 0:

```math
\mathcal{L}_s = \operatorname{mean}_{B}\,CE(h_s(F_s^i), y_s^i)
```

```math
\mathcal{L}_t = \operatorname{mean}_{B^*}\,CE(h_t(F_t^i), \hat y_t^i)
```

```math
\mathcal{L}_{t\rightarrow s}
= \underbrace{\operatorname{mean}_{B}\,CE(h_s(F_{t\rightarrow s}^i), y_s^i)}_{\text{t2s\_cross}}
+ \underbrace{\operatorname{mean}_{B^*}\,CE(h_s(F_t^i), \hat y_t^i)}_{\text{t2s\_native}}
```

```math
\mathcal{L}_{s\rightarrow t}
= \underbrace{\operatorname{mean}_{B^*}\,CE(h_t(F_{s\rightarrow t}^i), \hat y_t^i)}_{\text{s2t\_cross}}
+ \underbrace{\operatorname{mean}_{B}\,CE(h_t(F_s^i), y_s^i)}_{\text{s2t\_native}}
```

Ở đây $\operatorname{mean}_{B}$ và $\operatorname{mean}_{B^*}$ là hai kiểu trung bình ở 3.2. Bốn loss luôn được trả riêng. Không cộng $\mathcal{L}_s + \mathcal{L}_t$ thành $\mathcal{L}_{in}$ và không nhân trọng số bên trong task này.

### 3.4 Gradient

* Mặc định không có `detach` ở bất kỳ term nào. Gradient chảy về cả bốn feature, về encoder qua các feature đó, và về cả hai prototype.
* Hệ quả cần biết: $M_s$ nhận gradient từ term có $F_t$ và pseudo label, còn $M_t$ nhận gradient từ term có $F_s$ và nhãn nguồn. Đây là cách đọc của repo cho Eq. 8 và Eq. 9. POST không nói prototype có bị chặn gradient ở các term chéo hay không.
* Task này không đổi `requires_grad` của bất kỳ tham số nào và không gọi `backward()`.
* Việc đóng băng $M_t$ hay encoder trong giai đoạn nào là của Spec 6.

### 3.5 Kiểm tra đầu vào

Hàm raise lỗi rõ ràng (không tự sửa) khi:

* Bốn feature không cùng shape `[B, D]`, hoặc $D$ khác số cột của hai prototype, hoặc hai prototype khác shape nhau.
* `source_labels`, `pseudo_labels`, `trusted_mask` có độ dài khác $B$.
* `source_labels` hoặc `pseudo_labels` không phải kiểu số nguyên, hoặc `trusted_mask` không phải kiểu `bool`.
* `source_labels` có giá trị ngoài `[0, num_classes)`. `pseudo_labels` được kiểm tra phạm vi chỉ trên các vị trí có mask bằng true.
* Feature hoặc prototype chứa NaN hoặc Inf.

### 3.6 Log

Task trả `terms` và `num_trusted`. Caller (Spec 6) quyết định ghi log. Hai quan sát nên ghi mỗi bước để dễ chẩn đoán:

* Sáu giá trị term riêng, để biết term nào đang kéo loss.
* Tỉ lệ `num_trusted / B`. Nếu tỉ lệ này gần 0 kéo dài thì ba term có mask gần như không học được.

## 4. Giao diện với các task khác

| Task | Giao diện |
| --- | --- |
| Task 3 | Cung cấp $F_s, F_t, F_{t\rightarrow s}, F_{s\rightarrow t}$ cùng một lượt forward, cùng batch, đã ghép cặp theo chỉ số |
| Task 2.2 | Cung cấp $M_s$ và $M_t$ dạng tham số học được, shape `[7, 768]` |
| Task 4.1 | Cung cấp `pseudo_labels` và `trusted_mask` của các ảnh đích trong batch, qua hàm tra cứu theo `sample_id` |
| Spec 5 | STR có thể đổi thành phần batch. Nó phải giữ nguyên quy ước ghép cặp theo chỉ số $i$ và cung cấp nhãn, pseudo label, mask đúng theo chỉ số đó |
| Spec 6 | Gọi hàm này mỗi bước, nhân $\alpha, \beta, \gamma$, cộng $\mathcal{L}_{ad}$, gọi `backward()` và `step()` |

POST đặt $\alpha, \beta, \gamma$ lần lượt là 1, 0.5, 0.1 cho $\mathcal{L}_{t\rightarrow s}$, $\mathcal{L}_{s\rightarrow t}$, $\mathcal{L}_{ad}$. Các giá trị này nằm trong config của Spec 6, không nằm ở đây.

## 5. Phụ thuộc

* Task 2.2 phải cho hai prototype cùng thứ tự lớp với nhãn nguồn: `anger, disgust, fear, happiness, neutral, sadness, surprise`. Chỉ số 0 đến 6 trong `source_labels` phải khớp hàng trong $M_s$.
* Task 4.1 phải dùng cùng thứ tự lớp cho pseudo label.
* Task 3 phải trả feature có chiều bằng `feature_dim` của mô hình. Lệch chiều bị bắt ở 3.5.
* Spec 1 cung cấp batch nguồn dạng `(ảnh, nhãn)` với nhãn kiểu `long`, và batch đích dạng `(ảnh, sample_id)`. `sample_id` của ảnh đích dùng để tra 4.1.
* Spec 6 ghép hai batch này, lấy `pseudo_labels` và `trusted_mask` từ hàm tra cứu của 4.1 rồi gọi hàm loss. Nó phải đảm bảo batch nguồn và batch đích cùng kích thước $B$ và ghép cặp theo chỉ số. Nếu dùng `drop_last`, nó phải áp dụng đồng thời cho hai phía.

## 6. Lựa chọn của repo và hướng thử thay thế

| Lựa chọn | Mặc định của repo | Hướng thử thay thế |
| --- | --- | --- |
| Logit | Tích vô hướng thô, như Eq. 5 đến 9 | Chuẩn hóa L2 cả feature và prototype rồi nhân một temperature. Nếu đổi, 4.1 phải đổi theo để confidence cùng thang đo |
| Chuẩn hóa term có mask | Chia cho số mẫu tin cậy trong batch | Chia cho một hằng số cố định, hoặc cho kích thước tập tin cậy toàn cục $N_t^*$ như dạng viết của POST, để mỗi mẫu có trọng số ổn định giữa các batch |
| Gộp hai term của loss chéo | Hai trung bình riêng rồi cộng, như Eq. 8 và 9 | Gộp mẫu rồi trung bình chung. Không khuyến nghị vì tỉ lệ mẫu tin cậy sẽ đổi cân bằng hai term |
| Gradient của term chéo | Không `detach` | Chặn gradient vào prototype ở term native chéo, hoặc chặn gradient vào feature ở term đó, để xem prototype của miền kia có bị kéo lệch không |
| Tập tin cậy rỗng | Term bằng 0, gradient bằng 0 | Raise lỗi nếu rỗng quá nhiều bước liên tiếp, đặt ở Spec 6 |
| Độ chính xác số | `float32` cho cross entropy | Giữ mixed precision nếu đo được không đổi kết quả |
| Nhãn mềm | Không dùng (POST dùng hard label) | Label smoothing cho term nguồn |
| Trọng số lớp | Không có | Trọng số theo tần suất lớp trong tập tin cậy, nếu pseudo label lệch lớp mạnh |

## 7. Kịch bản test

Các ví dụ dưới đây dùng hai lớp và chiều 2 để tính tay được. Hàm thật vẫn phải chạy được với bảy lớp và chiều 768 (Scenario 8). Đặt $c = \ln(1 + e^{-2}) \approx 0.126928$. Các scenario 1, 2, 4, 5 và 6 dùng $M_s = M_t = \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}$. Scenario 3 dùng hai bank khác nhau, vì nếu hai bank bằng nhau thì việc chọn sai bank không làm đổi loss.

### Scenario 1: Một term nguồn

Với feature $F = (2, 0)$: logit là $(2, 0)$. Nhãn 0 cho cross entropy $c \approx 0.126928$. Nhãn 1 cho $2 + c \approx 2.126928$. Test $\mathcal{L}_s$ với $B = 1$, $F_s = (2, 0)$, $y_s = 0$ cho đúng $0.126928$.

### Scenario 2: Mask loại mẫu không tin cậy

$B = 2$, $F_t = [(0, 2), (2, 0)]$, $\hat y_t = [1, 1]$, mask `[true, false]`. Mẫu 0 có cross entropy $c$, mẫu 1 có $2 + c$ nhưng bị loại. $\mathcal{L}_t$ phải bằng $0.126928$, không phải trung bình của hai mẫu. Đổi $\hat y_t$ của mẫu 1 thành giá trị bất kỳ, kể cả 99, thì giá trị loss và gradient không đổi và không có lỗi.

### Scenario 3: Cặp feature, prototype, nhãn của loss chéo

Đặt $M_s = \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}$ và $M_t = \begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}$ (hai hàng đổi chỗ). Với $M_t$, logit của feature $(a, b)$ là $(b, a)$. Mask `[true]`, $B = 1$. Kiểm tra từng term riêng qua `terms`, không chỉ kiểm tra tổng, vì hai term có thể đổi chỗ giá trị mà tổng không đổi.

Ca $t \rightarrow s$: $F_{t\rightarrow s} = (2, 0)$, $y_s = 0$, $F_t = (0, 2)$, $\hat y_t = 0$. Đúng bank thì `t2s_cross` $= c \approx 0.126928$ (logit $(2, 0)$ trên $M_s$, nhãn 0) và `t2s_native` $= 2 + c \approx 2.126928$ (logit $(0, 2)$ trên $M_s$, nhãn 0). Nếu hai term bị tính nhầm trên $M_t$ thì chúng đổi chỗ giá trị cho nhau.

Ca $s \rightarrow t$: $F_{s\rightarrow t} = (2, 0)$, $\hat y_t = 0$, $F_s = (2, 0)$, $y_s = 1$. Đúng bank thì `s2t_cross` $= 2 + c \approx 2.126928$ (logit $(0, 2)$ trên $M_t$, nhãn 0) và `s2t_native` $= c \approx 0.126928$ (logit $(0, 2)$ trên $M_t$, nhãn 1). Nếu tính nhầm trên $M_s$ thì `s2t_cross` bằng $c$ và `s2t_native` bằng $2 + c$.

Chạy hai ca bằng hai lần gọi riêng vì $y_s$ khác nhau giữa chúng.

### Scenario 4: Chuẩn hóa riêng từng term

$B = 2$, mask `[true, false]`. Term có mask của $\mathcal{L}_{s\rightarrow t}$ chia cho 1 chứ không cho 2, còn term không mask chia cho 2. Test với giá trị cross entropy khác nhau giữa hai mẫu để thấy khác biệt, và xác nhận kết quả bằng tổng của hai trung bình riêng.

### Scenario 5: Tập tin cậy rỗng

Mask toàn false. Ba term có mask đều bằng 0, không NaN, không lỗi. `loss.backward()` chạy được và gradient của các feature và prototype chịu ảnh hưởng chỉ từ ba term không mask.

### Scenario 6: Gradient đi đúng hướng

Gọi `result.terms["t2s_native"].backward()` riêng, không qua loss tổng. Sau đó `M_s.grad` phải khác 0 và `F_t.grad` phải khác 0, trong khi `M_t.grad`, `F_s.grad` và `F_t2s.grad` phải là `None` hoặc bằng 0. Làm tương tự cho `s2t_native` với `M_t` và `F_s`. Mỗi lần `backward()` chạy trên một lần forward mới, hoặc dùng `retain_graph=True`. Test này xác nhận việc không `detach` và việc term native dùng đúng bank.

### Scenario 7: Kiểm tra đầu vào

Mỗi trường hợp ở 3.5 gây lỗi rõ ràng: lệch chiều giữa feature và prototype, nhãn nguồn bằng 7, mask kiểu `long`, feature chứa NaN, độ dài nhãn khác $B$. Pseudo label ngoài phạm vi ở vị trí mask false thì không lỗi.

### Scenario 8: Shape thật và không tác dụng phụ

Chạy với $B = 4$ hoặc lớn hơn, bảy lớp, chiều 768, `float32` và `float16` đầu vào. Bốn loss là vô hướng `float32`, hữu hạn, có `requires_grad`. Sau lần gọi, hàm không đổi `requires_grad`, không sửa tensor đầu vào, không đổi chế độ train/eval, và hai lần gọi liên tiếp với cùng đầu vào cho cùng kết quả.

## 8. Tiêu chí nghiệm thu

* Bốn loss khớp giá trị tính tay ở các scenario 1 đến 4.
* Cặp (feature, prototype, nhãn, mask) của cả sáu term đúng như bảng ở Section 0, kể cả hai term native của loss chéo.
* Mỗi term được trung bình riêng, và mẫu không tin cậy không ảnh hưởng giá trị hay gradient.
* Tập tin cậy rỗng cho loss bằng 0 và gradient bằng 0 mà không NaN.
* Gradient chảy về feature và prototype đúng theo 3.4, không có `detach` ngoài ý muốn.
* Mọi đầu vào sai ở 3.5 đều bị báo lỗi rõ ràng.
* Task trả bốn loss riêng, sáu giá trị term và `num_trusted`, không nhân trọng số, không gọi `backward()`, không thay đổi trạng thái nào bên ngoài.