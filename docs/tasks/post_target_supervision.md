# Task 4.1. Target supervision: pseudo label, k-means và tập tin cậy

## 0. Đọc nhanh

Ảnh đích không có nhãn, nên để huấn luyện bộ phân loại $M_t$ và các loss xuyên miền ta phải tự tạo nhãn tạm cho chúng, gọi là pseudo label. Task 4.1 làm đúng việc đó theo chu kỳ. Nó không tính loss, không huấn luyện gì, và không quyết định khi nào được chạy.

Một lần làm mới (refresh) diễn ra như sau:

1. Trích đặc trưng $F_t$ của toàn bộ tập đích bằng đường chỉ ảnh đích của Task 3, chế độ `eval()`, không augmentation.
2. Lấy bảy hàng của $M_t$ làm bảy tâm xuất phát của k-means.
3. Gán mỗi ảnh vào tâm gần nhất, tính lại tâm bằng trung bình của từng cụm, rồi gán lại. Nhãn sau lần gán lại là pseudo label.
4. Tính nhãn dự đoán và độ tin cậy từ softmax của $F_t$ trên $M_t$.
5. Một ảnh vào tập tin cậy khi nhãn dự đoán trùng pseudo label và độ tin cậy vượt ngưỡng.
6. Không giữ các tâm k-means vừa tính. Kết quả theo từng ảnh cùng danh sách `sample_id` và `epoch` được giữ lại.

Giữa hai lần làm mới (mặc định 10 epoch), pseudo label đứng yên và $M_t$ được huấn luyện ở nơi khác (Task 4.2 và Spec 6). Lần làm mới sau lại bắt đầu từ $M_t$ mới.

Các shape cần nhớ, với $N$ là số ảnh của tập thích nghi:

| Đại lượng | Shape | Ghi chú |
| --- | --- | --- |
| $M_t$ | `[7, 768]` | Bảy hàng, mỗi hàng là prototype của một lớp |
| $F_t$ của cả tập | `[N, 768]` | Một vector 768 chiều cho mỗi ảnh |
| Logit | `[N, 7]` | Tích vô hướng giữa $F_t$ và từng hàng của $M_t$ |
| Nhãn dự đoán | `[N]` | Argmax của softmax, số nguyên từ 0 đến 6 |
| Độ tin cậy | `[N]` | Xác suất lớn nhất của softmax |
| Pseudo label | `[N]` | Nhãn từ k-means, số nguyên từ 0 đến 6 |
| Mặt nạ tin cậy | `[N]` | Kiểu bool |

Ba điều dễ hiểu nhầm:

* $M_t$ chỉ là tâm xuất phát của k-means. Pseudo label vẫn do k-means gán, không phải đầu ra của bộ phân loại prototype.
* Mọi ảnh đích đều có pseudo label, nhưng loss dùng pseudo label chỉ được tính trên tập tin cậy.
* Các tâm tính lại không được giữ qua các lần làm mới. Thứ duy nhất đi tiếp là $M_t$ đã được huấn luyện.

## 1. Mục tiêu

Task 4.1 tạo, theo chu kỳ, bốn thông tin theo chỉ số ảnh đích: pseudo label, nhãn dự đoán, độ tin cậy và mặt nạ tin cậy. Task 4.2 dùng kết quả để mask loss. Spec 5 tự suy luận lại khi chọn ảnh cho STR.

POST chỉ nêu ba điều ở mức ý tưởng: k-means chạy "mỗi vài epoch", chỉ dùng mẫu có nhãn dự đoán trùng pseudo label và độ tin cậy vượt một ngưỡng, và prototype của miền đích khởi tạo từ prototype nguồn. Mọi chi tiết còn lại trong spec này (tâm xuất phát, số lượt, chuẩn hóa, ngưỡng, vòng đời) là lựa chọn của repo, được liệt kê ở Section 6.

Task này không tính loss, không huấn luyện, không lên lịch chạy (việc của Spec 6), và không cắt ghép landmark (việc của Spec 5).

## 2. Đầu vào và đầu ra

Hàm làm mới nhận:

* `feature_model`: mô-đun BCS của Task 3, chỉ dùng đường ảnh đích để lấy $F_t$.
* `prototype_banks`: bộ prototype của Task 2.2, chỉ đọc hàng của $M_t$. Hai đối tượng này được nhận riêng vì trong code chúng là hai mô-đun khác nhau. Nếu sau này có wrapper gộp cả hai thì hợp đồng vẫn giữ nguyên ý nghĩa của hai đầu vào.
* Một cách duyệt tập thích nghi cho k-means: cùng danh sách mẫu như `target_adaptation`, transform kiểu đánh giá, không shuffle, trả `(ảnh, sample_id)`. Cách duyệt này được yêu cầu ở Section 5.
* `expected_sample_ids`: danh sách `sample_id` chuẩn của tập thích nghi, lấy độc lập với loader quét, ví dụ từ danh sách mẫu của dataset `target_adaptation`. Đây là nguồn đối chiếu duy nhất để phát hiện loader thiếu, lặp hoặc đổi thứ tự ảnh. Cùng danh sách này được lưu cùng kết quả và dùng khi nạp lại checkpoint.
* Cấu hình: `num_classes`, `kmeans_iterations`, `trust_threshold`, `eps`.

Đầu ra là một kết quả có cấu trúc, mọi trường đều theo vị trí `0..N-1` của tập thích nghi:

| Trường | Shape | Kiểu |
| --- | --- | --- |
| `pseudo_labels` | `[N]` | `long` |
| `predicted_labels` | `[N]` | `long` |
| `confidence` | `[N]` | `float` |
| `trusted_mask` | `[N]` | `bool` |
| `sample_ids` | `N` phần tử | Danh sách `sample_id`, cùng thứ tự với các tensor trên |
| `epoch` | Số nguyên | Epoch tại thời điểm làm mới |

Ngoài ra có hàm tra cứu theo `sample_id` cho một batch, trả `pseudo_labels` và `trusted_mask` tương ứng, để các caller đang duyệt theo batch không phải tự đổi `sample_id` sang vị trí.

## 3. Hợp đồng chi tiết

### 3.1 Trích đặc trưng

* Chạy ở chế độ `eval()` và `torch.no_grad()`, rồi khôi phục chế độ ban đầu sau khi xong.
* Chỉ dùng đường chỉ ảnh đích của Task 3 ($x_t \rightarrow F_t$). Không cần ảnh nguồn và không chạy cross attention.
* Không augmentation, không lật ảnh, không shuffle, không bỏ batch cuối.
* Thứ tự `sample_id` của cách duyệt này phải khớp từng phần tử với `target_adaptation`.
* K-means chỉ chạy trên tập thích nghi, không bao giờ trên validation hoặc test.
* Trước khi xuất kết quả, kiểm tra các `sample_id` thu được trong lượt quét: đúng $N$ phần tử, không lặp, và khớp từng vị trí với `expected_sample_ids`. Thiếu, lặp hoặc sai thứ tự thì báo lỗi rõ ràng và không xuất kết quả. Kiểm tra này chạy ở mọi lần làm mới, không chỉ ở lần đầu.
* Nếu $F_t$ hoặc $M_t$ chứa giá trị không hữu hạn thì báo lỗi rõ ràng.

### 3.2 K-means khởi tạo từ $M_t$

Đặt $f_i$ là $F_t$ của ảnh thứ $i$ và $\mu_k$ là hàng thứ $k$ của $M_t$. Chuẩn hóa $L_2$ cả hai, chỉ trong bước k-means này:

```math
\hat f_i=\frac{f_i}{\lVert f_i\rVert_2},\qquad
\hat\mu_k=\frac{\mu_k}{\lVert \mu_k\rVert_2}.
```

Một lần làm mới gồm gán, tính lại tâm và gán lại:

```math
a_i^{(1)}=\arg\max_k \hat f_i^{\top}\hat\mu_k,\qquad
c_k=\frac{\sum_{i:\,a_i^{(1)}=k}\hat f_i}{\left\lVert\sum_{i:\,a_i^{(1)}=k}\hat f_i\right\rVert_2},\qquad
a_i^{(2)}=\arg\max_k \hat f_i^{\top}c_k.
```

Pseudo label của ảnh $i$ là $a_i^{(2)}$.

Quy tắc:

* Tâm xuất phát luôn là bảy hàng của $M_t$ hiện tại, ở mọi lần làm mới. Tâm tính lại của lần trước không được dùng và không được lưu vào mô hình hay checkpoint.
* Cụm $k$ là lớp $k$ theo cấu trúc, nên không có bước ghép nhãn. Thuật toán Hungarian không nằm trong luồng chính. Nếu bật chẩn đoán tùy chọn, Hungarian chỉ so khớp tâm cuối với $M_t$ để cảnh báo khi ánh xạ không phải ánh xạ đơn vị, và không được đổi nhãn.
* `kmeans_iterations` là số lượt tính lại tâm, mặc định 1. Giá trị 0 bị từ chối, vì khi đó pseudo label gần như trùng nhãn dự đoán và điều kiện trùng nhãn ở Section 3.4 mất ý nghĩa.
* Nếu một cụm rỗng ở lần gán thứ nhất, tâm của cụm đó giữ nguyên là $\hat\mu_k$ và sự việc được ghi log.
* Một cụm khác rỗng vẫn có thể có tổng vector bằng 0 hoặc rất nhỏ, khi các $\hat f_i$ triệt tiêu nhau. Nếu chuẩn của tổng nhỏ hơn `eps` (mặc định $10^{-8}$) thì xử lý như cụm rỗng: giữ tâm trước đó và ghi log riêng. Không để phép chia tạo tâm bằng 0 hay `NaN`.
* Phép chuẩn hóa $L_2$ của $f_i$ và $\mu_k$ dùng cùng `eps` ở mẫu số. Nếu chuẩn của một $f_i$ hoặc một hàng $\mu_k$ nhỏ hơn `eps` thì báo lỗi rõ ràng, vì pseudo label sẽ phụ thuộc vào cách chọn khi hòa chứ không vào dữ liệu.
* Khi hòa điểm ở argmax, chọn chỉ số lớp nhỏ nhất, để kết quả xác định.

### 3.3 Nhãn dự đoán và độ tin cậy

Tính tại cùng thời điểm với k-means, từ $f_i$ chưa chuẩn hóa và $M_t$, giống cách bài dùng tích vô hướng thô:

```math
z_i=f_i M_t^{\top}\in\mathbb{R}^{7},\qquad
p_i=\operatorname{softmax}(z_i),\qquad
\hat y_i=\arg\max_k p_{i,k},\qquad
\mathrm{conf}_i=\max_k p_{i,k}.
```

### 3.4 Tập tin cậy

```math
m_i=\mathbb{1}\!\left[\hat y_i=a_i^{(2)}\right]\cdot\mathbb{1}\!\left[\mathrm{conf}_i>thr_{trust}\right].
```

* `trust_threshold` là một số vô hướng dùng chung cho mọi lớp, mặc định 0.95.
* Bài POST chỉ nêu giá trị 0.95 cho ngưỡng resampling của STR. Giá trị cho tập tin cậy là suy đoán. Vì vậy cấu hình có hai khóa riêng, `trust_threshold` ở đây và `resample_threshold` của Spec 5.
* So sánh là lớn hơn hẳn. Tập tin cậy rỗng là hợp lệ, không phải lỗi.

### 3.5 Vòng đời

* Làm mới lần đầu ở epoch 0, trước khi huấn luyện. Lúc đó $M_t$ bằng $M_s$, nên tập tin cậy có thể nhỏ hoặc rỗng.
* Các lần sau do Spec 6 gọi theo `kmeans_interval`, mặc định 10 epoch. Giá trị 10 là mặc định chưa tinh chỉnh, chọn trong khoảng các bài tương tự dùng. Spec này không tự lên lịch.
* Kết quả đứng yên giữa hai lần làm mới. Mỗi lần làm mới thay toàn bộ kết quả trước, không trộn.
* Kết quả được lưu trong checkpoint cùng danh sách `sample_id`. Khi nạp lại, nếu danh sách khác tập thích nghi hiện tại thì báo lỗi rõ ràng, vì vị trí sẽ bị lệch âm thầm nếu bỏ qua.
* Hàm làm mới không đọc và không nhận nhãn thật của đích.
* Có một hàm chẩn đoán tách riêng, nhận nhãn thật từ caller chỉ để ghi log: độ chính xác của pseudo label và của tập tin cậy. Vòng huấn luyện không được gọi hàm này để ra quyết định.

### 3.6 Log mỗi lần làm mới

* Số mẫu tin cậy, tổng và theo từng lớp. Đây là tín hiệu sớm để thấy lớp khó bị thưa.
* Tỷ lệ trùng giữa nhãn dự đoán và pseudo label. Tỷ lệ này thấp bất thường là dấu hiệu cụm bị lệch.
* Số ảnh đổi pseudo label so với lần làm mới trước.
* Số cụm rỗng và số cụm có tổng vector gần 0.
* Tỷ lệ tin cậy theo từng lớp (số mẫu tin cậy chia số mẫu có pseudo label thuộc lớp đó). Ở epoch 0, $M_t$ mới chỉ copy từ $M_s$ nên tập tin cậy có thể rất nhỏ. Dùng log này để xem dữ liệu thực trước khi đổi `trust_threshold`.

## 4. Giao diện với các task khác

Tên trường dưới đây là hợp đồng, các task khác dùng đúng tên này:

| Người dùng | Trường dùng | Mục đích |
| --- | --- | --- |
| Task 4.2 | `pseudo_labels`, `trusted_mask` | Mask các số hạng loss dùng pseudo label, theo chỉ số ảnh |
| Spec 5 (STR) | Không bắt buộc | STR chạy suy luận lại trên toàn bộ tập đích bằng mô hình hiện tại ở mỗi vòng, nên tự tạo `predicted_labels` và `confidence` mới. `predicted_labels` và `confidence` của spec này đứng yên giữa hai lần làm mới nên không dùng thay cho suy luận của STR, trừ khi Spec 5 xác định một thời điểm hai bên dùng chung cùng một lượt suy luận. `pseudo_labels` và `trusted_mask` vẫn có thể được Spec 5 đọc nếu cần |
| Spec 6 | Hàm làm mới và lưu, nạp | Gọi theo lịch, lưu vào checkpoint |

Task 4.2 chỉ trích dẫn bảng này và không chép lại phần giải thích bên trong k-means.

## 5. Phụ thuộc

* **Spec 1, cách duyệt mới (yêu cầu bổ sung).** Cần một cách duyệt tập thích nghi theo thứ tự cố định, không augmentation, trả `(ảnh, sample_id)`, cùng danh sách mẫu với `target_adaptation`. Tên gợi ý là role `target_features`, quyết định cụ thể khi cài đặt. Cần một test kiểm tra thứ tự `sample_id` của hai role khớp nhau từng phần tử.
* **Spec 1, khóa mẫu.** `sample_id` là khóa ổn định. Vị trí `0..N-1` hình thành sau khi lọc mẫu không dùng được, nên cần ánh xạ `sample_id` sang vị trí nhất quán giữa các cách duyệt.
* **Spec 1, phạm vi hiện tại.** ExpW train (chỉ mẫu usable) là tập thích nghi chính thức. Bảy lớp theo thứ tự `anger, disgust, fear, happiness, neutral, sadness, surprise`. Spec này không phụ thuộc tập dữ liệu: $N$ lấy từ dataset, nên khi mở rộng benchmark (ví dụ CK+, JAFFE, FER2013, SFEW 2.0) việc cần làm nằm ở Spec 1 (reader, manifest, quy tắc nhãn) chứ không phải ở 4.1.
* **Task 2.2.** $M_t$ có shape `[7, 768]` và khởi tạo từ $M_s$.
* **Task 3.** Đường chỉ ảnh đích cho $F_t$ có shape `[B, 768]`.
* **Spec 6.** Gọi làm mới đúng lịch và định nghĩa epoch. Spec này không phụ thuộc cách ghép batch.

## 6. Lựa chọn của repo và hướng thử thay thế

Bài POST không xác định các điểm sau nên spec tự chọn. Khi kết quả lệch bài, giữ cấu hình này làm baseline và chỉ đổi một biến mỗi lần.

* Tâm xuất phát: baseline là $M_t$ hiện tại ở mỗi lần làm mới, để cụm $k$ luôn là lớp $k$. Thử giữ tâm của lần trước, đổi lại nhãn có thể trôi dần.
* Số lượt: baseline là 1 lượt tính lại tâm. Thử nhiều lượt, đổi lại cụm bám cấu trúc đích hơn nhưng xa ý nghĩa gốc của lớp.
* Chuẩn hóa: baseline là spherical k-means trong bước này. Thử khoảng cách Euclid trên đặc trưng chưa chuẩn hóa.
* Ngưỡng: baseline là một ngưỡng chung cho mọi lớp. Thử ngưỡng theo từng lớp (một bài FER gần đây dùng cách này) nếu log cho thấy lớp khó bị thưa.
* Chu kỳ: baseline là 10 epoch. Thử mỗi epoch hoặc cập nhật theo số bước.
* Điều kiện tin cậy: baseline là nhãn dự đoán phải trùng pseudo label cùng độ tin cậy vượt ngưỡng. Thử chỉ dùng ngưỡng.

## 7. Kịch bản test

### Scenario 1: Cụm $k$ là lớp $k$

Tạo ba nhóm điểm giả tách rõ, đặt $M_t$ gần tâm thật của từng nhóm, rồi kiểm tra pseudo label khớp nhóm thật mà không cần ghép nhãn. Sau đó hoán vị đồng thời thứ tự hàng của $M_t$ và nhãn thật, và kiểm tra pseudo label đổi theo đúng hoán vị. Điều này chứng minh ý nghĩa của cụm đến từ $M_t$ chứ không đến từ thứ tự cụm của thuật toán.

### Scenario 2: Một lượt tính lại tâm

Dùng ví dụ nhỏ tính tay, trong đó một ảnh ở rìa giữa hai cụm đổi nhãn sau khi tâm được tính lại. Kiểm tra tâm tính lại bằng trung bình của cụm (sau chuẩn hóa) và ảnh rìa đổi nhãn đúng như tính tay. Kiểm tra `kmeans_iterations = 0` bị từ chối.

### Scenario 3: Tập tin cậy

Kiểm tra một ảnh có độ tin cậy cao nhưng nhãn dự đoán khác pseudo label bị loại. Kiểm tra độ tin cậy đúng bằng ngưỡng bị loại. Kiểm tra tập tin cậy rỗng không gây lỗi và các trường vẫn đủ shape.

### Scenario 4: Cụm rỗng

Tạo dữ liệu để một cụm rỗng ở lần gán thứ nhất. Kiểm tra tâm của cụm đó giữ nguyên $\hat\mu_k$, kết quả không có `NaN`, và có ghi log.

### Scenario 5: Chỉ số và vòng đời

Kiểm tra thứ tự `sample_id` của cách duyệt mới khớp từng phần tử với `target_adaptation`. Lưu rồi nạp kết quả và kiểm tra các tensor khớp. Nạp với danh sách `sample_id` khác phải báo lỗi. Chạy làm mới hai lần với cùng $M_t$ và cùng đặc trưng, kết quả phải giống nhau bất kể lịch sử trước đó, để chứng minh tâm tính lại của lần trước không bị dùng lại.

### Scenario 6: Không rò nhãn thật

Kiểm tra hàm làm mới không có tham số nhận nhãn thật và không đọc trường nhãn từ dataset. Hàm chẩn đoán tách riêng và không được vòng huấn luyện gọi.

### Scenario 7: Không tác dụng phụ

Kiểm tra sau khi làm mới, chế độ `train()` hoặc `eval()` của mô hình được khôi phục, các tham số của mô hình và $M_t$ không đổi, và không có gradient nào được tích lũy.

### Scenario 8: Shape và tương thích

Kiểm tra lỗi rõ ràng khi `expected_sample_ids` không khớp độ dài hoặc thứ tự của lượt quét, và khi chiều của $F_t$ không khớp chiều của hàng $M_t$, hoặc số hàng của $M_t$ khác `num_classes`. Kiểm tra lỗi khi $F_t$ hoặc $M_t$ chứa `NaN`, `Inf` hoặc một vector có chuẩn bằng 0.

### Scenario 9: Tổng vector triệt tiêu

Tạo hai ảnh có $\hat f$ đối nhau cùng rơi vào một cụm, để tổng vector bằng 0. Kiểm tra tâm của cụm đó giữ nguyên tâm trước, kết quả không có `NaN`, và log ghi nhận cụm này.

### Scenario 10: Lượt quét sai ID

Dùng loader giả bị thiếu một ảnh, lặp một ảnh, hoặc đổi chỗ hai ảnh. Mỗi trường hợp phải báo lỗi và không có kết quả nào được xuất.

## 8. Tiêu chí nghiệm thu

Task hoàn thành khi hàm làm mới trả đủ sáu trường theo đúng tên, shape và kiểu ở Section 2, với pseudo label sinh từ k-means khởi tạo từ $M_t$ và một lượt tính lại tâm theo Section 3.2, nhãn dự đoán và độ tin cậy theo Section 3.3, và tập tin cậy theo Section 3.4. Cách duyệt mới của Spec 1 cho thứ tự `sample_id` khớp `target_adaptation`. Kết quả lưu và nạp được cùng danh sách `sample_id`, và nạp sai danh sách thì báo lỗi. Hàm làm mới không dùng nhãn thật, không có tác dụng phụ lên mô hình, và log đủ các chỉ số ở Section 3.6. Lượt quét sai ID, tâm triệt tiêu và đầu vào không hữu hạn đều được xử lý theo Section 3.1 và 3.2. Task 4.2 và Spec 5 dùng được kết quả mà không cần biết chi tiết bên trong k-means. Lịch gọi, định nghĩa epoch và việc lưu vào checkpoint ở mức vòng huấn luyện thuộc Spec 6.