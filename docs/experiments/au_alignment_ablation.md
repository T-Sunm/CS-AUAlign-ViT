# Task 8.2 — RAF-DB → ExpW ablation

Chọn checkpoint bằng ExpW validation, sau đó đánh giá một lần trên toàn bộ
ExpW test. Kết quả lấy từ `expw_test.json` trong thư mục run tương ứng của E0,
E1, E2, E3 và E4. E5 chưa có checkpoint hoàn chỉnh nên chưa được đánh giá.

| Experiment | AU attention | AU alignment | ExpW test accuracy | Δ accuracy với E0 | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|---:|---:|
| E0 — POST baseline | Off | Off | 63.67% | — | 42.98% | **44.20%** | 62.27% |
| E1 — AU attention | Dynamic | Off | 64.35% | +0.68 điểm % | 42.77% | 43.67% | 62.43% |
| E2 — AU alignment | Off | On | **65.86%** | **+2.20 điểm %** | 44.58% | **44.25%** | 64.21% |
| E3 — Both | Dynamic | On | 65.65% | +1.98 điểm % | 44.01% | 43.83% | 63.80% |
| E3-no-STR — Both, không STR | Dynamic | On | 65.78% | +2.11 điểm % | 44.12% | 43.39% | 63.81% |
| E4 — Alignment shuffled | Off | Shuffled | 65.64% | +1.97 điểm % | **44.82%** | 44.07% | **64.50%** |
| E5 — Attention shuffled | Dynamic shuffled | Off | — | — | — | — | — |

E0 dùng `runs/adaptation/run-001/best.pt` tại epoch 63 và phân loại đúng
5.133/8.062 ảnh. E3 dùng `runs/adaptation/e3-both-100epochs-seed42/best.pt`
tại epoch 21 và phân loại đúng 5.293/8.062 ảnh, tăng 160 ảnh. E3 cải thiện
accuracy, balanced accuracy và weighted F1. Macro F1 giảm 0,37 điểm phần trăm;
recall của class 1 và class 2 cũng giảm, nên kết quả chưa cho thấy cải thiện
đồng đều giữa các lớp.

E1, E2 và E4 lần lượt chọn best checkpoint tại epoch 7, 12 và 16. Trong các
run đã đánh giá, E2 có test accuracy cao nhất (`65.86%`), còn E4 có balanced
accuracy (`44.82%`) và weighted F1 (`64.50%`) cao nhất. Tuy nhiên, E1 và E2
chạy batch size 45 trong khi E3 và E4 chạy batch size 64. Vì vậy, chênh lệch
giữa các run này không chỉ phản ánh duy nhất AU attention/alignment và cần được
ghi nhận như một protocol confound.

E3-no-STR chọn best checkpoint tại epoch 34 và đạt `65.78%` accuracy, cao hơn
E3 `0.12` điểm phần trăm. Balanced accuracy và weighted F1 gần như không đổi,
còn macro F1 giảm `0.44` điểm phần trăm. Best checkpoint của E3 nằm ở epoch 21,
trước STR đầu tiên tại epoch 25, nên so sánh này cho thấy model selection không
thu được lợi ích rõ ràng từ phần training sau STR; nó không phải phép so sánh
hai checkpoint cùng epoch chỉ khác việc có hay không có ảnh STR.

## Thời điểm STR ảnh hưởng đến checkpoint

STR chạy sau khi hoàn thành epoch được lên lịch. Ảnh mới tạo ở boundary epoch
25 bắt đầu tham gia train từ epoch 26; tương tự, STR tại epoch 50 và 75 bắt
đầu ảnh hưởng từ epoch 51 và 76.

- E0 chọn best tại epoch 63. Checkpoint này đã train với dữ liệu STR tạo sau
  epoch 25 và 50.
- E3 chọn best tại epoch 21, trước STR đầu tiên. Run E3 vẫn thực hiện STR sau
  epoch 25, 50 và 75, nhưng các checkpoint sau đó không vượt validation
  accuracy của epoch 21.

E0 chạy 256 epoch, còn E3 và các ablation tiếp theo dùng horizon 100 epoch.
Cả hai best checkpoint đều nằm trong 100 epoch đầu. Checkpoint được chọn bằng
nhãn ExpW validation. Kết quả hiện có một seed; ExpW test không được dùng để
điều chỉnh hyperparameter hoặc chọn checkpoint.
