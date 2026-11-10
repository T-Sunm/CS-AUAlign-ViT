# Task 8.2 — RAF-DB → ExpW ablation

Chọn checkpoint bằng ExpW validation, sau đó đánh giá một lần trên toàn bộ
ExpW test. E0 lấy từ
[`run-001/expw_test.json`](../../runs/adaptation/run-001/expw_test.json), còn
E3 lấy từ
[`e3-both-100epochs-seed42/expw_test.json`](../../runs/adaptation/e3-both-100epochs-seed42/expw_test.json).

| Experiment | AU attention | AU alignment | ExpW test accuracy | Δ accuracy với E0 | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|---:|---:|
| E0 — POST baseline | Off | Off | 63.67% | — | 42.98% | **44.20%** | 62.27% |
| E1 — AU attention | Dynamic | Off | — | — | — | — | — |
| E2 — AU alignment | Off | On | — | — | — | — | — |
| E3 — Both | Dynamic | On | **65.65%** | **+1.98 điểm %** | **44.01%** | 43.83% | **63.80%** |
| E4 — Alignment shuffled | Off | Shuffled | — | — | — | — | — |
| E5 — Attention shuffled | Dynamic shuffled | Off | — | — | — | — | — |

E0 dùng `runs/adaptation/run-001/best.pt` tại epoch 63 và phân loại đúng
5.133/8.062 ảnh. E3 dùng `runs/adaptation/e3-both-100epochs-seed42/best.pt`
tại epoch 21 và phân loại đúng 5.293/8.062 ảnh, tăng 160 ảnh. E3 cải thiện
accuracy, balanced accuracy và weighted F1. Macro F1 giảm 0,37 điểm phần trăm;
recall của class 1 và class 2 cũng giảm, nên kết quả chưa cho thấy cải thiện
đồng đều giữa các lớp.

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
