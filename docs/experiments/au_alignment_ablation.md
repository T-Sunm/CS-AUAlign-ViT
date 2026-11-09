# Task 8.2 — RAF-DB → ExpW ablation

Chọn checkpoint bằng ExpW validation, sau đó đánh giá trên toàn bộ ExpW test. Bảng dưới ghi kết quả E0 từ [`run-001/expw_test.json`](../../runs/adaptation/run-001/expw_test.json); các cấu hình còn lại chưa chạy.

| Experiment | AU attention | AU alignment | ExpW test accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|---:|
| E0 — POST baseline | Off | Off | **63.67%** | **42.98%** | **44.20%** | **62.27%** |
| E1 — AU attention | Dynamic | Off | — | — | — | — |
| E2 — AU alignment | Off | On | — | — | — | — |
| E3 — Both | Dynamic | On | — | — | — | — |
| E4 — Alignment shuffled | Off | Shuffled | — | — | — | — |
| E5 — Attention shuffled | Dynamic shuffled | Off | — | — | — | — |

E0 dùng `runs/adaptation/run-001/best.pt` tại epoch 63; có **5.133/8.062** ảnh đúng. Các chỉ số balanced accuracy và F1 được tính từ confusion matrix của cùng lần đánh giá. Checkpoint được chọn bằng **nhãn ExpW validation**. Kết quả của một seed là thăm dò; không dùng ExpW test để chỉnh tham số hoặc chọn checkpoint cho E1–E5.
