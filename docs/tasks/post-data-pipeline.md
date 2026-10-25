# Task 1. Experimental Setup and Data Preparation

## 1. Objective

Xây dựng data pipeline cho setting:

$$
\text{RAF-DB} \rightarrow \text{ExpW}
$$

trong đó:

- RAF-DB là labeled source domain.
- ExpW là unlabeled target domain trong adaptation.
- Cả hai sử dụng chung 7 expression classes:

$$
\{\text{anger}, \text{disgust}, \text{fear}, \text{happiness}, \text{neutral}, \text{sadness}, \text{surprise}\}.
$$

Output của task phải có thể dùng trực tiếp cho source pretraining, target adaptation và STR ở các bước sau.

## 2. Dataset Preparation

### RAF-DB

- Sử dụng Basic Emotion Subset gồm 7 expressions.
- Giữ official train/test split.
- RAF-DB train được dùng làm labeled source training set.
- Remap label về common 7-class mapping.

### ExpW

- Remap label về cùng 7-class mapping với RAF-DB.
- Nếu có sẵn benchmark split file phù hợp thì sử dụng split đó.
- Nếu không có split file, tự chia train/validation/test bằng fixed random seed và lưu lại split manifest.
- ExpW train được sử dụng làm unlabeled target data trong adaptation.
- ExpW test chỉ được sử dụng để evaluation.
- Ground-truth label của ExpW không được trả cho adaptation training loop.

## 3. Offline Preprocessing

Áp dụng cùng preprocessing convention cho RAF-DB và ExpW:

1. Detect face bằng MTCNN.
2. Lấy face bounding box và 5 facial landmarks.
3. Loại ảnh không detect được face.
4. Loại detected face có resolution nhỏ hơn $20 \times 20$.
5. Crop vùng mặt khỏi ảnh gốc.
6. Lưu processed face và metadata cần thiết.

Không resize cố định về $224 \times 224$ ở bước offline.

Facial landmarks phải được lưu lại để dùng cho STR sau này mà không cần chạy lại MTCNN.

## 4. Online Transforms

### Training

$$
\text{face crop} \rightarrow \text{resize} \rightarrow \text{horizontal flip }(p=0.5) \rightarrow \text{tensor} \rightarrow \text{normalize}.
$$

Với configuration hiện tại:

$$
H = W = 224.
$$

Normalization để configurable.

Nếu sử dụng standard ImageNet-pretrained Swin thì mặc định:

$$
\mu = (0.485, 0.456, 0.406)
$$

$$
\sigma = (0.229, 0.224, 0.225).
$$

### Evaluation

$$
\text{face crop} \rightarrow \text{resize} \rightarrow \text{tensor} \rightarrow \text{normalize}.
$$

Không sử dụng random augmentation khi evaluation.

## 5. Dataset Interface

### Source Training Dataset

RAF-DB train trả về:

$$
(x_s, y_s).
$$

### Target Adaptation Dataset

ExpW train trả về:

$$
(x_t, \text{sample\_id}).
$$

Không trả target ground-truth label.

### Evaluation Dataset

RAF-DB test hoặc ExpW test trả về:

$$
(x, y).
$$

`batch_size` phải được truyền từ training configuration. Cách ghép source-target batch không thuộc task này.

## 6. Required Deliverables

### Preprocessing pipeline

Có thể chạy độc lập cho RAF-DB và ExpW.

### Split manifest

Ghi rõ sample thuộc train, validation hay test.

Nếu ExpW được tự chia:

- sử dụng fixed seed;
- lưu split ra file;
- không random split lại ở mỗi training run.

### Processed-data manifest

Tối thiểu gồm:

- sample ID hoặc image path;
- dataset;
- split;
- canonical label;
- processed image path;
- 5 facial landmarks;
- usable status.

### Dataset/DataLoader implementation

Cung cấp loader cho:

- source training;
- target adaptation;
- evaluation.

### Dataset summary

Report:

- số ảnh ban đầu;
- số ảnh usable;
- số ảnh bị loại;
- class distribution sau preprocessing theo từng split.

## 7. Acceptance Criteria

Task hoàn thành khi:

- RAF-DB và ExpW sử dụng cùng canonical 7-class mapping.
- Dataset splits cố định và reproduce được.
- ExpW adaptation loader không expose target labels.
- Với configuration hiện tại, sample sau online transform có shape:

$$
[3, 224, 224].
$$

- Training và evaluation transforms được tách riêng.
- Facial landmarks có thể truy xuất lại theo sample.
- Dataset có thể được load mà không cần chạy lại MTCNN.
- Có summary số lượng dữ liệu sau preprocessing.
