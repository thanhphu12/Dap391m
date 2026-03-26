# Heart Disease Risk Prediction

## Logistic Regression vs SVC with Threshold Analysis and Interpretability

Đây là file README mô tả notebook:

**`[Topic_11]_HEART_DISEASE_Nhom5_LeNhatTruong_TranThanhPhu.ipynb`**

Project này xây dựng một pipeline machine learning để **dự đoán nguy cơ bệnh tim** từ dữ liệu lâm sàng, đồng thời phân tích:
- hiệu năng mô hình
- ngưỡng dự đoán (threshold)
- độ calibration của xác suất
- khả năng giải thích của Logistic Regression

---

## 1. Mục tiêu của bài toán

Bài toán gốc có nhãn mức độ bệnh tim từ `0–4`.  
Trong notebook, nhãn được chuyển về **bài toán phân loại nhị phân**:

- `0` = không mắc bệnh tim
- `1` = có bệnh tim (`num > 0`)

Điều này giúp:
- đơn giản hóa bài toán
- phù hợp với mục tiêu phát hiện nguy cơ bệnh
- dễ so sánh giữa các mô hình cổ điển

---

## 2. Dataset sử dụng

Notebook đọc dữ liệu từ file:

```python
DATA_PATH = "data.csv"
```

Bạn cần đặt file CSV cùng thư mục với notebook, hoặc sửa lại `DATA_PATH` cho đúng đường dẫn trên máy.

### Nhóm đặc trưng sử dụng

**Biến số (Numerical features):**
- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

**Biến phân loại (Categorical features):**
- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

---

## 3. Các bước xử lý dữ liệu

Notebook có cả phần **kiểm tra dữ liệu**, **EDA**, và **preprocessing chính thức** cho training.

### 3.1. Kiểm tra và làm sạch dữ liệu
- đọc file CSV
- thay giá trị `"?"` thành `NaN`
- phân tích missing values trước xử lý
- tạo bản dữ liệu sạch tạm thời để phục vụ EDA

### 3.2. Chia tập dữ liệu
- train/test split theo tỷ lệ **80/20**
- dùng `stratify=y` để giữ cân bằng tỷ lệ lớp

### 3.3. Preprocessing pipeline
Pipeline chính thức dùng trong training gồm:

**Numerical pipeline**
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

**Categorical pipeline**
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Lý do dùng pipeline:
- tránh data leakage
- xử lý thiếu dữ liệu đúng quy trình
- chuẩn hóa dữ liệu cho mô hình
- mã hóa biến phân loại tự động

---

## 4. Các mô hình được so sánh

Notebook so sánh 5 mô hình:

1. **Majority Baseline**  
   Luôn dự đoán theo lớp xuất hiện nhiều nhất.

2. **Logistic Regression L2**

3. **Logistic Regression L1**

4. **SVC Linear**

5. **SVC RBF**

---

## 5. Nội dung chính trong notebook

Notebook được viết theo kiểu **paper/report**, gồm các phần sau:

### 5.1. EDA
- missing values trước và sau fix
- class balance
- boxplot
- histogram
- KDE
- countplot cho biến phân loại
- correlation heatmap
- phân bố theo target
- pairplot

### 5.2. So sánh mô hình
Đánh giá mô hình bằng:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision

### 5.3. Threshold Analysis
Phân tích sự thay đổi của:
- precision
- recall
- F1-score

theo các threshold khác nhau.

Notebook còn có:
- bảng threshold tuning tại các mốc `0.3, 0.4, 0.5, 0.6, 0.7`
- confusion matrix theo threshold
- cost function để chọn threshold phù hợp cho bài toán y tế

### 5.4. Calibration Analysis
Ngoài biểu đồ calibration, notebook còn tính:
- **Brier Score**
- **Log Loss**
- **ECE (Expected Calibration Error)**

Đồng thời có:
- reliability table theo từng bin
- **Platt Scaling**
- **Isotonic Regression**

### 5.5. Interpretability
Với Logistic Regression, notebook trích xuất:
- hệ số `coef`
- dấu dương / âm của hệ số
- odds ratio thông qua `exp(coef)`

Điều này giúp giải thích yếu tố nào làm tăng hoặc giảm nguy cơ bệnh tim.

### 5.6. Reproducibility
Notebook có thêm:
- bootstrap 95% CI cho ROC-AUC
- export bảng kết quả ra Excel
- phần checklist tái lập kết quả

---

## 6. Kết quả nổi bật

### 6.1. Test results
| Model | Accuracy | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| LogReg_L2 | 0.885246 | 0.966450 | 0.963435 | 0.881356 | 0.838710 | 0.928571 |
| SVC_RBF | 0.885246 | 0.964286 | 0.955836 | 0.881356 | 0.838710 | 0.928571 |
| SVC_Linear | 0.852459 | 0.961039 | 0.957194 | 0.852459 | 0.787879 | 0.928571 |
| LogReg_L1 | 0.868852 | 0.958874 | 0.954618 | 0.862069 | 0.833333 | 0.892857 |
| Majority | 0.540984 | 0.500000 | 0.459016 | 0.000000 | 0.000000 | 0.000000 |

### 6.2. Mô hình tốt nhất
Theo notebook, mô hình tốt nhất là:

**Logistic Regression L2**
- ROC-AUC = **0.966450**
- PR-AUC = **0.963435**
- F1 = **0.881356**
- Recall = **0.928571**

Điều này cho thấy Logistic Regression L2 đạt hiệu năng rất mạnh, đồng thời còn có lợi thế về khả năng giải thích.

### 6.3. Threshold tuning
Bảng threshold tuning của `LogReg_L2`:

| Threshold | Precision | Recall | F1 | FP | FN | Cost |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.729730 | 0.964286 | 0.830769 | 10 | 1 | 15 |
| 0.4 | 0.818182 | 0.964286 | 0.885246 | 6 | 1 | 11 |
| 0.5 | 0.838710 | 0.928571 | 0.881356 | 5 | 2 | 15 |
| 0.6 | 0.862069 | 0.892857 | 0.877193 | 4 | 3 | 19 |
| 0.7 | 0.923077 | 0.857143 | 0.888889 | 2 | 4 | 22 |

Nhận xét:
- threshold thấp → recall cao hơn
- threshold cao → precision cao hơn
- threshold `0.4` cho cost tốt trong bảng hiện tại
- threshold `0.5` là điểm cân bằng khá tốt giữa precision và recall

### 6.4. Calibration metrics
| Model | Brier Score | Log Loss | ECE |
|---|---:|---:|---:|
| LogReg_L2 | 0.079737 | 0.259499 | 0.082536 |
| LogReg_L1 | 0.083293 | 0.272178 | 0.097331 |
| SVC_RBF | 0.086401 | 0.299525 | 0.108663 |
| SVC_Linear | 0.089112 | 0.298365 | 0.108011 |
| Majority | 0.459016 | 16.544628 | 0.459016 |

Kết quả cho thấy `LogReg_L2` không chỉ mạnh về phân loại mà còn có calibration tốt nhất trong nhóm mô hình chính.

### 6.5. Bootstrap 95% CI for AUC
| Model | AUC Bootstrap Mean | AUC 95% CI Lower | AUC 95% CI Upper |
|---|---:|---:|---:|
| LogReg_L2 | 0.966139 | 0.923077 | 0.994444 |
| SVC_RBF | 0.963848 | 0.912902 | 0.995699 |
| SVC_Linear | 0.960644 | 0.911762 | 0.992376 |
| LogReg_L1 | 0.958471 | 0.910981 | 0.989515 |
| Majority | 0.500000 | 0.500000 | 0.500000 |

---

## 7. Cấu trúc pipeline tổng quan

```text
Raw Data
   │
   ├── Replace '?' -> NaN
   │
   ├── Target Transformation
   │      num = 0   -> y = 0
   │      num > 0   -> y = 1
   │
   ├── Preprocessing
   │      ├── Numeric   -> Median Imputation -> StandardScaler
   │      └── Categorical -> Most Frequent Imputation -> OneHotEncoder
   │
   ├── Models
   │      ├── Majority Class
   │      ├── Logistic Regression (L1 / L2)
   │      └── SVC (Linear / RBF)
   │
   ├── Research Question 1
   │      Compare ROC-AUC / PR-AUC / F1
   │
   ├── Research Question 2
   │      Threshold trade-off
   │
   ├── Research Question 3
   │      Calibration check
   │
   └── Final Recommendation
```

---

## 8. Cách chạy notebook

### Bước 1. Cài thư viện
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

### Bước 2. Chuẩn bị dữ liệu
- đặt file dữ liệu thành `data.csv`
- hoặc sửa lại biến `DATA_PATH` trong notebook

### Bước 3. Chạy notebook
Mở Jupyter Notebook hoặc VS Code rồi chạy lần lượt từng cell.

---

## 9. Output của notebook

Sau khi chạy, notebook có thể tạo ra:
- biểu đồ EDA
- ROC curve
- Precision-Recall curve
- calibration plot
- confusion matrix
- bảng kết quả test
- bảng threshold tuning
- bảng calibration metrics
- bảng bootstrap CI
- file Excel tổng hợp kết quả
- thư mục hình như `figures/`

---

## 10. Ý nghĩa thực tế

Project này có thể dùng làm nền tảng cho:
- hệ thống hỗ trợ sàng lọc nguy cơ bệnh tim
- công cụ minh họa trong học tập và nghiên cứu ML y tế
- mô hình demo cho dashboard dự đoán sức khỏe
- bài báo cáo / paper so sánh mô hình cổ điển

**Lưu ý:** đây là mô hình học máy phục vụ học tập và nghiên cứu, **không thay thế chẩn đoán y khoa thực tế**.

---

## 11. Điểm mạnh của notebook

- có đầy đủ từ EDA đến modeling
- có baseline để so sánh
- có threshold analysis
- có calibration analysis
- có interpretability
- có bootstrap confidence interval
- có reproducibility và export kết quả
- phù hợp để viết báo cáo, làm slide và thuyết trình

---

## 12. Gợi ý cải tiến thêm

Nếu muốn nâng cấp project, có thể bổ sung:
- GridSearchCV / Optuna
- SHAP cho interpretability nâng cao
- external validation trên tập dữ liệu khác
- web app nhập feature để dự đoán
- lưu model bằng `joblib`
- triển khai bằng Streamlit hoặc Flask

---

## 13. File chính

- Notebook chính: **`[Topic_11]_HEART_DISEASE_Nhom5_LeNhatTruong_TranThanhPhu.ipynb`**

---

## 14. Tóm tắt ngắn

Đây là một notebook hoàn chỉnh về **dự đoán bệnh tim bằng machine learning**, tập trung vào:
- Logistic Regression
- SVC
- threshold optimization
- calibration reliability
- interpretability
- reproducibility

Mô hình nổi bật nhất trong notebook là **Logistic Regression L2**.

