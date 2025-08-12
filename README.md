# Elliptic_AI

## Mô tả dự án
Dự án này nghiên cứu và huấn luyện mô hình AI để dự đoán các tính chất của đường cong elliptic trên trường hữu hạn.

## Đường cong Elliptic
Đường cong elliptic có dạng: y² = x³ + Ax + B (mod p)
- p: số nguyên tố (kích thước trường)
- A, B: các tham số của đường cong
- Điều kiện: 4A³ + 27B² ≠ 0 (mod p) để tránh suy biến

## Mục tiêu
Dự đoán giá trị tilde_delta = (p + 1 - N) / (2√p)
- N: số điểm trên đường cong elliptic
- tilde_delta: đại lượng liên quan đến phân phối điểm

## Cấu trúc dự án
- `generate_data.py`: Sinh dữ liệu huấn luyện
- `model.py`: Định nghĩa và huấn luyện neural network
- `train.py`: Script huấn luyện chính
- `evaluate.py`: Đánh giá model
- `requirements.txt`: Các thư viện cần thiết

## Cài đặt và chạy

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ quá trình
```bash
# 1. Sinh dữ liệu
python generate_data_simple.py

# 2. Huấn luyện model
python model_simple.py

# 3. Demo sử dụng
python demo.py
```

### Chạy từng bước
```bash
# Chỉ sinh dữ liệu
python generate_data_simple.py

# Chỉ huấn luyện model
python model_simple.py

# Chỉ demo
python demo.py
```

## Kết quả
Dự án sử dụng nhiều model khác nhau:
- Linear Regression
- Ridge Regression  
- Lasso Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)

Model tốt nhất được lưu vào `best_elliptic_model.pkl`

## Cấu trúc file
- `generate_data_simple.py`: Sinh dữ liệu elliptic curve
- `model_simple.py`: Huấn luyện và so sánh các model
- `demo.py`: Demo sử dụng model
- `elliptic_data_X.npy`: Dữ liệu đầu vào
- `elliptic_data_y.npy`: Dữ liệu đầu ra
- `best_elliptic_model.pkl`: Model tốt nhất
- `model_comparison.png`: Biểu đồ so sánh model
- `predictions_comparison.png`: Biểu đồ dự đoán