# Neural Network cho Elliptic Curve Analysis

## Mô tả
Chương trình sử dụng Deep Neural Network để phân tích và dự đoán các thuộc tính của đường cong elliptic.

## Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
- File `input.txt` chứa dữ liệu đầu vào với format: `p a b order`
- Trong đó:
  - `p`: prime field size
  - `a, b`: coefficients của elliptic curve y² = x³ + ax + b
  - `order`: thứ tự của điểm trên curve

## Sử dụng

### Chạy chương trình
```bash
python nn88.py
```

### Input
- Nhập tỷ lệ train/test (ví dụ: 0.8 cho 80% train, 20% test)

### Output
- Kết quả được lưu trong file `output_nn.txt`
- Bao gồm: |diff|, p, a, b, ord (thứ tự thực), est (ước lượng)

## Kiến trúc Neural Network
- 7 hidden layers với kiến trúc giảm dần: 512→256→128→64→32→16→8→1
- Activation: ReLU cho hidden layers, Sigmoid cho output
- BatchNormalization và Dropout để tránh overfitting
- Optimizer: Adam với learning rate 0.005
- Loss function: MSLE (Mean Squared Logarithmic Error)

## Dữ liệu mẫu
File `input.txt` đã chứa dữ liệu mẫu với các prime field p=17 và p=23.
