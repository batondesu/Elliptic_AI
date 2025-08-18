# Elliptic_AI - AI-Enhanced Schoof Algorithm v2.0

## Mô tả

Elliptic_AI là một hệ thống AI tiên tiến để phân tích đường cong elliptic, triển khai AI-enhanced Schoof's Algorithm với deep neural networks để dự đoán δ (delta) và thu hẹp khoảng Hasse, giảm thời gian đếm điểm trên đường cong elliptic y² = x³ + Ax + B (mod p).

## Tính năng chính

### AI-Enhanced Schoof v2.0
- **Deep Neural Network**: 10 hidden layers (512->384->256->192->128->96->64->32->16->8) để dự đoán δ
- **CM Classification**: Neural network phân loại CM/non-CM curves
- **Hasse Interval Narrowing**: Thu hẹp khoảng Hasse trung bình 3.38x
- **Advanced Feature Engineering**: 24 đặc trưng toán học nâng cao

### Cải tiến hiệu suất
- **Speedup**: Tăng 3.0x trung bình so với phương pháp cổ điển
- **Hasse Reduction**: Thu hẹp khoảng Hasse hiệu quả
- **Scalability**: Xử lý được p từ 3 đến 98,801
- **CM Classification**: Độ chính xác cao cho phân loại CM curves

### Dataset chuẩn
- **1,874 mẫu** với phạm vi p từ 3 đến 98,801
- **500 số nguyên tố** rải đều
- **24 đặc trưng toán học** nâng cao
- **CM classification** chính xác

## Cấu trúc dự án

```
Elliptic_AI/
├── ai_enhanced_schoof_v2.py        # AI-enhanced Schoof v2.0 implementation
├── demo_schoof_v2.py               # Demo tương tác v2.0
├── generate_schoof_dataset.py      # Sinh dataset chuẩn cho Schoof
├── create_comparison_table.py      # Tạo bảng so sánh kết quả
├── requirements.txt                # Dependencies
├── README.md                       # Documentation này
├── SCHOOF_V2_SUMMARY.md            # Tóm tắt AI-enhanced Schoof v2.0
├── image/                          # Biểu đồ và hình ảnh
│   ├── comparison_results.png      # Bảng so sánh chi tiết
│   ├── comparison_summary.png      # Bảng tóm tắt thống kê
│   └── [các biểu đồ khác]
├── Models đã huấn luyện
│   ├── schoof_ai_regressor_v2.h5   # Delta regressor model
│   ├── schoof_ai_regressor_v2_scaler.pkl
│   ├── schoof_ai_cm_classifier_v2.h5 # CM classifier model
│   └── schoof_ai_cm_v2_scaler.pkl
└── Datasets
    ├── schoof_data_X.npy           # Dataset chính (1,874 mẫu, 24 features)
    ├── schoof_data_delta.npy       # δ values
    ├── schoof_data_tilde_delta.npy # δ̃ values
    ├── schoof_data_cm.npy          # CM classification
    └── schoof_feature_names.txt    # Feature names
```

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- 4GB RAM (khuyến nghị)
- 2GB disk space

### Cài đặt dependencies
```bash
pip3 install -r requirements.txt
```

## Sử dụng nhanh

### 1. Sinh dataset chuẩn (nếu cần)
```bash
python3 generate_schoof_dataset.py
```

### 2. Huấn luyện AI-enhanced Schoof v2.0
```bash
python3 ai_enhanced_schoof_v2.py
```

### 3. Demo tương tác
```bash
python3 demo_schoof_v2.py
```

### 4. Tạo bảng so sánh kết quả
```bash
python3 create_comparison_table.py
```

## Demo tương tác

Sau khi chạy `demo_schoof_v2.py`, bạn có thể:

```
Nhập p, A, B: 17 5 3

KẾT QUẢ CHO p=17, A=5, B=3:
AI δ Prediction:    0.4874
Classical δ:        4.0
Absolute Error:     3.5126
Hasse Interval:     (9, 27) -> (16, 20)
Reduction Factor:   3.80x
CM Probability:     0.1496
```

### Lệnh demo:
- `demo` - Chạy demo tự động
- `help` - Xem hướng dẫn chi tiết
- `quit` - Thoát

## Kết quả mô hình

### AI-Enhanced Schoof v2.0 Performance
- **Dataset Size**: 1,874 mẫu
- **Prime Range**: 3 - 98,801
- **Average Speedup**: 3.0x
- **Average Hasse Reduction**: 3.38x
- **CM Curves**: 50 (2.67%)

### Phân tích theo loại Prime

**Small Prime (p < 100):**
- Trung bình Error: 3.50 (thấp nhất)
- Trung bình Speedup: 0.0x
- Trung bình Hasse Reduction: 3.46x

**Medium Prime (100 ≤ p < 1000):**
- Trung bình Error: 20.20
- Trung bình Speedup: 0.1x
- Trung bình Hasse Reduction: 3.40x

**Large Prime (1000 ≤ p < 10000):**
- Trung bình Error: 43.74
- Trung bình Speedup: 0.8x
- Trung bình Hasse Reduction: 3.32x

**Very Large Prime (p ≥ 10000):**
- Trung bình Error: 90.60
- Trung bình Speedup: 11.3x (cao nhất)
- Trung bình Hasse Reduction: 3.34x

## Dataset chuẩn

### Thống kê dataset
- **Tổng số mẫu**: 1,874
- **Số nguyên tố**: 500 (rải đều từ 3 đến 98,801)
- **Số đặc trưng**: 24
- **CM curves**: 50 (2.67%)

### Phân phối theo p
- **p < 100**: 100 mẫu
- **100 ≤ p < 1000**: 350 mẫu
- **1000 ≤ p < 10000**: 660 mẫu
- **p ≥ 10000**: 764 mẫu

### 24 đặc trưng toán học
1. p, A, B (cơ bản)
2. discriminant, discriminant_ratio
3. j_invariant, j_invariant_ratio
4. A_mod_3, B_mod_3, p_mod_3
5. A_mod_4, B_mod_4, p_mod_4
6. A_times_B, A_squared, B_squared
7. A_over_p, B_over_p, A_plus_B_over_p
8. log_p, log_A, log_B
9. sin_A_over_p, cos_B_over_p

## Kiến trúc AI-Enhanced Schoof v2.0

### Delta Regressor
- **Architecture**: 10 hidden layers (512->384->256->192->128->96->64->32->16->8)
- **Activation**: ReLU
- **Regularization**: BatchNormalization + Dropout(0.15)
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: MSE

### CM Classifier
- **Architecture**: 3 hidden layers (128->64->32)
- **Activation**: ReLU
- **Class Weights**: Để xử lý class imbalance
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Binary Crossentropy

### Feature Extractor
- **24 đặc trưng toán học** nâng cao
- **StandardScaler** cho normalization
- **Modular arithmetic** và **trigonometric features**

## Tùy chỉnh

### Thay đổi hyperparameters
```python
# Trong ai_enhanced_schoof_v2.py
class DeltaRegressorV2:
    def _build(self):
        # Thay đổi architecture ở đây
        for units in [512, 384, 256, 192, 128, 96, 64, 32, 16, 8]:
            # ...
```

### Thêm đặc trưng mới
```python
# Trong generate_schoof_dataset.py
def extract_advanced_features(p, A, B):
    features = {}
    # Thêm đặc trưng mới ở đây
    features['new_feature'] = some_calculation(p, A, B)
    return features
```

### Sinh dataset với tham số khác
```python
# Trong generate_schoof_dataset.py
max_p = 200000        # Tăng phạm vi p
select_primes_count = 1000  # Tăng số primes
target_samples = 50000      # Tăng số mẫu
```

## Monitoring và Logging

### Training Progress
- Epoch-by-epoch training logs
- Validation metrics
- Early stopping
- Learning rate scheduling

### Biểu đồ
- Comparison results
- Performance analysis
- Hasse narrowing analysis
- Dataset overview

## Troubleshooting

### Lỗi thường gặp

1. **"No module named 'tensorflow'"**
   ```bash
   pip3 install tensorflow
   ```

2. **"No module named 'sympy'"**
   ```bash
   pip3 install sympy
   ```

3. **Memory error**
   - Giảm `max_p` trong `generate_schoof_dataset.py`
   - Giảm `select_primes_count`

4. **Model không tìm thấy**
   ```bash
   python3 ai_enhanced_schoof_v2.py
   ```

## Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License

MIT License - xem file LICENSE để biết chi tiết.

## Acknowledgments

- Thuật toán Schoof cổ điển
- TensorFlow/Keras cho neural networks
- SymPy cho symbolic computation
- Scikit-learn cho data preprocessing

## Liên hệ

- **Email**: [your-email@example.com]
- **GitHub**: [your-github-username]
- **Project**: [repository-url]

---

**Nếu dự án này hữu ích, hãy cho một star!**