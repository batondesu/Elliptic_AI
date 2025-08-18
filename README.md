# Elliptic_AI - AI-Enhanced Schoof Algorithm

## Mô tả

Elliptic_AI là một hệ thống AI tiên tiến để phân tích đường cong elliptic, triển khai AI-enhanced Schoof's Algorithm với deep neural networks để dự đoán δ (delta) và thu hẹp khoảng Hasse, giảm thời gian đếm điểm trên đường cong elliptic y² = x³ + Ax + B (mod p).

## Tính năng chính

### AI-Enhanced Schoof 
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

### Lệnh demo:
- `demo` - Chạy demo tự động
- `quit` - Thoát
