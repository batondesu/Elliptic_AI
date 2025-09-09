# Elliptic_AI - AI-Enhanced Schoof Algorithm

## Mô tả

Elliptic_AI là một hệ thống AI tiên tiến để phân tích đường cong elliptic, triển khai AI-enhanced Schoof's Algorithm với deep neural networks để dự đoán δ (delta) và thu hẹp khoảng Hasse, giảm thời gian đếm điểm trên đường cong elliptic y² = x³ + Ax + B (mod p).

## Cài đặt

### Cài đặt dependencies
```bash
pip3 install -r requirements.txt
```

## Sử dụng nhanh

### 1. Sinh dataset chuẩn (nếu cần)
```bash
python3 generate_more_data.py
```

### 2. Huấn luyện AI-enhanced Schoof v2.0
```bash
python3 ai_enhanced_schoof_v2.py
```

### 3. Demo tương tác
```bash
python3 demo_schoof_v2.py
```

### Lệnh demo:
- `demo` - Chạy demo tự động
- `quit` - Thoát
