# AI-Enhanced Schoof Algorithm

## 🎯 Mục Tiêu Dự Án

Sử dụng Deep Learning để **tăng tốc thuật toán Schoof** tính số điểm trên đường cong elliptic bằng cách:
1. Dự đoán **Trace Frobenius** từ tham số đường cong (p, a, b)
2. Thu hẹp **khoảng Hasse** từ [-2√p, 2√p] xuống khoảng nhỏ hơn
3. Giảm **số lượng số nguyên tố ℓ** cần tính
4. **Early termination** khi đã đủ confidence

## ✅ Kết Quả Chính

### Model AI
- **Architecture**: Deep Neural Network (7 layers: 512→256→128→64→32→16→8→1)
- **Input**: (p, a, b)
- **Output**: Trace Frobenius (normalized)

## 🚀 Sử Dụng
```bash
python3 ai_training.py
```

