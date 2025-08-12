#!/usr/bin/env python3
"""
Script huấn luyện chính cho Elliptic_AI
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from model import train_model, plot_training_history, plot_predictions
from generate_data import generate_elliptic_data

def main():
    print("=" * 60)
    print("ELLIPTIC_AI - HUẤN LUYỆN MODEL")
    print("=" * 60)
    
    # Bước 1: Sinh dữ liệu
    print("\n1. SINH DỮ LIỆU")
    print("-" * 30)
    
    # Kiểm tra xem có dữ liệu đã lưu chưa
    if os.path.exists('elliptic_data_X.npy') and os.path.exists('elliptic_data_y.npy'):
        print("Tìm thấy dữ liệu đã lưu, đang tải...")
        X = np.load('elliptic_data_X.npy')
        y = np.load('elliptic_data_y.npy')
        print(f"Đã tải {len(X)} mẫu dữ liệu")
    else:
        print("Không tìm thấy dữ liệu, đang sinh dữ liệu mới...")
        X, y = generate_elliptic_data(max_p=300, samples_per_p=15)
        
        # Lưu dữ liệu
        np.save('elliptic_data_X.npy', X)
        np.save('elliptic_data_y.npy', y)
        print("Đã lưu dữ liệu")
    
    # Bước 2: Huấn luyện model
    print("\n2. HUẤN LUYỆN MODEL")
    print("-" * 30)
    
    start_time = time.time()
    model, history, data = train_model(X, y, epochs=100, patience=20)
    training_time = time.time() - start_time
    
    print(f"\nThời gian huấn luyện: {training_time:.2f} giây")
    
    # Bước 3: Vẽ biểu đồ
    print("\n3. TẠO BIỂU ĐỒ")
    print("-" * 30)
    
    print("Đang vẽ biểu đồ quá trình huấn luyện...")
    plot_training_history(history)
    
    print("Đang vẽ biểu đồ phân tích dự đoán...")
    plot_predictions(model, data[2], data[5], data[6])
    
    # Bước 4: Lưu model
    print("\n4. LƯU MODEL")
    print("-" * 30)
    
    model.save('elliptic_model.h5')
    print("Đã lưu model vào elliptic_model.h5")
    
    # Bước 5: Tóm tắt kết quả
    print("\n5. TÓM TẮT KẾT QUẢ")
    print("-" * 30)
    
    # Đánh giá cuối cùng
    X_test = data[2]
    y_test = data[5]
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Số mẫu huấn luyện: {len(data[0])}")
    print(f"Số mẫu validation: {len(data[1])}")
    print(f"Số mẫu test: {len(data[2])}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    # Phạm vi giá trị
    print(f"\nPhạm vi giá trị:")
    print(f"  Thực tế: [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  Dự đoán: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH HUẤN LUYỆN!")
    print("=" * 60)
    print("\nCác file đã tạo:")
    print("- elliptic_model.h5: Model đã huấn luyện")
    print("- elliptic_data_X.npy: Dữ liệu đầu vào")
    print("- elliptic_data_y.npy: Dữ liệu đầu ra")
    print("- training_history.png: Biểu đồ quá trình huấn luyện")
    print("- predictions_analysis.png: Phân tích dự đoán")

if __name__ == "__main__":
    main() 