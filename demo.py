#!/usr/bin/env python3
"""
Script demo sử dụng model Elliptic_AI
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_model_and_scaler():
    """Tải model và scaler"""
    try:
        model = joblib.load('best_elliptic_model.pkl')
        print("Đã tải model thành công")
        return model
    except FileNotFoundError:
        print("Không tìm thấy model, vui lòng chạy model_simple.py trước")
        return None

def predict_tilde_delta(model, p, A, B):
    """Dự đoán tilde_delta cho một bộ tham số"""
    # Chuẩn hóa dữ liệu
    X = np.array([[p, A, B]], dtype=np.float32)
    
    # Dự đoán
    prediction = model.predict(X)[0]
    return prediction

def demo_predictions():
    """Demo một số dự đoán"""
    print("=" * 50)
    print("DEMO DỰ ĐOÁN TILDE_DELTA")
    print("=" * 50)
    
    model = load_model_and_scaler()
    if model is None:
        return
    
    # Một số ví dụ
    examples = [
        (17, 5, 3),
        (23, 7, 11),
        (31, 13, 19),
        (41, 17, 23),
        (53, 29, 31),
        (67, 37, 43),
        (83, 47, 59),
        (97, 61, 71),
        (113, 73, 89),
        (127, 79, 101)
    ]
    
    print(f"{'p':<5} {'A':<5} {'B':<5} {'Predicted tilde_delta':<20}")
    print("-" * 40)
    
    predictions = []
    for p, A, B in examples:
        pred = predict_tilde_delta(model, p, A, B)
        predictions.append(pred)
        print(f"{p:<5} {A:<5} {B:<5} {pred:<20.6f}")
    
    print(f"\nPhạm vi dự đoán: [{min(predictions):.6f}, {max(predictions):.6f}]")
    print(f"Trung bình: {np.mean(predictions):.6f}")

def interactive_prediction():
    """Chế độ tương tác để người dùng nhập tham số"""
    print("\n" + "=" * 50)
    print("CHẾ ĐỘ TƯƠNG TÁC")
    print("=" * 50)
    
    model = load_model_and_scaler()
    if model is None:
        return
    
    print("Nhập các tham số để dự đoán tilde_delta")
    print("Nhập 'quit' để thoát")
    
    while True:
        try:
            user_input = input("\nNhập p, A, B (cách nhau bởi dấu phẩy): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            p, A, B = map(int, user_input.split(','))
            
            # Kiểm tra điều kiện
            if p <= 2:
                print("Lỗi: p phải là số nguyên tố > 2")
                continue
                
            if A < 0 or A >= p:
                print(f"Lỗi: A phải trong khoảng [0, {p-1}]")
                continue
                
            if B < 0 or B >= p:
                print(f"Lỗi: B phải trong khoảng [0, {p-1}]")
                continue
            
            # Kiểm tra điều kiện không suy biến
            if (4*A**3 + 27*B**2) % p == 0:
                print("Lỗi: 4A³ + 27B² ≡ 0 (mod p) - đường cong suy biến")
                continue
            
            # Dự đoán
            prediction = predict_tilde_delta(model, p, A, B)
            print(f"Dự đoán tilde_delta: {prediction:.6f}")
            
        except ValueError:
            print("Lỗi: Vui lòng nhập 3 số nguyên cách nhau bởi dấu phẩy")
        except KeyboardInterrupt:
            break
    
    print("Cảm ơn bạn đã sử dụng!")

def analyze_model_performance():
    """Phân tích hiệu suất model"""
    print("\n" + "=" * 50)
    print("PHÂN TÍCH HIỆU SUẤT MODEL")
    print("=" * 50)
    
    try:
        # Tải dữ liệu test
        X_test = np.load('elliptic_data_X.npy')
        y_test = np.load('elliptic_data_y.npy')
        
        model = load_model_and_scaler()
        if model is None:
            return
        
        # Dự đoán trên toàn bộ dữ liệu
        y_pred = model.predict(X_test)
        
        # Tính metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"R² Score: {r2:.6f} ({r2*100:.2f}%)")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MSE: {mse:.6f}")
        
        # Phân tích lỗi
        residuals = y_test - y_pred
        print(f"\nPhân tích lỗi:")
        print(f"  Lỗi trung bình: {np.mean(residuals):.6f}")
        print(f"  Độ lệch chuẩn lỗi: {np.std(residuals):.6f}")
        print(f"  Lỗi tuyệt đối trung bình: {np.mean(np.abs(residuals)):.6f}")
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6, s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Giá trị thực tế')
        plt.ylabel('Giá trị dự đoán')
        plt.title(f'Dự đoán vs Thực tế\nR² = {r2:.4f}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Giá trị dự đoán')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Tần suất')
        plt.title('Phân phối Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.hist(y_test, bins=30, alpha=0.7, label='Thực tế', edgecolor='black')
        plt.hist(y_pred, bins=30, alpha=0.7, label='Dự đoán', edgecolor='black')
        plt.xlabel('tilde_delta')
        plt.ylabel('Tần suất')
        plt.title('So sánh phân phối')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print("Không tìm thấy dữ liệu test")

def main():
    print("ELLIPTIC_AI - DEMO")
    print("=" * 50)
    
    while True:
        print("\nChọn chức năng:")
        print("1. Demo dự đoán")
        print("2. Chế độ tương tác")
        print("3. Phân tích hiệu suất")
        print("4. Thoát")
        
        choice = input("\nNhập lựa chọn (1-4): ").strip()
        
        if choice == '1':
            demo_predictions()
        elif choice == '2':
            interactive_prediction()
        elif choice == '3':
            analyze_model_performance()
        elif choice == '4':
            print("Cảm ơn bạn đã sử dụng Elliptic_AI!")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại")

if __name__ == "__main__":
    main() 