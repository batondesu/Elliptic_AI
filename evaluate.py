#!/usr/bin/env python3
"""
Script đánh giá model Elliptic_AI
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data_and_model():
    """Tải dữ liệu và model đã huấn luyện"""
    try:
        # Tải dữ liệu
        X = np.load('elliptic_data_X.npy')
        y = np.load('elliptic_data_y.npy')
        print(f"Đã tải {len(X)} mẫu dữ liệu")
        
        # Tải model
        model = load_model('elliptic_model.h5')
        print("Đã tải model thành công")
        
        return X, y, model
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file {e}")
        return None, None, None

def evaluate_model(X, y, model):
    """Đánh giá model chi tiết"""
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ MODEL")
    print("=" * 50)
    
    # Dự đoán
    y_pred = model.predict(X)
    
    # Tính các metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Tính RMSE
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    # Phân tích lỗi
    residuals = y - y_pred.flatten()
    print(f"\nPhân tích lỗi:")
    print(f"  Lỗi trung bình: {np.mean(residuals):.6f}")
    print(f"  Độ lệch chuẩn lỗi: {np.std(residuals):.6f}")
    print(f"  Lỗi tuyệt đối trung bình: {np.mean(np.abs(residuals)):.6f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'residuals': residuals
    }

def plot_comprehensive_analysis(y_true, y_pred, residuals):
    """Vẽ biểu đồ phân tích toàn diện"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scatter plot dự đoán vs thực tế
    plt.subplot(3, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('Dự đoán vs Thực tế')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    plt.subplot(3, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # 3. Histogram residuals
    plt.subplot(3, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Tần suất')
    plt.title('Phân phối Residuals')
    plt.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    plt.subplot(3, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal Distribution)')
    plt.grid(True, alpha=0.3)
    
    # 5. Distribution comparison
    plt.subplot(3, 3, 5)
    plt.hist(y_true, bins=30, alpha=0.7, label='Thực tế', edgecolor='black')
    plt.hist(y_pred.flatten(), bins=30, alpha=0.7, label='Dự đoán', edgecolor='black')
    plt.xlabel('tilde_delta')
    plt.ylabel('Tần suất')
    plt.title('So sánh phân phối')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Error vs predicted value
    plt.subplot(3, 3, 6)
    plt.scatter(y_pred, np.abs(residuals), alpha=0.6, s=20)
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Lỗi tuyệt đối')
    plt.title('Lỗi tuyệt đối vs Dự đoán')
    plt.grid(True, alpha=0.3)
    
    # 7. Box plot residuals
    plt.subplot(3, 3, 7)
    plt.boxplot(residuals)
    plt.ylabel('Residuals')
    plt.title('Box Plot Residuals')
    plt.grid(True, alpha=0.3)
    
    # 8. Cumulative distribution
    plt.subplot(3, 3, 8)
    sorted_residuals = np.sort(residuals)
    cumulative = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
    plt.plot(sorted_residuals, cumulative)
    plt.xlabel('Residuals')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # 9. Error distribution by magnitude
    plt.subplot(3, 3, 9)
    error_magnitude = np.abs(residuals)
    plt.hist(error_magnitude, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Lỗi tuyệt đối')
    plt.ylabel('Tần suất')
    plt.title('Phân phối lỗi tuyệt đối')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_by_features(X, y, y_pred):
    """Phân tích lỗi theo từng đặc trưng"""
    print("\n" + "=" * 50)
    print("PHÂN TÍCH THEO ĐẶC TRƯNG")
    print("=" * 50)
    
    feature_names = ['p', 'A', 'B']
    residuals = y - y_pred.flatten()
    
    for i, feature in enumerate(feature_names):
        print(f"\nĐặc trưng {feature}:")
        print(f"  Phạm vi: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
        print(f"  Trung bình: {X[:, i].mean():.2f}")
        print(f"  Độ lệch chuẩn: {X[:, i].std():.2f}")
        
        # Tính correlation với residuals
        corr = np.corrcoef(X[:, i], residuals)[0, 1]
        print(f"  Correlation với residuals: {corr:.4f}")
        
        # Phân tích lỗi theo nhóm
        if feature == 'p':
            # Nhóm theo p
            p_groups = []
            p_errors = []
            for p_val in sorted(set(X[:, i].astype(int))):
                mask = X[:, i].astype(int) == p_val
                if np.sum(mask) > 5:  # Chỉ xét nhóm có ít nhất 5 mẫu
                    p_groups.append(p_val)
                    p_errors.append(np.mean(np.abs(residuals[mask])))
            
            if len(p_groups) > 1:
                print(f"  Lỗi trung bình theo p:")
                for p_val, error in zip(p_groups[:5], p_errors[:5]):  # Chỉ hiển thị 5 nhóm đầu
                    print(f"    p={p_val}: {error:.6f}")
                if len(p_groups) > 5:
                    print(f"    ... và {len(p_groups)-5} nhóm khác")

def generate_report(results):
    """Tạo báo cáo tổng hợp"""
    print("\n" + "=" * 50)
    print("BÁO CÁO TỔNG HỢP")
    print("=" * 50)
    
    print(f"Hiệu suất model:")
    print(f"  R² Score: {results['r2']:.4f} ({results['r2']*100:.1f}%)")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE: {results['mae']:.6f}")
    
    # Đánh giá chất lượng
    if results['r2'] > 0.9:
        quality = "Xuất sắc"
    elif results['r2'] > 0.8:
        quality = "Tốt"
    elif results['r2'] > 0.7:
        quality = "Khá"
    elif results['r2'] > 0.5:
        quality = "Trung bình"
    else:
        quality = "Cần cải thiện"
    
    print(f"\nĐánh giá chất lượng: {quality}")
    
    # Khuyến nghị
    print(f"\nKhuyến nghị:")
    if results['r2'] < 0.8:
        print("  - Cần thu thập thêm dữ liệu")
        print("  - Thử nghiệm kiến trúc model khác")
        print("  - Tối ưu hóa hyperparameters")
    else:
        print("  - Model có hiệu suất tốt")
        print("  - Có thể sử dụng cho dự đoán thực tế")

def main():
    print("ELLIPTIC_AI - ĐÁNH GIÁ MODEL")
    print("=" * 50)
    
    # Tải dữ liệu và model
    X, y, model = load_data_and_model()
    if X is None:
        return
    
    # Đánh giá model
    results = evaluate_model(X, y, model)
    
    # Phân tích theo đặc trưng
    analyze_by_features(X, y, results['y_pred'])
    
    # Vẽ biểu đồ phân tích toàn diện
    print("\nĐang tạo biểu đồ phân tích...")
    plot_comprehensive_analysis(y, results['y_pred'], results['residuals'])
    
    # Tạo báo cáo
    generate_report(results)
    
    print("\nHoàn thành đánh giá!")

if __name__ == "__main__":
    main() 