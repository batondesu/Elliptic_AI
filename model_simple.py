import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

def load_data():
    """Tải dữ liệu đã sinh"""
    try:
        X = np.load('elliptic_data_X.npy')
        y = np.load('elliptic_data_y.npy')
        print(f"Đã tải {len(X)} mẫu dữ liệu")
        return X, y
    except FileNotFoundError:
        print("Không tìm thấy dữ liệu, vui lòng chạy generate_data_simple.py trước")
        return None, None

def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Chuẩn bị dữ liệu train/validation/test"""
    # Chia train/test trước
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Chia train thành train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler)

def create_models():
    """Tạo các model khác nhau để so sánh"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    return models

def train_and_evaluate_models(X, y):
    """Huấn luyện và đánh giá tất cả các model"""
    print("\n" + "=" * 60)
    print("HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MODEL")
    print("=" * 60)
    
    # Chuẩn bị dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y)
    
    # Tạo các model
    models = create_models()
    
    results = {}
    
    for name, model in models.items():
        print(f"\nĐang huấn luyện {name}...")
        start_time = time.time()
        
        # Huấn luyện
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Tính metrics
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        training_time = time.time() - start_time
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'training_time': training_time,
            'y_pred_test': y_pred_test
        }
        
        print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"  Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
        print(f"  Thời gian huấn luyện: {training_time:.2f}s")
    
    return results, (X_train, X_val, X_test, y_train, y_val, y_test, scaler)

def plot_model_comparison(results):
    """Vẽ biểu đồ so sánh các model"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(results.keys())
    
    # R² scores
    train_r2 = [results[name]['train_r2'] for name in model_names]
    val_r2 = [results[name]['val_r2'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax1.bar(x - width, train_r2, width, label='Train', alpha=0.8)
    ax1.bar(x, val_r2, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, test_r2, width, label='Test', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE scores
    test_mse = [results[name]['test_mse'] for name in model_names]
    ax2.bar(model_names, test_mse, alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('MSE')
    ax2.set_title('Test MSE Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # MAE scores
    test_mae = [results[name]['test_mae'] for name in model_names]
    ax3.bar(model_names, test_mae, alpha=0.8)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('MAE')
    ax3.set_title('Test MAE Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Training time
    training_times = [results[name]['training_time'] for name in model_names]
    ax4.bar(model_names, training_times, alpha=0.8)
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Training Time (s)')
    ax4.set_title('Training Time Comparison')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(results, y_test):
    """Vẽ biểu đồ dự đoán của các model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        if i < 6:  # Chỉ vẽ 6 model đầu
            ax = axes[i]
            y_pred = result['y_pred_test']
            
            ax.scatter(y_test, y_pred, alpha=0.6, s=20)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Giá trị thực tế')
            ax.set_ylabel('Giá trị dự đoán')
            ax.set_title(f'{name}\nR² = {result["test_r2"]:.4f}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_best_model(results):
    """Lưu model tốt nhất"""
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    import joblib
    joblib.dump(best_model, 'best_elliptic_model.pkl')
    
    print(f"\nĐã lưu model tốt nhất ({best_model_name}) vào best_elliptic_model.pkl")
    print(f"R² Score: {results[best_model_name]['test_r2']:.4f}")

def generate_report(results):
    """Tạo báo cáo tổng hợp"""
    print("\n" + "=" * 60)
    print("BÁO CÁO TỔNG HỢP")
    print("=" * 60)
    
    # Sắp xếp theo R² score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    
    print(f"{'Model':<20} {'Test R²':<10} {'Test MSE':<12} {'Test MAE':<12} {'Time(s)':<8}")
    print("-" * 70)
    
    for name, result in sorted_results:
        print(f"{name:<20} {result['test_r2']:<10.4f} {result['test_mse']:<12.6f} "
              f"{result['test_mae']:<12.6f} {result['training_time']:<8.2f}")
    
    best_model_name = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"\nModel tốt nhất: {best_model_name}")
    print(f"R² Score: {best_result['test_r2']:.4f} ({best_result['test_r2']*100:.1f}%)")
    print(f"RMSE: {np.sqrt(best_result['test_mse']):.6f}")
    print(f"MAE: {best_result['test_mae']:.6f}")
    
    # Đánh giá chất lượng
    if best_result['test_r2'] > 0.9:
        quality = "Xuất sắc"
    elif best_result['test_r2'] > 0.8:
        quality = "Tốt"
    elif best_result['test_r2'] > 0.7:
        quality = "Khá"
    elif best_result['test_r2'] > 0.5:
        quality = "Trung bình"
    else:
        quality = "Cần cải thiện"
    
    print(f"\nĐánh giá chất lượng: {quality}")

def main():
    print("ELLIPTIC_AI - MODEL ĐƠN GIẢN")
    print("=" * 60)
    
    # Tải dữ liệu
    X, y = load_data()
    if X is None:
        return
    
    # Huấn luyện và đánh giá
    results, data = train_and_evaluate_models(X, y)
    
    # Vẽ biểu đồ
    print("\nĐang tạo biểu đồ so sánh...")
    plot_model_comparison(results)
    
    print("\nĐang tạo biểu đồ dự đoán...")
    plot_predictions(results, data[5]) 
    
    save_best_model(results)
    
    generate_report(results)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)

if __name__ == "__main__":
    main() 