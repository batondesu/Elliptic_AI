#!/usr/bin/env python3
"""
Phân tích mối quan hệ giữa (p,A,B) và hệ số δ̃ trong đường cong Elliptic
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import math
from sympy import primerange, legendre_symbol

class DeltaRelationshipAnalyzer:
    """Phân tích mối quan hệ δ̃ = (p + 1 - N) / (2√p)"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def count_points(self, A: int, B: int, p: int) -> int:
        """Đếm số điểm trên đường cong elliptic"""
        count = 1  # Điểm vô cực
        for x in range(p):
            y_squared = (x**3 + A*x + B) % p
            if y_squared == 0:
                count += 1
            elif legendre_symbol(y_squared, p) == 1:
                count += 2
        return count
    
    def calculate_tilde_delta(self, A: int, B: int, p: int) -> float:
        """Tính δ̃ = (p + 1 - N) / (2√p)"""
        N = self.count_points(A, B, p)
        delta = p + 1 - N
        tilde_delta = delta / (2 * math.sqrt(p))
        return tilde_delta
    
    def generate_data(self, max_p: int = 100, samples_per_p: int = 10):
        """Sinh dữ liệu để phân tích"""
        print(f"Sinh dữ liệu: p < {max_p}, {samples_per_p} mẫu/p")
        
        data = []
        primes = list(primerange(3, max_p))
        
        for p in primes:
            for _ in range(samples_per_p):
                A = np.random.randint(0, p)
                B = np.random.randint(0, p)
                
                if (4*A**3 + 27*B**2) % p == 0:
                    continue
                
                tilde_delta = self.calculate_tilde_delta(A, B, p)
                
                # Đặc trưng cơ bản
                features = [
                    p, A, B,
                    p % 4, p % 8,
                    A % p, B % p,
                    A / p, B / p,
                    (A * A) % p, (B * B) % p,
                    (A * B) % p,
                    math.log(p), math.sqrt(p)
                ]
                
                data.append(features + [tilde_delta])
        
        data = np.array(data)
        X = data[:, :-1]
        y = data[:, -1]
        
        print(f"Hoàn thành! {len(data)} mẫu")
        return X, y
    
    def analyze_relationships(self, X: np.ndarray, y: np.ndarray):
        """Phân tích mối quan hệ"""
        print("\n" + "=" * 50)
        print("PHÂN TÍCH MỐI QUAN HỆ")
        print("=" * 50)
        
        feature_names = ['p', 'A', 'B', 'p_mod_4', 'p_mod_8', 'A_mod_p', 'B_mod_p',
                        'A_over_p', 'B_over_p', 'A_squared', 'B_squared', 'A_times_B',
                        'log_p', 'sqrt_p']
        
        # Tính correlation
        data_matrix = np.column_stack([X, y])
        corr_matrix = np.corrcoef(data_matrix.T)
        
        print("\nCorrelation với δ̃:")
        correlations = []
        for i, name in enumerate(feature_names):
            corr = corr_matrix[i, -1]
            correlations.append((name, corr))
            print(f"  {name:<15}: {corr:>8.4f}")
        
        # Sắp xếp theo độ mạnh
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 5 đặc trưng quan trọng:")
        for i, (name, corr) in enumerate(correlations[:5]):
            print(f"  {i+1}. {name:<15}: {corr:>8.4f}")
        
        return correlations
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện models"""
        print("\n" + "=" * 50)
        print("HUẤN LUYỆN MODELS")
        print("=" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        y_pred = rf.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Random Forest R²: {r2:.4f}")
        
        # Feature importance
        importance = rf.feature_importances_
        feature_names = ['p', 'A', 'B', 'p_mod_4', 'p_mod_8', 'A_mod_p', 'B_mod_p',
                        'A_over_p', 'B_over_p', 'A_squared', 'B_squared', 'A_times_B',
                        'log_p', 'sqrt_p']
        
        print(f"\nFeature importance:")
        for name, imp in zip(feature_names, importance):
            print(f"  {name:<15}: {imp:.4f}")
        
        return rf, r2, y_pred, y_test
    
    def visualize(self, X: np.ndarray, y: np.ndarray, y_pred=None, y_test=None):
        """Vẽ biểu đồ phân tích"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # δ̃ vs p
        axes[0, 0].scatter(X[:, 0], y, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('p')
        axes[0, 0].set_ylabel('δ̃')
        axes[0, 0].set_title('δ̃ vs p')
        axes[0, 0].grid(True)
        
        # δ̃ vs A
        axes[0, 1].scatter(X[:, 1], y, alpha=0.6, s=20)
        axes[0, 1].set_xlabel('A')
        axes[0, 1].set_ylabel('δ̃')
        axes[0, 1].set_title('δ̃ vs A')
        axes[0, 1].grid(True)
        
        # δ̃ vs B
        axes[0, 2].scatter(X[:, 2], y, alpha=0.6, s=20)
        axes[0, 2].set_xlabel('B')
        axes[0, 2].set_ylabel('δ̃')
        axes[0, 2].set_title('δ̃ vs B')
        axes[0, 2].grid(True)
        
        # A vs B với màu theo δ̃
        scatter = axes[1, 0].scatter(X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=axes[1, 0])
        axes[1, 0].set_xlabel('A')
        axes[1, 0].set_ylabel('B')
        axes[1, 0].set_title('A vs B (màu theo δ̃)')
        axes[1, 0].grid(True)
        
        # Phân phối δ̃
        axes[1, 1].hist(y, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('δ̃')
        axes[1, 1].set_ylabel('Tần suất')
        axes[1, 1].set_title('Phân phối δ̃')
        axes[1, 1].grid(True)
        
        # Dự đoán vs thực tế
        if y_pred is not None and y_test is not None:
            axes[1, 2].scatter(y_test, y_pred, alpha=0.6, s=20)
            axes[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[1, 2].set_xlabel('δ̃ thực tế')
            axes[1, 2].set_ylabel('δ̃ dự đoán')
            axes[1, 2].set_title('Dự đoán vs Thực tế')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('delta_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Demo phân tích mối quan hệ δ̃"""
    print("=" * 60)
    print("PHÂN TÍCH MỐI QUAN HỆ (p,A,B) VÀ δ̃")
    print("=" * 60)
    
    analyzer = DeltaRelationshipAnalyzer()
    
    # Sinh dữ liệu
    X, y = analyzer.generate_data(max_p=80, samples_per_p=8)
    
    # Phân tích mối quan hệ
    correlations = analyzer.analyze_relationships(X, y)
    
    # Huấn luyện model
    model, r2, y_pred, y_test = analyzer.train_models(X, y)
    
    # Vẽ biểu đồ
    analyzer.visualize(X, y, y_pred, y_test)
    
    print(f"\nKết quả:")
    print(f"R² Score: {r2:.4f}")
    print(f"Đã lưu biểu đồ vào delta_analysis.png")

if __name__ == "__main__":
    main() 