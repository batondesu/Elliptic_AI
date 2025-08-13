#!/usr/bin/env python3
"""
Module cải tiến để học mối quan hệ δ̃
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import math
from sympy import primerange, legendre_symbol
import joblib

class ImprovedDeltaLearner:
    """Model cải tiến để học δ̃ = (p + 1 - N) / (2√p)"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def count_points(self, A: int, B: int, p: int) -> int:
        """Đếm số điểm trên đường cong elliptic"""
        count = 1
        for x in range(p):
            y_squared = (x**3 + A*x + B) % p
            if y_squared == 0:
                count += 1
            elif legendre_symbol(y_squared, p) == 1:
                count += 2
        return count
    
    def calculate_tilde_delta(self, A: int, B: int, p: int) -> float:
        """Tính δ̃"""
        N = self.count_points(A, B, p)
        delta = p + 1 - N
        tilde_delta = delta / (2 * math.sqrt(p))
        return tilde_delta
    
    def extract_features(self, p: int, A: int, B: int) -> np.ndarray:
        """Trích xuất đặc trưng cải tiến"""
        features = [
            p, A, B,
            p % 4, p % 8, p % 3,
            A % p, B % p,
            A / p, B / p,  # Quan trọng
            (A * B) % p,   # Quan trọng nhất
            (A * A) % p, (B * B) % p,
            (A + B) % p, (A - B) % p,
            math.log(p), math.sqrt(p),
            (A / p) * (B / p),  # Interaction
            abs(A / p - B / p),  # Khoảng cách
            (A / p) ** 2 + (B / p) ** 2  # Tổng bình phương
        ]
        return np.array(features)
    
    def generate_data(self, max_p: int = 150, samples_per_p: int = 15):
        """Sinh dữ liệu"""
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
                features = self.extract_features(p, A, B)
                
                data.append(np.concatenate([features, [tilde_delta]]))
        
        data = np.array(data)
        X = data[:, :-1]
        y = data[:, -1]
        
        print(f"Hoàn thành! {len(data)} mẫu")
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện model"""
        print("\nHuấn luyện Improved Delta Model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest với hyperparameters tối ưu
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test R²: {r2:.4f}")
        return r2
    
    def predict(self, p: int, A: int, B: int) -> float:
        """Dự đoán δ̃"""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện")
        
        features = self.extract_features(p, A, B).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]
    
    def test_predictions(self):
        """Test dự đoán"""
        print("\nTest dự đoán:")
        test_cases = [(17, 5, 3), (23, 7, 11), (31, 13, 19), (41, 17, 23)]
        
        for p, A, B in test_cases:
            actual = self.calculate_tilde_delta(A, B, p)
            predicted = self.predict(p, A, B)
            error = abs(actual - predicted)
            
            print(f"p={p}, A={A}, B={B}: Actual={actual:.6f}, Predicted={predicted:.6f}, Error={error:.6f}")
    
    def save_model(self, filepath: str):
        """Lưu model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filepath)
        print(f"Đã lưu model vào {filepath}")

def main():
    """Demo Improved Delta Learning"""
    print("=" * 60)
    print("IMPROVED DELTA LEARNING")
    print("=" * 60)
    
    learner = ImprovedDeltaLearner()
    
    # Sinh dữ liệu
    X, y = learner.generate_data(max_p=120, samples_per_p=12)
    
    # Huấn luyện
    r2 = learner.train(X, y)
    
    # Test dự đoán
    learner.test_predictions()
    
    # Lưu model
    learner.save_model('improved_delta_model.pkl')
    
    print(f"\nKết quả: R² = {r2:.4f}")

if __name__ == "__main__":
    main() 