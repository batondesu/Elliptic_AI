#!/usr/bin/env python3
"""
AI-Enhanced Schoof Algorithm cho Elliptic Curve Point Counting
Kết hợp thuật toán Schoof cổ điển với AI để tối ưu hóa và dự đoán
"""

import numpy as np
import math
import time
from sympy import primerange, legendre_symbol, isprime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchoofAIEnhanced:
    """
    AI-Enhanced Schoof Algorithm
    Kết hợp thuật toán Schoof cổ điển với machine learning
    """
    
    def __init__(self, model_type='ensemble', use_cache=True):
        """
        Khởi tạo AI-Enhanced Schoof
        
        Args:
            model_type: Loại model ('ensemble', 'neural_network', 'gradient_boosting')
            use_cache: Có sử dụng cache để lưu kết quả không
        """
        self.model_type = model_type
        self.use_cache = use_cache
        self.cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = ['p', 'A', 'B', 'p_mod_4', 'p_mod_8', 'A_mod_p', 'B_mod_p']
        
        # Khởi tạo models
        self._initialize_models()
        
    def _initialize_models(self):
        """Khởi tạo các model AI"""
        if self.model_type == 'ensemble':
            self.models['rf'] = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.models['mlp'] = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.models['mlp'] = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=2000,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
    
    def extract_features(self, p: int, A: int, B: int) -> np.ndarray:
        """
        Trích xuất đặc trưng cho model AI
        
        Args:
            p: Số nguyên tố
            A, B: Tham số đường cong elliptic
            
        Returns:
            Mảng đặc trưng
        """
        features = [
            p,
            A,
            B,
            p % 4,  # p mod 4
            p % 8,  # p mod 8
            A % p,  # A mod p
            B % p   # B mod p
        ]
        
        # Thêm đặc trưng phái sinh
        features.extend([
            A / p,  # Tỷ lệ A/p
            B / p,  # Tỷ lệ B/p
            (A * A) % p,  # A² mod p
            (B * B) % p,  # B² mod p
            (A * B) % p,  # A×B mod p
            math.log(p),  # log(p)
            math.sqrt(p), # sqrt(p)
            p ** 0.25,    # p^(1/4)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def classical_schoof(self, A: int, B: int, p: int) -> int:
        """
        Triển khai thuật toán Schoof cổ điển (đơn giản hóa)
        
        Args:
            A, B: Tham số đường cong elliptic
            p: Số nguyên tố
            
        Returns:
            Số điểm trên đường cong elliptic
        """
        # Đếm điểm trực tiếp (thay thế cho Schoof phức tạp)
        count = 1  # Điểm vô cực
        
        for x in range(p):
            # Tính y² = x³ + Ax + B (mod p)
            y_squared = (x**3 + A*x + B) % p
            
            if y_squared == 0:
                count += 1  # y = 0
            elif legendre_symbol(y_squared, p) == 1:
                count += 2  # Hai điểm y và -y
        
        return count
    
    def ai_predict_tilde_delta(self, p: int, A: int, B: int) -> float:
        """
        Dự đoán tilde_delta bằng AI
        
        Args:
            p, A, B: Tham số đường cong elliptic
            
        Returns:
            Dự đoán tilde_delta
        """
        if not self.models:
            raise ValueError("Models chưa được huấn luyện")
        
        # Trích xuất đặc trưng
        features = self.extract_features(p, A, B).reshape(1, -1)
        
        # Chuẩn hóa
        features_scaled = self.scaler.transform(features)
        
        # Dự đoán từ các model
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions.append(pred)
        
        # Kết hợp dự đoán (ensemble)
        if len(predictions) > 1:
            return np.mean(predictions)
        else:
            return predictions[0]
    
    def hybrid_count_points(self, A: int, B: int, p: int, 
                          threshold: int = 100) -> Tuple[int, str]:
        """
        Đếm điểm bằng phương pháp hybrid: AI + Classical
        
        Args:
            A, B: Tham số đường cong elliptic
            p: Số nguyên tố
            threshold: Ngưỡng để quyết định dùng AI hay classical
            
        Returns:
            (số điểm, phương pháp sử dụng)
        """
        cache_key = f"{p}_{A}_{B}"
        
        # Kiểm tra cache
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Quyết định phương pháp
        if p <= threshold and self.models:
            # Dùng AI cho p nhỏ
            try:
                tilde_delta_pred = self.ai_predict_tilde_delta(p, A, B)
                N_pred = p + 1 - 2 * math.sqrt(p) * tilde_delta_pred
                N = int(round(N_pred))
                method = "AI"
            except:
                N = self.classical_schoof(A, B, p)
                method = "Classical (AI failed)"
        else:
            # Dùng classical cho p lớn
            N = self.classical_schoof(A, B, p)
            method = "Classical"
        
        # Lưu cache
        if self.use_cache:
            self.cache[cache_key] = (N, method)
        
        return N, method
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    validation_split: float = 0.2) -> Dict:
        """
        Huấn luyện các model AI
        
        Args:
            X: Dữ liệu đặc trưng
            y: Giá trị tilde_delta
            validation_split: Tỷ lệ validation
            
        Returns:
            Dict chứa kết quả huấn luyện
        """
        logger.info("Bắt đầu huấn luyện AI models...")
        
        # Chuẩn bị dữ liệu
        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        # Chuẩn hóa
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        results = {}
        
        # Huấn luyện từng model
        for name, model in self.models.items():
            logger.info(f"Huấn luyện {name}...")
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Đánh giá
            y_pred_train = model.predict(X_train_scaled)
            y_pred_val = model.predict(X_val_scaled)
            
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            train_mse = mean_squared_error(y_train, y_pred_train)
            val_mse = mean_squared_error(y_val, y_pred_val)
            
            results[name] = {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'training_time': training_time
            }
            
            logger.info(f"{name}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
        
        return results
    
    def generate_training_data(self, max_p: int = 500, 
                             samples_per_p: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sinh dữ liệu huấn luyện cho AI models
        
        Args:
            max_p: Số nguyên tố lớn nhất
            samples_per_p: Số mẫu cho mỗi p
            
        Returns:
            (X, y) - Dữ liệu huấn luyện
        """
        logger.info(f"Sinh dữ liệu huấn luyện: p < {max_p}, {samples_per_p} mẫu/p")
        
        data = []
        primes = list(primerange(3, max_p))
        
        for i, p in enumerate(primes):
            if i % 10 == 0:
                logger.info(f"Xử lý p = {p} ({i+1}/{len(primes)})")
            
            valid_samples = 0
            attempts = 0
            max_attempts = samples_per_p * 10
            
            while valid_samples < samples_per_p and attempts < max_attempts:
                A = np.random.randint(0, p)
                B = np.random.randint(0, p)
                
                # Kiểm tra điều kiện không suy biến
                if (4*A**3 + 27*B**2) % p == 0:
                    attempts += 1
                    continue
                
                # Đếm điểm bằng classical method
                N = self.classical_schoof(A, B, p)
                delta = p + 1 - N
                tilde_delta = delta / (2 * math.sqrt(p))
                
                # Trích xuất đặc trưng
                features = self.extract_features(p, A, B)
                
                data.append(np.concatenate([features, [tilde_delta]]))
                valid_samples += 1
                attempts += 1
        
        data = np.array(data)
        X = data[:, :-1]  # Đặc trưng
        y = data[:, -1]   # tilde_delta
        
        logger.info(f"Hoàn thành! {len(data)} mẫu")
        return X, y
    
    def save_models(self, filepath: str):
        """Lưu models và scaler"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Đã lưu models vào {filepath}")
    
    def load_models(self, filepath: str):
        """Tải models và scaler"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        logger.info(f"Đã tải models từ {filepath}")
    
    def benchmark_performance(self, test_cases: List[Tuple[int, int, int]]) -> Dict:
        """
        Đánh giá hiệu suất trên test cases
        
        Args:
            test_cases: List các (p, A, B) để test
            
        Returns:
            Dict chứa kết quả benchmark
        """
        results = {
            'ai_predictions': [],
            'classical_results': [],
            'ai_times': [],
            'classical_times': [],
            'methods_used': []
        }
        
        for p, A, B in test_cases:
            # Test AI method
            start_time = time.time()
            try:
                tilde_delta_ai = self.ai_predict_tilde_delta(p, A, B)
                ai_time = time.time() - start_time
                results['ai_predictions'].append(tilde_delta_ai)
                results['ai_times'].append(ai_time)
            except:
                results['ai_predictions'].append(None)
                results['ai_times'].append(None)
            
            # Test classical method
            start_time = time.time()
            N_classical = self.classical_schoof(A, B, p)
            classical_time = time.time() - start_time
            tilde_delta_classical = (p + 1 - N_classical) / (2 * math.sqrt(p))
            
            results['classical_results'].append(tilde_delta_classical)
            results['classical_times'].append(classical_time)
            
            # Test hybrid method
            N_hybrid, method = self.hybrid_count_points(A, B, p)
            results['methods_used'].append(method)
        
        return results
    
    def plot_benchmark_results(self, results: Dict):
        """Vẽ biểu đồ kết quả benchmark"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # So sánh dự đoán AI vs Classical
        valid_ai = [i for i, pred in enumerate(results['ai_predictions']) if pred is not None]
        if valid_ai:
            ai_preds = [results['ai_predictions'][i] for i in valid_ai]
            classical_preds = [results['classical_results'][i] for i in valid_ai]
            
            axes[0, 0].scatter(classical_preds, ai_preds, alpha=0.6)
            axes[0, 0].plot([min(classical_preds), max(classical_preds)], 
                           [min(classical_preds), max(classical_preds)], 'r--')
            axes[0, 0].set_xlabel('Classical tilde_delta')
            axes[0, 0].set_ylabel('AI tilde_delta')
            axes[0, 0].set_title('AI vs Classical Predictions')
            axes[0, 0].grid(True)
        
        # So sánh thời gian
        valid_times = [i for i, t in enumerate(results['ai_times']) if t is not None]
        if valid_times:
            ai_times = [results['ai_times'][i] for i in valid_times]
            classical_times = [results['classical_times'][i] for i in valid_times]
            
            axes[0, 1].scatter(classical_times, ai_times, alpha=0.6)
            axes[0, 1].set_xlabel('Classical Time (s)')
            axes[0, 1].set_ylabel('AI Time (s)')
            axes[0, 1].set_title('AI vs Classical Time')
            axes[0, 1].grid(True)
        
        # Phân phối methods used
        method_counts = {}
        for method in results['methods_used']:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            axes[1, 0].bar(methods, counts)
            axes[1, 0].set_title('Methods Used Distribution')
            axes[1, 0].set_ylabel('Count')
        
        # Histogram classical times
        axes[1, 1].hist(results['classical_times'], bins=20, alpha=0.7)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Classical Method Time Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('schoof_ai_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Demo AI-Enhanced Schoof Algorithm"""
    print("=" * 60)
    print("AI-ENHANCED SCHOOF ALGORITHM")
    print("=" * 60)
    
    # Khởi tạo
    schoof_ai = SchoofAIEnhanced(model_type='ensemble')
    
    # Sinh dữ liệu huấn luyện
    print("\n1. SINH DỮ LIỆU HUẤN LUYỆN")
    X, y = schoof_ai.generate_training_data(max_p=300, samples_per_p=15)
    
    # Huấn luyện models
    print("\n2. HUẤN LUYỆN AI MODELS")
    results = schoof_ai.train_models(X, y)
    
    # Lưu models
    print("\n3. LƯU MODELS")
    schoof_ai.save_models('schoof_ai_models.pkl')
    
    # Test cases
    print("\n4. BENCHMARK PERFORMANCE")
    test_cases = [
        (17, 5, 3), (23, 7, 11), (31, 13, 19), (41, 17, 23),
        (53, 29, 31), (67, 37, 43), (83, 47, 59), (97, 61, 71),
        (113, 73, 89), (127, 79, 101), (137, 89, 103), (149, 97, 107)
    ]
    
    benchmark_results = schoof_ai.benchmark_performance(test_cases)
    
    # Vẽ kết quả
    print("\n5. VẼ KẾT QUẢ")
    schoof_ai.plot_benchmark_results(benchmark_results)
    
    # Tóm tắt
    print("\n6. TÓM TẮT KẾT QUẢ")
    print(f"Số mẫu huấn luyện: {len(X)}")
    print(f"Số test cases: {len(test_cases)}")
    
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Val R²: {metrics['val_r2']:.4f}")
        print(f"  Training time: {metrics['training_time']:.2f}s")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)

if __name__ == "__main__":
    main() 