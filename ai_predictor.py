"""
AI Predictor - Dự đoán TRACE Frobenius (đã sửa đúng mục đích)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

class TracePredictor:
    def __init__(self, weights_path='018weights.hdf5'):
        """Load model dự đoán trace"""
        self.model = self._build_model()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Không tìm thấy: {weights_path}")
        self.model.load_weights(weights_path)
        
    def _build_model(self):
        return keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_dim=3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.15),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def predict_hasse_interval(self, p, a, b, n_samples=1):
        X = np.array([[float(p), float(a), float(b)]], dtype=np.float32)
        sqrt_p = np.sqrt(float(p))
        
        # MONTE CARLO DROPOUT: Predict nhiều lần với dropout enabled
        # Điều này cho phép estimate uncertainty từ model chính nó
        predictions_norm = []
        
        # Predict với training=True để enable dropout layers
        # Thử cả 2 cách: model() và model.predict() với dropout
        try:
            for _ in range(n_samples):
                # Cách 1: Dùng model call trực tiếp
                y_norm = self.model(X, training=True).numpy()[0][0]
                if not (np.isnan(y_norm) or np.isinf(y_norm)):
                    predictions_norm.append(y_norm)
        except:
            # Fallback: predict bình thường (không có dropout)
            y_norm = self.model.predict(X, verbose=0)[0][0]
            predictions_norm = [y_norm] * n_samples  # Lặp lại giá trị
        
        # Kiểm tra có đủ predictions không
        if len(predictions_norm) == 0:
            # Fallback: predict 1 lần
            y_norm = self.model.predict(X, verbose=0)[0][0]
            predictions_norm = np.array([y_norm])
        
        predictions_norm = np.array(predictions_norm)
        
        # Kiểm tra NaN hoặc inf trong predictions
        if len(predictions_norm) == 0 or np.any(np.isnan(predictions_norm)) or np.any(np.isinf(predictions_norm)):
            # Fallback: dùng delta tối thiểu
            delta_min = max(int(sqrt_p * 0.1), 100)
            delta_max = int(2 * sqrt_p)
            return max(delta_min, min(int(0.85 * sqrt_p * 2), delta_max))
        
        # Tính std dev từ model predictions (uncertainty)
        std_norm = np.std(predictions_norm)
        
        # Xử lý trường hợp std = 0 hoặc NaN (model quá confident hoặc lỗi)
        if std_norm == 0 or np.isnan(std_norm) or np.isinf(std_norm):
            # Fallback: dùng delta dựa trên √p (tương tự avg_diff)
            std_norm = 0.85 / (2.5 * 4)  # Tương đương với delta ≈ 0.85 * √p
            std_trace = std_norm * 4 * sqrt_p
        else:
            # Chuyển sang không gian trace: std_trace = std_norm * 4√p
            std_trace = std_norm * 4 * sqrt_p
        
        # Delta = confidence_factor * std_trace
        # Dùng 2.5 sigma để bao phủ ~99% predictions
        confidence_factor = 2.5
        delta = confidence_factor * std_trace
        
        # Kiểm tra NaN trước khi convert
        if np.isnan(delta) or np.isinf(delta):
            delta = 0.85 * sqrt_p * 2  # Fallback
        
        delta = int(delta)
        
        # Đảm bảo delta hợp lý:
        # - Tối thiểu: 10% của √p hoặc 100
        # - Tối đa: 2√p (trong Hasse bound)
        delta_min = max(int(sqrt_p * 0.1), 100)
        delta_max = int(2 * sqrt_p)
        delta = max(delta_min, min(delta, delta_max))
        
        return delta
    
    def predict_trace(self, p, a, b, n_samples=1):
        """
        Dự đoán TRACE Frobenius và tính DELTA (khoảng Hasse thu hẹp) từ MODEL
        
        (Wrapper function để tương thích với code cũ)
        """
        X = np.array([[float(p), float(a), float(b)]], dtype=np.float32)
        sqrt_p = np.sqrt(float(p))
        
        # MONTE CARLO DROPOUT
        predictions_norm = []
        for _ in range(n_samples):
            y_norm = self.model(X, training=True).numpy()[0][0]
            predictions_norm.append(y_norm)
        
        predictions_norm = np.array(predictions_norm)
        
        # Kiểm tra NaN hoặc inf
        if np.any(np.isnan(predictions_norm)) or np.any(np.isinf(predictions_norm)):
            # Fallback: predict 1 lần không có dropout
            y_norm = self.model.predict(X, verbose=0)[0][0]
            mean_norm = y_norm
            std_norm = 0.85 / (2.5 * 4)  # Fallback
        else:
            mean_norm = np.mean(predictions_norm)
            std_norm = np.std(predictions_norm)
            if std_norm == 0 or np.isnan(std_norm) or np.isinf(std_norm):
                std_norm = 0.85 / (2.5 * 4)  # Fallback
        
        # Trace prediction
        trace_pred = (mean_norm - 0.5) * 4 * sqrt_p
        
        # Delta từ uncertainty
        std_trace = std_norm * 4 * sqrt_p
        confidence_factor = 2.5
        delta = confidence_factor * std_trace
    
        # Kiểm tra NaN
        if np.isnan(delta) or np.isinf(delta):
            delta = 0.85 * sqrt_p * 2  # Fallback
        
        delta = int(delta)
        
        # Giới hạn delta
        delta_min = max(int(sqrt_p * 0.1), 100)
        delta_max = int(2 * sqrt_p)
        delta = max(delta_min, min(delta, delta_max))
        
        return int(round(trace_pred)), delta
