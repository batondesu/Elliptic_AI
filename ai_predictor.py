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
    def __init__(self, weights_path='018weights_trace.hdf5'):
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
    
    def predict_trace(self, p, a, b):
        """
        Dự đoán TRACE Frobenius (ĐÃ SỬA ĐÚNG)
        
        Model output: trace_normalized ∈ [0,1]
        Denormalize: trace = (y_norm - 0.5) * 4√p
        """
        X = np.array([[float(p), float(a), float(b)]], dtype=np.float32)
        y_norm = self.model.predict(X, verbose=0)[0][0]
        
        sqrt_p = np.sqrt(float(p))
        trace_pred = (y_norm - 0.5) * 4 * sqrt_p
        
        # Delta dựa trên kết quả training
        # avg_diff ≈ 0.85 * √p từ kết quả train
        delta = int(2.0 * 0.85 * sqrt_p)  # ±2×avg_diff
        
        return int(round(trace_pred)), max(delta, 100)
