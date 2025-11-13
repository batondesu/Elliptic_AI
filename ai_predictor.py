"""
AI Predictor - Dự đoán TRACE Frobenius (đã sửa đúng mục đích)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl
from pathlib import Path

class TracePredictor:
    def __init__(self, weights_path='018weights.hdf5', bit_size=128, max_a=None, max_b=None):
        """
        Load model dự đoán trace
        
        Args:
            weights_path: đường dẫn đến file weights
            bit_size: kích thước bit của p (default 128)
            max_a, max_b: giá trị max của a, b để normalize (nếu None sẽ dùng giá trị lớn)
        """
        self.bit_size = bit_size
        # Nếu không có max_a, max_b, dùng giá trị lớn để đảm bảo normalize đúng
        # Với số lớn (256-bit), tránh tính 2^bit_size trực tiếp để không overflow
        if max_a is not None:
            self.max_a = float(max_a)
        else:
            # Sử dụng giá trị lớn nhưng an toàn (float64 max ~ 1.8e308)
            # Với 256-bit, p max ~ 2^256 ≈ 1.16e77, nên dùng giá trị này
            if bit_size <= 128:
                self.max_a = 2.0**bit_size
            else:
                # Với bit_size > 128, dùng log scale để tránh overflow
                self.max_a = np.exp2(bit_size) if bit_size <= 1024 else 1e77
        
        if max_b is not None:
            self.max_b = float(max_b)
        else:
            if bit_size <= 128:
                self.max_b = 2.0**bit_size
            else:
                self.max_b = np.exp2(bit_size) if bit_size <= 1024 else 1e77
        
        self.model = self._build_model()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Không tìm thấy: {weights_path}")
        
        # Compile model với cùng config như training (cần cho predict)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Load weights
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
        
        # Warm-up prediction to initialize BatchNormalization layers
        try:
            dummy = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
            _ = self.model.predict(dummy, verbose=0)
        except Exception:
            pass
        
    def _build_model(self):
        """Kiến trúc model như training script"""
        model = tf.keras.Sequential([
            tfl.Dense(units=512, activation='relu', input_dim=3),
            tfl.BatchNormalization(),

            tfl.Dense(units=256, activation='relu'),
            tfl.Dropout(0.1),
            tfl.BatchNormalization(),

            tfl.Dense(units=128, activation='relu'),
            tfl.Dropout(0.15),
            tfl.BatchNormalization(),

            tfl.Dense(units=64, activation='relu'),
            tfl.Dropout(0.2),
            tfl.BatchNormalization(),

            tfl.Dense(units=32, activation='relu'),
            tfl.Dropout(0.2),
            tfl.BatchNormalization(),

            tfl.Dense(units=16, activation='relu'),
            tfl.Dropout(0.25),
            tfl.BatchNormalization(),

            tfl.Dense(units=8, activation='relu'),
            tfl.Dropout(0.3),
            tfl.BatchNormalization(),

            tfl.Dense(units=1, activation='sigmoid')
        ])
        return model
    
    def predict_hasse_interval(self, p, a, b, n_samples=16):
        _, delta = self.predict_trace(p, a, b, n_samples=n_samples)
        return delta
    
    def predict_trace(self, p, a, b, n_samples=16):
        """
        Dự đoán TRACE Frobenius và tính DELTA (khoảng Hasse thu hẹp) từ MODEL
        
        (Wrapper function để tương thích với code cũ)
        """
        # Chuẩn hóa đầu vào giống lúc huấn luyện
        p_log2 = None
        if isinstance(p, (int, np.integer)) and p > 0:
            try:
                p_log2 = math.log2(p)
            except (OverflowError, ValueError):
                p_log2 = None
        if p_log2 is None:
            try:
                p_float = float(p)
                if np.isfinite(p_float) and p_float > 0:
                    p_log2 = math.log2(p_float)
            except Exception:
                p_log2 = None
        if p_log2 is None:
            try:
                p_int = int(p)
                p_log2 = p_int.bit_length() - 1
            except Exception:
                p_log2 = float(self.bit_size)

        p_norm = float(np.clip(p_log2 / self.bit_size, 0.0, 1.0))

        try:
            a_norm = float(a) / self.max_a if self.max_a else 0.0
        except (OverflowError, ValueError, ZeroDivisionError):
            a_norm = 0.0
        try:
            b_norm = float(b) / self.max_b if self.max_b else 0.0
        except (OverflowError, ValueError, ZeroDivisionError):
            b_norm = 0.0

        a_norm = float(np.clip(a_norm, 0.0, 1.0))
        b_norm = float(np.clip(b_norm, 0.0, 1.0))

        X = np.array([[p_norm, a_norm, b_norm]], dtype=np.float32)

        try:
            sqrt_p = math.sqrt(float(p))
        except (OverflowError, ValueError):
            sqrt_p = math.sqrt(2 ** self.bit_size)

        predictions = []
        samples = max(n_samples, 8)
        for _ in range(samples):
            try:
                val = float(self.model(X, training=True).numpy()[0][0])
                if np.isfinite(val):
                    predictions.append(val)
            except Exception:
                continue

        if len(predictions) < 2:
            try:
                val = float(self.model.predict(X, verbose=0)[0][0])
            except Exception:
                val = 0.5
            if not np.isfinite(val):
                val = 0.5
            predictions = [val]

        preds = np.array(predictions, dtype=float)

        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            mean_norm = 0.5
            std_norm = 0.25
        else:
            mean_norm = float(np.mean(preds))
            std_norm = float(np.std(preds, ddof=0))
            if not np.isfinite(mean_norm):
                mean_norm = 0.5
            if not np.isfinite(std_norm) or std_norm < 1e-4:
                std_norm = 0.25

        mean_norm = float(np.clip(mean_norm, 0.0, 1.0))

        trace_pred = (mean_norm - 0.5) * 4.0 * sqrt_p
        if not np.isfinite(trace_pred):
            trace_pred = 0.0

        hasse_half = 2.0 * sqrt_p
        trace_pred = float(np.clip(trace_pred, -hasse_half, hasse_half))

        std_trace = max(std_norm * 4.0 * sqrt_p, sqrt_p * 0.05)
        delta = std_trace * 2.5
        if not np.isfinite(delta):
            delta = 0.85 * hasse_half
            print(f"=============== Delta is not finite, using {delta} ===============")

        delta_min = max(sqrt_p * 0.1, 50.0)
        delta_max = hasse_half
        delta = float(np.clip(delta, delta_min, delta_max))

        return int(round(trace_pred)), int(round(delta))
        