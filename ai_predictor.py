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
        
        print(f"Predictions norm: {predictions_norm}")

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
        # Normalize input giống như training
        # p_norm = log2(p) / bit_size
        # a_norm = a / max_a
        # b_norm = b / max_b
        
        # Xử lý overflow với số lớn (256-bit)
        # Sử dụng log2 trực tiếp từ integer để tránh overflow khi convert sang float
        p_log2 = None
        try:
            # Thử tính log2 từ integer trước (math.log2 có thể xử lý integer lớn)
            if isinstance(p, (int, np.integer)) and p > 0:
                try:
                    p_log2 = math.log2(p)
                except (OverflowError, ValueError):
                    # Fallback: dùng bit_length để ước lượng log2
                    p_log2 = p.bit_length() - 1 if hasattr(p, 'bit_length') else 0
            else:
                # Nếu p đã là float, thử convert
                p_float = float(p)
                if np.isfinite(p_float) and p_float > 0:
                    p_log2 = np.log2(p_float)
                else:
                    # Fallback: dùng bit_length
                    p_int = int(p) if hasattr(p, '__int__') else p
                    p_log2 = p_int.bit_length() - 1 if hasattr(p_int, 'bit_length') else 0
            p_norm = p_log2 / self.bit_size
        except Exception:
            # Ultimate fallback: ước lượng từ bit_length
            try:
                p_int = int(p) if hasattr(p, '__int__') else p
                p_log2 = p_int.bit_length() - 1 if hasattr(p_int, 'bit_length') else self.bit_size
            except Exception:
                p_log2 = self.bit_size
            p_norm = p_log2 / self.bit_size if p_log2 is not None else 1.0
        
        # Normalize a, b với xử lý overflow
        # Với số lớn, normalize theo p thay vì max_a/max_b để tránh overflow
        try:
            a_float = float(a)
            p_float = float(p)
            if not np.isinf(a_float) and not np.isinf(p_float) and p_float > 0:
                # Ưu tiên normalize theo p (a < p nên a/p < 1)
                a_norm = a_float / p_float
            elif self.max_a > 0 and not np.isinf(a_float):
                a_norm = a_float / self.max_a
            else:
                a_norm = 0.0
        except (OverflowError, ValueError):
            # Fallback: normalize theo p sử dụng log scale
            try:
                if isinstance(p, (int, np.integer)) and isinstance(a, (int, np.integer)) and p > 0 and a > 0:
                    # Sử dụng log scale để tránh overflow
                    a_log2 = math.log2(a)
                    p_log2_val = p_log2 if p_log2 is not None else math.log2(p)
                    a_norm = a_log2 / p_log2_val if p_log2_val > 0 else 0.5
                else:
                    a_norm = 0.5  # Default value
            except Exception:
                a_norm = 0.5  # Ultimate fallback
        
        try:
            b_float = float(b)
            p_float = float(p)
            if not np.isinf(b_float) and not np.isinf(p_float) and p_float > 0:
                # Ưu tiên normalize theo p (b < p nên b/p < 1)
                b_norm = b_float / p_float
            elif self.max_b > 0 and not np.isinf(b_float):
                b_norm = b_float / self.max_b
            else:
                b_norm = 0.0
        except (OverflowError, ValueError):
            # Fallback: normalize theo p sử dụng log scale
            try:
                if isinstance(p, (int, np.integer)) and isinstance(b, (int, np.integer)) and p > 0 and b > 0:
                    b_log2 = math.log2(b)
                    p_log2_val = p_log2 if p_log2 is not None else math.log2(p)
                    b_norm = b_log2 / p_log2_val if p_log2_val > 0 else 0.5
                else:
                    b_norm = 0.5  # Default value
            except Exception:
                b_norm = 0.5  # Ultimate fallback
        
        # Clip để đảm bảo trong [0, 1] (hoặc phạm vi hợp lệ)
        p_norm = np.clip(p_norm, 0.0, 1.0)
        a_norm = np.clip(a_norm, 0.0, 1.0)
        b_norm = np.clip(b_norm, 0.0, 1.0)
        
        X = np.array([[p_norm, a_norm, b_norm]], dtype=np.float32)
        
        # Tính sqrt_p với xử lý overflow
        try:
            sqrt_p = np.sqrt(float(p))
            if np.isinf(sqrt_p) or np.isnan(sqrt_p):
                # Fallback: ước lượng sqrt từ log2
                if p_log2 is not None:
                    sqrt_p = np.exp2(p_log2 / 2.0)
                else:
                    sqrt_p = np.exp2(self.bit_size / 2.0)
        except (OverflowError, ValueError):
            # Fallback: ước lượng sqrt từ log2
            if p_log2 is not None:
                sqrt_p = np.exp2(p_log2 / 2.0)
            else:
                sqrt_p = np.exp2(self.bit_size / 2.0)
        
        
        # MONTE CARLO DROPOUT
        predictions_norm = []
        for _ in range(n_samples):
            try:
                y_norm = self.model(X, training=True).numpy()[0][0]
                y_norm_float = float(y_norm)
                # Only add valid predictions
                if np.isfinite(y_norm_float):
                    predictions_norm.append(y_norm_float)
            except Exception:
                # Skip invalid predictions
                continue
        
        # If no valid predictions, use fallback
        if len(predictions_norm) == 0:
            try:
                # Last resort: use model.predict
                y_norm = self.model.predict(X, verbose=0)[0][0]
                y_norm_float = float(y_norm)
                if np.isfinite(y_norm_float):
                    predictions_norm = [y_norm_float]
                else:
                    predictions_norm = [0.5]  # Ultimate fallback
            except Exception:
                predictions_norm = [0.5]  # Ultimate fallback
        
        predictions_norm = np.array(predictions_norm)
        
        # Kiểm tra NaN hoặc inf
        if np.any(np.isnan(predictions_norm)) or np.any(np.isinf(predictions_norm)):
            # Fallback: predict 1 lần không có dropout
            try:
                y_norm = self.model.predict(X, verbose=0)[0][0]
                if np.isnan(y_norm) or np.isinf(y_norm):
                    y_norm = 0.5  # Center value
                mean_norm = float(y_norm)
            except Exception:
                mean_norm = 0.5  # Ultimate fallback
            std_norm = 0.85 / (2.5 * 4)  # Fallback
        else:
            mean_norm = float(np.mean(predictions_norm))
            std_norm = float(np.std(predictions_norm))
            if std_norm == 0 or np.isnan(std_norm) or np.isinf(std_norm):
                std_norm = 0.85 / (2.5 * 4)  # Fallback
        
        # Validate mean_norm
        if not np.isfinite(mean_norm):
            mean_norm = 0.5  # Center value (corresponds to trace = 0)
        mean_norm = float(np.clip(mean_norm, 0.0, 1.0))  # Ensure in [0, 1] for sigmoid output
        
        # Trace prediction
        trace_pred = (mean_norm - 0.5) * 4 * sqrt_p
        
        # Validate trace_pred
        if not np.isfinite(trace_pred):
            trace_pred = 0.0  # Fallback: trace = 0 (center of Hasse bound)
        
        # Clip trace_pred to Hasse bound [-2√p, 2√p]
        hasse_half = 2 * sqrt_p
        trace_pred = float(np.clip(trace_pred, -hasse_half, hasse_half))
        
        # Delta từ uncertainty
        std_trace = std_norm * 4 * sqrt_p
        confidence_factor = 2.5
        delta = confidence_factor * std_trace
    
        # Kiểm tra NaN
        if not np.isfinite(delta):
            delta = 0.85 * hasse_half  # Fallback
        
        delta = float(delta)
        
        # Giới hạn delta
        delta_min = max(sqrt_p * 0.1, 100.0)
        delta_max = hasse_half
        delta = float(np.clip(delta, delta_min, delta_max))
        
        return int(round(trace_pred)), int(round(delta))
        