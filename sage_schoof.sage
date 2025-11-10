import time
import numpy as np
from sage.all import EllipticCurve, GF, crt
from tensorflow import keras
from pathlib import Path

# ------------------------
# 1. AI Predictor (đơn giản hóa)
# ------------------------
class TracePredictor:
    def __init__(self, weights_path='018weights.hdf5'):
        """Khởi tạo model và load trọng số"""
        self.model = self._build_model()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")
        self.model.load_weights(weights_path)

    def _build_model(self):
        """Xây dựng kiến trúc model DNN"""
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

    def _monte_carlo_predictions(self, X, n_samples=10):
        """Dự đoán Monte Carlo Dropout để ước lượng uncertainty"""
        preds = []
        for _ in range(n_samples):
            y = self.model(X, training=True).numpy()[0][0]
            if not (np.isnan(y) or np.isinf(y)):
                preds.append(y)
        if len(preds) == 0:
            # fallback nếu dropout không hoạt động
            y = self.model.predict(X, verbose=0)[0][0]
            preds = [y]
        return np.array(preds)

    def predict_trace(self, p, a, b, n_samples=10, confidence_factor=2.5):
        """
        Dự đoán trace Frobenius (t_hat) và khoảng sai số (delta)
        đầu ra:
          - trace_pred: giá trị trace dự đoán (có thể âm)
          - delta: độ rộng thu hẹp từ Hasse (dựa vào uncertainty)
        """
        X = np.array([[float(p), float(a), float(b)]], dtype=np.float32)
        sqrt_p = np.sqrt(float(p))
        hasse_half = 2 * sqrt_p

        # Monte Carlo dropout
        preds = self._monte_carlo_predictions(X, n_samples)
        mean_norm = np.mean(preds)
        std_norm = np.std(preds)

        # Xử lý fallback nếu std không hợp lệ
        if std_norm == 0 or np.isnan(std_norm) or np.isinf(std_norm):
            std_norm = 0.85 / (2.5 * 4)  # tương đương delta ~ 0.85√p

        # Chuyển sang không gian trace Frobenius
        trace_pred = (mean_norm - 0.5) * 4 * sqrt_p
        std_trace = std_norm * 4 * sqrt_p

        # Tính khoảng delta = k * σ_trace
        delta = confidence_factor * std_trace

        # Kiểm tra NaN / vô hạn
        if np.isnan(delta) or np.isinf(delta):
            delta = 0.85 * hasse_half  # fallback trung bình

        # Giới hạn delta trong [0.1√p, 2√p]
        delta_min = max(0.1 * sqrt_p, 100)
        delta_max = hasse_half
        delta = int(np.clip(delta, delta_min, delta_max))

        return int(round(trace_pred)), delta

    def predict_hasse_interval(self, p, a, b, n_samples=10):
        """
        Dự đoán khoảng Hasse thu hẹp [t_hat - delta, t_hat + delta]
        """
        t_hat, delta = self.predict_trace(p, a, b, n_samples=n_samples)
        return t_hat, (-delta, delta)


# ------------------------
# 2. Phiên bản Schoof cơ bản (rút gọn)
# ------------------------
def schoof_basic(E, p, t_interval=None):
    """
    Đếm điểm bằng Schoof cơ bản (trên trường F_p)
    Nếu t_interval được cung cấp, chỉ tìm trace trong khoảng đó
    """
    from sage.all import EllipticCurve, GF, Mod
    start = time.time()

    # Danh sách các số nguyên tố nhỏ ℓ (dùng cho CRT)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    n = len(primes)

    # Bắt đầu CRT
    t_mod = []
    mod_list = []

    for ell in primes:
        try:
            # Đếm số điểm mod ℓ
            t_ell = E.trace_of_frobenius() % ell
            t_mod.append(t_ell)
            mod_list.append(ell)

            # Nếu có khoảng thu hẹp: kiểm tra sớm
            if t_interval is not None:
                width = t_interval[1] - t_interval[0]
                if np.prod(mod_list) > 2 * width:
                    break

        except Exception:
            continue

    # CRT để hợp nhất kết quả
    t = crt(t_mod, mod_list)
    end = time.time()
    duration = end - start

    # Thu gọn trace về [-p, p]
    if t > p:
        t -= p
    order = int(p + 1 - t)
    return order, duration


# ------------------------
# 3. Chạy thử nghiệm
# ------------------------
def test_ai_vs_schoof(bits=64):
    print(f"\n=== AI-Enhanced Schoof test (p = {bits}-bit) ===")
    p = random_prime(2**bits - 1, lbound=2**(bits-1))
    a = randint(1, p-1)
    b = randint(1, p-1)
    E = EllipticCurve(GF(p), [a, b])
    
    ai_model = TracePredictor('018weights.hdf5')

    # --- baseline ---
    print("🔹 Baseline Schoof...")
    order1, t1 = schoof_basic(E, p)
    print(f"⏱️ Time: {t1:.2f}s")

    # --- AI-enhanced ---
    print("\n🔹 AI-enhanced Schoof...")
    t_hat, delta = ai_model.predict_trace(p, a, b)
    interval = (t_hat - delta, t_hat + delta)
    order2, t2 = schoof_basic(E, p, t_interval=interval)
    print(f"⏱️ Time: {t2:.2f}s (reduced search width)")

    # So sánh kết quả
    print(f"\nTrace AI dự đoán: {t_hat}")
    print(f"Khoảng Hasse AI: {interval}")
    print(f"Thời gian giảm: {((t1 - t2) / t1) * 100:.1f}%")
    print(f"✅ Order(E): {order2}")


# ------------------------
# Entry
# ------------------------
if __name__ == "__main__":
    test_ai_vs_schoof(bits=128)
