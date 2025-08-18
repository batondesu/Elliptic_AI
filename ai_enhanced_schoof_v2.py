#!/usr/bin/env python3
"""
AI-enhanced Schoof's Algorithm v2.0
- Sử dụng dataset chuẩn với 24 đặc trưng toán học
- Deep Neural Network (10 hidden layers: 512->384->256->192->128->96->64->32->16->8) dự đoán δ
- CM/non-CM classifier chính xác hơn
- Thu hẹp khoảng Hasse dựa trên dự đoán để hỗ trợ tăng tốc đếm điểm
"""

import os
import math
import time
import numpy as np
from typing import Tuple, Dict, Any, List

# TF/Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

np.random.seed(42)

def ensure_image_dir():
    os.makedirs('image', exist_ok=True)

class SchoofFeatureExtractor:
    """Trích xuất đặc trưng từ dataset Schoof chuẩn."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
    
    def extract_from_raw(self, p: int, A: int, B: int) -> np.ndarray:
        """Trích xuất đặc trưng từ p, A, B (tương tự generate_schoof_dataset)"""
        # Đây là phiên bản đơn giản, trong thực tế cần tính toán đầy đủ
        features = np.zeros(self.feature_count, dtype=np.float32)
        
        # Đặc trưng cơ bản
        features[0] = float(p)  # p
        features[1] = float(A)  # A
        features[2] = float(B)  # B
        
        # Discriminant
        discriminant = (4 * pow(A, 3, p) + 27 * pow(B, 2, p)) % p
        features[3] = float(discriminant)  # discriminant
        features[4] = discriminant / p  # discriminant_ratio
        
        # J-invariant (đơn giản)
        if A != 0 or B != 0:
            try:
                num = (1728 * (4 * pow(A, 3, p))) % p
                den = discriminant
                if den != 0:
                    inv = pow(den, -1, p)
                    j_inv = (num * inv) % p
                    features[5] = float(j_inv)  # j_invariant
                    features[6] = float(j_inv) / p  # j_invariant_ratio
            except:
                pass
        
        # Modular arithmetic
        features[7] = float(A % 3)  # A_mod_3
        features[8] = float(B % 3)  # B_mod_3
        features[9] = float(p % 3)  # p_mod_3
        features[10] = float(A % 4)  # A_mod_4
        features[11] = float(B % 4)  # B_mod_4
        features[12] = float(p % 4)  # p_mod_4
        
        # Tương tác
        features[13] = float((A * B) % p)  # A_times_B
        features[14] = float((A * A) % p)  # A_squared
        features[15] = float((B * B) % p)  # B_squared
        
        # Tỷ lệ
        features[16] = A / p  # A_over_p
        features[17] = B / p  # B_over_p
        features[18] = (A + B) / p  # A_plus_B_over_p
        
        # Logarit
        features[19] = math.log(p)  # log_p
        features[20] = math.log(abs(A) + 1)  # log_A
        features[21] = math.log(abs(B) + 1)  # log_B
        
        # Trigonometric
        features[22] = math.sin(A / p * math.pi)  # sin_A_over_p
        features[23] = math.cos(B / p * math.pi)  # cos_B_over_p
        
        return features

class DeltaRegressorV2:
    """Deep NN dự đoán δ với dataset chuẩn."""
    
    def __init__(self, feature_count: int = 24):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_count = feature_count

    def _build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.feature_count,), name='features')
        x = inputs
        
        # 10 hidden layers: 512->384->256->192->128->96->64->32->16->8
        for units in [512, 384, 256, 192, 128, 96, 64, 32, 16, 8]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.15)(x)
        
        outputs = layers.Dense(1, activation='linear', name='delta')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='delta_regressor_v2')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3), 
            loss='mse', 
            metrics=['mae']
        )
        return model

    def fit(self, X: np.ndarray, y_delta: np.ndarray, epochs: int = 50, batch_size: int = 256) -> Dict[str, Any]:
        print(f"Training Delta Regressor v2.0 (24 features, 10 hidden layers)...")
        print(f"Dataset: X={X.shape}, y={y_delta.shape}")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y_delta, test_size=0.2, random_state=42)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        
        self.model = self._build()
        
        # Callbacks
        early = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        start_time = time.time()
        hist = self.model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early, reduce],
            verbose=2
        )
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.1f}s")
        
        return {'history': hist.history, 'training_time': training_time}

    def evaluate(self, X: np.ndarray, y_delta: np.ndarray) -> Dict[str, float]:
        X_s = self.scaler.transform(X)
        loss, mae = self.model.evaluate(X_s, y_delta, verbose=0)
        return {
            'mse': float(loss), 
            'mae': float(mae), 
            'rmse': float(np.sqrt(loss))
        }

    def predict_delta(self, features: np.ndarray) -> float:
        features_s = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_s, verbose=0)[0, 0])

    def save(self, model_path='schoof_ai_regressor_v2.h5', scaler_path='schoof_ai_regressor_v2_scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path='schoof_ai_regressor_v2.h5', scaler_path='schoof_ai_regressor_v2_scaler.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

class CMClassifierV2:
    """CM/non-CM classifier với dataset chuẩn."""
    
    def __init__(self, feature_count: int = 24):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_count = feature_count

    def _build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.feature_count,))
        x = inputs
        
        # 3 hidden layers cho classifier
        for units in [128, 64, 32]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='cm_classifier_v2')
        model.compile(
            optimizer=optimizers.Adam(1e-3), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def fit(self, X: np.ndarray, y_cm: np.ndarray, epochs: int = 30, batch_size: int = 256):
        print(f"Training CM Classifier v2.0...")
        print(f"CM distribution: {int(y_cm.sum())} / {len(y_cm)} ({100*y_cm.sum()/len(y_cm):.2f}%)")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cm, test_size=0.2, random_state=42, 
            stratify=y_cm if y_cm.sum() > 0 else None
        )
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        
        self.model = self._build()
        
        # Class weights để xử lý imbalanced data
        class_weights = {0: 1.0, 1: len(y_cm) / (2 * y_cm.sum())} if y_cm.sum() > 0 else {0: 1.0, 1: 1.0}
        
        early = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        
        hist = self.model.fit(
            X_train_s, y_train, 
            validation_data=(X_val_s, y_val), 
            epochs=epochs, 
            batch_size=batch_size, 
            class_weight=class_weights,
            verbose=2, 
            callbacks=[early]
        )
        
        return {'history': hist.history}

    def predict_is_cm(self, features: np.ndarray) -> float:
        features_s = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_s, verbose=0)[0, 0])

    def save(self, model_path='schoof_ai_cm_classifier_v2.h5', scaler_path='schoof_ai_cm_v2_scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path='schoof_ai_cm_classifier_v2.h5', scaler_path='schoof_ai_cm_v2_scaler.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

class AISchoofAssistantV2:
    """Kết hợp Regressor + CM Classifier v2 để thu hẹp khoảng Hasse."""
    
    def __init__(self, regressor: DeltaRegressorV2, classifier: CMClassifierV2, feature_extractor: SchoofFeatureExtractor):
        self.regressor = regressor
        self.classifier = classifier
        self.feature_extractor = feature_extractor

    @staticmethod
    def hasse_interval(p: int) -> Tuple[int, int]:
        T = int(np.ceil(2.0 * np.sqrt(p)))
        return (p + 1 - T, p + 1 + T)

    def narrowed_hasse(self, p: int, A: int, B: int, k_sigma: float = 0.3) -> Tuple[int, int]:
        """Thu hẹp khoảng Hasse dựa trên δ dự đoán."""
        features = self.feature_extractor.extract_from_raw(p, A, B)
        delta_pred = self.regressor.predict_delta(features)
        
        N_est = int(round(p + 1 - delta_pred))
        low, high = self.hasse_interval(p)
        
        # Thu hẹp đối xứng xung quanh N_est
        T = int(np.ceil(2.0 * np.sqrt(p)))
        radius = max(1, int(k_sigma * T))
        low_n = max(low, N_est - radius)
        high_n = min(high, N_est + radius)
        
        return (low_n, high_n)

    def suggest_speedup(self, p: int, A: int, B: int) -> Dict[str, Any]:
        features = self.feature_extractor.extract_from_raw(p, A, B)
        
        low_h, high_h = self.hasse_interval(p)
        low_n, high_n = self.narrowed_hasse(p, A, B)
        
        width_h = high_h - low_h + 1
        width_n = high_n - low_n + 1
        
        cm_prob = self.classifier.predict_is_cm(features)
        delta_pred = self.regressor.predict_delta(features)
        
        return {
            'hasse_interval': (low_h, high_h),
            'narrowed_interval': (low_n, high_n),
            'width_reduction_factor': width_h / max(1, width_n),
            'cm_probability': cm_prob,
            'delta_prediction': delta_pred,
            'N_est': int(round((low_n + high_n) / 2)),
        }

def load_schoof_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Tải dataset Schoof chuẩn."""
    if not all(os.path.exists(f) for f in ['schoof_data_X.npy', 'schoof_data_delta.npy', 'schoof_data_cm.npy']):
        raise FileNotFoundError('Không tìm thấy dataset Schoof. Hãy chạy generate_schoof_dataset.py trước.')
    
    X = np.load('schoof_data_X.npy')
    y_delta = np.load('schoof_data_delta.npy')
    y_tilde_delta = np.load('schoof_data_tilde_delta.npy')
    y_cm = np.load('schoof_data_cm.npy')
    
    # Đọc tên đặc trưng
    with open('schoof_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return X, y_delta, y_tilde_delta, y_cm, feature_names

def plot_training_history(hist_reg, hist_clf):
    """Vẽ biểu đồ lịch sử huấn luyện."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Regressor loss
    axes[0, 0].plot(hist_reg['loss'], label='Train')
    axes[0, 0].plot(hist_reg['val_loss'], label='Validation')
    axes[0, 0].set_title('Delta Regressor - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Regressor MAE
    axes[0, 1].plot(hist_reg['mae'], label='Train')
    axes[0, 1].plot(hist_reg['val_mae'], label='Validation')
    axes[0, 1].set_title('Delta Regressor - MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Classifier loss
    axes[1, 0].plot(hist_clf['loss'], label='Train')
    axes[1, 0].plot(hist_clf['val_loss'], label='Validation')
    axes[1, 0].set_title('CM Classifier - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Binary Crossentropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Classifier accuracy
    axes[1, 1].plot(hist_clf['accuracy'], label='Train')
    axes[1, 1].plot(hist_clf['val_accuracy'], label='Validation')
    axes[1, 1].set_title('CM Classifier - Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('image/schoof_v2_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    ensure_image_dir()
    print('AI-ENHANCED SCHOOF v2.0 - TRAINING')
    print('=' * 60)
    
    # Tải dataset
    X, y_delta, y_tilde_delta, y_cm, feature_names = load_schoof_dataset()
    print(f'Loaded Schoof dataset: X={X.shape}, features={len(feature_names)}')
    print(f'Feature names: {feature_names[:5]}...{feature_names[-5:]}')
    
    # Khởi tạo feature extractor
    feature_extractor = SchoofFeatureExtractor(feature_names)
    
    # Huấn luyện Regressor (δ)
    reg = DeltaRegressorV2(feature_count=len(feature_names))
    print('\nTraining Delta Regressor v2.0...')
    start = time.time()
    reg_hist = reg.fit(X, y_delta, epochs=40, batch_size=256)
    eval_reg = reg.evaluate(X, y_delta)
    reg.save()
    print(f"Regressor v2.0 saved. Eval: {eval_reg}")
    print(f'Time: {time.time()-start:.1f}s')
    
    # Huấn luyện CM Classifier
    clf = CMClassifierV2(feature_count=len(feature_names))
    print('\nTraining CM Classifier v2.0...')
    clf_hist = clf.fit(X, y_cm, epochs=25, batch_size=256)
    clf.save()
    print('CM Classifier v2.0 saved.')
    
    # Vẽ biểu đồ
    plot_training_history(reg_hist['history'], clf_hist['history'])
    
    # Demo thu hẹp khoảng Hasse
    assistant = AISchoofAssistantV2(reg, clf, feature_extractor)
    
    # Test với một số mẫu
    test_cases = [
        (17, 5, 3),
        (101, 23, 45),
        (257, 67, 89),
        (499, 123, 456)
    ]
    
    print('\nHasse narrowing demo:')
    for p, A, B in test_cases:
        sug = assistant.suggest_speedup(p, A, B)
        print(f'p={p}, A={A}, B={B}:')
        print(f'  Hasse: {sug["hasse_interval"]} (width: {sug["hasse_interval"][1] - sug["hasse_interval"][0] + 1})')
        print(f'  Narrowed: {sug["narrowed_interval"]} (width: {sug["narrowed_interval"][1] - sug["narrowed_interval"][0] + 1})')
        print(f'  Reduction: {sug["width_reduction_factor"]:.2f}x')
        print(f'  CM prob: {sug["cm_probability"]:.4f}')
        print(f'  δ pred: {sug["delta_prediction"]:.2f}')
        print()
    
    print('🎉 Done.')

if __name__ == '__main__':
    main() 