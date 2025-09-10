#!/usr/bin/env python3
"""
AI-enhanced Schoof's Algorithm v2.0 (Enhanced)
- Sử dụng dataset với 92 đặc trưng toán học nâng cao
- Deep Neural Network (12 hidden layers với residual connections) dự đoán δ
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
    """Trích xuất đặc trưng từ dataset Schoof với 40 features đã cleaned."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
    
    def extract_from_raw(self, p: int, A: int, B: int) -> np.ndarray:
        from feature_explanation import extract_features
        return extract_features(p, A, B)

class DeltaRegressorV2:
    """Deep NN dự đoán δ với 40 features đã cleaned."""
    
    def __init__(self, feature_count: int = 40):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_count = feature_count

    def _build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.feature_count,), name='features')
        x = inputs
        
        # 12 hidden layers với residual connections và regularization mạnh hơn
        # 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64 -> 128
        layer_sizes = [1024, 512, 256, 128, 64, 32, 16, 8, 16, 32, 64, 128]
        
        for i, units in enumerate(layer_sizes):
            # Residual connection cho layers lớn
            if i < 6 and units >= 64:
                residual = x
                x = layers.Dense(units, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)  # Tăng dropout
                # Add residual connection
                if x.shape[-1] == residual.shape[-1]:
                    x = layers.Add()([x, residual])
            else:
                x = layers.Dense(units, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.25)(x)  # Tăng dropout
        
        outputs = layers.Dense(1, activation='linear', name='delta')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='delta_regressor_v2')
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=2e-3, weight_decay=1e-3),  # Tăng learning rate và weight decay
            loss='huber',  # Huber loss cho robustness
            metrics=['mae', 'mse']
        )
        return model

    def fit(self, X: np.ndarray, y_delta: np.ndarray, epochs: int = 150, batch_size: int = 64, 
            use_early_stopping: bool = True, patience: int = 25) -> Dict[str, Any]:  # Tăng epochs, giảm batch size
        print(f"Training Delta Regressor v2.0 (40 features, 12 hidden layers)...")
        print(f"Dataset: X={X.shape}, y={y_delta.shape}")
        
        # Tăng validation split để có nhiều dữ liệu validation hơn
        X_train, X_val, y_train, y_val = train_test_split(X, y_delta, test_size=0.25, random_state=42)  # Tăng từ 0.2 lên 0.25
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        
        self.model = self._build()
        
        # Callbacks
        callbacks_list = []
        if use_early_stopping:
            early = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks_list.append(early)
        
        # Learning rate scheduling với patience thấp hơn
        reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)  # Tăng min_lr
        callbacks_list.append(reduce)
        
        start_time = time.time()
        hist = self.model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=2
        )
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.1f}s")
        print(f"Completed epochs: {len(hist.history['loss'])}/{epochs}")
        
        return {'history': hist.history, 'training_time': training_time}

    def evaluate(self, X: np.ndarray, y_delta: np.ndarray) -> Dict[str, float]:
        X_s = self.scaler.transform(X)
        loss, mae, mse = self.model.evaluate(X_s, y_delta, verbose=0)
        return {
            'mse': float(mse), 
            'mae': float(mae), 
            'rmse': float(np.sqrt(mse))
        }

    def predict_delta(self, features: np.ndarray) -> float:
        features_s = self.scaler.transform(features)
        return float(self.model.predict(features_s, verbose=0)[0, 0])

    def save(self, model_path='schoof_ai_regressor_v2.h5', scaler_path='schoof_ai_regressor_v2_scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path='schoof_ai_regressor_v2.h5', scaler_path='schoof_ai_regressor_v2_scaler.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

class CMClassifierV2:
    """CM/non-CM classifier với 40 features đã cleaned."""
    
    def __init__(self, feature_count: int = 40):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_count = feature_count

    def _build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.feature_count,))
        x = inputs
        
        # 5 hidden layers cho classifier
        for units in [256, 128, 64, 32, 16]:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)  # Tăng dropout
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='cm_classifier_v2')
        model.compile(
            optimizer=optimizers.AdamW(1e-3, weight_decay=1e-3),  # Tăng learning rate và weight decay
            loss='binary_crossentropy', 
            metrics=['accuracy']  # Chỉ dùng accuracy, bỏ precision, recall
        )
        return model

    def fit(self, X: np.ndarray, y_cm: np.ndarray, epochs: int = 100, batch_size: int = 128,
            use_early_stopping: bool = True, patience: int = 15):  # Tăng epochs, giảm batch size
        print(f"Training CM Classifier v2.0...")
        print(f"CM distribution: {int(y_cm.sum())} / {len(y_cm)} ({100*y_cm.sum()/len(y_cm):.2f}%)")
        
        # Tăng validation split để có nhiều dữ liệu validation hơn
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cm, test_size=0.25, random_state=42,  # Tăng từ 0.2 lên 0.25
            stratify=y_cm if y_cm.sum() > 0 else None
        )
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        
        self.model = self._build()
        
        # Class weights để xử lý imbalanced data
        class_weights = {0: 1.0, 1: len(y_cm) / (2 * y_cm.sum())} if y_cm.sum() > 0 else {0: 1.0, 1: 1.0}
        
        callbacks_list = []
        if use_early_stopping:
            early = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks_list.append(early)
        
        hist = self.model.fit(
            X_train_s, y_train, 
            validation_data=(X_val_s, y_val), 
            epochs=epochs, 
            batch_size=batch_size, 
            class_weight=class_weights,
            verbose=2, 
            callbacks=callbacks_list
        )
        
        print(f"Completed epochs: {len(hist.history['loss'])}/{epochs}")
        
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
    """Tải dataset Schoof đã được làm sạch với 40 features."""
    if not all(os.path.exists(f) for f in ['schoof_data_X_cleaned.npy', 'schoof_data_delta.npy', 'schoof_data_cm.npy']):
        raise FileNotFoundError('Không tìm thấy dataset Schoof cleaned. Hãy chạy fix_dataset_issues.py trước.')
    
    X = np.load('schoof_data_X_cleaned.npy')
    y_delta = np.load('schoof_data_delta.npy')
    y_tilde_delta = np.load('schoof_data_tilde_delta.npy')
    y_cm = np.load('schoof_data_cm.npy')
    
    # Đọc tên đặc trưng
    with open('schoof_feature_names_cleaned.txt', 'r') as f:
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
    print('AI-ENHANCED SCHOOF v2.0 (ENHANCED) - TRAINING')
    print('=' * 70)
    
    # Tải dataset
    X, y_delta, y_tilde_delta, y_cm, feature_names = load_schoof_dataset()
    print(f'Loaded Enhanced Schoof dataset: X={X.shape}, features={len(feature_names)}')
    print(f'Feature names: {feature_names[:5]}...{feature_names[-5:]}')
    
    # Khởi tạo feature extractor
    feature_extractor = SchoofFeatureExtractor(feature_names)
    
    # Cấu hình huấn luyện với learning rate cao hơn
    USE_EARLY_STOPPING = True   # Bật early stopping
    REGRESSOR_EPOCHS = 100      # Tăng epochs
    CLASSIFIER_EPOCHS = 100     # Tăng epochs
    REGRESSOR_PATIENCE = 25     # Giảm patience
    CLASSIFIER_PATIENCE = 15    # Giảm patience
    
    print(f'\nTraining configuration:')
    print(f'  Early stopping: {USE_EARLY_STOPPING}')
    print(f'  Regressor epochs: {REGRESSOR_EPOCHS}')
    print(f'  Classifier epochs: {CLASSIFIER_EPOCHS}')
    print(f'  Regressor patience: {REGRESSOR_PATIENCE}')
    print(f'  Classifier patience: {CLASSIFIER_PATIENCE}')
    print(f'  Learning rate: 2e-3 (increased for better learning)')
    print(f'  Weight decay: 1e-3 (increased regularization)')
    print(f'  Validation split: 25% (increased from 20%)')
    
    # Huấn luyện Regressor (δ)
    reg = DeltaRegressorV2(feature_count=len(feature_names))
    print('\nTraining Delta Regressor v2.0 (Enhanced)...')
    start = time.time()
    reg_hist = reg.fit(X, y_delta, epochs=REGRESSOR_EPOCHS, batch_size=64, 
                      use_early_stopping=USE_EARLY_STOPPING, patience=REGRESSOR_PATIENCE)
    eval_reg = reg.evaluate(X, y_delta)
    reg.save()
    print(f"Regressor v2.0 (Enhanced) saved. Eval: {eval_reg}")
    print(f'Time: {time.time()-start:.1f}s')
    
    # Huấn luyện CM Classifier
    clf = CMClassifierV2(feature_count=len(feature_names))
    print('\nTraining CM Classifier v2.0 (Enhanced)...')
    clf_hist = clf.fit(X, y_cm, epochs=CLASSIFIER_EPOCHS, batch_size=128,
                      use_early_stopping=USE_EARLY_STOPPING, patience=CLASSIFIER_PATIENCE)
    clf.save()
    print('CM Classifier v2.0 (Enhanced) saved.')
    
    # Vẽ biểu đồ
    plot_training_history(reg_hist['history'], clf_hist['history'])
    
    # Demo thu hẹp khoảng Hasse
    assistant = AISchoofAssistantV2(reg, clf, feature_extractor)
    
    # Test với một số mẫu
    test_cases = [
        (17, 5, 3),
        (101, 23, 45),
        (257, 67, 89),
        (499, 123, 456),
        (503, 127, 461),
        (857, 281, 733),
        (1003, 341, 881),
        (2011, 523, 1033),
        (4003, 991, 2011),
        (6001, 1627, 3041),
        (7001, 1979, 3581),
        (8009, 2407, 4129),
        (9001, 2729, 4661),
        (13007, 4261, 8811),
        (15001, 4261, 8811),
        (16001, 4261, 8811),
        (19001, 4261, 8811)
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
        print(f'  Ket qua thuc te: {sug["N_est"]}')
        print(f'  Chenh lech: {abs(sug["N_est"] - (p + 1))} (so diem)')
        print(f'  Delta Reduction: {abs(sug["delta_prediction"] - (p + 1)):.2f}')
        print()
    
    print('🎉 AI-Enhanced Schoof v2.0 (Enhanced) training completed!')

if __name__ == '__main__':
    main() 