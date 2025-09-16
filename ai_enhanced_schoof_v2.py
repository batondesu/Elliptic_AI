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
from sklearn.metrics import r2_score, mean_squared_error, precision_recall_fscore_support, roc_auc_score, average_precision_score
import joblib
import matplotlib.pyplot as plt

np.random.seed(42)

def ensure_image_dir():
    os.makedirs('image', exist_ok=True)

def ensure_log_dir(subdir: str) -> str:
    base = os.path.join('logs', subdir)
    os.makedirs(base, exist_ok=True)
    return base

def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

class RegressionMetricsCallback(callbacks.Callback):
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, scaler: StandardScaler, tb_log_dir: str = None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler
        self.tb_log_dir = tb_log_dir
        self.writer = None
        if tb_log_dir is not None:
            self.writer = tf.summary.create_file_writer(tb_log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        try:
            Xs = self.scaler.transform(self.X_val)
            y_pred = self.model.predict(Xs, verbose=0).flatten()
            mse = mean_squared_error(self.y_val, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(self.y_val, y_pred))
            logs['val_rmse'] = rmse
            logs['val_r2'] = r2
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.scalar('val_rmse', rmse, step=epoch)
                    tf.summary.scalar('val_r2', r2, step=epoch)
        except Exception:
            pass

class ClassificationMetricsCallback(callbacks.Callback):
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, scaler: StandardScaler, tb_log_dir: str = None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler
        self.tb_log_dir = tb_log_dir
        self.writer = None
        if tb_log_dir is not None:
            self.writer = tf.summary.create_file_writer(tb_log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        try:
            Xs = self.scaler.transform(self.X_val)
            y_prob = self.model.predict(Xs, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_val, y_pred, average='binary', zero_division=0)
            try:
                auc = float(roc_auc_score(self.y_val, y_prob))
            except Exception:
                auc = float('nan')
            try:
                aupr = float(average_precision_score(self.y_val, y_prob))
            except Exception:
                aupr = float('nan')
            logs['val_precision'] = float(precision)
            logs['val_recall'] = float(recall)
            logs['val_f1'] = float(f1)
            logs['val_auc'] = auc
            logs['val_aupr'] = aupr
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.scalar('val_precision', float(precision), step=epoch)
                    tf.summary.scalar('val_recall', float(recall), step=epoch)
                    tf.summary.scalar('val_f1', float(f1), step=epoch)
                    if not np.isnan(auc):
                        tf.summary.scalar('val_auc', auc, step=epoch)
                    if not np.isnan(aupr):
                        tf.summary.scalar('val_aupr', aupr, step=epoch)
        except Exception:
            pass

class SchoofFeatureExtractor:
    """Trích xuất đặc trưng từ dataset Schoof (kết hợp features gốc + rich)."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
    
    def extract_from_raw(self, p: int, A: int, B: int) -> np.ndarray:
        from feature_explanation import extract_features, extract_features_rich
        base_feats = extract_features(p, A, B)
        rich_feats = extract_features_rich(p, A, B, sample_x=16)
        return np.array(list(base_feats) + list(rich_feats), dtype=np.float32)

class DeltaRegressorV2:
    """Deep NN dự đoán δ với 94 features đã cleaned."""
    
    def __init__(self, feature_count: int = 94):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_count = feature_count

    def _build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.feature_count,), name='features')
        x = inputs
        
        # 8 hidden layers dạng residual block "chuẩn" với projection shortcut khi cần
        layer_sizes = [512, 256, 128, 64, 32, 16, 8, 16, 32]

        for units in layer_sizes:
            shortcut = x
            x = layers.Dense(units, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            # Projection cho shortcut nếu dimension không khớp
            if int(shortcut.shape[-1]) != units:
                shortcut = layers.Dense(units, activation=None, use_bias=False)(shortcut)
            x = layers.Add()([x, shortcut])

        outputs = layers.Dense(1, activation='linear', name='delta')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='delta_regressor_v2')
        # Compile tạm; sẽ re-compile trong fit với CosineDecayRestarts
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=2e-3, weight_decay=1e-3),
            loss='huber',
            metrics=['mae', 'mse', r2_metric]
        )
        return model

    def fit(self, X: np.ndarray, y_delta: np.ndarray, epochs: int = 300, batch_size: int = 1024, 
            use_early_stopping: bool = True, patience: int = 25, resume: bool = True) -> Dict[str, Any]:  # Tăng epochs, giảm batch size
        print(f"Training Delta Regressor v2.0 (40 features, 8 hidden layers)...")
        print(f"Dataset: X={X.shape}, y={y_delta.shape}")
        
        # Tách test set cố định rồi mới tách train/val
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_delta, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

        # Scale: nếu resume được và đã có scaler thì dùng transform, ngược lại fit mới
        resumed = False
        
        # Kiểm tra xem có model cũ để resume không (ưu tiên best checkpoint)
        model_path = 'schoof_ai_regressor_v2.h5'
        best_model_path = 'schoof_ai_regressor_v2_best.h5'
        scaler_path = 'schoof_ai_regressor_v2_scaler.pkl'
        
        if resume and os.path.exists(scaler_path) and (os.path.exists(best_model_path) or os.path.exists(model_path)):
            print("🔄 RESUMING Delta Regressor từ model cũ (ưu tiên best checkpoint)...")
            try:
                load_path = best_model_path if os.path.exists(best_model_path) else model_path
                self.model = tf.keras.models.load_model(load_path, custom_objects={'r2_metric': r2_metric})
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded: {load_path} và scaler!")
                resumed = True
                # Kiểm tra đổi số features (ví dụ 40 -> 94)
                loaded_in_features = int(self.model.input_shape[-1]) if self.model is not None else self.feature_count
                current_in_features = X.shape[1]
                if loaded_in_features != current_in_features:
                    print(f"⚠️ Feature count mismatch (loaded={loaded_in_features}, current={current_in_features}). Rebuilding fresh model...")
                    self.model = self._build()
                    self.scaler = StandardScaler()
                    resumed = False
                    # reset optimizer (compile lại với schedule mới)
                    initial_lr = 2e-3
                    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                        initial_learning_rate=initial_lr,
                        first_decay_steps=epochs // 5,
                        t_mul=2.0,
                        m_mul=0.9,
                        alpha=1e-4
                    )
                    self.model.compile(
                        optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-3),
                        loss='huber',
                        metrics=['mae', 'mse', r2_metric]
                    )
            except Exception as e:
                print(f"⚠️ Không thể load Delta Regressor cũ: {e}")
                print("🆕 Tạo Delta Regressor mới...")
                self.model = self._build()
        else:
            print("🆕 Tạo Delta Regressor mới...")
            self.model = self._build()

        if resumed:
            try:
                X_train_s = self.scaler.transform(X_train)
                X_val_s = self.scaler.transform(X_val)
                X_test_s = self.scaler.transform(X_test)
            except Exception as e:
                self.model = self._build()
                self.scaler = StandardScaler()
                X_train_s = self.scaler.fit_transform(X_train)
                X_val_s = self.scaler.transform(X_val)
                X_test_s = self.scaler.transform(X_test)
                resumed = False
        else:
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            X_test_s = self.scaler.transform(X_test)

        print(f"Splits: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

        # Callbacks
        callbacks_list = []
        if use_early_stopping:
            early = callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
            callbacks_list.append(early)
        # Learning rate scheduling
        # Dùng CosineDecayRestarts thay cho ReduceLROnPlateau
        initial_lr = 2e-3
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=epochs // 5,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-4
        )
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-3),
            loss='huber',
            metrics=['mae', 'mse', r2_metric]
        )
        # ModelCheckpoint (best model)
        ckpt_path = 'schoof_ai_regressor_v2_best.h5'
        ckpt = callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
        callbacks_list.append(ckpt)
        # TensorBoard + custom regression metrics
        log_dir = ensure_log_dir(os.path.join('regressor', time.strftime("%Y%m%d-%H%M%S")))
        tb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)
        callbacks_list.append(tb)
        callbacks_list.append(RegressionMetricsCallback(X_val, y_val, self.scaler, tb_log_dir=log_dir))
        
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
        
        # Đánh giá trên test set (compile có thêm r2_metric nên trả về 4 giá trị)
        test_loss, test_mae, test_mse, test_r2_keras = self.model.evaluate(X_test_s, y_test, verbose=0)
        y_pred_test = self.model.predict(X_test_s, verbose=0).flatten()
        test_r2 = float(r2_score(y_test, y_pred_test))
        print(f"Test MSE: {float(test_mse):.4f} | Test MAE: {float(test_mae):.4f} | Test R² (sklearn): {test_r2:.4f} | Test R² (keras): {float(test_r2_keras):.4f}")
        
        # Load lại best trước khi save chính thức
        if os.path.exists(ckpt_path):
            try:
                self.model = tf.keras.models.load_model(ckpt_path, custom_objects={'r2_metric': r2_metric})
                print("✅ Loaded best regressor checkpoint for final save.")
            except Exception as e:
                print(f"⚠️ Không thể load best checkpoint regressor: {e}")

        return {'history': hist.history, 'training_time': training_time}

    def evaluate(self, X: np.ndarray, y_delta: np.ndarray) -> Dict[str, float]:
        X_s = self.scaler.transform(X)
        # compile có thêm r2_metric nên evaluate trả về 4 giá trị
        loss, mae, mse, r2k = self.model.evaluate(X_s, y_delta, verbose=0)
        # Tính thêm R² theo sklearn để đối chiếu
        y_pred = self.model.predict(X_s, verbose=0).flatten()
        r2s = float(r2_score(y_delta, y_pred))
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2_keras': float(r2k),
            'r2_sklearn': r2s
        }

    def predict_delta(self, features: np.ndarray) -> float:
        features_s = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_s, verbose=0)[0, 0])

    def save(self, model_path='schoof_ai_regressor_v2.h5', scaler_path='schoof_ai_regressor_v2_scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path='schoof_ai_regressor_v2.h5', scaler_path='schoof_ai_regressor_v2_scaler.pkl'):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'r2_metric': r2_metric})
        self.scaler = joblib.load(scaler_path)

class CMClassifierV2:
    """CM/non-CM classifier với 94 features đã cleaned. 8 hidden layers."""
    
    def __init__(self, feature_count: int = 94):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_count = feature_count
        self.best_threshold = 0.5

    def _build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.feature_count,))
        x = inputs
        
        # Residual blocks chuẩn với projection
        for units in [512, 256, 128, 64, 32, 16]:
            shortcut = x
            x = layers.Dense(units, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            if int(shortcut.shape[-1]) != units:
                shortcut = layers.Dense(units, activation=None, use_bias=False)(shortcut)
            x = layers.Add()([x, shortcut])
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='cm_classifier_v2')
        model.compile(
            optimizer=optimizers.AdamW(1e-3, weight_decay=1e-3),  # Tăng learning rate và weight decay
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def fit(self, X: np.ndarray, y_cm: np.ndarray, epochs: int = 200, batch_size: int = 2048,
            use_early_stopping: bool = True, patience: int = 25, resume: bool = True):  # Cấu hình theo đề xuất
        print(f"Training CM Classifier v2.0...")
        print(f"CM distribution: {int(y_cm.sum())} / {len(y_cm)} ({100*y_cm.sum()/len(y_cm):.2f}%)")
        
        # Tách test set cố định rồi mới tách train/val, stratify nếu có thể
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_cm, test_size=0.1, random_state=42, stratify=y_cm if y_cm.sum() > 0 else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp if y_cm.sum() > 0 else None
        )
        
        resumed = False
        
        # Kiểm tra xem có model cũ để resume không (ưu tiên best checkpoint)
        model_path = 'schoof_ai_cm_classifier_v2.h5'
        best_model_path = 'schoof_ai_cm_classifier_v2_best.h5'
        scaler_path = 'schoof_ai_cm_v2_scaler.pkl'
        
        if resume and os.path.exists(scaler_path) and (os.path.exists(best_model_path) or os.path.exists(model_path)):
            print("🔄 RESUMING CM Classifier từ model cũ (ưu tiên best checkpoint)...")
            try:
                load_path = best_model_path if os.path.exists(best_model_path) else model_path
                self.model = tf.keras.models.load_model(load_path)
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded: {load_path} và scaler!")
                resumed = True
                # Kiểm tra thay đổi số features
                loaded_in_features = int(self.model.input_shape[-1]) if self.model is not None else self.feature_count
                current_in_features = X.shape[1]
                if loaded_in_features != current_in_features:
                    self.model = self._build()
                    self.scaler = StandardScaler()
                    resumed = False
            except Exception as e:
                print(f"⚠️ Không thể load CM Classifier cũ: {e}")
                print("🆕 Tạo CM Classifier mới...")
                self.model = self._build()
        else:
            print("🆕 Tạo CM Classifier mới...")
            self.model = self._build()
        
        if resumed:
            try:
                X_train_s = self.scaler.transform(X_train)
                X_val_s = self.scaler.transform(X_val)
                X_test_s = self.scaler.transform(X_test)
            except Exception as e:
                self.model = self._build()
                self.scaler = StandardScaler()
                X_train_s = self.scaler.fit_transform(X_train)
                X_val_s = self.scaler.transform(X_val)
                X_test_s = self.scaler.transform(X_test)
                resumed = False
        else:
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            X_test_s = self.scaler.transform(X_test)

        print(f"Splits: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

        # class_weight: upweight lớp thiểu số
        n_pos = float(y_cm.sum())
        n_neg = float(len(y_cm) - n_pos)
        if n_pos == 0 or n_neg == 0:
            class_weights = None
        else:
            cap = 1e5
            if n_pos < n_neg:
                # Minority = 1 (CM)
                factor = min(cap, (n_neg / n_pos))
                class_weights = {0: 1.0, 1: factor}
            else:
                # Minority = 0 (non-CM)
                factor = min(cap, (n_pos / n_neg))
                class_weights = {0: factor, 1: 1.0}
        
        callbacks_list = []
        if use_early_stopping:
            early = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks_list.append(early)
        # ReduceLROnPlateau
        reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
        callbacks_list.append(reduce)
        # ModelCheckpoint + TensorBoard + custom metrics (AUC, Precision, Recall, F1)
        ckpt_path = 'schoof_ai_cm_classifier_v2_best.h5'
        ckpt = callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
        callbacks_list.append(ckpt)
        log_dir = ensure_log_dir(os.path.join('classifier', time.strftime("%Y%m%d-%H%M%S")))
        tb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)
        callbacks_list.append(tb)
        callbacks_list.append(ClassificationMetricsCallback(X_val, y_val, self.scaler, tb_log_dir=log_dir))

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
        # Tìm ngưỡng tối ưu theo F1 trên validation và lưu
        try:
            y_prob_val = self.model.predict(X_val_s, verbose=0).flatten()
            import numpy as _np
            from sklearn.metrics import precision_recall_fscore_support as _prfs
            thresholds = _np.unique(_np.concatenate(([0.0], _np.percentile(y_prob_val, _np.linspace(0,100,1001)))))
            best = (0.5, 0.0)
            for t in thresholds:
                y_hat = (y_prob_val >= t).astype(int)
                _, _, f1, _ = _prfs(y_val, y_hat, average='binary', zero_division=0)
                if f1 > best[1]:
                    best = (float(t), float(f1))
            self.best_threshold = best[0]
            try:
                with open('schoof_ai_cm_threshold.txt', 'w') as f:
                    f.write(str(self.best_threshold))
            except Exception:
                pass
            print(f"Best F1 threshold on val: t={self.best_threshold:.6f} (F1={best[1]:.4f})")
        except Exception as _e:
            print(f"⚠️ Không thể tính threshold tối ưu: {_e}")
        # Đánh giá trên test set
        # Load lại best trước khi evaluate + save
        if os.path.exists(ckpt_path):
            try:
                self.model = tf.keras.models.load_model(ckpt_path)
                print("✅ Loaded best classifier checkpoint for evaluation.")
            except Exception as e:
                print(f"⚠️ Không thể load best checkpoint classifier: {e}")
        test_loss, test_acc = self.model.evaluate(X_test_s, y_test, verbose=0)
        y_prob_test = self.model.predict(X_test_s, verbose=0).flatten()
        # dùng threshold tốt nhất nếu có
        thr = getattr(self, 'best_threshold', 0.5)
        y_pred_test = (y_prob_test >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary', zero_division=0)
        try:
            auc = float(roc_auc_score(y_test, y_prob_test))
        except Exception:
            auc = float('nan')
        try:
            aupr = float(average_precision_score(y_test, y_prob_test))
        except Exception:
            aupr = float('nan')
        print(f"Test Acc: {float(test_acc):.4f} | P: {float(precision):.4f} | R: {float(recall):.4f} | F1: {float(f1):.4f} | AUC: {auc if not np.isnan(auc) else 'nan'} | AUCPR: {aupr if not np.isnan(aupr) else 'nan'} | thr: {thr:.4f}")
        
        return {'history': hist.history}

    def predict_is_cm(self, features: np.ndarray) -> float:
        features_s = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_s, verbose=0)[0, 0])

    def save(self, model_path='schoof_ai_cm_classifier_v2.h5', scaler_path='schoof_ai_cm_v2_scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        try:
            with open('schoof_ai_cm_threshold.txt', 'w') as f:
                f.write(str(getattr(self, 'best_threshold', 0.5)))
        except Exception:
            pass

    def load(self, model_path='schoof_ai_cm_classifier_v2.h5', scaler_path='schoof_ai_cm_v2_scaler.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        try:
            with open('schoof_ai_cm_threshold.txt', 'r') as f:
                self.best_threshold = float(f.read().strip())
        except Exception:
            self.best_threshold = 0.5

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
    needed = ['schoof_data_X_cleaned.npy', 'schoof_data_delta.npy', 'schoof_data_cm.npy']
    if not all(os.path.exists(f) for f in needed):
        raise FileNotFoundError('Không tìm thấy dataset Schoof cleaned. Hãy chạy fix_dataset_issues.py trước.')
    
    X = np.load('schoof_data_X_cleaned.npy')
    y_delta = np.load('schoof_data_delta.npy')
    # y_tilde_delta là tuỳ chọn: nếu không có, tạo mảng zeros để giữ API
    if os.path.exists('schoof_data_tilde_delta.npy'):
        y_tilde_delta = np.load('schoof_data_tilde_delta.npy')
    else:
        y_tilde_delta = np.zeros_like(y_delta)
    y_cm = np.load('schoof_data_cm.npy')
    
    with open('schoof_feature_names_cleaned.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return X, y_delta, y_tilde_delta, y_cm, feature_names

def plot_training_history(hist_reg, hist_clf):
    """Vẽ biểu đồ lịch sử huấn luyện."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Regressor loss (Huber)
    axes[0, 0].plot(hist_reg['loss'], label='Train')
    axes[0, 0].plot(hist_reg['val_loss'], label='Validation')
    axes[0, 0].set_title('Delta Regressor - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Huber loss')
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

def main(resume_training=True):
    ensure_image_dir()
    print('AI-ENHANCED SCHOOF v2.0 (ENHANCED) - TRAINING')
    print('=' * 70)
    
    if resume_training:
        print("🔄 Sẽ tiếp tục training từ model cũ (nếu có)")
    else:
        print("🆕 Sẽ train từ đầu (bỏ qua model cũ)")
    
    # Tải dataset
    X, y_delta, y_tilde_delta, y_cm, feature_names = load_schoof_dataset()
    print(f'Loaded Enhanced Schoof dataset: X={X.shape}, features={len(feature_names)}')
    print(f'Feature names: {feature_names[:5]}...{feature_names[-5:]}')
    
    # Khởi tạo feature extractor
    feature_extractor = SchoofFeatureExtractor(feature_names)
    
    # Cấu hình huấn luyện cho dataset ~200k (theo đề xuất)
    USE_EARLY_STOPPING = True
    REGRESSOR_EPOCHS = 300
    CLASSIFIER_EPOCHS = 200
    REGRESSOR_PATIENCE = 40
    CLASSIFIER_PATIENCE = 25
    REGRESSOR_BATCH = 1024
    CLASSIFIER_BATCH = 2048
    
    print(f'\nTraining configuration:')
    print(f'  Early stopping: {USE_EARLY_STOPPING}')
    print(f'  Regressor epochs: {REGRESSOR_EPOCHS}')
    print(f'  Classifier epochs: {CLASSIFIER_EPOCHS}')
    print(f'  Regressor patience: {REGRESSOR_PATIENCE}')
    print(f'  Classifier patience: {CLASSIFIER_PATIENCE}')
    print(f'  Regressor batch size: {REGRESSOR_BATCH}')
    print(f'  Classifier batch size: {CLASSIFIER_BATCH}')
    print(f'  Learning rate: 2e-3 (AdamW)')
    print(f'  Weight decay: 1e-3')
    print(f'  Split: test 10% | val 20% của phần train')
    
    # Huấn luyện Regressor (δ) chỉ với non-CM
    noncm_mask = (y_cm == 0)
    X_noncm = X[noncm_mask]
    y_delta_noncm = y_delta[noncm_mask]
    print(f"\nTraining Delta Regressor v2.0 (Enhanced) on non-CM only: {X_noncm.shape[0]} samples...")
    reg = DeltaRegressorV2(feature_count=X.shape[1])
    start = time.time()
    reg_hist = reg.fit(X_noncm, y_delta_noncm, epochs=REGRESSOR_EPOCHS, batch_size=REGRESSOR_BATCH, 
                      use_early_stopping=USE_EARLY_STOPPING, patience=REGRESSOR_PATIENCE, resume=resume_training)
    eval_reg = reg.evaluate(X_noncm, y_delta_noncm)
    reg.save()
    print(f"Regressor v2.0 (Enhanced) saved. Eval: {eval_reg}")
    print(f'Time: {time.time()-start:.1f}s')
    
    # Huấn luyện CM Classifier
    clf = CMClassifierV2(feature_count=X.shape[1])
    print('\nTraining CM Classifier v2.0 (Enhanced)...')
    clf_hist = clf.fit(X, y_cm, epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH,
                      use_early_stopping=USE_EARLY_STOPPING, patience=CLASSIFIER_PATIENCE, resume=resume_training)
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
        # Gợi ý: cần so với ground truth delta hoặc N thật nếu có
        print()
    
    print('🎉 AI-Enhanced Schoof v2.0 (Enhanced) training completed!')

if __name__ == '__main__':
    import sys
    
    # Kiểm tra tham số command line
    resume_training = True
    if len(sys.argv) > 1:
        if sys.argv[1] == '--fresh' or sys.argv[1] == '-f':
            resume_training = False
            print("🆕 Fresh training mode - sẽ train từ đầu")
        elif sys.argv[1] == '--resume' or sys.argv[1] == '-r':
            resume_training = True
            print("🔄 Resume training mode - sẽ tiếp tục từ model cũ")
    
    main(resume_training) 