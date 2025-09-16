#!/usr/bin/env python3
"""
Simple Training System cho Schoof - Phiên bản cuối cùng
- Một model duy nhất (final_model) học kỹ
- Thứ tự layer: BatchNorm -> ReLU -> Dropout
- Tối ưu hóa hyperparameters
- Monitoring chi tiết
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import json

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l2

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Utils
import joblib

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class SimpleTrainingModel:
    """Model đơn giản nhưng học kỹ cho Schoof."""
    
    def __init__(self, feature_count: int = 40, model_name: str = "final_model"):
        self.feature_count = feature_count
        self.model_name = model_name
        self.model = None
        self.scaler = RobustScaler()
        
        # Tạo thư mục lưu model
        os.makedirs('models', exist_ok=True)
        self.model_path = f'models/{model_name}.h5'
        self.scaler_path = f'models/{model_name}_scaler.pkl'
        self.y_scaler = RobustScaler()
        self.y_scaler_path = f'models/{model_name}_y_scaler.pkl'
        
    def create_model(self, 
                    hidden_layers: list = [512, 256, 128, 64, 32, 16],
                    dropout_rate: float = 0.3,
                    l2_reg: float = 1e-5,
                    learning_rate: float = 3e-4) -> tf.keras.Model:
        """Tạo model với cấu hình tùy chỉnh."""
        
        print(f"🔧 Tạo model với cấu hình:")
        print(f"   Hidden layers: {hidden_layers}")
        print(f"   Dropout rate: {dropout_rate}")
        print(f"   L2 regularization: {l2_reg}")
        print(f"   Learning rate: {learning_rate}")
        
        inputs = layers.Input(shape=(self.feature_count,), name='features')
        x = inputs
        
        # Xây dựng hidden layers theo thứ tự: BatchNorm -> ReLU -> Dropout
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, 
                           kernel_regularizer=l2(l2_reg),
                           name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)  # BatchNorm trước
            x = layers.ReLU(name=f'relu_{i+1}')(x)             # ReLU sau
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)  # Dropout cuối
        
        # Output layer
        outputs = layers.Dense(1, activation='linear', name='delta_output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='schoof_training_model')
        
        # Compile với optimizer
        optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=l2_reg
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Huber loss cho robustness
            metrics=['mae', 'mse']
        )
        
        return model
    
    def load_existing_model(self) -> bool:
        """Tải model hiện có nếu tồn tại."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                if os.path.exists(self.y_scaler_path):
                    self.y_scaler = joblib.load(self.y_scaler_path)
                    print(f"✅ Đã tải model và scalers hiện có: {self.model_path}")
                else:
                    print(f"✅ Đã tải model hiện có: {self.model_path}")
                return True
            except Exception as e:
                print(f"❌ Lỗi khi tải model: {e}")
                return False
        return False
    
    def save_model(self):
        """Lưu model và scaler."""
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"💾 Đã lưu model: {self.model_path}")
    
    def train(self, X: np.ndarray, y: np.ndarray,
              hidden_layers: list = [512, 256, 128, 64, 32, 16],
              dropout_rate: float = 0.3,
              l2_reg: float = 1e-5,
              learning_rate: float = 3e-4,
              batch_size: int = 64,
              epochs: int = 200,
              patience: int = 20) -> Dict[str, Any]:
        """Training với monitoring chi tiết."""
        
        print(f"🚀 Bắt đầu Training")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Patience: {patience}")
        
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        print(f"   Training set: {X_train.shape}")
        print(f"   Validation set: {X_val.shape}")
        
        # Tạo model
        self.model = self.create_model(
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate
        )
        
        # Custom callback để in kết quả mỗi epoch
        class EpochLogger(callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best_val_loss = float('inf')
                self.best_val_mae = float('inf')
            
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                epoch_num = epoch + 1
                train_loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                val_mae = logs.get('val_mae', 0)
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                
                # Cập nhật best values
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                
                print(f"Epoch {epoch_num:3d}/{epochs} - "
                      f"loss: {train_loss:.4f} - "
                      f"val_loss: {val_loss:.4f} - "
                      f"val_mae: {val_mae:.4f} - "
                      f"best_val_loss: {self.best_val_loss:.4f} - "
                      f"best_val_mae: {self.best_val_mae:.4f} - "
                      f"lr: {lr:.2e} - "
                      f"time elapsed: {time.time() - start_time:.2f}s")

        # Callbacks
        callbacks_list = [
            EpochLogger(),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0  # Tắt verbose để không in thông báo
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=patience//2,
                min_lr=1e-6,
                verbose=0  # Tắt verbose để không in thông báo
            ),
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0  # Tắt verbose để không in thông báo
            )
        ]
        
        # Training
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=0  # Chỉ in kết quả epoch, không in quá trình
        )
        training_time = time.time() - start_time
        
        # Đánh giá
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        best_val_loss = min(history.history['val_loss'])
        best_val_mae = min(history.history['val_mae'])
        best_epoch = np.argmin(history.history['val_loss']) + 1
        
        print(f"\n📊 KẾT QUẢ TRAINING:")
        print(f"   Best validation loss: {val_loss:.6f}")
        print(f"   Best validation MAE: {val_mae:.6f}")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Best validation MAE: {best_val_mae:.6f}")
        print(f"   Best epoch: {best_epoch}")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Epochs completed: {len(history.history['loss'])}")
        
        # Lưu model và y_scaler
        self.save_model()
        joblib.dump(self.y_scaler, self.y_scaler_path)
        print(f"💾 Đã lưu y_scaler: {self.y_scaler_path}")
        
        return {
            'val_loss': val_loss,
            'val_mae': val_mae,
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
            'training_time': training_time,
            'epochs_completed': len(history.history['loss']),
            'history': history.history
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Đánh giá model."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled, verbose=0)
        y_pred_scaled = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        
        mse = mean_squared_error(y, y_pred_scaled)
        mae = mean_absolute_error(y, y_pred_scaled)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán."""
        X_scaled = self.scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    def plot_training_history(self, training_result: Dict[str, Any]):
        """Vẽ biểu đồ lịch sử training."""
        history = training_result['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(history['mae'], label='Train MAE', color='blue')
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='red')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MSE
        axes[1, 0].plot(history['mse'], label='Train MSE', color='blue')
        axes[1, 0].plot(history['val_mse'], label='Validation MSE', color='red')
        axes[1, 0].set_title('Model MSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Thông tin model
        info_text = f"""
Training Results:
• Best epoch: {training_result['best_epoch']}
• Best val loss: {training_result['best_val_loss']:.6f}
• Best val MAE: {training_result['best_val_mae']:.6f}
• Training time: {training_result['training_time']:.1f}s
• Epochs completed: {training_result['epochs_completed']}
        """
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_model_summary(self):
        """Hiển thị tóm tắt model."""
        if self.model is not None:
            print(f"\n📋 MODEL SUMMARY:")
            print(f"   Total parameters: {self.model.count_params():,}")
            print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
            print(f"   Non-trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]):,}")
            print(f"   Number of layers: {len(self.model.layers)}")

def load_improved_dataset():
    """Tải dataset chính và lọc chỉ non-CM data."""
    # Ưu tiên sử dụng dataset chính (đã được cập nhật)
    if os.path.exists('schoof_data_X_cleaned.npy'):
        print(f"📂 Loading MAIN dataset from current directory")
        
        # Load dataset chính
        X = np.load('schoof_data_X_cleaned.npy')
        y_delta = np.load('schoof_data_delta.npy')
        y_cm = np.load('schoof_data_cm.npy')
    else:
        # Fallback: tìm dataset cải thiện mới nhất
        backup_dirs = [d for d in os.listdir('backups') if d.startswith('dataset_improved_')]
        if not backup_dirs:
            print("❌ Không tìm thấy dataset!")
            print("   Hãy chạy update_main_dataset.py trước")
            return None
        
        latest_backup = sorted(backup_dirs)[-1]
        backup_path = f'backups/{latest_backup}'
        
        print(f"📂 Loading improved dataset from: {backup_path}")
        
        # Load dataset
        X = np.load(f'{backup_path}/schoof_data_X_improved.npy')
        y_delta = np.load(f'{backup_path}/schoof_data_delta_improved.npy')
        y_cm = np.load(f'{backup_path}/schoof_data_cm_improved.npy')
    
    print(f"✅ Loaded dataset: X={X.shape}, y_delta={y_delta.shape}, y_cm={y_cm.shape}")
    
    # Lọc chỉ non-CM data (y_cm = 0)
    non_cm_mask = (y_cm == 0)
    X_non_cm = X[non_cm_mask]
    y_delta_non_cm = y_delta[non_cm_mask]
    y_cm_non_cm = y_cm[non_cm_mask]
    
    print(f"🎯 Filtered to NON-CM data only:")
    print(f"   Original: {X.shape[0]:,} samples")
    print(f"   Non-CM: {X_non_cm.shape[0]:,} samples ({X_non_cm.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   CM samples excluded: {X.shape[0] - X_non_cm.shape[0]:,} samples")
    
    return X_non_cm, y_delta_non_cm, y_cm_non_cm

def main():
    """Hàm chính."""
    print("🎯 SIMPLE TRAINING SYSTEM - FINAL VERSION")
    print("🎯 DELTA REGRESSOR - NON-CM DATA ONLY")
    print("=" * 50)
    
    # Tải dataset (chỉ non-CM data)
    dataset = load_improved_dataset()
    if dataset is None:
        return
    
    X, y_delta, y_cm = dataset
    
    # Khởi tạo model
    model = SimpleTrainingModel(feature_count=X.shape[1], model_name="final_model")
    
    # Kiểm tra model hiện có
    if model.load_existing_model():
        print("🔄 Tiếp tục training từ model hiện có...")
    else:
        print("🆕 Bắt đầu training từ đầu...")
    
    # Cấu hình training - CÓ THỂ ĐIỀU CHỈNH
    config = {
        'hidden_layers': [256, 128, 64, 32, 16],
        'dropout_rate': 0.3,
        'l2_reg': 5e-4,
        'learning_rate': 3e-4,
        'batch_size': 64,  # Tăng batch size để giảm số batches
        'epochs': 300,
        'patience': 25
    }

    
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Training
    training_result = model.train(X, y_delta, **config)
    
    # Hiển thị model summary
    model.get_model_summary()
    
    # Đánh giá cuối cùng
    print(f"\n📊 ĐÁNH GIÁ CUỐI CÙNG:")
    final_eval = model.evaluate(X, y_delta)
    print(f"   MSE: {final_eval['mse']:.6f}")
    print(f"   MAE: {final_eval['mae']:.6f}")
    print(f"   RMSE: {final_eval['rmse']:.6f}")
    
    # Vẽ biểu đồ
    model.plot_training_history(training_result)
    
    print(f"\n🎉 Training hoàn thành!")
    print(f"📁 Model saved: {model.model_path}")
    print(f"📊 Training history: models/training_history.png")
    print(f"\n💡 Để thay đổi độ phức tạp, hãy sửa config trong hàm main()")

if __name__ == '__main__':
    main()
