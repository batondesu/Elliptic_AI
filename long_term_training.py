#!/usr/bin/env python3
"""
Framework cho Long-term Training và Model Expansion
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import math
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from sympy import primerange

# Import từ module Schoof AI
from schoof_ai_enhanced import SchoofAIEnhanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongTermTrainingFramework:
    """Framework cho huấn luyện lâu dài"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Khởi tạo Schoof AI
        self.schoof_ai = SchoofAIEnhanced(model_type='ensemble')
        
        # Lưu trữ lịch sử
        self.training_history = []
        self.current_epoch = 0
        
    def generate_incremental_data(self, p_range: Tuple[int, int], 
                                samples_per_p: int = 10) -> np.ndarray:
        """Sinh dữ liệu tăng dần"""
        min_p, max_p = p_range
        logger.info(f"Sinh dữ liệu: p từ {min_p} đến {max_p}")
        
        data = []
        primes = list(primerange(min_p, max_p))
        
        for p in primes:
            for _ in range(samples_per_p):
                A = np.random.randint(0, p)
                B = np.random.randint(0, p)
                
                if (4*A**3 + 27*B**2) % p == 0:
                    continue
                
                N = self.schoof_ai.classical_schoof(A, B, p)
                delta = p + 1 - N
                tilde_delta = delta / (2 * math.sqrt(p))
                
                features = self.schoof_ai.extract_features(p, A, B)
                data.append(np.concatenate([features, [tilde_delta]]))
        
        return np.array(data)
    
    def incremental_training_step(self, new_data: np.ndarray) -> Dict:
        """Bước huấn luyện tăng dần"""
        X_new = new_data[:, :-1]
        y_new = new_data[:, -1]
        
        # Chia train/validation
        n_val = max(1, len(X_new) // 5)
        X_train, X_val = X_new[:-n_val], X_new[-n_val:]
        y_train, y_val = y_new[:-n_val], y_new[-n_val:]
        
        # Chuẩn hóa
        X_train_scaled = self.schoof_ai.scaler.fit_transform(X_train)
        X_val_scaled = self.schoof_ai.scaler.transform(X_val)
        
        start_time = time.time()
        
        # Huấn luyện models
        results = {}
        for name, model in self.schoof_ai.models.items():
            model.fit(X_train_scaled, y_train)
            
            y_pred_train = model.predict(X_train_scaled)
            y_pred_val = model.predict(X_val_scaled)
            
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            train_loss = mean_squared_error(y_train, y_pred_train)
            val_loss = mean_squared_error(y_val, y_pred_val)
            
            results[name] = {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
        
        training_time = time.time() - start_time
        
        # Metrics tổng hợp
        avg_train_r2 = np.mean([r['train_r2'] for r in results.values()])
        avg_val_r2 = np.mean([r['val_r2'] for r in results.values()])
        
        metrics = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'train_r2': avg_train_r2,
            'val_r2': avg_val_r2,
            'training_time': training_time,
            'data_size': len(new_data)
        }
        
        self.training_history.append(metrics)
        self.current_epoch += 1
        
        return {'metrics': metrics, 'detailed_results': results}
    
    def continuous_training(self, max_epochs: int = 100, save_interval: int = 10):
        """Huấn luyện liên tục"""
        logger.info(f"Bắt đầu huấn luyện liên tục: {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            
            # Sinh dữ liệu mới với p tăng dần
            p_start = 3 + epoch * 5
            p_end = p_start + 20
            new_data = self.generate_incremental_data((p_start, p_end), samples_per_p=5)
            
            # Huấn luyện
            results = self.incremental_training_step(new_data)
            metrics = results['metrics']
            
            logger.info(f"Epoch {epoch + 1}: Train R²={metrics['train_r2']:.4f}, "
                       f"Val R²={metrics['val_r2']:.4f}")
            
            # Lưu checkpoint định kỳ
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")
    
    def save_checkpoint(self, name: str):
        """Lưu checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Lưu model
        model_path = os.path.join(checkpoint_path, "model.pkl")
        self.schoof_ai.save_models(model_path)
        
        # Lưu history
        history_path = os.path.join(checkpoint_path, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Đã lưu checkpoint: {name}")
    
    def plot_training_progress(self):
        """Vẽ tiến trình huấn luyện"""
        if not self.training_history:
            return
        
        epochs = [m['epoch'] for m in self.training_history]
        train_r2s = [m['train_r2'] for m in self.training_history]
        val_r2s = [m['val_r2'] for m in self.training_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_r2s, label='Train R²')
        plt.plot(epochs, val_r2s, label='Val R²')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        training_times = [m['training_time'] for m in self.training_history]
        plt.plot(epochs, training_times)
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Training Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        data_sizes = [m['data_size'] for m in self.training_history]
        cumulative_data = np.cumsum(data_sizes)
        plt.plot(epochs, cumulative_data)
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative Data Size')
        plt.title('Data Growth')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_progress.png'))
        plt.show()

def main():
    """Demo Long-term Training"""
    print("=" * 60)
    print("LONG-TERM TRAINING FRAMEWORK")
    print("=" * 60)
    
    # Khởi tạo framework
    framework = LongTermTrainingFramework()
    
    # Huấn luyện liên tục
    framework.continuous_training(max_epochs=30, save_interval=5)
    
    # Vẽ tiến trình
    framework.plot_training_progress()
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)

if __name__ == "__main__":
    main() 