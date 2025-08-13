#!/usr/bin/env python3
"""
Script demo AI-Enhanced Schoof Algorithm
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from schoof_ai_enhanced import SchoofAIEnhanced
import os

def main():
    print("AI-ENHANCED SCHOOF ALGORITHM")
    print("=" * 60)
    
    # Khởi tạo
    schoof_ai = SchoofAIEnhanced(model_type='ensemble')
    
    # Sinh dữ liệu
    print("\n1. SINH DỮ LIỆU")
    X, y = schoof_ai.generate_training_data(max_p=200, samples_per_p=10)
    print(f"Đã sinh {len(X)} mẫu")
    
    # Huấn luyện
    print("\n2. HUẤN LUYỆN")
    results = schoof_ai.train_models(X, y)
    
    # Hiển thị kết quả
    print("\n3. KẾT QUẢ:")
    for name, metrics in results.items():
        print(f"{name}: R²={metrics['val_r2']:.4f}")
    
    # Test
    print("\n4. TEST:")
    test_cases = [(17, 5, 3), (23, 7, 11), (31, 13, 19)]
    
    for p, A, B in test_cases:
        N_classical = schoof_ai.classical_schoof(A, B, p)
        tilde_delta_classical = (p + 1 - N_classical) / (2 * np.sqrt(p))
        
        try:
            tilde_delta_ai = schoof_ai.ai_predict_tilde_delta(p, A, B)
            print(f"p={p}: Classical={tilde_delta_classical:.4f}, AI={tilde_delta_ai:.4f}")
        except:
            print(f"p={p}: Classical={tilde_delta_classical:.4f}, AI=Failed")
    
    # Lưu model
    schoof_ai.save_models('schoof_ai_models.pkl')
    print("\nĐã lưu model!")

if __name__ == "__main__":
    main() 