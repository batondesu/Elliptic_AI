#!/usr/bin/env python3
"""
Giải thích chi tiết về Features và ảnh hưởng với Model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ai_enhanced_schoof_v2 import load_schoof_dataset

def explain_features():
    """Giải thích chi tiết về features"""
    print("GIẢI THÍCH CHI TIẾT VỀ FEATURES VÀ ẢNH HƯỞNG VỚI MODEL")
    print("=" * 70)
    
    # Tải dataset
    X, y_delta, y_tilde_delta, y_cm, feature_names = load_schoof_dataset()
    
    print(f"\n1. FEATURES LÀ GÌ?")
    print("-" * 50)
    print("Features (đặc trưng) là các thuộc tính hoặc đặc điểm của dữ liệu")
    print("được sử dụng để mô tả và phân biệt các mẫu khác nhau.")
    print()
    print("Trong trường hợp elliptic curves:")
    print("- Input: p (số nguyên tố), A, B (hệ số của curve)")
    print("- Output: δ (delta) - số điểm trên curve")
    print("- Features: Các đặc trưng toán học được tính từ p, A, B")
    
    print(f"\n2. PHÂN LOẠI FEATURES TRONG DỰ ÁN")
    print("-" * 50)
    
    # Phân loại features
    feature_categories = {
        "Basic Features": [
            "p", "A", "B"
        ],
        "Discriminant & J-invariant": [
            "discriminant", "discriminant_ratio", "j_invariant", "j_invariant_ratio"
        ],
        "Modular Arithmetic": [
            "A_mod_3", "B_mod_3", "p_mod_3", "A_mod_4", "B_mod_4", "p_mod_4", "A_mod_5", "B_mod_5", "p_mod_5"
        ],
        "Quadratic Interactions": [
            "A_times_B", "A_squared", "B_squared", "A_times_p", "B_times_p", "A_plus_B_mod_p"
        ],
        "Ratios & Normalization": [
            "A_over_p", "B_over_p", "A_plus_B_over_p", "abs_A_over_p", "abs_B_over_p"
        ],
        "Logarithmic & Exponential": [
            "log_p", "log_abs_A", "log_abs_B", "log_abs_A_times_B"
        ],
        "Trigonometric": [
            "sin_A_over_p", "cos_B_over_p", "tan_A_over_p", "sin_A_plus_B_over_p"
        ],
        "Legendre Symbols": [
            "legendre_A", "legendre_B", "legendre_A_times_B", "legendre_discriminant"
        ],
        "Polynomial": [
            "A_squared_mod_p", "B_squared_mod_p", "A_cubed_mod_p", "B_cubed_mod_p"
        ],
        "Statistical": [
            "mean_A_B", "half_range_A_B", "geometric_mean_A_B", "mean_squares_A_B"
        ],
        "Advanced Elliptic": [
            "discriminant_mod_p", "A_cubed_mod_p_2", "B_cubed_mod_p_2", "A_squared_times_B_mod_p"
        ],
        "Modular Multiplicative": [
            "A_inverse_mod_p", "B_inverse_mod_p", "A_times_B_inverse_mod_p"
        ],
        "Hasse Interval": [
            "hasse_lower", "hasse_upper", "hasse_width", "hasse_width_over_p"
        ],
        "Prime-specific": [
            "p_mod_6", "p_mod_8", "p_mod_12", "p_mod_24"
        ],
        "Advanced Mathematical": [
            "gcd_A_p", "gcd_B_p", "gcd_A_B", "sqrt_p", "sqrt_abs_A_times_B"
        ]
    }
    
    print("Features được chia thành các nhóm:")
    for category, features in feature_categories.items():
        count = len([f for f in features if f in feature_names])
        if count > 0:
            print(f"  {category}: {count} features")
    
    print(f"\n3. TÍNH TOÁN CORRELATION VỚI TARGET")
    print("-" * 50)
    
    # Tính correlation với target
    correlations = []
    for i, name in enumerate(feature_names):
        try:
            corr = np.corrcoef(X[:, i], y_delta)[0, 1]
            if not np.isnan(corr):
                correlations.append((name, abs(corr)))
            else:
                correlations.append((name, 0.0))
        except:
            correlations.append((name, 0.0))
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 features có correlation cao nhất với target (δ):")
    for i, (name, corr) in enumerate(correlations[:15]):
        print(f"  {i+1:2d}. {name}: {corr:.4f}")
    
    print(f"\nBottom 10 features có correlation thấp nhất:")
    for i, (name, corr) in enumerate(correlations[-10:]):
        print(f"  {i+1:2d}. {name}: {corr:.4f}")
    
    print(f"\n4. ẢNH HƯỞNG CỦA FEATURES VỚI MODEL")
    print("-" * 50)
    
    print("A. FEATURES TỐT (correlation cao):")
    print("  ✅ Giúp model học được pattern")
    print("  ✅ Cải thiện accuracy và R²")
    print("  ✅ Giảm training time")
    print("  ✅ Tăng generalization")
    
    print("\nB. FEATURES XẤU (correlation thấp):")
    print("  ❌ Tạo noise, làm model khó học")
    print("  ❌ Giảm performance")
    print("  ❌ Tăng overfitting risk")
    print("  ❌ Tốn computational resources")
    
    print("\nC. FEATURES CÓ VẤN ĐỀ (NaN/Inf):")
    print("  ⚠️ Làm model crash hoặc cho kết quả sai")
    print("  ⚠️ Cần được xử lý trước khi training")
    
    print(f"\n5. PHÂN TÍCH FEATURE IMPORTANCE")
    print("-" * 50)
    
    # Phân tích feature importance theo correlation
    high_corr = [name for name, corr in correlations if corr > 0.01]
    medium_corr = [name for name, corr in correlations if 0.005 < corr <= 0.01]
    low_corr = [name for name, corr in correlations if corr <= 0.005]
    
    print(f"Feature importance distribution:")
    print(f"  High importance (corr > 0.01): {len(high_corr)} features")
    print(f"  Medium importance (0.005 < corr ≤ 0.01): {len(medium_corr)} features")
    print(f"  Low importance (corr ≤ 0.005): {len(low_corr)} features")
    
    print(f"\n6. TÁC ĐỘNG CỦA FEATURE SELECTION")
    print("-" * 50)
    
    print("Trước khi clean dataset:")
    print(f"  - Tổng features: 92")
    print(f"  - Features có vấn đề: 52 (57%)")
    print(f"  - Delta R²: -0.0165")
    
    print("\nSau khi clean dataset:")
    print(f"  - Tổng features: {len(feature_names)}")
    print(f"  - Features chất lượng: {len(high_corr) + len(medium_corr)}")
    print(f"  - Dự kiến Delta R²: > 0")
    
    print(f"\n7. VÍ DỤ CỤ THỂ VỀ FEATURES")
    print("-" * 50)
    
    # Lấy một số mẫu để minh họa
    sample_indices = [0, 1000, 5000]
    
    for idx in sample_indices:
        if idx < len(X):
            print(f"\nMẫu {idx}:")
            print(f"  p = {X[idx, 0]:.0f}, A = {X[idx, 1]:.0f}, B = {X[idx, 2]:.0f}")
            print(f"  Target δ = {y_delta[idx]:.2f}")
            
            # Hiển thị một số features quan trọng
            important_features = [
                ("discriminant", 3),
                ("A_times_B", 5),
                ("legendre_B", 7),
                ("hasse_width", 13)
            ]
            
            for name, feat_idx in important_features:
                if feat_idx < len(feature_names) and name in feature_names:
                    actual_idx = feature_names.index(name)
                    print(f"  {name} = {X[idx, actual_idx]:.2f}")
    
    print(f"\n8. KHUYẾN NGHỊ VỀ FEATURE ENGINEERING")
    print("-" * 50)
    
    print("A. Feature Selection:")
    print("  ✅ Loại bỏ features có correlation < 0.005")
    print("  ✅ Giữ lại features có correlation > 0.01")
    print("  ✅ Cân nhắc features có 0.005 < corr < 0.01")
    
    print("\nB. Feature Preprocessing:")
    print("  ✅ Xử lý NaN/Inf values")
    print("  ✅ Normalize features về cùng scale")
    print("  ✅ Kiểm tra outliers")
    
    print("\nC. Feature Creation:")
    print("  ✅ Tạo interaction features (A*B, A², B²)")
    print("  ✅ Tạo ratio features (A/p, B/p)")
    print("  ✅ Tạo modular features (A mod p, B mod p)")
    
    print("\nD. Feature Validation:")
    print("  ✅ Kiểm tra correlation với target")
    print("  ✅ Kiểm tra multicollinearity")
    print("  ✅ Cross-validation với different feature sets")
    
    return {
        'total_features': len(feature_names),
        'high_importance': len(high_corr),
        'medium_importance': len(medium_corr),
        'low_importance': len(low_corr),
        'max_correlation': correlations[0][1] if correlations else 0,
        'min_correlation': correlations[-1][1] if correlations else 0
    }

if __name__ == '__main__':
    explain_features() 