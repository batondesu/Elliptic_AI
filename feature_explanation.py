#!/usr/bin/env python3
"""
Giải thích chi tiết về Features và ảnh hưởng với Model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ai_enhanced_schoof_v2 import load_schoof_dataset
import math
from sympy import legendre_symbol
from typing import List

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

def j_invariant_mod_p(A: int, B: int, p: int) -> int:
    """Tính j-invariant mod p"""
    try:
        discriminant = (-16 * (4 * pow(A, 3, p) + 27 * pow(B, 2, p))) % p
        if discriminant == 0:
            return 0
        inv_disc = pow(discriminant, p - 2, p)
        j = (1728 * 4 * pow(A, 3, p) * inv_disc) % p
        return j
    except:
        return 0

def extract_features(p: int, A: int, B: int) -> List[float]:
    """Trích xuất 40 features từ (p, A, B)"""
    features = []
    
    # Basic features (6)
    features.extend([
        float(p), float(A), float(B),
        float(A % p), float(B % p),
        float((4 * A**3 + 27 * B**2) % p)  # discriminant
    ])
    
    # Logarithmic features (3)
    features.extend([
        math.log10(p), math.log10(max(1, abs(A))), math.log10(max(1, abs(B)))
    ])
    
    # Ratios (4)
    features.extend([
        float(A / p), float(B / p),
        float(A / max(1, abs(B))), float(B / max(1, abs(A)))
    ])
    
    # Quadratic residues (3)
    try:
        features.extend([
            float(legendre_symbol(A, p) if A % p != 0 else 0),
            float(legendre_symbol(B, p) if B % p != 0 else 0),
            float(legendre_symbol((4 * A**3 + 27 * B**2) % p, p))
        ])
    except:
        features.extend([0.0, 0.0, 0.0])
    
    # Modular arithmetic (6)
    features.extend([
        float(A % 3), float(A % 5), float(A % 7),
        float(B % 3), float(B % 5), float(B % 7)
    ])
    
    # Powers mod p (6)
    features.extend([
        float(pow(A, 2, p)), float(pow(A, 3, p)), float(pow(A, 4, p)),
        float(pow(B, 2, p)), float(pow(B, 3, p)), float(pow(B, 4, p))
    ])
    
    # Combinations (6)
    features.extend([
        float((A + B) % p), float((A - B) % p), float((A * B) % p),
        float((A**2 + B**2) % p), float((A**2 - B**2) % p), float((A**3 + B**3) % p)
    ])
    
    # Advanced features (6)
    j_inv = j_invariant_mod_p(A, B, p)
    features.extend([
        float(j_inv),
        float((A + p) % (p + 1)), float((B + p) % (p + 1)),
        float(math.gcd(abs(A), p)), float(math.gcd(abs(B), p)),
        float(bin(p).count('1'))  # Hamming weight of p
    ])
    
    return features[:40]  # Đảm bảo đúng 40 features

def extract_features_rich(p: int, A: int, B: int, sample_x: int = 64) -> List[float]:
    """Sinh bộ features toán học giàu thông tin hơn (KHÔNG dùng trực tiếp cho model hiện tại).

    Mục đích: phục vụ phân tích/so sánh/chuẩn bị mở rộng dataset về sau.
    Vẫn đảm bảo chi phí tính toán thấp (O(sample_x)).
    """
    features: List[float] = []
    # 1) Tham số cơ bản
    features.extend([
        float(p), float(A), float(B)
    ])
    # 2) Invariants cổ điển (Weierstrass short form y^2 = x^3 + Ax + B)
    #    c4 = -48A, c6 = -864B, Δ = -16(4A^3 + 27B^2), j ≈ c4^3 / Δ (tránh Δ=0)
    c4 = -48.0 * A
    c6 = -864.0 * B
    disc = -16.0 * (4.0 * (A ** 3) + 27.0 * (B ** 2))
    j_val = 0.0
    try:
        if disc != 0:
            j_val = float((c4 ** 3) / disc)
    except Exception:
        j_val = 0.0
    features.extend([
        float(c4), float(c6), float(disc), float(j_val),
        abs(float(c4)), abs(float(c6)), abs(float(disc))
    ])
    # log-invariants tránh log(0)
    features.extend([
        math.log10(max(1.0, abs(float(c4)))),
        math.log10(max(1.0, abs(float(c6)))),
        math.log10(max(1.0, abs(float(disc)))),
        math.log10(max(1.0, abs(float(j_val))))
    ])
    # 3) Hasse interval
    T = int(math.ceil(2.0 * math.sqrt(p)))
    hasse_lower = p + 1 - T
    hasse_upper = p + 1 + T
    hasse_width = hasse_upper - hasse_lower + 1
    features.extend([
        float(T), float(hasse_lower), float(hasse_upper), float(hasse_width)
    ])
    # 4) Phân lớp modulo của p (giúp bắt tính chất số học của trường)
    features.extend([
        float(p % 3), float(p % 4), float(p % 5), float(p % 7),
        float(p % 8), float(p % 12), float(p % 24)
    ])
    # 5) Ký hiệu Legendre cho A, B, Δ (0 nếu chia hết mod p)
    def safe_legendre(x: int, p: int) -> int:
        try:
            return int(legendre_symbol(x % p, p)) if (x % p) != 0 else 0
        except Exception:
            return 0
    features.extend([
        float(safe_legendre(A, p)),
        float(safe_legendre(B, p)),
        float(safe_legendre(int(disc) % p, p))
    ])
    # Một số tổ hợp Legendre bổ sung
    features.extend([
        float(safe_legendre((A + B) % p, p)),
        float(safe_legendre((A - B) % p, p)),
        float(safe_legendre((A * B) % p, p)),
        float(safe_legendre(int(c4) % p, p)),
        float(safe_legendre(int(c6) % p, p)),
        float(safe_legendre(int(-disc) % p, p)),
        float(safe_legendre(2, p)),
        float(safe_legendre(p - 1, p))  # legendre(-1, p)
    ])
    # 6) Mật độ nghiệm bậc hai của r(x) = x^3 + A x + B (mod p) trên một mẫu nhỏ
    #    (tương quan với số điểm y^2 = r(x) tồn tại) → proxy nhẹ cho độ khó/hình dạng đường cong
    sample = min(sample_x, p)
    qres_count = 0
    zeros_count = 0
    leg_sum = 0  # Σ χ(r(x)) trên mẫu (proxy liên quan tới trace)
    for x in range(sample):
        r = (x * x % p * x % p + (A % p) * x + (B % p)) % p
        if r == 0:
            zeros_count += 1
            qres_count += 1  # giữ nguyên định nghĩa cũ cho mật độ
        else:
            try:
                chi = int(legendre_symbol(int(r), p))
                if chi == 1:
                    qres_count += 1
                leg_sum += chi  # chi ∈ {-1,0,1} nhưng r!=0 nên chi∈{-1,1}
            except Exception:
                continue
    qres_density = qres_count / float(sample) if sample > 0 else 0.0
    features.extend([
        float(qres_density), float(qres_count), float(zeros_count)
    ])
    # 6b) Proxy Frobenius trace mod ℓ nhỏ: dùng leg_sum (mẫu) % ℓ và chuẩn hóa
    for ell in [3, 5, 7, 11]:
        try:
            val_mod = ((leg_sum % ell) + ell) % ell
        except Exception:
            val_mod = 0
        features.extend([
            float(val_mod),
            float(val_mod) / float(ell)
        ])
    # 7) Chuẩn hóa theo sqrt(p) cho một vài đại lượng (scale-invariant-ish)
    sqrtp = math.sqrt(p)
    features.extend([
        float(A / max(1.0, sqrtp)), float(B / max(1.0, sqrtp)),
        float(T / max(1.0, sqrtp))
    ])
    # 8) j-invariant modulo các cơ số nhỏ (chuẩn hoá về [0,1])
    try:
        j_mods = []
        for m in [3, 5, 7, 11]:
            j_mods.append(float((int(j_val) % m) / m))
        features.extend(j_mods)
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])
    return features

if __name__ == '__main__':
    explain_features() 