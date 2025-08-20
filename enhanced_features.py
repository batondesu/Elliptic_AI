#!/usr/bin/env python3
"""
Enhanced Features cho AI-enhanced Schoof v2.0
Tăng từ 24 lên 92 features phức tạp hơn
"""

import numpy as np
import math
from sympy import legendre_symbol
from typing import List, Tuple

def extract_enhanced_features(p: int, A: int, B: int) -> np.ndarray:
    """Trích xuất 92 features nâng cao cho elliptic curves"""
    features = []
    
    # 1. Features cơ bản (3)
    features.extend([p, A, B])
    
    # 2. Discriminant và J-invariant (4)
    discriminant = (4 * pow(A, 3, p) + 27 * pow(B, 2, p)) % p
    features.append(discriminant)
    features.append(discriminant / p)
    
    # J-invariant
    if A != 0 or B != 0:
        try:
            num = (1728 * (4 * pow(A, 3, p))) % p
            den = discriminant
            if den != 0:
                inv = pow(den, -1, p)
                j_inv = (num * inv) % p
                features.append(j_inv)
                features.append(j_inv / p)
            else:
                features.extend([0, 0])
        except:
            features.extend([0, 0])
    else:
        features.extend([0, 0])
    
    # 3. Modular arithmetic cơ bản (9)
    features.extend([
        A % 3, B % 3, p % 3,
        A % 4, B % 4, p % 4,
        A % 5, B % 5, p % 5
    ])
    
    # 4. Tương tác bậc 2 (6)
    features.extend([
        (A * B) % p,
        (A * A) % p,
        (B * B) % p,
        (A * p) % p,
        (B * p) % p,
        (A + B) % p
    ])
    
    # 5. Tỷ lệ và chuẩn hóa (8)
    features.extend([
        A / p, B / p, (A + B) / p,
        abs(A) / p, abs(B) / p,
        (A - B) / p, (A * B) / (p * p),
        discriminant / (p * p)
    ])
    
    # 6. Logarit và exponential (6)
    features.extend([
        math.log(p),
        math.log(abs(A) + 1),
        math.log(abs(B) + 1),
        math.log(abs(A * B) + 1),
        math.exp(-abs(A) / p),
        math.exp(-abs(B) / p)
    ])
    
    # 7. Trigonometric functions (8)
    features.extend([
        math.sin(A / p * math.pi),
        math.cos(B / p * math.pi),
        math.tan(A / p * math.pi) if abs(math.cos(A / p * math.pi)) > 1e-10 else 0,
        math.sin((A + B) / p * math.pi),
        math.cos((A - B) / p * math.pi),
        math.sin(discriminant / p * math.pi),
        math.cos(j_inv / p * math.pi) if 'j_inv' in locals() else 0,
        math.sin(math.log(p))
    ])
    
    # 8. Legendre symbols (6)
    try:
        features.extend([
            legendre_symbol(A, p),
            legendre_symbol(B, p),
            legendre_symbol(A * B, p),
            legendre_symbol(discriminant, p),
            legendre_symbol(A + B, p),
            legendre_symbol(A - B, p)
        ])
    except:
        features.extend([0, 0, 0, 0, 0, 0])
    
    # 9. Polynomial features (8)
    features.extend([
        pow(A, 2, p),
        pow(B, 2, p),
        pow(A, 3, p),
        pow(B, 3, p),
        (pow(A, 2, p) + pow(B, 2, p)) % p,
        (pow(A, 3, p) + pow(B, 3, p)) % p,
        (A * pow(B, 2, p)) % p,
        (B * pow(A, 2, p)) % p
    ])
    
    # 10. Statistical features (6)
    features.extend([
        (A + B) / 2,  # mean
        abs(A - B) / 2,  # half range
        math.sqrt(abs(A * B)),  # geometric mean
        (A * A + B * B) / 2,  # mean of squares
        abs(A - B),  # range
        (A + B + p) / 3  # mean of all
    ])
    
    # 11. Advanced elliptic curve features (8)
    features.extend([
        (4 * A * A * A + 27 * B * B) % p,  # discriminant again
        (A * A * A) % p,
        (B * B * B) % p,
        (A * A * B) % p,
        (A * B * B) % p,
        (A * A + B * B) % p,
        (A * A - B * B) % p,
        (A * A * A + B * B * B) % p
    ])
    
    # 12. Modular multiplicative features (6)
    try:
        features.extend([
            pow(A, -1, p) if A != 0 else 0,
            pow(B, -1, p) if B != 0 else 0,
            pow(A * B, -1, p) if A * B != 0 else 0,
            pow(discriminant, -1, p) if discriminant != 0 else 0,
            pow(A + B, -1, p) if (A + B) % p != 0 else 0,
            pow(A - B, -1, p) if (A - B) % p != 0 else 0
        ])
    except:
        features.extend([0, 0, 0, 0, 0, 0])
    
    # 13. Hasse interval related features (4)
    hasse_lower = p + 1 - 2 * math.sqrt(p)
    hasse_upper = p + 1 + 2 * math.sqrt(p)
    hasse_width = hasse_upper - hasse_lower
    features.extend([
        hasse_lower,
        hasse_upper,
        hasse_width,
        hasse_width / p
    ])
    
    # 14. Prime-specific features (4)
    features.extend([
        p % 6,  # p mod 6
        p % 8,  # p mod 8
        p % 12,  # p mod 12
        p % 24  # p mod 24
    ])
    
    # 15. Advanced mathematical features (6)
    features.extend([
        math.gcd(A, p),
        math.gcd(B, p),
        math.gcd(A, B),
        (A * A + B * B + p * p) % p,
        math.sqrt(p),
        math.sqrt(abs(A * B))
    ])
    
    return np.array(features, dtype=np.float32)

def get_enhanced_feature_names() -> List[str]:
    """Lấy tên của 92 features nâng cao"""
    return [
        # 1. Basic (3)
        'p', 'A', 'B',
        
        # 2. Discriminant & J-invariant (4)
        'discriminant', 'discriminant_ratio', 'j_invariant', 'j_invariant_ratio',
        
        # 3. Modular arithmetic (9)
        'A_mod_3', 'B_mod_3', 'p_mod_3', 'A_mod_4', 'B_mod_4', 'p_mod_4', 'A_mod_5', 'B_mod_5', 'p_mod_5',
        
        # 4. Quadratic interactions (6)
        'A_times_B', 'A_squared', 'B_squared', 'A_times_p', 'B_times_p', 'A_plus_B_mod_p',
        
        # 5. Ratios & normalization (8)
        'A_over_p', 'B_over_p', 'A_plus_B_over_p', 'abs_A_over_p', 'abs_B_over_p', 'A_minus_B_over_p', 'A_times_B_over_p_squared', 'discriminant_over_p_squared',
        
        # 6. Logarithmic & exponential (6)
        'log_p', 'log_abs_A', 'log_abs_B', 'log_abs_A_times_B', 'exp_neg_abs_A_over_p', 'exp_neg_abs_B_over_p',
        
        # 7. Trigonometric (8)
        'sin_A_over_p', 'cos_B_over_p', 'tan_A_over_p', 'sin_A_plus_B_over_p', 'cos_A_minus_B_over_p', 'sin_discriminant_over_p', 'cos_j_invariant_over_p', 'sin_log_p',
        
        # 8. Legendre symbols (6)
        'legendre_A', 'legendre_B', 'legendre_A_times_B', 'legendre_discriminant', 'legendre_A_plus_B', 'legendre_A_minus_B',
        
        # 9. Polynomial (8)
        'A_squared_mod_p', 'B_squared_mod_p', 'A_cubed_mod_p', 'B_cubed_mod_p', 'A_squared_plus_B_squared_mod_p', 'A_cubed_plus_B_cubed_mod_p', 'A_times_B_squared_mod_p', 'B_times_A_squared_mod_p',
        
        # 10. Statistical (6)
        'mean_A_B', 'half_range_A_B', 'geometric_mean_A_B', 'mean_squares_A_B', 'range_A_B', 'mean_A_B_p',
        
        # 11. Advanced elliptic (8)
        'discriminant_mod_p', 'A_cubed_mod_p_2', 'B_cubed_mod_p_2', 'A_squared_times_B_mod_p', 'A_times_B_squared_mod_p', 'A_squared_plus_B_squared_mod_p_2', 'A_squared_minus_B_squared_mod_p', 'A_cubed_plus_B_cubed_mod_p_2',
        
        # 12. Modular multiplicative (6)
        'A_inverse_mod_p', 'B_inverse_mod_p', 'A_times_B_inverse_mod_p', 'discriminant_inverse_mod_p', 'A_plus_B_inverse_mod_p', 'A_minus_B_inverse_mod_p',
        
        # 13. Hasse interval (4)
        'hasse_lower', 'hasse_upper', 'hasse_width', 'hasse_width_over_p',
        
        # 14. Prime-specific (4)
        'p_mod_6', 'p_mod_8', 'p_mod_12', 'p_mod_24',
        
        # 15. Advanced mathematical (6)
        'gcd_A_p', 'gcd_B_p', 'gcd_A_B', 'A_squared_plus_B_squared_plus_p_squared_mod_p', 'sqrt_p', 'sqrt_abs_A_times_B'
    ]

def test_enhanced_features():
    """Test enhanced features"""
    print("TESTING ENHANCED FEATURES")
    print("=" * 50)
    
    test_cases = [
        (17, 5, 3),
        (101, 23, 45),
        (257, 67, 89),
        (503, 127, 461)
    ]
    
    feature_names = get_enhanced_feature_names()
    print(f"Total features: {len(feature_names)}")
    
    for p, A, B in test_cases:
        features = extract_enhanced_features(p, A, B)
        print(f"p={p}, A={A}, B={B}: {len(features)} features")
        print(f"  Range: {features.min():.3f} to {features.max():.3f}")
        print(f"  Mean: {features.mean():.3f}")
        print(f"  Std: {features.std():.3f}")
        print()

if __name__ == '__main__':
    test_enhanced_features() 