#!/usr/bin/env python3
"""
Demo AI-enhanced Schoof v2.0
Giao diện tương tác để test thu hẹp khoảng Hasse và dự đoán δ
"""

import numpy as np
import time
import os
from typing import Tuple, Dict, Any

# Import từ ai_enhanced_schoof_v2
from ai_enhanced_schoof_v2 import (
    DeltaRegressorV2, CMClassifierV2, SchoofFeatureExtractor, AISchoofAssistantV2
)

def load_models() -> Tuple[DeltaRegressorV2, CMClassifierV2, SchoofFeatureExtractor]:
    """Tải các model đã huấn luyện"""
    print("🔄 Đang tải AI-enhanced Schoof v2.0 models...")
    
    # Kiểm tra file tồn tại
    required_files = [
        'schoof_ai_regressor_v2.h5',
        'schoof_ai_regressor_v2_scaler.pkl',
        'schoof_ai_cm_classifier_v2.h5',
        'schoof_ai_cm_v2_scaler.pkl',
        'schoof_feature_names.txt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Thiếu các file: {missing_files}")
        print("💡 Hãy chạy ai_enhanced_schoof_v2.py trước để huấn luyện models.")
        return None, None, None
    
    # Đọc feature names
    with open('schoof_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Tải models
    regressor = DeltaRegressorV2(feature_count=len(feature_names))
    regressor.load()
    
    classifier = CMClassifierV2(feature_count=len(feature_names))
    classifier.load()
    
    feature_extractor = SchoofFeatureExtractor(feature_names)
    
    print("Đã tải models thành công!")
    return regressor, classifier, feature_extractor

def classical_count_points(A: int, B: int, p: int) -> int:
    """Đếm điểm theo phương pháp cổ điển"""
    from sympy import legendre_symbol
    
    c = 1  # Điểm vô cực
    for x in range(p):
        r = (x**3 + A*x + B) % p
        if r == 0:
            c += 1
        elif legendre_symbol(r, p) == 1:
            c += 2
    return c

def calculate_delta_classical(A: int, B: int, p: int) -> float:
    """Tính δ theo phương pháp cổ điển"""
    N = classical_count_points(A, B, p)
    return float(p + 1 - N)

def compare_methods(p: int, A: int, B: int, assistant: AISchoofAssistantV2) -> Dict[str, Any]:
    """So sánh AI vs Classical methods"""
    print(f"\nSO SÁNH AI vs CLASSICAL CHO p={p}, A={A}, B={B}")
    print("=" * 60)
    
    # AI prediction
    start_time = time.time()
    sug = assistant.suggest_speedup(p, A, B)
    ai_time = time.time() - start_time
    
    # Classical calculation
    start_time = time.time()
    delta_classical = calculate_delta_classical(A, B, p)
    classical_time = time.time() - start_time
    
    # Kết quả
    delta_ai = sug['delta_prediction']
    error = abs(delta_ai - delta_classical)
    speedup = classical_time / ai_time if ai_time > 0 else float('inf')
    
    print(f"AI Prediction:")
    print(f"   δ = {delta_ai:.4f}")
    print(f"   CM probability = {sug['cm_probability']:.4f}")
    print(f"   Time = {ai_time:.6f}s")
    
    print(f"\n📐 Classical Calculation:")
    print(f"   δ = {delta_classical:.4f}")
    print(f"   Time = {classical_time:.6f}s")
    
    print(f"\nComparison:")
    print(f"   Error = {error:.4f}")
    print(f"   Speedup = {speedup:.1f}x")
    
    print(f"\nHasse Interval Narrowing:")
    print(f"   Original: {sug['hasse_interval']} (width: {sug['hasse_interval'][1] - sug['hasse_interval'][0] + 1})")
    print(f"   Narrowed: {sug['narrowed_interval']} (width: {sug['narrowed_interval'][1] - sug['narrowed_interval'][0] + 1})")
    print(f"   Reduction: {sug['width_reduction_factor']:.2f}x")
    
    return {
        'ai_delta': delta_ai,
        'classical_delta': delta_classical,
        'error': error,
        'speedup': speedup,
        'hasse_reduction': sug['width_reduction_factor'],
        'cm_probability': sug['cm_probability']
    }

def interactive_demo():
    """Demo tương tác"""
    print("AI-ENHANCED SCHOOF v2.0 - INTERACTIVE DEMO")
    print("=" * 60)
    
    # Tải models
    regressor, classifier, feature_extractor = load_models()
    if regressor is None:
        return
    
    # Tạo assistant
    assistant = AISchoofAssistantV2(regressor, classifier, feature_extractor)
    
    print("\nHƯỚNG DẪN:")
    print("• Nhập: p A B (ví dụ: 17 5 3)")
    print("• Nhập: 'demo' để chạy demo tự động")
    print("• Nhập: 'quit' để thoát")
    print("• Nhập: 'help' để xem hướng dẫn")
    
    while True:
        try:
            user_input = input("\nNhập p, A, B: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Tạm biệt!")
                break
            elif user_input.lower() == 'help':
                print("\n📖 HƯỚNG DẪN CHI TIẾT:")
                print("• p: số nguyên tố (ví dụ: 17, 101, 257)")
                print("• A, B: hệ số của đường cong elliptic y² = x³ + Ax + B (mod p)")
                print("• AI sẽ dự đoán δ và thu hẹp khoảng Hasse")
                print("• So sánh với phương pháp cổ điển")
                continue
            elif user_input.lower() == 'demo':
                run_auto_demo(assistant)
                continue
            
            # Parse input
            parts = user_input.split()
            if len(parts) != 3:
                print("❌ Vui lòng nhập đúng định dạng: p A B")
                continue
            
            p, A, B = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Validation
            if p < 3:
                print("❌ p phải >= 3")
                continue
            if A < 0 or A >= p or B < 0 or B >= p:
                print(f"❌ A, B phải trong khoảng [0, {p-1}]")
                continue
            
            # Kiểm tra discriminant
            discriminant = (4 * pow(A, 3, p) + 27 * pow(B, 2, p)) % p
            if discriminant == 0:
                print("❌ Đường cong có điểm kỳ dị (discriminant = 0)")
                continue
            
            # So sánh methods
            compare_methods(p, A, B, assistant)
            
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ")
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

def run_auto_demo(assistant: AISchoofAssistantV2):
    """Chạy demo tự động với các test cases"""
    print("\nCHẠY DEMO TỰ ĐỘNG")
    print("=" * 60)
    
    test_cases = [
        (17, 5, 3),
        (101, 23, 45),
        (257, 67, 89),
        (499, 123, 456),
        (1009, 234, 567),
        (2003, 456, 789)
    ]
    
    results = []
    
    for i, (p, A, B) in enumerate(test_cases, 1):
        print(f"\nTest case {i}/{len(test_cases)}: p={p}, A={A}, B={B}")
        result = compare_methods(p, A, B, assistant)
        results.append(result)
    
    # Tóm tắt
    print(f"\nTÓM TẮT DEMO:")
    print("=" * 60)
    
    avg_error = np.mean([r['error'] for r in results])
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_hasse_reduction = np.mean([r['hasse_reduction'] for r in results])
    
    print(f"Trung bình:")
    print(f"   Error: {avg_error:.4f}")
    print(f"   Speedup: {avg_speedup:.1f}x")
    print(f"   Hasse reduction: {avg_hasse_reduction:.2f}x")
    
    print(f"\n🏆 Kết quả tốt nhất:")
    best_speedup = max(results, key=lambda x: x['speedup'])
    best_reduction = max(results, key=lambda x: x['hasse_reduction'])
    best_accuracy = min(results, key=lambda x: x['error'])
    
    print(f"   Speedup cao nhất: {best_speedup['speedup']:.1f}x")
    print(f"   Hasse reduction cao nhất: {best_reduction['hasse_reduction']:.2f}x")
    print(f"   Error thấp nhất: {best_accuracy['error']:.4f}")

def main():
    """Main function"""
    interactive_demo()

if __name__ == "__main__":
    main() 