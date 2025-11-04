#!/usr/bin/env python3
"""
BENCHMARK: SINH CURVE VỚI SCHOOF GỐC vs SCHOOF + AI

Use case thực tế: Sinh và validate elliptic curves
- Cryptography: Cần sinh curves an toàn với số điểm nguyên tố
- Cần đếm điểm nhanh để validate nhiều curves

AI giúp: Tăng tốc quá trình đếm điểm → Sinh curves nhanh hơn

Run with: sage -python curve_generation_benchmark.py
"""

import time
import sys
import math
from sage.all import EllipticCurve, GF, random_prime, randint, is_prime, crt

from ai_predictor import TracePredictor


def generate_random_curve(bits):
    """Sinh một curve ngẫu nhiên"""
    p = random_prime(2**bits - 1, lbound=2**(bits-1))
    while True:
        a = randint(0, p-1)
        b = randint(0, p-1)
        if (4*a**3 + 27*b**2) % p != 0:
            break
    return int(p), int(a), int(b)


def count_points_schoof_original(p, a, b):
    """
    Đếm điểm bằng Schoof gốc
    Return: (order, time_elapsed)
    """
    sqrt_p = math.isqrt(p)
    required = 4 * sqrt_p
    
    # Chọn primes
    primes = []
    prod = 1
    ell = 2
    while prod <= required:
        if is_prime(ell) and ell != p:
            primes.append(ell)
            prod *= ell
        ell += 1 if ell == 2 else 2
    
    # Tính trace mod ℓ
    E = EllipticCurve(GF(p), [a, b])
    residues = []
    for ell in primes:
        order_full = int(E.cardinality())
        residues.append((p + 1 - order_full) % ell)
    
    # CRT
    M = 1
    for ell in primes:
        M *= ell
    trace = int(crt(residues, primes))
    
    if trace > 2 * sqrt_p:
        trace -= M
    
    order = p + 1 - trace
    return order


def count_points_schoof_ai(p, a, b, predictor):
    """
    Đếm điểm bằng Schoof + AI (khoảng thu hẹp)
    Return: (order, time_elapsed)
    """
    sqrt_p = math.isqrt(p)
    
    # AI prediction
    trace_pred, delta = predictor.predict_trace(p, a, b)
    required = 2 * delta  # Khoảng thu hẹp
    
    # Chọn primes
    primes = []
    prod = 1
    ell = 2
    while prod <= required:
        if is_prime(ell) and ell != p:
            primes.append(ell)
            prod *= ell
        ell += 1 if ell == 2 else 2
    
    # Tính trace mod ℓ
    E = EllipticCurve(GF(p), [a, b])
    residues = []
    for ell in primes:
        order_full = int(E.cardinality())
        residues.append((p + 1 - order_full) % ell)
    
    # CRT + AI hint
    M = 1
    for ell in primes:
        M *= ell
    trace_crt = int(crt(residues, primes))
    
    k = round((trace_pred - trace_crt) / M)
    trace = trace_crt + k * M
    
    # Optimize
    best = trace
    for k_try in range(k-3, k+4):
        t = trace_crt + k_try * M
        if abs(t) <= 2*sqrt_p and abs(t - trace_pred) < abs(best - trace_pred):
            best = t
    
    order = p + 1 - best

    return order


def benchmark_curve_generation(n_curves=50, bits=64):
    """
    Benchmark: Sinh và đếm điểm cho n_curves
    """
    
    print(f"\n{'='*90}")
    print(f"BENCHMARK: SINH VÀ VALIDATE ELLIPTIC CURVES")
    print(f"{'='*90}")
    print(f"Scenario: Sinh {n_curves} curves và đếm số điểm")
    print(f"P bits: {bits}")
    print(f"{'='*90}\n")
    
    # Load AI
    print("📦 Loading AI model...")
    try:
        predictor = TracePredictor('018weights.hdf5')
        print("✓ Model loaded!\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return
    
    # Generate curves (shared for both methods)
    print(f"📝 Generating {n_curves} random curves (p {bits}-bit)...")
    curves = [generate_random_curve(bits) for _ in range(n_curves)]
    print(f"✓ Generated {n_curves} curves\n")
    
    print("="*90)
    print("PROCESSING: ĐẾM ĐIỂM CHO TẤT CẢ CURVES")
    print("="*90 + "\n")
    
    # Schoof + AI
    print("⏱️  Method: SCHOOF + AI (KHOẢNG THU HẸP)")
    
    t2_start = time.time()
    results_ai = []
    
    for i, (p, a, b) in enumerate(curves):
        order = count_points_schoof_ai(p, a, b, predictor)
        results_ai.append(order)
        
        if (i+1) % 100 == 0:
            print(f"   Progress: {i+1}/{n_curves}...")
    
    t2_total = time.time() - t2_start
    
    print(f"   Tổng thời gian: {t2_total:.3f}s")

    #save results to "curve.txt" file
    with open('curve.txt', 'w') as f:
        for order in results_ai:
            f.write(f"{order}\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    
    benchmark_curve_generation(n_curves=n, bits=bits)

