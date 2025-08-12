import tensorflow as tf
import numpy as np
import math, random
from sympy import primerange, legendre_symbol
from sklearn.model_selection import train_test_split
import time

def count_points(A, B, p):
    """Đếm số điểm trên đường cong elliptic y² = x³ + Ax + B (mod p)"""
    c = 1  
    for x in range(p):
        r = (x**3 + A*x + B) % p
        if r == 0:
            c += 1        
        elif legendre_symbol(r, p) == 1:
            c += 2       
    return c

def generate_elliptic_data(max_p=500, samples_per_p=20, seed=42):
    """Sinh dữ liệu cho đường cong elliptic"""
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Đang sinh dữ liệu với p < {max_p}, {samples_per_p} mẫu mỗi p...")
    start_time = time.time()
    
    data = []
    primes = list(primerange(2, max_p))
    
    for i, p in enumerate(primes):
        if i % 10 == 0:
            print(f"Đang xử lý p = {p} ({i+1}/{len(primes)})")
        
        valid_samples = 0
        attempts = 0
        max_attempts = samples_per_p * 10 
        
        while valid_samples < samples_per_p and attempts < max_attempts:
            A = random.randrange(p)
            B = random.randrange(p)
            
            if (4*A**3 + 27*B**2) % p == 0:
                attempts += 1
                continue
                
            N = count_points(A, B, p)
            delta = p + 1 - N
            tilde_delta = delta / (2 * math.sqrt(p))
            
            data.append([p, A, B, tilde_delta])
            valid_samples += 1
            attempts += 1
    
    data = np.array(data)
    X = data[:, :3].astype(np.float32)   
    y = data[:, 3].astype(np.float32)  
    
    elapsed_time = time.time() - start_time
    print(f"Hoàn thành! Sinh được {len(data)} mẫu trong {elapsed_time:.2f} giây")
    print(f"Kích thước dữ liệu: X={X.shape}, y={y.shape}")
    print(f"Phạm vi tilde_delta: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y


if __name__ == "__main__":
    X, y = generate_elliptic_data(max_p=300, samples_per_p=15)
    
    
    np.save('elliptic_data_X.npy', X)
    np.save('elliptic_data_y.npy', y)
    print("Đã lưu dữ liệu vào elliptic_data_X.npy và elliptic_data_y.npy")
else:
    X, y = generate_elliptic_data(max_p=200, samples_per_p=10)

