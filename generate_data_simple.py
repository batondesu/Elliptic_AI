import numpy as np
import math, random
from sympy import primerange, legendre_symbol
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
    primes = list(primerange(3, max_p))  
    
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

def analyze_data(X, y):
    """Phân tích dữ liệu"""
    print("\n" + "=" * 50)
    print("PHÂN TÍCH DỮ LIỆU")
    print("=" * 50)
    
    feature_names = ['p', 'A', 'B']
    
    for i, feature in enumerate(feature_names):
        print(f"\nĐặc trưng {feature}:")
        print(f"  Phạm vi: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
        print(f"  Trung bình: {X[:, i].mean():.2f}")
        print(f"  Độ lệch chuẩn: {X[:, i].std():.2f}")
    
    print(f"\nĐặc trưng tilde_delta:")
    print(f"  Phạm vi: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Trung bình: {y.mean():.4f}")
    print(f"  Độ lệch chuẩn: {y.std():.4f}")
    
    # Tính correlation
    print(f"\nCorrelation matrix:")
    data_matrix = np.column_stack([X, y])
    corr_matrix = np.corrcoef(data_matrix.T)
    
    for i, name1 in enumerate(feature_names + ['tilde_delta']):
        for j, name2 in enumerate(feature_names + ['tilde_delta']):
            if i <= j:
                print(f"  {name1} vs {name2}: {corr_matrix[i, j]:.4f}")

def save_data(X, y, filename_prefix='elliptic_data'):
    """Lưu dữ liệu"""
    np.save(f'{filename_prefix}_X.npy', X)
    np.save(f'{filename_prefix}_y.npy', y)
    print(f"\nĐã lưu dữ liệu vào {filename_prefix}_X.npy và {filename_prefix}_y.npy")

if __name__ == "__main__":
    # Sinh dữ liệu mẫu
    X, y = generate_elliptic_data(max_p=300, samples_per_p=15)
    
    # Phân tích dữ liệu
    analyze_data(X, y)
    
    # Lưu dữ liệu
    save_data(X, y)
    
    print("\n" + "=" * 50)
    print("HOÀN THÀNH SINH DỮ LIỆU!")
    print("=" * 50) 