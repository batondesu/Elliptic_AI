#!/usr/bin/env python3
"""
Sinh thêm dữ liệu chuẩn cho Schoof dataset
- Tăng số lượng samples cho từng khoảng p
- Cải thiện độ phủ dữ liệu
- Đảm bảo chất lượng cao
"""

import numpy as np
import math
import random
from sympy import primerange, legendre_symbol
import time
import os
import pickle
from typing import List, Tuple, Dict, Optional

def count_points_accurate(A: int, B: int, p: int) -> int:
    """Đếm số điểm chính xác trên đường cong elliptic."""
    c = 1  # Điểm vô cực
    # Vector hóa tính r = x^3 + A x + B (mod p)
    x_values = np.arange(p, dtype=np.int64)
    r_values = (x_values * x_values % p * x_values % p + (A % p) * x_values + (B % p)) % p
    # Đếm r == 0 (mỗi điểm cộng 1)
    c += int(np.count_nonzero(r_values == 0))
    # Với r != 0: nếu là bình phương (Legendre symbol = 1) thì có 2 nghiệm
    non_zero = r_values[r_values != 0]
    for r in non_zero:
        try:
            if legendre_symbol(int(r), p) == 1:
                c += 2
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            continue
    return c

def calculate_delta(A: int, B: int, p: int) -> float:
    """Tính δ = p + 1 - N"""
    N = count_points_accurate(A, B, p)
    return float(p + 1 - N)

def calculate_tilde_delta(A: int, B: int, p: int) -> float:
    """Tính δ̃ = δ / (2√p)"""
    delta = calculate_delta(A, B, p)
    return delta / (2.0 * math.sqrt(p))

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

def extract_features(A: int, B: int, p: int) -> List[float]:
    """Trích xuất 40 features từ (A, B, p)"""
    features = []
    
    # Basic features (6)
    features.extend([
        float(A), float(B), float(p),
        float(A % p), float(B % p),
        float((4 * A**3 + 27 * B**2) % p)  # discriminant
    ])
    
    # Logarithmic features (3)
    features.extend([
        math.log10(p), math.log10(max(1, A)), math.log10(max(1, B))
    ])
    
    # Ratios (4)
    features.extend([
        float(A / p), float(B / p),
        float(A / max(1, B)), float(B / max(1, A))
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
        float(math.gcd(A, p)), float(math.gcd(B, p)),
        float(bin(p).count('1'))  # Hamming weight of p
    ])
    
    return features[:40]  # Đảm bảo đúng 40 features

def is_valid_curve(A: int, B: int, p: int) -> bool:
    """Kiểm tra đường cong elliptic hợp lệ"""
    return (4 * A**3 + 27 * B**2) % p != 0

def save_checkpoint(X_list, y_delta_list, y_tilde_list, y_cm_list, 
                   total_generated, current_range_idx, checkpoint_file="checkpoint.pkl"):
    """Lưu checkpoint để có thể resume sau"""
    checkpoint_data = {
        'X_list': X_list,
        'y_delta_list': y_delta_list, 
        'y_tilde_list': y_tilde_list,
        'y_cm_list': y_cm_list,
        'total_generated': total_generated,
        'current_range_idx': current_range_idx,
        'timestamp': time.time()
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"💾 Checkpoint saved: {total_generated} samples")

def load_checkpoint(checkpoint_file="checkpoint.pkl"):
    """Load checkpoint nếu có"""
    if not os.path.exists(checkpoint_file):
        return None, None, None, None, 0, 0
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"📂 Loaded checkpoint: {checkpoint_data['total_generated']} samples")
        return (checkpoint_data['X_list'], 
                checkpoint_data['y_delta_list'],
                checkpoint_data['y_tilde_list'], 
                checkpoint_data['y_cm_list'],
                checkpoint_data['total_generated'],
                checkpoint_data['current_range_idx'])
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        return None, None, None, None, 0, 0

def save_to_existing_dataset(X_list, y_delta_list, y_tilde_list, y_cm_list):
    """Lưu dữ liệu trực tiếp vào dataset hiện có"""
    if len(X_list) == 0:
        return
    
    # Convert to arrays
    X_new = np.array(X_list, dtype=np.float64)
    y_delta_new = np.array(y_delta_list, dtype=np.float64)
    y_tilde_new = np.array(y_tilde_list, dtype=np.float64)
    y_cm_new = np.array(y_cm_list, dtype=np.float64)
    
    try:
        # Load existing data
        existing_X = np.load('schoof_data_X_cleaned.npy')
        existing_y_delta = np.load('schoof_data_delta.npy')
        existing_y_tilde = np.load('schoof_data_tilde_delta.npy')
        existing_y_cm = np.load('schoof_data_cm.npy')
        
        print(f"📂 Found existing data: {existing_X.shape[0]} samples")
        
        # Append new data
        X_combined = np.vstack([existing_X, X_new])
        y_delta_combined = np.concatenate([existing_y_delta, y_delta_new])
        y_tilde_combined = np.concatenate([existing_y_tilde, y_tilde_new])
        y_cm_combined = np.concatenate([existing_y_cm, y_cm_new])
        
    except FileNotFoundError:
        print("📂 No existing data found, creating new dataset")
        X_combined = X_new
        y_delta_combined = y_delta_new
        y_tilde_combined = y_tilde_new
        y_cm_combined = y_cm_new
    
    # Save combined data
    np.save('schoof_data_X_cleaned.npy', X_combined)
    np.save('schoof_data_delta.npy', y_delta_combined)
    np.save('schoof_data_tilde_delta.npy', y_tilde_combined)
    np.save('schoof_data_cm.npy', y_cm_combined)
    
    print(f"💾 Saved {len(X_list)} new samples to existing dataset")
    print(f"📊 Total dataset now: {X_combined.shape[0]} samples")

def merge_all_batches():
    """Merge tất cả batch files thành dataset cuối cùng"""
    batch_files = []
    batch_num = 1
    
    # Tìm tất cả batch files
    while os.path.exists(f'batch_X_{batch_num}.npy'):
        batch_files.append(batch_num)
        batch_num += 1
    
    if not batch_files:
        print("📂 No batch files found")
        return 0
    
    print(f"🔄 Merging {len(batch_files)} batch files...")
    
    # Load và merge tất cả batches
    X_list = []
    y_delta_list = []
    y_tilde_list = []
    y_cm_list = []
    
    for batch_num in batch_files:
        X_batch = np.load(f'batch_X_{batch_num}.npy')
        y_delta_batch = np.load(f'batch_delta_{batch_num}.npy')
        y_tilde_batch = np.load(f'batch_tilde_{batch_num}.npy')
        y_cm_batch = np.load(f'batch_cm_{batch_num}.npy')
        
        X_list.append(X_batch)
        y_delta_list.append(y_delta_batch)
        y_tilde_list.append(y_tilde_batch)
        y_cm_list.append(y_cm_batch)
        
        print(f"✅ Loaded batch {batch_num}: {len(X_batch)} samples")
    
    # Merge all batches
    X = np.vstack(X_list)
    y_delta = np.concatenate(y_delta_list)
    y_tilde = np.concatenate(y_tilde_list)
    y_cm = np.concatenate(y_cm_list)
    
    # Merge with existing data
    try:
        existing_X = np.load('schoof_data_X_cleaned.npy')
        existing_y_delta = np.load('schoof_data_delta.npy')
        existing_y_tilde = np.load('schoof_data_tilde_delta.npy')
        existing_y_cm = np.load('schoof_data_cm.npy')
        
        print(f"📂 Found existing data: {existing_X.shape[0]} samples")
        
        # Merge
        X = np.vstack([existing_X, X])
        y_delta = np.concatenate([existing_y_delta, y_delta])
        y_tilde = np.concatenate([existing_y_tilde, y_tilde])
        y_cm = np.concatenate([existing_y_cm, y_cm])
        
    except FileNotFoundError:
        print("📂 No existing data found")
    
    # Save final dataset
    np.save('schoof_data_X_cleaned.npy', X)
    np.save('schoof_data_delta.npy', y_delta)
    np.save('schoof_data_tilde_delta.npy', y_tilde)
    np.save('schoof_data_cm.npy', y_cm)
    
    # Save feature names
    feature_names = [
        'A', 'B', 'p', 'A_mod_p', 'B_mod_p', 'discriminant',
        'log_p', 'log_A', 'log_B',
        'A_over_p', 'B_over_p', 'A_over_B', 'B_over_A',
        'legendre_A', 'legendre_B', 'legendre_disc',
        'A_mod_3', 'A_mod_5', 'A_mod_7', 'B_mod_3', 'B_mod_5', 'B_mod_7',
        'A_pow_2', 'A_pow_3', 'A_pow_4', 'B_pow_2', 'B_pow_3', 'B_pow_4',
        'A_plus_B', 'A_minus_B', 'A_times_B', 'A2_plus_B2', 'A2_minus_B2', 'A3_plus_B3',
        'j_invariant', 'A_plus_p_mod', 'B_plus_p_mod', 'gcd_A_p', 'gcd_B_p', 'hamming_p'
    ]
    
    with open('schoof_feature_names_cleaned.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Clean up batch files
    for batch_num in batch_files:
        os.remove(f'batch_X_{batch_num}.npy')
        os.remove(f'batch_delta_{batch_num}.npy')
        os.remove(f'batch_tilde_{batch_num}.npy')
        os.remove(f'batch_cm_{batch_num}.npy')
    
    print(f"🎉 Final dataset: {X.shape[0]} samples")
    print("🧹 Cleaned up batch files")
    
    return X.shape[0]

def generate_enhanced_data(target_samples: int = 50000, max_p: int = 200000, 
                          batch_size: int = 1000, auto_save_interval: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Sinh data chuẩn với nhiều mẫu và checkpoint support"""
    print(f"🚀 Generating {target_samples} high-quality samples...")
    print(f"📦 Batch size: {batch_size}, Auto-save every: {auto_save_interval} samples")
    
    # Try to load checkpoint first
    X_list, y_delta_list, y_tilde_list, y_cm_list, total_generated, start_range_idx = load_checkpoint()
    
    if X_list is None:
        # Start fresh
        X_list = []
        y_delta_list = []
        y_tilde_list = []
        y_cm_list = []
        total_generated = 0
        start_range_idx = 0
        print("🆕 Starting fresh generation")
    else:
        print(f"🔄 Resuming from checkpoint: {total_generated} samples")
    
    # Improved sampling strategy
    prime_ranges = [
        (11, 100, 1000),      # Small primes: nhiều mẫu
        (101, 1000, 1000),   # Medium primes: nhiều mẫu 
        (1001, 10000, 1000), # Large primes: nhiều mẫu
        (10001, 50000, 10000),# Very large: ít hơn
        (50001, max_p, 5000)  # Huge primes: ít nhất
    ]
    
    batch_num = 1
    last_save_time = time.time()
    
    for range_idx, (min_p, max_p_range, samples_target) in enumerate(prime_ranges):
        # Skip ranges that were already completed
        if range_idx < start_range_idx:
            continue
            
        print(f"\n📊 Range {min_p}-{max_p_range}: Target {samples_target} samples")
        
        primes = list(primerange(min_p, min(max_p_range + 1, max_p + 1)))
        if not primes:
            continue
            
        range_samples = 0
        attempts = 0
        max_attempts = samples_target * 10
        
        while range_samples < samples_target and attempts < max_attempts:
            p = random.choice(primes)
            
            # Better A, B sampling
            if p < 100:
                A = random.randint(1, p - 1)
                B = random.randint(1, p - 1)
            elif p < 1000:
                A = random.randint(1, min(1000, p - 1))
                B = random.randint(1, min(1000, p - 1))
            else:
                # Biased towards smaller values for large p
                if random.random() < 0.7:
                    A = random.randint(1, min(10000, p // 10))
                    B = random.randint(1, min(10000, p // 10))
                else:
                    A = random.randint(1, p - 1)
                    B = random.randint(1, p - 1)
            
            attempts += 1
            
            if not is_valid_curve(A, B, p):
                continue
                
            try:
                # Calculate targets
                delta = calculate_delta(A, B, p)
                tilde_delta = calculate_tilde_delta(A, B, p)
                
                # Improved CM detection
                j_inv = j_invariant_mod_p(A, B, p)
                
                # Known CM j-invariants mod small primes
                known_cm_j = [0, 1728]  # j=0 (A=0), j=1728 (B=0)
                if p > 3:
                    known_cm_j.extend([8000, 54000])  # Other common CM values
                
                has_cm = j_inv in known_cm_j or abs(tilde_delta) > 1.8
                
                # Extract features
                features = extract_features(A, B, p)
                
                # Add to dataset
                X_list.append(features)
                y_delta_list.append(delta)
                y_tilde_list.append(tilde_delta)
                y_cm_list.append(1.0 if has_cm else 0.0)
                
                range_samples += 1
                total_generated += 1
                
                # Auto-save checkpoint
                if total_generated % auto_save_interval == 0:
                    save_checkpoint(X_list, y_delta_list, y_tilde_list, y_cm_list, 
                                  total_generated, range_idx)
                
                # Save to existing dataset when reaching batch_size
                if len(X_list) >= batch_size:
                    save_to_existing_dataset(X_list, y_delta_list, y_tilde_list, y_cm_list)
                    X_list = []
                    y_delta_list = []
                    y_tilde_list = []
                    y_cm_list = []
                    batch_num += 1
                    print(f"💾 Batch {batch_num-1} saved to existing dataset")
                
                if total_generated % 5000 == 0:
                    print(f"✅ Generated {total_generated} samples...")
                    
            except Exception as e:
                continue
        
        print(f"✅ Range {min_p}-{max_p_range}: Generated {range_samples} samples")
        
        # Save checkpoint after each range
        save_checkpoint(X_list, y_delta_list, y_tilde_list, y_cm_list, 
                       total_generated, range_idx + 1)
    
    # Save remaining data to existing dataset
    if len(X_list) > 0:
        save_to_existing_dataset(X_list, y_delta_list, y_tilde_list, y_cm_list)
        print(f"💾 Final batch saved to existing dataset: {len(X_list)} samples")
    
    # Clean up checkpoint
    if os.path.exists("checkpoint.pkl"):
        os.remove("checkpoint.pkl")
        print("🧹 Cleaned up checkpoint file")
    
    # Feature names
    feature_names = [
        'A', 'B', 'p', 'A_mod_p', 'B_mod_p', 'discriminant',
        'log_p', 'log_A', 'log_B',
        'A_over_p', 'B_over_p', 'A_over_B', 'B_over_A',
        'legendre_A', 'legendre_B', 'legendre_disc',
        'A_mod_3', 'A_mod_5', 'A_mod_7', 'B_mod_3', 'B_mod_5', 'B_mod_7',
        'A_pow_2', 'A_pow_3', 'A_pow_4', 'B_pow_2', 'B_pow_3', 'B_pow_4',
        'A_plus_B', 'A_minus_B', 'A_times_B', 'A2_plus_B2', 'A2_minus_B2', 'A3_plus_B3',
        'j_invariant', 'A_plus_p_mod', 'B_plus_p_mod', 'gcd_A_p', 'gcd_B_p', 'hamming_p'
    ]
    
    print(f"\n🎉 Generated {total_generated} new samples!")
    
    # Load final dataset for return
    try:
        X = np.load('schoof_data_X_cleaned.npy')
        y_delta = np.load('schoof_data_delta.npy')
        y_tilde = np.load('schoof_data_tilde_delta.npy')
        y_cm = np.load('schoof_data_cm.npy')
        
        print(f"📊 Total dataset: {X.shape[0]} samples")
        print(f"🎯 Delta range: [{y_delta.min():.2f}, {y_delta.max():.2f}]")
        print(f"🎯 Tilde delta range: [{y_tilde.min():.3f}, {y_tilde.max():.3f}]")
        print(f"🎯 CM ratio: {y_cm.mean():.3f}")
        
        return X, y_delta, y_tilde, y_cm, feature_names
    except FileNotFoundError:
        print("❌ Error loading final dataset")
        return np.array([]), np.array([]), np.array([]), np.array([]), feature_names

def merge_with_existing(new_X, new_y_delta, new_y_tilde, new_y_cm, new_feature_names):
    """Merge với dataset hiện có"""
    
    # Load existing data
    try:
        existing_X = np.load('schoof_data_X_cleaned.npy')
        existing_y_delta = np.load('schoof_data_delta.npy')
        existing_y_tilde = np.load('schoof_data_tilde_delta.npy')
        existing_y_cm = np.load('schoof_data_cm.npy')
        
        with open('schoof_feature_names_cleaned.txt', 'r') as f:
            existing_feature_names = [line.strip() for line in f.readlines()]
        
        print(f"📂 Loaded existing data: {existing_X.shape[0]} samples")
        
    except FileNotFoundError:
        print("📂 No existing data found, creating new dataset")
        existing_X = np.array([]).reshape(0, len(new_feature_names))
        existing_y_delta = np.array([])
        existing_y_tilde = np.array([])
        existing_y_cm = np.array([])
        existing_feature_names = new_feature_names
    
    # Merge datasets
    if existing_X.shape[0] > 0:
        merged_X = np.vstack([existing_X, new_X])
        merged_y_delta = np.concatenate([existing_y_delta, new_y_delta])
        merged_y_tilde = np.concatenate([existing_y_tilde, new_y_tilde])
        merged_y_cm = np.concatenate([existing_y_cm, new_y_cm])
    else:
        merged_X = new_X
        merged_y_delta = new_y_delta
        merged_y_tilde = new_y_tilde
        merged_y_cm = new_y_cm
    
    print(f"🔄 Merged dataset: {merged_X.shape[0]} total samples")
    
    # Save merged data
    np.save('schoof_data_X_cleaned.npy', merged_X)
    np.save('schoof_data_delta.npy', merged_y_delta)
    np.save('schoof_data_tilde_delta.npy', merged_y_tilde)
    np.save('schoof_data_cm.npy', merged_y_cm)
    
    with open('schoof_feature_names_cleaned.txt', 'w') as f:
        for name in existing_feature_names:
            f.write(f"{name}\n")
    
    print("💾 Saved enhanced dataset!")
    return merged_X.shape[0]

def main():
    """Main function với checkpoint support"""
    print("=" * 60)
    print("🚀 ENHANCED SCHOOF DATA GENERATION (WITH CHECKPOINT)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Generate new data với checkpoint support
    new_X, new_y_delta, new_y_tilde, new_y_cm, feature_names = generate_enhanced_data(
        target_samples=50000,  # Sinh 50k samples mới
        max_p=200000,
        batch_size=1000,       # Save mỗi 5k samples vào dataset hiện có
        auto_save_interval=1000  # Checkpoint mỗi 1k samples
    )
    
    if len(new_X) == 0:
        print("❌ Failed to generate data!")
        return
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.1f}s")
    print("✅ Data generation completed!")
    print("\n💡 Tips:")
    print("  - Nếu bị ngắt, chạy lại script sẽ tự động resume từ checkpoint")
    print("  - Dữ liệu được lưu trực tiếp vào dataset hiện có, không bị mất khi ngắt")
    print("  - Checkpoint được lưu mỗi 1000 samples")
    print("  - Dữ liệu được append vào file dataset hiện có")

def merge_batches_only():
    """Chỉ merge các batch files thành dataset cuối cùng"""
    print("🔄 Merging existing batch files...")
    total_samples = merge_all_batches()
    print(f"✅ Merged {total_samples} samples from batch files")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        # Chỉ merge batch files
        merge_batches_only()
    else:
        # Chạy generation bình thường
        main()