#!/usr/bin/env python3
"""
Sinh dataset chuẩn cho AI-enhanced Schoof Algorithm
- Tính toán chính xác δ (delta) thay vì δ̃ (tilde delta)
- Thêm đặc trưng toán học quan trọng: j-invariant, discriminant, CM properties
- Phân loại CM/non-CM chính xác hơn
- Tạo dataset đa dạng với phạm vi p rộng hơn
"""

import numpy as np
import math
import random
from sympy import primerange, legendre_symbol
import time
import os
from typing import List, Tuple, Dict, Optional

def count_points_accurate(A: int, B: int, p: int) -> int:
	"""Đếm số điểm chính xác trên đường cong elliptic (tối ưu hóa vector hóa r_values)."""
	c = 1  # Điểm vô cực
	# Vector hóa tính r = x^3 + A x + B (mod p)
	x_values = np.arange(p, dtype=np.int64)
	r_values = (x_values * x_values % p * x_values % p + (A % p) * x_values + (B % p)) % p
	# Đếm r == 0 (mỗi điểm cộng 1)
	c += int(np.count_nonzero(r_values == 0))
	# Với r != 0: nếu là bình phương (Legendre symbol = 1) thì có 2 nghiệm
	non_zero = r_values[r_values != 0]
	for r in non_zero:
		if legendre_symbol(int(r), p) == 1:
			c += 2
	return c

def calculate_delta(A: int, B: int, p: int) -> float:
	"""Tính δ = p + 1 - N (không chia cho 2√p)"""
	N = count_points_accurate(A, B, p)
	return float(p + 1 - N)

def calculate_tilde_delta(A: int, B: int, p: int) -> float:
	"""Tính δ̃ = δ / (2√p)"""
	delta = calculate_delta(A, B, p)
	return delta / (2.0 * math.sqrt(p))

def j_invariant_mod_p(A: int, B: int, p: int) -> int:
	"""Tính j-invariant mod p"""
	# j = 1728 * 4A^3 / (4A^3 + 27B^2) mod p
	num = (1728 * (4 * pow(A % p, 3, p))) % p
	den = (4 * pow(A % p, 3, p) + 27 * pow(B % p, 2, p)) % p
	if den == 0:
		return -1
	try:
		inv = pow(den, -1, p)
		return (num * inv) % p
	except ValueError:
		return -1

def is_cm_curve(A: int, B: int, p: int) -> bool:
	"""CM đơn giản theo j-invariant đặc biệt."""
	j = j_invariant_mod_p(A, B, p)
	return j != -1 and (j == 0 or j == 1728 % p)

def calculate_discriminant(A: int, B: int, p: int) -> int:
	"""Δ = 4A³ + 27B² mod p"""
	return (4 * pow(A % p, 3, p) + 27 * pow(B % p, 2, p)) % p

def extract_advanced_features(p: int, A: int, B: int) -> Dict[str, float]:
	"""Trích xuất đặc trưng toán học nâng cao"""
	features: Dict[str, float] = {}
	features['p'] = float(p)
	features['A'] = float(A)
	features['B'] = float(B)
	discriminant = calculate_discriminant(A, B, p)
	features['discriminant'] = float(discriminant)
	features['discriminant_ratio'] = discriminant / p
	j_inv = j_invariant_mod_p(A, B, p)
	features['j_invariant'] = float(j_inv) if j_inv != -1 else 0.0
	features['j_invariant_ratio'] = (float(j_inv) / p) if j_inv != -1 else 0.0
	features['A_mod_3'] = float(A % 3)
	features['B_mod_3'] = float(B % 3)
	features['p_mod_3'] = float(p % 3)
	features['A_mod_4'] = float(A % 4)
	features['B_mod_4'] = float(B % 4)
	features['p_mod_4'] = float(p % 4)
	features['A_times_B'] = float((A * B) % p)
	features['A_squared'] = float((A * A) % p)
	features['B_squared'] = float((B * B) % p)
	features['A_over_p'] = A / p
	features['B_over_p'] = B / p
	features['A_plus_B_over_p'] = (A + B) / p
	features['log_p'] = math.log(p)
	features['log_A'] = math.log(abs(A) + 1)
	features['log_B'] = math.log(abs(B) + 1)
	features['sin_A_over_p'] = math.sin(A / p * math.pi)
	features['cos_B_over_p'] = math.cos(B / p * math.pi)
	return features

def generate_schoof_dataset(max_p: int = 1000,
						  samples_per_p: int = 50,
						  target_samples: int = 50000,
						  select_primes_count: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
	"""Sinh dataset chuẩn cho Schoof algorithm (hỗ trợ chọn ~N primes rải đều)."""
	    print(f"SINH DATASET CHUẨN CHO SCHOOF ALGORITHM")
	    print(f"Tham số: max_p={max_p:,}, samples_per_p≈{samples_per_p}, target={target_samples:,}, select_primes_count={select_primes_count}")
	print("=" * 60)

	start_time = time.time()
	all_primes = list(primerange(3, max_p + 1))
	if select_primes_count and len(all_primes) > select_primes_count:
		step = max(1, len(all_primes) // select_primes_count)
		primes = all_primes[::step][:select_primes_count]
		        print(f"Chọn rải đều {len(primes)}/{len(all_primes)} primes")
	else:
		primes = all_primes
		        print(f"Sử dụng toàn bộ {len(primes)} primes")

	data: List[List[float]] = []
	feature_names: Optional[List[str]] = None
	valid_samples = 0

	for i, p in enumerate(primes):
		if i % 10 == 0:
			elapsed = time.time() - start_time
			            print(f"p={p:,} ({i+1}/{len(primes)}) — đã sinh {valid_samples:,} mẫu — {elapsed:.1f}s")

		# Số mẫu động theo kích thước p để đảm bảo thời gian
		if p >= 50000:
			current_samples = random.randint(1, 2)
		elif p >= 10000:
			current_samples = random.randint(1, 3)
		elif p >= 1000:
			current_samples = random.randint(5, 20)
		else:
			current_samples = min(samples_per_p, 50)

		attempts = 0
		max_attempts = current_samples * 100
		p_samples = 0

		while p_samples < current_samples and attempts < max_attempts:
			A = random.randint(1, p - 1)
			B = random.randint(1, p - 1)
			delta_disc = calculate_discriminant(A, B, p)
			if delta_disc == 0:
				attempts += 1
				continue

			# Tính targets
			delta = calculate_delta(A, B, p)
			tilde_delta = delta / (2.0 * math.sqrt(p))
			is_cm = 1.0 if is_cm_curve(A, B, p) else 0.0

			# Features
			feat_map = extract_advanced_features(p, A, B)
			if feature_names is None:
				feature_names = list(feat_map.keys())
			row = list(feat_map.values()) + [delta, tilde_delta, is_cm]
			data.append(row)

			p_samples += 1
			valid_samples += 1
			attempts += 1

		if target_samples and valid_samples >= target_samples:
			break
	if target_samples and valid_samples >= target_samples:
		pass

	data_np = np.array(data, dtype=np.float32)
	feat_cnt = len(feature_names) if feature_names else 0
	X = data_np[:, :feat_cnt]
	y_delta = data_np[:, feat_cnt]
	y_tilde = data_np[:, feat_cnt + 1]
	y_cm = data_np[:, feat_cnt + 2].astype(np.int32)

	elapsed_time = time.time() - start_time
	print("\n" + "=" * 60)
	    print("HOÀN THÀNH SINH DATASET!")
	print("=" * 60)
	    print(f"Số mẫu đã sinh: {len(data_np):,}")
	    print(f"Số đặc trưng: {feat_cnt}")
	    print(f"Thời gian: {elapsed_time:.1f}s")
	if elapsed_time > 0:
		        print(f"Tốc độ: {len(data_np)/elapsed_time:.1f} mẫu/giây")
	print(f"   Phạm vi p: [{X[:,0].min():.0f}, {X[:,0].max():.0f}]")
	print(f"   CM curves: {int(y_cm.sum())} / {len(y_cm)} ({100*y_cm.sum()/len(y_cm):.2f}%)")
	return X, y_delta, y_tilde, y_cm, (feature_names or [])

def save_schoof_dataset(X: np.ndarray, y_delta: np.ndarray, y_tilde_delta: np.ndarray,
						 y_cm: np.ndarray, feature_names: List[str]):
	"""Lưu dataset Schoof"""
	    print("\nĐANG LƯU DATASET...")
	np.save('schoof_data_X.npy', X)
	np.save('schoof_data_delta.npy', y_delta)
	np.save('schoof_data_tilde_delta.npy', y_tilde_delta)
	np.save('schoof_data_cm.npy', y_cm)
	with open('schoof_feature_names.txt', 'w') as f:
		for name in feature_names:
			f.write(name + '\n')
	    print("Đã lưu dataset & feature names")

def main():
	"""Main function"""
	    print("SCHOOF DATASET GENERATOR")
	print("=" * 60)
	# Cấu hình lớn: max_p=100000, chọn ~500 primes, mục tiêu 30k-50k mẫu (thực tế tùy p)
	max_p = 100000
	select_primes_count = 500
	target_samples = 30000  # có thể vượt/thiếu một chút do dynamic sampling
	samples_per_p = 50  # chỉ áp dụng cho p nhỏ, p lớn sẽ giảm tự động
	X, y_delta, y_tilde, y_cm, feature_names = generate_schoof_dataset(
		max_p=max_p,
		samples_per_p=samples_per_p,
		target_samples=target_samples,
		select_primes_count=select_primes_count
	)
	save_schoof_dataset(X, y_delta, y_tilde, y_cm, feature_names)
	    print("\nDataset sẵn sàng cho huấn luyện v2!")

if __name__ == "__main__":
	main() 