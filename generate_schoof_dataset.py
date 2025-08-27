#!/usr/bin/env python3
"""
Sinh thêm dữ liệu và nối vào dataset hiện có
- Giữ nguyên dataset cũ
- Sinh thêm dữ liệu mới
- Nối vào cuối dataset hiện có
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
		try:
			if legendre_symbol(int(r), p) == 1:
				c += 2
		except (KeyboardInterrupt, SystemExit):
			raise
		except Exception:
			# Bỏ qua lỗi legendre_symbol, coi như không phải bình phương
			continue
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

def load_existing_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
	"""Tải dataset hiện có"""
	print("Đang tải dataset hiện có...")
	
	if not os.path.exists('schoof_data_X_cleaned.npy'):
		print("Không tìm thấy dataset hiện có. Tạo dataset mới.")
		return np.array([]), np.array([]), np.array([]), np.array([]), []
	
	X = np.load('schoof_data_X_cleaned.npy')
	y_delta = np.load('schoof_data_delta.npy')
	y_tilde = np.load('schoof_data_tilde_delta.npy')
	y_cm = np.load('schoof_data_cm.npy')
	
	feature_names = []
	if os.path.exists('schoof_feature_names_cleaned.txt'):
		with open('schoof_feature_names_cleaned.txt', 'r') as f:
			feature_names = [line.strip() for line in f.readlines()]
	
	print(f"Dataset hiện có: {len(X):,} mẫu")
	return X, y_delta, y_tilde, y_cm, feature_names

def generate_additional_data(existing_primes: set, max_p: int = 10000,
						   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
	"""Sinh thêm dữ liệu mới với primes chưa có"""
	print(f"SINH THÊM DỮ LIỆU MỚI")
	print(f"Tham số: max_p={max_p:,}")
	print("=" * 60)

	start_time = time.time()
	all_primes = list(primerange(3, max_p + 1))
	
	# Lọc ra primes chưa có trong dataset hiện tại
	new_primes = [p for p in all_primes if p not in existing_primes]
	print(f"Tìm thấy {len(new_primes):,} primes mới (từ {len(all_primes):,} primes tổng)")
	
	if len(new_primes) == 0:
		print("Không có primes mới để sinh dữ liệu!")
		return np.array([]), np.array([]), np.array([]), np.array([]), []

	data: List[List[float]] = []
	feature_names: Optional[List[str]] = None
	valid_samples = 0

	for i, p in enumerate(new_primes):
		if i % 20 == 0:
			elapsed = time.time() - start_time
			print(f"p={p:,} ({i+1}/{len(new_primes)}) — đã sinh {valid_samples:,} mẫu — {elapsed:.1f}s")

		# Số mẫu động theo kích thước p (tăng để có nhiều data hơn)
		if p >= 200000:
			current_samples = random.randint(5, 10)  # Tăng cho p rất lớn
		elif p >= 100000:
			current_samples = random.randint(10, 20)
		elif p >= 50000:
			current_samples = random.randint(15, 30)
		elif p >= 10000:
			current_samples = random.randint(30, 60)
		elif p >= 1000:
			current_samples = random.randint(100, 200)
		elif p >= 100:
			current_samples = random.randint(500, 1000)
		else:
			current_samples = random.randint(100, 200)

		attempts = 0
		max_attempts = current_samples * 200
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


	data_np = np.array(data, dtype=np.float32)
	feat_cnt = len(feature_names) if feature_names else 0
	X = data_np[:, :feat_cnt]
	y_delta = data_np[:, feat_cnt]
	y_tilde = data_np[:, feat_cnt + 1]
	y_cm = data_np[:, feat_cnt + 2].astype(np.int32)

	elapsed_time = time.time() - start_time
	print("\n" + "=" * 60)
	print("HOÀN THÀNH SINH DỮ LIỆU MỚI!")
	print("=" * 60)
	print(f"Số mẫu mới: {len(data_np):,}")
	print(f"Số đặc trưng: {feat_cnt}")
	print(f"Thời gian: {elapsed_time:.1f}s")
	if elapsed_time > 0:
		print(f"Tốc độ: {len(data_np)/elapsed_time:.1f} mẫu/giây")
	print(f"   Phạm vi p: [{X[:,0].min():.0f}, {X[:,0].max():.0f}]")
	print(f"   CM curves: {int(y_cm.sum())} / {len(y_cm)} ({100*y_cm.sum()/len(y_cm):.2f}%)")
	return X, y_delta, y_tilde, y_cm, (feature_names or [])

def merge_and_save_datasets(existing_X: np.ndarray, existing_y_delta: np.ndarray, existing_y_tilde: np.ndarray, existing_y_cm: np.ndarray,
						   new_X: np.ndarray, new_y_delta: np.ndarray, new_y_tilde: np.ndarray, new_y_cm: np.ndarray,
						   feature_names: List[str]):
	"""Nối dataset mới vào dataset hiện có và lưu"""
	print("\nĐANG NỐI VÀ LƯU DATASET...")
	
	if len(existing_X) == 0:
		# Nếu không có dataset cũ, chỉ lưu dataset mới
		final_X = new_X
		final_y_delta = new_y_delta
		final_y_tilde = new_y_tilde
		final_y_cm = new_y_cm
	else:
		# Nối dataset mới vào cuối dataset cũ
		final_X = np.vstack([existing_X, new_X])
		final_y_delta = np.concatenate([existing_y_delta, new_y_delta])
		final_y_tilde = np.concatenate([existing_y_tilde, new_y_tilde])
		final_y_cm = np.concatenate([existing_y_cm, new_y_cm])
	
	# Lưu dataset tổng hợp
	np.save('schoof_data_X_cleaned.npy', final_X)
	np.save('schoof_data_delta.npy', final_y_delta)
	np.save('schoof_data_tilde_delta.npy', final_y_tilde)
	np.save('schoof_data_cm.npy', final_y_cm)
	
	# Lưu feature names
	with open('schoof_feature_names_cleaned.txt', 'w') as f:
		for name in feature_names:
			f.write(name + '\n')
	
	print(f"Dataset tổng hợp: {len(final_X):,} mẫu")
	print(f"  - Dataset cũ: {len(existing_X):,} mẫu")
	print(f"  - Dataset mới: {len(new_X):,} mẫu")
	print("Đã lưu dataset tổng hợp thành công!")

def main():
	"""Main function"""
	print("SINH THÊM DỮ LIỆU VÀ NỐI VÀO DATASET HIỆN CÓ")
	print("=" * 60)
	
	# Tải dataset hiện có
	existing_X, existing_y_delta, existing_y_tilde, existing_y_cm, existing_feature_names = load_existing_dataset()
	
	# Lấy danh sách primes hiện có
	existing_primes = set()
	if len(existing_X) > 0:
		existing_primes = set(existing_X[:, 0].astype(int))
		print(f"Dataset hiện có chứa {len(existing_primes):,} primes khác nhau")
	
	# Cấu hình sinh thêm dữ liệu
	max_p = 50000  # Tăng phạm vi p để sinh thêm nhiều data
	print(f"Cấu hình sinh thêm:")
	print(f"  - max_p: {max_p:,}")
	print("=" * 60)
	
	# Sinh thêm dữ liệu
	new_X, new_y_delta, new_y_tilde, new_y_cm, new_feature_names = generate_additional_data(
		existing_primes=existing_primes,
		max_p=max_p,
	)
	
	if len(new_X) == 0:
		print("Không sinh được dữ liệu mới!")
		return
	
	# Nối và lưu dataset
	feature_names = new_feature_names if len(existing_feature_names) == 0 else existing_feature_names
	merge_and_save_datasets(
		existing_X, existing_y_delta, existing_y_tilde, existing_y_cm,
		new_X, new_y_delta, new_y_tilde, new_y_cm,
		feature_names
	)
	
	print("\nHoàn thành sinh thêm dữ liệu!")

if __name__ == "__main__":
	main()