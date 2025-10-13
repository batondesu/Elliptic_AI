import csv
import random
from sage.all import *

def generate_ec_dataset(bits=32, num_samples=1000, filename="ec_dataset_24bit.txt"):
    data = []
    p_min = 2**(bits-1)
    p_max = 2**bits - 1

    # Vòng lặp chính
    for i in range(num_samples):
        # Reset bộ nhớ PARI định kỳ (tránh tràn stack)
        if i % 700 == 0:
            pari.allocatemem(10**7)

        # Sinh số nguyên tố p
        p = random_prime(p_max, lbound=p_min)

        # Sinh hệ số a, b sao cho không suy biến (Δ ≠ 0)
        while True:
            a = randint(0, p - 1)
            b = randint(0, p - 1)
            if (4 * a^3 + 27 * b^2) % p != 0:
                break

        # Tạo elliptic curve trên trường F_p
        E = EllipticCurve(GF(p), [a, b])

        # Tính số điểm n = |E(F_p)|
        try:
            n = E.cardinality()  # an toàn hơn với PARI
        except Exception as e:
            print(f"⚠️ Bỏ qua p={p} do lỗi: {e}")
            continue

        data.append((p, a, b, n))
        
    with open(filename, "w") as f:
        for p, a, b, n in data:
            f.write(f"{p} {a} {b} {n}\n")

    print(f"✅ Đã sinh {len(data)} elliptic curves {bits}-bit và lưu vào {filename}")

# Ví dụ chạy:
generate_ec_dataset(bits=24, num_samples=7000, filename="ec_dataset_24bit.txt")