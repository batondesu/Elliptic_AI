import csv
import random
from sage.all import *

def generate_ec_dataset(bits=64, num_samples=1000, filename="input64.txt"):
    data = []
    p_min = 2**(bits-1)
    p_max = 2**bits - 1

    # Vòng lặp chính
    for i in range(num_samples):
        # Reset bộ nhớ PARI định kỳ (tránh tràn stack)
        if i % 100 == 0:
            pari.allocatemem(4 * 10**9)

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

    print(f"✅ Đã sinh {len(data)} elliptica curves {bits}-bit và lưu vào {filename}")

# Sinh 70,000 dữ liệu vào input32.txt
generate_ec_dataset(bits=256, num_samples=25000, filename="input1281.txt")