import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from math import log2, floor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""
Model mới: Dự đoán TRACE Frobenius thay vì ORDER

Lý do:
- Mục đích: Thu hẹp khoảng Hasse từ [-2√p, 2√p] xuống [t_pred - δ, t_pred + δ]
- Trace có phân bố tập trung hơn quanh 0, dễ học hơn
- Order = p + 1 - trace, nên dự đoán trace trực tiếp tốt hơn
"""

def read_raw_data():
	with open('input64.txt') as file:
		lines = file.readlines()
		return lines

def proccess_raw_data(raw_data):
	"""
	Xử lý dữ liệu:
	Input: p a b order
	→ Chuyển thành: p a b trace
	→ Normalize trace theo khoảng Hasse
	"""
	data = []
	for line in raw_data:
		data.append(list(map(int, line.split())))
	data = np.array(data, dtype=np.longdouble)
	
	# Lưu order gốc
	original_orders = np.copy(data[:, 3])
	
	# Chuyển order → trace
	# trace = p + 1 - order
	traces = data[:, 0] + 1 - data[:, 3]
	
	# Normalize trace về [0, 1]
	# Trace ∈ [-2√p, 2√p] theo định lý Hasse
	# Normalize: (trace + 2√p) / (4√p) = trace/(4√p) + 0.5
	sqrt_p = np.sqrt(data[:, 0])
	traces_normalized = traces / (4 * sqrt_p) + 0.5
	
	# Thay cột thứ 4 bằng trace đã normalize
	data[:, 3] = traces_normalized
	
	return data, original_orders, traces

def generate_X_Y_sets(data):
	"""
	X = [p, a, b]
	Y = trace_normalized
	"""
	X = data[:, :-1]  # p, a, b
	Y = data[:, -1]   # trace normalized
	n = len(data)
	return X, Y, n

def Model():
	"""Kiến trúc model như cũ"""
	model = tf.keras.Sequential([
		tfl.Dense(units=512, activation='relu', input_dim=3),
		tfl.BatchNormalization(),

		tfl.Dense(units=256, activation='relu'),
		tfl.Dropout(0.1),
		tfl.BatchNormalization(),

		tfl.Dense(units=128, activation='relu'),
		tfl.Dropout(0.15),
		tfl.BatchNormalization(),

		tfl.Dense(units=64, activation='relu'),
		tfl.Dropout(0.2),
		tfl.BatchNormalization(),

		tfl.Dense(units=32, activation='relu'),
		tfl.Dropout(0.2),
		tfl.BatchNormalization(),

		tfl.Dense(units=16, activation='relu'),
		tfl.Dropout(0.25),
		tfl.BatchNormalization(),

		tfl.Dense(units=8, activation='relu'),
		tfl.Dropout(0.3),
		tfl.BatchNormalization(),

		tfl.Dense(units=1, activation='sigmoid')
	])
	return model

# Main training
print("\n" + "="*70)
print("TRAINING MODEL DỰ ĐOÁN TRACE FROBENIUS")
print("="*70)

data = read_raw_data()
data, original_orders, original_traces = proccess_raw_data(data)
X, Y, n = generate_X_Y_sets(data)

print(f'\n Số examples: {n}')
print(f' Dữ liệu: input32.txt')
P_bits = 1 + floor(log2(X[0, 0]))
print(f' p bits: {P_bits}')
print(f' √p ≈ {int(X[0, 0]**0.5):,}')
print(f' Range p: [{int(np.min(X[:, 0])):,}, {int(np.max(X[:, 0])):,}]')

ratio = float(input('\n Nhập tỷ lệ train/test (vd: 0.18): '))
split = floor(ratio * n)

X_train, Y_train = X[:split, :], Y[:split]
X_test, Y_test = X[split:, :], Y[split:]
original_orders_test = original_orders[split:]
original_traces_test = original_traces[split:]

print(f'\n Train: {len(X_train)}, Test: {len(X_test)}')

# Build model
model = Model()

# Try load weights
weights_file = str(ratio).replace('.', '1') + 'weights_trace.hdf5'
try:
	model.load_weights(weights_file)
	print(f' ✓ Loaded weights: {weights_file}\n')
except:
	print(f' • Khởi tạo weights mới: {weights_file}\n')

# Compile & train
# Giảm learning rate để training ổn định hơn
model.compile(
	optimizer=tf.keras.optimizers.Adam(
		learning_rate=0.001,  # Giảm từ 0.005 xuống 0.001
		clipnorm=1.0  # Clip gradients để tránh exploding
	), 
	loss='mse',
	metrics=['mae']  # Thêm MAE để theo dõi
)

print("Đang training...")
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")

# Training với early stopping và reduce LR

callbacks = [
	EarlyStopping(
		monitor='loss',
		patience=5,
		restore_best_weights=True,
		verbose=1
	),
	ReduceLROnPlateau(
		monitor='loss',
		factor=0.5,
		patience=3,
		min_lr=0.0001,
		verbose=1
	)
]

history = model.fit(
	X_train, Y_train, 
	epochs=50, 
	batch_size=min(64, len(X_train)), 
	shuffle=True,  # Shuffle để training ổn định hơn
	callbacks=callbacks,
	verbose=1
)

# Save
model.save_weights(weights_file)
print(f'\n✓ Saved weights: {weights_file}\n')

# Evaluate
print("="*70)
print("ĐÁNH GIÁ KẾT QUẢ")
print("="*70)

eval_results = model.evaluate(X_test, Y_test, verbose=0)
if isinstance(eval_results, list):
	loss, mae = eval_results
	print(f'\nTest loss (MSE): {loss:.6f}, MAE: {mae:.6f}')
else:
	loss = eval_results
	print(f'\nTest loss (MSE): {loss:.6f}')

# Predictions
n_test = np.shape(X_test)[0]
predictions_normalized = np.reshape(model.predict(X_test, verbose=0), n_test)

# Denormalize về trace thực
# trace = (y_norm - 0.5) * 4√p
sqrt_p_test = np.sqrt(X_test[:, 0])
traces_pred = (predictions_normalized - 0.5) * 4 * sqrt_p_test

# So sánh với trace thực
Delta = np.absolute(original_traces_test - traces_pred)
avg_diff = np.sum(Delta) / n_test

print(f'\n√p trung bình: {np.mean(sqrt_p_test):,.0f}')
print(f'Sai số trung bình (avg_diff): {avg_diff:,.0f}')
print(f'Sai số / √p: {avg_diff / np.mean(sqrt_p_test):.2f}')

print(f'\n2 × avg_diff = {2*avg_diff:,.0f}')
print(f'4 × avg_diff = {4*avg_diff:,.0f}')

print(f'\nPhân bố sai số (18 examples đầu):')
print(Delta[:18])

# Phần trăm trong khoảng 2×avg_diff
within_2avg = (Delta < 2*avg_diff).sum() / n_test * 100
print(f'\n{within_2avg:.1f}% predictions trong khoảng ±2×avg_diff')

# Thu hẹp khoảng Hasse
hasse_original = 4 * np.mean(sqrt_p_test)
hasse_ai = 2 * 2 * avg_diff  # ±2×avg_diff
reduction = hasse_original / hasse_ai

print(f'\n' + "="*70)
print("THU HẸP KHOẢNG HASSE")
print("="*70)
print(f'Khoảng Hasse gốc: [-2√p, 2√p] = {hasse_original:,.0f}')
print(f'Khoảng AI (±2×avg): [t_pred - δ, t_pred + δ] = {hasse_ai:,.0f}')
print(f'Thu hẹp: {reduction:.2f}x')
print("="*70)

# Lưu kết quả
with open('./output_trace_nn.txt', 'w') as file:
	file.write('|diff|    p a b trace_true trace_pred\n')
	for i in range(n_test):
		file.write(f'\n  {int(Delta[i])}   {int(X_test[i][0])} '
				  f'{int(X_test[i][1])} {int(X_test[i][2])} '
				  f'{int(original_traces_test[i])} {int(traces_pred[i])}')

print(f'\n✓ Kết quả lưu tại: output_trace_nn.txt\n')

