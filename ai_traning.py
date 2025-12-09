import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from math import log2, floor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

BIT_SIZE = 32  

def read_raw_data():
	intputFile = f'input{BIT_SIZE}.txt'
	with open(intputFile) as file:
		lines = file.readlines()
		return lines

def proccess_raw_data(raw_data):
    data = []
    traces = []
    original_orders = []
    p_list, a_list, b_list = [], [], []

    for line in raw_data:
        parts = list(map(int, line.split()))
        if len(parts) >= 4:
            p, a, b, order = parts[:4]
            p_list.append(p)
            a_list.append(a)
            b_list.append(b)

    max_a = max(a_list)
    max_b = max(b_list)

    for idx, line in enumerate(raw_data):
        parts = list(map(int, line.split()))
        if len(parts) >= 4:
            p, a, b, order = parts[:4]

            trace = p + 1 - order
            sqrt_p = np.sqrt(float(p))

            p_norm = np.log2(float(p)) / BIT_SIZE
            a_norm = float(a) / max_a if max_a != 0 else 0.0
            b_norm = float(b) / max_b if max_b != 0 else 0.0
            trace_norm = (trace / (4 * sqrt_p)) + 0.5  

            data.append([p_norm, a_norm, b_norm, trace_norm])
            traces.append(trace)
            original_orders.append(order)

    data = np.array(data, dtype=np.float64)
    traces = np.array(traces, dtype=np.float64)
    original_orders = np.array(original_orders, dtype=np.float64)

    return data, original_orders, traces

def generate_X_Y_sets(data):
	X = data[:, :-1]  
	Y = data[:, -1]  
	n = len(data)
	return X, Y, n

def Model():
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

data = read_raw_data()
data, original_orders, original_traces = proccess_raw_data(data)
X, Y, n = generate_X_Y_sets(data)

ratio = float(input('\n Nhập tỷ lệ train (vd: 0.8): '))
split = floor(ratio * n)

X_train, Y_train = X[:split, :], Y[:split]
X_test, Y_test = X[split:, :], Y[split:]
original_orders_test = original_orders[split:]
original_traces_test = original_traces[split:]

p_test_original = np.power(2.0, X_test[:, 0] * BIT_SIZE)

print(f'\n Train: {len(X_train)}, Test: {len(X_test)}')

model = Model()
weights_file = str(ratio).replace('.', '1') + 'new_weights.hdf5'
try: 
	model.load_weights(weights_file)
	print(f' ✓ Loaded weights: {weights_file}\n')
	print(f"NaN in weights: {np.any(np.isnan(model.get_weights()[0]))}")
except:
	print(f' • Init new weights: {weights_file}\n')

model.compile(
	optimizer=tf.keras.optimizers.Adam(
		learning_rate=0.001,  
		clipnorm=1.0  
	), 
	loss='mse',
	metrics=['mae']  
)

print("Đang training...")
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")

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
	),
	ModelCheckpoint(weights_file, save_best_only=True)
]

history = model.fit(
	X_train, Y_train, 
	epochs=50, 
	batch_size=min(64, len(X_train)), 
	shuffle=True,  
	callbacks=callbacks,
	verbose=1
)

model.save_weights(weights_file)

print("="*70)
print("ĐÁNH GIÁ KẾT QUẢ")
print("="*70)

eval_results = model.evaluate(X_test, Y_test, verbose=0)
if isinstance(eval_results, list):
	loss, mae = eval_results
	print(f'\nTest loss (MSE) hai chjam: {loss:.6f}, MAE: {mae:.6f}')
else:
	loss = eval_results
	print(f'\nTest loss (MSE): {loss:.6f}')

n_test = np.shape(X_test)[0]
trace_norm = np.reshape(model.predict(X_test, verbose=0), n_test)
sqrt_p_test = np.sqrt(p_test_original)
traces_pred = (trace_norm - 0.5) * 4 * sqrt_p_test

n_epsilon = np.absolute(original_traces_test - traces_pred)
epsilon = np.sum(n_epsilon) / n_test

print(f'\n√p trung bình: {np.mean(sqrt_p_test):,.0f}')
print(f'Sai số trung bình (epsilon): {epsilon:,.0f}')

print(f'\n2 × epsilon = {2*epsilon:,.0f}')
print(f'4 × epsilon = {4*epsilon:,.0f}')

within_2avg = (n_epsilon < 2*epsilon).sum() / n_test * 100
print(f'\n{within_2avg:.1f}% n_epsilon trong khoảng ±2×epsilon')

hasse_original = 4 * np.mean(sqrt_p_test)
hasse_ai = 2 * 2 * epsilon
reduction = (1 - hasse_ai / hasse_original) * 100

print(f'\n' + "="*70)
print("THU HẸP KHOẢNG HASSE")
print("="*70)
print(f'Khoảng Hasse gốc: [-2√p, 2√p] = {hasse_original:,.0f}')
print(f'Khoảng AI: [-2δ, 2δ] = {hasse_ai:,.0f}')
print(f'Thu hẹp: {reduction:.2f}%')
print("="*70)

with open('./output_trace_nn.txt', 'w') as file:
	file.write('|diff|    p a b trace_true trace_pred\n')
	for i in range(n_test):
		file.write(f'\n  {int(n_epsilon[i])}   {int(X_test[i][0])} '
				  f'{int(X_test[i][1])} {int(X_test[i][2])} '
				  f'{int(original_traces_test[i])} {int(traces_pred[i])}')

