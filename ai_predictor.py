import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl
from pathlib import Path

BIT_SIZE = 32

def read_raw_data(filename=None):
	if filename is None:
		filename = f'input{BIT_SIZE}_test.txt'
	with open(filename) as file:
		lines = file.readlines()
		return lines

def get_max_a_b_from_data(filename=None):
	"""Tính max_a và max_b từ training data để normalize đúng"""
	raw_data = read_raw_data(filename)
	a_list, b_list = [], []
	for line in raw_data:
		parts = list(map(int, line.split()))
		if len(parts) >= 4:
			a_list.append(parts[1])
			b_list.append(parts[2])
	max_a = max(a_list) if a_list else 2**BIT_SIZE
	max_b = max(b_list) if b_list else 2**BIT_SIZE
	return max_a, max_b

def proccess_raw_data(raw_data, max_a=None, max_b=None):
	"""Xử lý dữ liệu giống nn88_trace.py"""
	data = []
	traces = []
	original_orders = []
	
	# Tính max_a, max_b nếu chưa có
	if max_a is None or max_b is None:
		a_list, b_list = [], []
		for line in raw_data:
			parts = list(map(int, line.split()))
			if len(parts) >= 4:
				a_list.append(parts[1])
				b_list.append(parts[2])
		max_a = max(a_list) if a_list else 2**BIT_SIZE
		max_b = max(b_list) if b_list else 2**BIT_SIZE
	
	for line in raw_data:
		parts = list(map(int, line.split()))
		if len(parts) >= 4:
			p, a, b, order = parts[:4]
			trace = p + 1 - order
			sqrt_p = np.sqrt(float(p))
			
			# Normalize giống nn88_trace.py
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
	X = data[:, :-1]  # p_norm, a_norm, b_norm
	Y = data[:, -1]   # trace_norm
	n = len(data)
	return X, Y, n

class TracePredictor:
    def __init__(self, weights_path='018new_weights.hdf5', bit_size=32, max_a=None, max_b=None):
        self.bit_size = bit_size
        # Tính max_a, max_b từ training data nếu chưa có
        if max_a is None or max_b is None:
            try:
                max_a, max_b = get_max_a_b_from_data()
            except:
                max_a = 2**bit_size
                max_b = 2**bit_size
        self.max_a = float(max_a)
        self.max_b = float(max_b)
        
        self.model = self._build_model()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0  
            ), 
            loss='mse',
            metrics=['mae']
        )
        
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
        
    def _build_model(self):
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
    
    def predict_trace(self, p, a, b, n_samples=16):
        # Convert Sage Integer/any type to Python float
        p_float = float(p)
        a_float = float(a)
        b_float = float(b)
        
        # Normalize giống training (nn88_trace.py)
        p_norm = np.log2(p_float) / self.bit_size
        a_norm = a_float / self.max_a if self.max_a != 0 else 0.0
        b_norm = b_float / self.max_b if self.max_b != 0 else 0.0
        
        # Clip để đảm bảo trong [0, 1]
        p_norm = np.clip(p_norm, 0.0, 1.0)
        a_norm = np.clip(a_norm, 0.0, 1.0)
        b_norm = np.clip(b_norm, 0.0, 1.0)
        
        X = np.array([[p_norm, a_norm, b_norm]] * n_samples, dtype=np.float32)
        predictions = np.reshape(self.model.predict(X, verbose=0), n_samples)
        
        trace_norm = np.mean(predictions)
        
        # Denormalize giống nn88_trace.py: trace = (pred_norm - 0.5) * 4 * sqrt(p)
        sqrt_p = math.sqrt(p_float)
        trace_pred = (trace_norm - 0.5) * 4.0 * sqrt_p
        
        hasse_half = 2.0 * sqrt_p
        trace_pred = float(np.clip(trace_pred, -hasse_half, hasse_half))
        delta = hasse_half * 0.85

        interval = (int(np.clip(trace_pred - delta, -hasse_half, hasse_half)), 
                   int(np.clip(trace_pred + delta, -hasse_half, hasse_half)))

        return int(round(trace_pred)), interval

    def predict_trace_list(self, test_file=None) :
        raw_data = read_raw_data(test_file if test_file else f'input{BIT_SIZE}_test.txt')
        data, original_orders, original_traces = proccess_raw_data(raw_data, self.max_a, self.max_b)
        X, Y, n = generate_X_Y_sets(data)
        
        predictions_norm = np.reshape(self.model.predict(X, verbose=0), n)
        
        p_original = np.power(2.0, X[:, 0] * self.bit_size)
        sqrt_p = np.sqrt(p_original)
        
        trace_pred = (predictions_norm - 0.5) * 4.0 * sqrt_p
        
        hasse_half = 2.0 * sqrt_p
        delta = hasse_half * 0.85
        
        in_interval = (original_traces >= trace_pred - delta) & (original_traces <= trace_pred + delta)
        
        return predictions_norm

if __name__ == "__main__":
    ai_model = TracePredictor(bit_size=BIT_SIZE)
    ai_model.predict_trace_list()