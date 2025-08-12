import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_model(input_shape=(3,)):
    """Tạo neural network model cho dự đoán tilde_delta"""
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Chuẩn bị dữ liệu train/validation/test"""
    # Chia train/test trước
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Chia train thành train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler)

def train_model(X, y, epochs=100, batch_size=32, patience=15):
    """Huấn luyện model"""
    # Chuẩn bị dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y)
    
    # Tạo model
    model = create_model()
    print("Kiến trúc model:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Huấn luyện
    print("\nBắt đầu huấn luyện...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Đánh giá
    print("\nĐánh giá model:")
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Train Loss: {train_loss[0]:.6f}, MAE: {train_loss[1]:.6f}")
    print(f"Val Loss: {val_loss[0]:.6f}, MAE: {val_loss[1]:.6f}")
    print(f"Test Loss: {test_loss[0]:.6f}, MAE: {test_loss[1]:.6f}")
    
    # Dự đoán và tính R²
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2:.6f}")
    
    return model, history, (X_train, X_val, X_test, y_train, y_val, y_test, scaler)

def plot_training_history(history):
    """Vẽ biểu đồ quá trình huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE
    ax2.plot(history.history['mae'], label='Train MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(model, X_test, y_test, scaler):
    """Vẽ biểu đồ so sánh dự đoán và thực tế"""
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('So sánh dự đoán vs thực tế')
    plt.grid(True)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = y_test - y_pred.flatten()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    
    # Histogram of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Tần suất')
    plt.title('Phân phối Residuals')
    plt.grid(True)
    
    # Distribution comparison
    plt.subplot(2, 2, 4)
    plt.hist(y_test, bins=30, alpha=0.7, label='Thực tế', edgecolor='black')
    plt.hist(y_pred.flatten(), bins=30, alpha=0.7, label='Dự đoán', edgecolor='black')
    plt.xlabel('tilde_delta')
    plt.ylabel('Tần suất')
    plt.title('So sánh phân phối')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Import dữ liệu
    try:
        X = np.load('elliptic_data_X.npy')
        y = np.load('elliptic_data_y.npy')
        print("Đã tải dữ liệu từ file")
    except:
        from generate_data import X, y
        print("Sinh dữ liệu mới")
    
    # Huấn luyện model
    model, history, data = train_model(X, y, epochs=100)
    
    # Vẽ biểu đồ
    plot_training_history(history)
    plot_predictions(model, data[2], data[5], data[6])
    
    # Lưu model
    model.save('elliptic_model.h5')
    print("Đã lưu model vào elliptic_model.h5")
