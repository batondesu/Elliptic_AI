#!/usr/bin/env python3
"""
Kiểm tra kết quả AI-enhanced Schoof v2.0 (Enhanced) với 92 features
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ai_enhanced_schoof_v2 import DeltaRegressorV2, CMClassifierV2, SchoofFeatureExtractor, AISchoofAssistantV2, load_schoof_dataset

def check_enhanced_v2():
    """Kiểm tra kết quả model v2.0 enhanced"""
    print("KIỂM TRA AI-ENHANCED SCHOOF v2.0 (ENHANCED)")
    print("=" * 60)
    
    # Tải dataset
    X, y_delta, y_tilde_delta, y_cm, feature_names = load_schoof_dataset()
    print(f"Dataset: {X.shape}, {len(feature_names)} features")
    
    # Tải models
    print("\nĐang tải models v2.0 (Enhanced)...")
    reg = DeltaRegressorV2(feature_count=len(feature_names))
    clf = CMClassifierV2(feature_count=len(feature_names))
    
    try:
        reg.load()
        clf.load()
        print("✅ Models v2.0 (Enhanced) đã tải thành công")
    except Exception as e:
        print(f"❌ Lỗi tải models: {e}")
        return
    
    # Đánh giá performance
    print("\nĐÁNH GIÁ PERFORMANCE:")
    print("-" * 40)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_delta_train, y_delta_test, y_cm_train, y_cm_test = train_test_split(
        X, y_delta, y_cm, test_size=0.2, random_state=42
    )
    
    # Delta regression evaluation
    X_test_s = reg.scaler.transform(X_test)
    y_delta_pred = reg.model.predict(X_test_s, verbose=0).flatten()
    
    delta_mse = mean_squared_error(y_delta_test, y_delta_pred)
    delta_mae = mean_absolute_error(y_delta_test, y_delta_pred)
    delta_r2 = r2_score(y_delta_test, y_delta_pred)
    delta_rmse = np.sqrt(delta_mse)
    
    print(f"Delta Regression v2.0 (Enhanced):")
    print(f"  MSE: {delta_mse:.4f}")
    print(f"  MAE: {delta_mae:.4f}")
    print(f"  RMSE: {delta_rmse:.4f}")
    print(f"  R²: {delta_r2:.4f}")
    
    # CM classification evaluation
    y_cm_pred_proba = clf.model.predict(X_test_s, verbose=0).flatten()
    y_cm_pred = (y_cm_pred_proba > 0.5).astype(int)
    cm_accuracy = np.mean(y_cm_pred == y_cm_test)
    
    print(f"\nCM Classification v2.0 (Enhanced):")
    print(f"  Accuracy: {cm_accuracy:.4f}")
    print(f"  CM curves detected: {np.sum(y_cm_pred)}/{len(y_cm_pred)}")
    
    # So sánh với v2.0 cũ
    print(f"\nSO SÁNH VỚI v2.0 cũ:")
    print(f"  v2.0 cũ Delta R²: -0.0001")
    print(f"  v2.0 Enhanced Delta R²: {delta_r2:.4f}")
    print(f"  Cải thiện: {delta_r2 - (-0.0001):.4f}")
    
    if delta_r2 > -0.0001:
        print("  ✅ Có cải thiện về Delta R²!")
    else:
        print("  ⚠️ Chưa cải thiện về Delta R²")
    
    # Test Hasse narrowing
    print(f"\nTEST HASSE NARROWING:")
    print("-" * 40)
    
    feature_extractor = SchoofFeatureExtractor(feature_names)
    assistant = AISchoofAssistantV2(reg, clf, feature_extractor)
    
    test_cases = [
        (503, 127, 461),
        (857, 281, 733),
        (1003, 341, 881),
        (2011, 523, 1033)
    ]
    
    total_reduction = 0
    for p, A, B in test_cases:
        sug = assistant.suggest_speedup(p, A, B)
        hasse_width = sug["hasse_interval"][1] - sug["hasse_interval"][0] + 1
        narrowed_width = sug["narrowed_interval"][1] - sug["narrowed_interval"][0] + 1
        reduction = (1 - narrowed_width / hasse_width) * 100
        total_reduction += reduction
        
        print(f"p={p}: Giảm {reduction:.1f}% ({hasse_width} → {narrowed_width})")
        print(f"  δ pred: {sug['delta_prediction']:.2f}")
        print(f"  CM prob: {sug['cm_probability']:.4f}")
    
    avg_reduction = total_reduction / len(test_cases)
    print(f"\nTrung bình giảm khoảng: {avg_reduction:.1f}%")
    
    # Kết luận
    print(f"\nKẾT LUẬN:")
    print(f"  Delta R²: {delta_r2:.4f} (v2.0 cũ: -0.0001)")
    print(f"  CM Accuracy: {cm_accuracy:.4f}")
    print(f"  Hasse Reduction: {avg_reduction:.1f}%")
    
    if delta_r2 > 0:
        print("  🎉 Model v2.0 (Enhanced) đã cải thiện Delta R²!")
    elif delta_r2 > -0.0001:
        print("  ✅ Model v2.0 (Enhanced) có cải thiện nhỏ!")
    else:
        print("  ⚠️ Model v2.0 (Enhanced) chưa cải thiện Delta R²")
    
    return {
        'delta_r2': delta_r2,
        'cm_accuracy': cm_accuracy,
        'hasse_reduction': avg_reduction
    }

if __name__ == '__main__':
    check_enhanced_v2() 