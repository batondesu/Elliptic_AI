#!/usr/bin/env python3
"""
Tạo bảng ví dụ so sánh kết quả AI vs Classical
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sympy import legendre_symbol
from ai_enhanced_schoof_v2 import (
    DeltaRegressorV2, CMClassifierV2, SchoofFeatureExtractor, AISchoofAssistantV2
)

def classical_count_points(A: int, B: int, p: int) -> int:
    """Đếm điểm theo phương pháp cổ điển"""
    c = 1
    for x in range(p):
        r = (x**3 + A*x + B) % p
        if r == 0:
            c += 1
        elif legendre_symbol(r, p) == 1:
            c += 2
    return c

def calculate_delta_classical(A: int, B: int, p: int) -> float:
    """Tính δ theo phương pháp cổ điển"""
    N = classical_count_points(A, B, p)
    return float(p + 1 - N)

def create_comparison_visualization(df: pd.DataFrame):
    """Tạo bảng so sánh dưới dạng hình ảnh"""
    print("Đang tạo bảng so sánh dưới dạng hình ảnh...")
    
    # Tạo figure với subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('AI-Enhanced Schoof v2.0: Comparison Results', fontsize=20, fontweight='bold')
    
    # 1. Bảng so sánh chính
    ax1.axis('tight')
    ax1.axis('off')
    
    # Chuẩn bị dữ liệu cho bảng
    table_data = []
    headers = ['Category', 'p', 'AI_δ', 'Classical_δ', 'Error', 'Speedup', 'Hasse_Reduction']
    
    for _, row in df.iterrows():
        table_data.append([
            row['Category'],
            str(row['p']),
            f"{row['AI_δ']:.3f}",
            f"{row['Classical_δ']:.1f}",
            f"{row['Absolute_Error']:.1f}",
            f"{row['Speedup']:.1f}x" if row['Speedup'] != 'N/A' else 'N/A',
            f"{row['Reduction_Factor']:.1f}x"
        ])
    
    # Tạo table
    table = ax1.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Style table
    for i in range(len(table_data) + 1):
        for j in range(len(headers)):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#2E86AB')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F7F7F7')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
    
    ax1.set_title('Detailed Comparison Table', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Biểu đồ Error theo Category
    categories = df['Category'].unique()
    avg_errors = [df[df['Category'] == cat]['Absolute_Error'].mean() for cat in categories]
    
    bars = ax2.bar(categories, avg_errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_xlabel('Prime Category', fontsize=12)
    ax2.set_ylabel('Average Absolute Error', fontsize=12)
    ax2.set_title('Average Error by Prime Category', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Thêm giá trị trên bars
    for bar, error in zip(bars, avg_errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{error:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Biểu đồ Speedup theo Category
    valid_speedups = []
    valid_categories = []
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        speedups = [x for x in cat_df['Speedup'] if x != 'N/A']
        if speedups:
            valid_speedups.append(np.mean(speedups))
            valid_categories.append(cat)
    
    if valid_speedups:
        bars = ax3.bar(valid_categories, valid_speedups, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_xlabel('Prime Category', fontsize=12)
        ax3.set_ylabel('Average Speedup Factor', fontsize=12)
        ax3.set_title('Average Speedup by Prime Category', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Thêm giá trị trên bars
        for bar, speedup in zip(bars, valid_speedups):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 4. Biểu đồ Hasse Reduction
    avg_reductions = [df[df['Category'] == cat]['Reduction_Factor'].mean() for cat in categories]
    
    bars = ax4.bar(categories, avg_reductions, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_xlabel('Prime Category', fontsize=12)
    ax4.set_ylabel('Average Hasse Reduction Factor', fontsize=12)
    ax4.set_title('Average Hasse Interval Narrowing', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Thêm giá trị trên bars
    for bar, reduction in zip(bars, avg_reductions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{reduction:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('image/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Đã lưu comparison_results.png")

def create_summary_visualization(df: pd.DataFrame):
    """Tạo bảng tóm tắt dạng hình ảnh"""
    print("Đang tạo bảng tóm tắt...")
    
    # Tính toán thống kê tổng hợp
    valid_speedups = [x for x in df['Speedup'] if x != 'N/A']
    valid_errors = [x for x in df['Error_%'] if x != 'N/A']
    
    summary_stats = {
        'Tổng số test cases': len(df),
        'Trung bình Absolute Error': round(df['Absolute_Error'].mean(), 4),
        'Trung bình Error %': round(np.mean(valid_errors), 2) if valid_errors else 'N/A',
        'Trung bình Speedup': round(np.mean(valid_speedups), 1) if valid_speedups else 'N/A',
        'Trung bình Hasse Reduction': round(df['Reduction_Factor'].mean(), 2),
        'Trung bình CM Probability': round(df['CM_Probability'].mean(), 4),
        'Test case có Error thấp nhất': df.loc[df['Absolute_Error'].idxmin(), 'Category'],
        'Test case có Speedup cao nhất': df.loc[df['Speedup'].idxmax() if df['Speedup'].dtype != 'object' else 0, 'Category'],
        'Test case có Hasse Reduction tốt nhất': df.loc[df['Reduction_Factor'].idxmax(), 'Category']
    }
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo table data
    table_data = []
    for key, value in summary_stats.items():
        table_data.append([key, str(value)])
    
    # Tạo table
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Style table
    for i in range(len(table_data) + 1):
        for j in range(2):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
    
    ax.set_title('AI-Enhanced Schoof v2.0 - Summary Statistics', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('image/comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Đã lưu comparison_summary.png")

def create_comparison_table():
    """Tạo bảng so sánh kết quả"""
    print("ĐANG TẠO BẢNG SO SÁNH KẾT QUẢ")
    print("=" * 60)
    
    # Tải models
    print("Đang tải models...")
    with open('schoof_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    regressor = DeltaRegressorV2(feature_count=len(feature_names))
    regressor.load()
    
    classifier = CMClassifierV2(feature_count=len(feature_names))
    classifier.load()
    
    feature_extractor = SchoofFeatureExtractor(feature_names)
    assistant = AISchoofAssistantV2(regressor, classifier, feature_extractor)
    
    # Test cases đa dạng
    test_cases = [
        # Small primes
        (17, 5, 3, "Small Prime"),
        (23, 7, 11, "Small Prime"),
        (31, 13, 19, "Small Prime"),
        
        # Medium primes
        (101, 23, 45, "Medium Prime"),
        (257, 67, 89, "Medium Prime"),
        (499, 123, 456, "Medium Prime"),
        
        # Large primes
        (1009, 234, 567, "Large Prime"),
        (2003, 456, 789, "Large Prime"),
        (5003, 1234, 2345, "Large Prime"),
        
        # Very large primes
        (10007, 2345, 3456, "Very Large Prime"),
        (20011, 4567, 6789, "Very Large Prime"),
        (50021, 12345, 23456, "Very Large Prime")
    ]
    
    results = []
    
    print("Đang tính toán kết quả...")
    for i, (p, A, B, category) in enumerate(test_cases):
        print(f"  Test {i+1}/{len(test_cases)}: p={p}, A={A}, B={B}")
        
        # AI Prediction
        start_time = time.time()
        sug = assistant.suggest_speedup(p, A, B)
        ai_time = time.time() - start_time
        
        delta_ai = sug['delta_prediction']
        cm_prob_ai = sug['cm_probability']
        hasse_reduction = sug['width_reduction_factor']
        
        # Classical Calculation
        start_time = time.time()
        delta_classical = calculate_delta_classical(A, B, p)
        classical_time = time.time() - start_time
        
        # Tính toán metrics
        error = abs(delta_ai - delta_classical)
        error_percentage = (error / abs(delta_classical)) * 100 if delta_classical != 0 else float('inf')
        speedup = classical_time / ai_time if ai_time > 0 else float('inf')
        
        # Hasse interval info
        hasse_width = sug['hasse_interval'][1] - sug['hasse_interval'][0] + 1
        narrowed_width = sug['narrowed_interval'][1] - sug['narrowed_interval'][0] + 1
        
        results.append({
            'Category': category,
            'p': p,
            'A': A,
            'B': B,
            'AI_δ': round(delta_ai, 4),
            'Classical_δ': round(delta_classical, 4),
            'Absolute_Error': round(error, 4),
            'Error_%': round(error_percentage, 2) if error_percentage != float('inf') else 'N/A',
            'AI_Time(s)': round(ai_time, 6),
            'Classical_Time(s)': round(classical_time, 6),
            'Speedup': round(speedup, 1) if speedup != float('inf') else 'N/A',
            'CM_Probability': round(cm_prob_ai, 4),
            'Hasse_Width': hasse_width,
            'Narrowed_Width': narrowed_width,
            'Reduction_Factor': round(hasse_reduction, 2),
            'Reduction_%': round(100 * (1 - narrowed_width / hasse_width), 1)
        })
    
    # Tạo DataFrame
    df = pd.DataFrame(results)
    
    # In kết quả
    print("\n" + "=" * 100)
    print("BẢNG SO SÁNH KẾT QUẢ AI vs CLASSICAL")
    print("=" * 100)
    
    # Bảng chính
    print("\nBẢNG CHI TIẾT:")
    print(df.to_string(index=False))
    
    # Thống kê tổng hợp
    print("\n" + "=" * 100)
    print("THỐNG KÊ TỔNG HỢP:")
    print("=" * 100)
    
    # Tính trung bình (loại bỏ 'N/A')
    valid_speedups = [x for x in df['Speedup'] if x != 'N/A']
    valid_errors = [x for x in df['Error_%'] if x != 'N/A']
    
    summary_stats = {
        'Tổng số test cases': len(df),
        'Trung bình Absolute Error': round(df['Absolute_Error'].mean(), 4),
        'Trung bình Error %': round(np.mean(valid_errors), 2) if valid_errors else 'N/A',
        'Trung bình Speedup': round(np.mean(valid_speedups), 1) if valid_speedups else 'N/A',
        'Trung bình Hasse Reduction': round(df['Reduction_Factor'].mean(), 2),
        'Trung bình CM Probability': round(df['CM_Probability'].mean(), 4),
        'Test case có Error thấp nhất': df.loc[df['Absolute_Error'].idxmin(), 'Category'],
        'Test case có Speedup cao nhất': df.loc[df['Speedup'].idxmax() if df['Speedup'].dtype != 'object' else 0, 'Category'],
        'Test case có Hasse Reduction tốt nhất': df.loc[df['Reduction_Factor'].idxmax(), 'Category']
    }
    
    for key, value in summary_stats.items():
        print(f"{key}: {value}")
    
    # Phân tích theo category
    print("\n" + "=" * 100)
    print("PHÂN TÍCH THEO LOẠI PRIME:")
    print("=" * 100)
    
    for category in df['Category'].unique():
        cat_df = df[df['Category'] == category]
        print(f"\n{category}:")
        print(f"  Số test cases: {len(cat_df)}")
        print(f"  Trung bình Error: {round(cat_df['Absolute_Error'].mean(), 4)}")
        print(f"  Trung bình Speedup: {round(cat_df['Speedup'].mean(), 1) if cat_df['Speedup'].dtype != 'object' else 'N/A'}")
        print(f"  Trung bình Hasse Reduction: {round(cat_df['Reduction_Factor'].mean(), 2)}")
    
    # Tạo hình ảnh
    create_comparison_visualization(df)
    create_summary_visualization(df)
    
    print(f"\nĐã lưu các file hình ảnh trong thư mục 'image/':")
    print(f"  • comparison_results.png")
    print(f"  • comparison_summary.png")
    
    return df

if __name__ == "__main__":
    create_comparison_table() 