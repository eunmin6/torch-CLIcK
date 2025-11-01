"""
베이스라인 및 Fine-tuned 모델 결과 시각화 스크립트

이 스크립트는:
1. 베이스라인 및 Fine-tuned 모델의 메트릭 로드
2. 비교 그래프 생성 (카테고리별, 출처별 정확도)
3. 개선도 시각화
4. 결과를 PNG 파일로 저장
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 경로 설정
BASELINE_METRICS_PATH = "./benchmarks/baseline/overall_metrics.json"
FINETUNED_METRICS_PATH = "./benchmarks/finetuned/overall_metrics.json"
OUTPUT_DIR = "./benchmarks/visualizations"

def load_metrics(metrics_path):
    """메트릭 파일 로드"""
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_overall_comparison(baseline_metrics, finetuned_metrics=None):
    """전체 정확도 비교 바 차트"""
    print("[INFO] Creating overall accuracy comparison...")

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Baseline']
    accuracies = [baseline_metrics['overall']['accuracy']]

    if finetuned_metrics:
        models.append('Fine-tuned')
        accuracies.append(finetuned_metrics['overall']['accuracy'])

    colors = ['#3498db', '#2ecc71']
    bars = ax.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, edgecolor='black')

    # 값 레이블 추가
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('전체 정확도 비교', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "overall_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def plot_category_comparison(baseline_metrics, finetuned_metrics=None):
    """카테고리별 정확도 비교"""
    print("[INFO] Creating category comparison...")

    categories = list(baseline_metrics['by_category'].keys())
    baseline_accs = [baseline_metrics['by_category'][cat]['accuracy'] for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2 if finetuned_metrics else x, baseline_accs,
                   width, label='Baseline', color='#3498db', alpha=0.8, edgecolor='black')

    if finetuned_metrics:
        finetuned_accs = [finetuned_metrics['by_category'][cat]['accuracy'] for cat in categories]
        bars2 = ax.bar(x + width/2, finetuned_accs,
                       width, label='Fine-tuned', color='#2ecc71', alpha=0.8, edgecolor='black')

        # 개선도 표시
        for i, (b_acc, f_acc) in enumerate(zip(baseline_accs, finetuned_accs)):
            improvement = f_acc - b_acc
            if improvement > 0:
                ax.text(i, max(b_acc, f_acc) + 2,
                        f'+{improvement:.1f}%',
                        ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')

    # 값 레이블
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    if finetuned_metrics:
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('카테고리별 정확도 비교', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "category_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def plot_subcategory_comparison(baseline_metrics, finetuned_metrics=None):
    """서브카테고리별 정확도 비교 (Top 10)"""
    print("[INFO] Creating subcategory comparison...")

    # 샘플 수로 정렬하여 상위 10개 추출
    sorted_subcats = sorted(baseline_metrics['by_subcategory'].items(),
                           key=lambda x: x[1]['total'], reverse=True)[:10]

    subcats = [s[0] for s in sorted_subcats]
    baseline_accs = [s[1]['accuracy'] for s in sorted_subcats]

    fig, ax = plt.subplots(figsize=(14, 8))

    y = np.arange(len(subcats))
    height = 0.35

    bars1 = ax.barh(y + height/2 if finetuned_metrics else y, baseline_accs,
                    height, label='Baseline', color='#3498db', alpha=0.8, edgecolor='black')

    if finetuned_metrics:
        finetuned_accs = [finetuned_metrics['by_subcategory'][cat]['accuracy'] for cat in subcats]
        bars2 = ax.barh(y - height/2, finetuned_accs,
                        height, label='Fine-tuned', color='#2ecc71', alpha=0.8, edgecolor='black')

    # 값 레이블
    for bar in bars1:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontsize=9)

    if finetuned_metrics:
        for bar in bars2:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%',
                    ha='left', va='center', fontsize=9)

    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('Subcategory', fontsize=12)
    ax.set_title('서브카테고리별 정확도 비교 (Top 10)', fontsize=16, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(subcats)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "subcategory_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def plot_source_comparison(baseline_metrics, finetuned_metrics=None):
    """출처별 정확도 비교"""
    print("[INFO] Creating source comparison...")

    sources = list(baseline_metrics['by_source'].keys())
    baseline_accs = [baseline_metrics['by_source'][src]['accuracy'] for src in sources]

    # 정확도순으로 정렬
    sorted_data = sorted(zip(sources, baseline_accs), key=lambda x: x[1], reverse=True)
    sources = [s[0] for s in sorted_data]
    baseline_accs = [s[1] for s in sorted_data]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sources))
    width = 0.35

    bars1 = ax.bar(x - width/2 if finetuned_metrics else x, baseline_accs,
                   width, label='Baseline', color='#3498db', alpha=0.8, edgecolor='black')

    if finetuned_metrics:
        finetuned_accs = [finetuned_metrics['by_source'][src]['accuracy'] for src in sources]
        bars2 = ax.bar(x + width/2, finetuned_accs,
                       width, label='Fine-tuned', color='#2ecc71', alpha=0.8, edgecolor='black')

    # 값 레이블
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    if finetuned_metrics:
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Source', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('출처별 정확도 비교', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "source_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def plot_improvement_heatmap(baseline_metrics, finetuned_metrics):
    """카테고리/서브카테고리별 개선도 히트맵"""
    if not finetuned_metrics:
        print("[INFO] Skipping improvement heatmap (no fine-tuned metrics)")
        return

    print("[INFO] Creating improvement heatmap...")

    # Top 10 서브카테고리
    sorted_subcats = sorted(baseline_metrics['by_subcategory'].items(),
                           key=lambda x: x[1]['total'], reverse=True)[:10]

    subcats = [s[0] for s in sorted_subcats]
    improvements = []

    for subcat in subcats:
        baseline_acc = baseline_metrics['by_subcategory'][subcat]['accuracy']
        finetuned_acc = finetuned_metrics['by_subcategory'][subcat]['accuracy']
        improvement = finetuned_acc - baseline_acc
        improvements.append(improvement)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['red' if imp < 0 else 'green' for imp in improvements]
    y_pos = np.arange(len(subcats))

    bars = ax.barh(y_pos, improvements, color=colors, alpha=0.6, edgecolor='black')

    # 값 레이블
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{imp:+.1f}%',
                ha='left' if width > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subcats)
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_title('서브카테고리별 개선도 (Fine-tuned vs Baseline)', fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "improvement_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")

def main():
    print("="*80)
    print("[START] Results Visualization")
    print("="*80 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 베이스라인 메트릭 로드
    print("[INFO] Loading baseline metrics...")
    baseline_metrics = load_metrics(BASELINE_METRICS_PATH)

    if not baseline_metrics:
        print(f"[ERROR] Baseline metrics not found: {BASELINE_METRICS_PATH}")
        print("[INFO] Please run baseline evaluation first.")
        return

    print("[OK] Baseline metrics loaded")

    # 2. Fine-tuned 메트릭 로드 (선택적)
    print("[INFO] Loading fine-tuned metrics...")
    finetuned_metrics = load_metrics(FINETUNED_METRICS_PATH)

    if finetuned_metrics:
        print("[OK] Fine-tuned metrics loaded")
    else:
        print("[INFO] Fine-tuned metrics not found. Will visualize baseline only.")

    # 3. 시각화 생성
    plot_overall_comparison(baseline_metrics, finetuned_metrics)
    plot_category_comparison(baseline_metrics, finetuned_metrics)
    plot_subcategory_comparison(baseline_metrics, finetuned_metrics)
    plot_source_comparison(baseline_metrics, finetuned_metrics)

    if finetuned_metrics:
        plot_improvement_heatmap(baseline_metrics, finetuned_metrics)

    print("\n" + "="*80)
    print("[DONE] Visualization complete!")
    print("="*80)
    print(f"\n시각화 결과: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
