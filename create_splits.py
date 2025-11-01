"""
CLIcK 데이터셋 Train/Val/Test Split 생성

Stratified sampling을 사용하여:
- Train: 70%
- Validation: 15%
- Test: 15%

카테고리와 서브카테고리 비율을 유지하면서 분할
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 경로 설정
RAW_DATA_PATH = "./data/raw/all_data.json"
SPLIT_DIR = "./data/splits"

# Split 비율
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

def load_data():
    """저장된 데이터 로드"""
    print("[INFO] Loading data from:", RAW_DATA_PATH)
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] Loaded {len(data)} samples")
    return data

def create_stratified_split(data):
    """Stratified sampling으로 데이터 분할"""
    print("\n[INFO] Creating stratified splits...")

    # stratify를 위한 레이블 생성 (category + subcategory 조합)
    stratify_labels = []
    for sample in data:
        # category와 subcategory를 조합한 레이블
        label = f"{sample['category']}_{sample['subcategory']}"
        stratify_labels.append(label)

    # 레이블 분포 확인
    label_counts = defaultdict(int)
    for label in stratify_labels:
        label_counts[label] += 1

    print(f"\n[INFO] Found {len(label_counts)} unique category-subcategory combinations")

    # 샘플이 너무 적은 그룹 확인 (최소 3개 필요)
    small_groups = {label: count for label, count in label_counts.items() if count < 3}
    if small_groups:
        print(f"\n[WARN] Some groups have very few samples (< 3):")
        for label, count in sorted(small_groups.items(), key=lambda x: x[1]):
            print(f"  {label}: {count} samples")
        print("\n[INFO] These will be handled by merging to parent category")

        # 작은 그룹은 상위 카테고리로 병합
        stratify_labels_adjusted = []
        for i, label in enumerate(stratify_labels):
            if label in small_groups:
                # subcategory 대신 category만 사용
                stratify_labels_adjusted.append(data[i]['category'])
            else:
                stratify_labels_adjusted.append(label)
        stratify_labels = stratify_labels_adjusted

    # 첫 번째 split: train + (val+test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data,
        stratify_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=stratify_labels
    )

    print(f"[OK] Train split: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")

    # 두 번째 split: val과 test
    # val과 test의 비율을 맞추기 위해 계산
    val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data,
        temp_labels,
        test_size=(1 - val_ratio_of_temp),
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    print(f"[OK] Validation split: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"[OK] Test split: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")

    return train_data, val_data, test_data

def analyze_split_distribution(train_data, val_data, test_data):
    """각 split의 분포 분석"""
    print("\n" + "="*80)
    print("[ANALYSIS] Split Distribution Analysis")
    print("="*80)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    # 카테고리별 분포
    print("\n--- Category Distribution ---")
    print(f"{'Category':<20} {'Train':<15} {'Val':<15} {'Test':<15}")
    print("-" * 65)

    all_categories = set()
    for split_data in splits.values():
        for sample in split_data:
            all_categories.add(sample['category'])

    for category in sorted(all_categories):
        counts = []
        for split_name, split_data in splits.items():
            count = sum(1 for s in split_data if s['category'] == category)
            percentage = count / len(split_data) * 100 if len(split_data) > 0 else 0
            counts.append(f"{count:>5} ({percentage:>4.1f}%)")

        print(f"{category:<20} {counts[0]:<15} {counts[1]:<15} {counts[2]:<15}")

    # 서브카테고리별 분포 (상위 10개만)
    print("\n--- Top 10 Subcategory Distribution ---")
    print(f"{'Subcategory':<25} {'Train':<15} {'Val':<15} {'Test':<15}")
    print("-" * 70)

    # 전체에서 가장 많은 서브카테고리 10개 추출
    all_subcategories = defaultdict(int)
    for split_data in splits.values():
        for sample in split_data:
            all_subcategories[sample['subcategory']] += 1

    top_subcategories = sorted(all_subcategories.items(), key=lambda x: x[1], reverse=True)[:10]

    for subcategory, _ in top_subcategories:
        counts = []
        for split_name, split_data in splits.items():
            count = sum(1 for s in split_data if s['subcategory'] == subcategory)
            counts.append(f"{count:>5}")

        print(f"{subcategory:<25} {counts[0]:<15} {counts[1]:<15} {counts[2]:<15}")

    # 출처별 분포
    print("\n--- Source Distribution ---")
    print(f"{'Source':<15} {'Train':<15} {'Val':<15} {'Test':<15}")
    print("-" * 60)

    all_sources = set()
    for split_data in splits.values():
        for sample in split_data:
            all_sources.add(sample['source'])

    for source in sorted(all_sources):
        counts = []
        for split_name, split_data in splits.items():
            count = sum(1 for s in split_data if s['source'] == source)
            percentage = count / len(split_data) * 100 if len(split_data) > 0 else 0
            counts.append(f"{count:>5} ({percentage:>4.1f}%)")

        print(f"{source:<15} {counts[0]:<15} {counts[1]:<15} {counts[2]:<15}")

def save_splits(train_data, val_data, test_data):
    """Split 데이터 저장"""
    print("\n" + "="*80)
    print("[SAVE] Saving splits to disk...")
    print("="*80)

    os.makedirs(SPLIT_DIR, exist_ok=True)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        # JSON 형식으로 저장
        json_path = os.path.join(SPLIT_DIR, f"{split_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] {split_name}.json saved: {len(split_data)} samples")

    # Split 정보 요약 저장
    summary = {
        'total_samples': len(train_data) + len(val_data) + len(test_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'train_ratio': len(train_data) / (len(train_data) + len(val_data) + len(test_data)),
        'val_ratio': len(val_data) / (len(train_data) + len(val_data) + len(test_data)),
        'test_ratio': len(test_data) / (len(train_data) + len(val_data) + len(test_data)),
        'random_seed': RANDOM_SEED
    }

    summary_path = os.path.join(SPLIT_DIR, "split_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] split_summary.json saved")

    print(f"\n[INFO] All splits saved to: {SPLIT_DIR}/")

    return summary

def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("[START] CLIcK Dataset Split Creation")
    print("="*80 + "\n")

    # 데이터 로드
    data = load_data()

    # Stratified split 생성
    train_data, val_data, test_data = create_stratified_split(data)

    # 분포 분석
    analyze_split_distribution(train_data, val_data, test_data)

    # 저장
    summary = save_splits(train_data, val_data, test_data)

    # 최종 요약
    print("\n" + "="*80)
    print("[DONE] Split Creation Complete!")
    print("="*80)

    print(f"\n[SUMMARY]")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Train: {summary['train_samples']} ({summary['train_ratio']*100:.1f}%)")
    print(f"  Val: {summary['val_samples']} ({summary['val_ratio']*100:.1f}%)")
    print(f"  Test: {summary['test_samples']} ({summary['test_ratio']*100:.1f}%)")
    print(f"  Random seed: {summary['random_seed']}")
    print(f"\n  Splits saved to: {SPLIT_DIR}/")

if __name__ == "__main__":
    main()
