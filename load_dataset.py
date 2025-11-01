"""
CLIcK 데이터셋 로드 및 분석 스크립트

이 스크립트는:
1. HuggingFace Hub에서 모든 JSON 파일 다운로드
2. 데이터 통합 및 품질 검증
3. 통계 분석 수행
"""

from huggingface_hub import HfApi, hf_hub_download
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# 데이터셋 정보
REPO_ID = "EunsuKim/CLIcK"
CACHE_DIR = "./data/raw"

def get_all_json_files():
    """HuggingFace 레포지토리에서 모든 JSON 파일 목록 가져오기"""
    print("[INFO] Fetching file list from HuggingFace...")
    api = HfApi()
    dataset_info = api.dataset_info(REPO_ID)

    json_files = []
    for sibling in dataset_info.siblings:
        if sibling.rfilename.endswith('.json'):
            json_files.append(sibling.rfilename)

    print(f"[OK] Found {len(json_files)} JSON files")
    return json_files

def download_and_load_file(file_path):
    """개별 JSON 파일 다운로드 및 로드"""
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset"
        )

        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data, None
    except Exception as e:
        return None, str(e)

def extract_metadata(file_path):
    """파일 경로에서 메타데이터 추출"""
    # Dataset/Culture/Korean History/History_KHB.json -> Culture, Korean History, History, KHB
    parts = file_path.split('/')

    if len(parts) >= 4:
        category = parts[1]  # Culture or Language
        subcategory = parts[2]  # Korean History, Grammar, etc.
        filename = parts[3]  # History_KHB.json

        # 파일명에서 주제와 출처 추출
        name_parts = filename.replace('.json', '').split('_')
        if len(name_parts) == 2:
            topic, source = name_parts
        else:
            topic = name_parts[0] if name_parts else "Unknown"
            source = "Unknown"

        return {
            'category': category,
            'subcategory': subcategory,
            'topic': topic,
            'source': source,
            'file_path': file_path
        }
    else:
        return {
            'category': 'Unknown',
            'subcategory': 'Unknown',
            'topic': 'Unknown',
            'source': 'Unknown',
            'file_path': file_path
        }

def validate_sample(sample, idx, file_path):
    """개별 샘플 검증"""
    errors = []
    warnings = []

    # 필수 필드 확인
    required_fields = ['id', 'question', 'choices', 'answer']
    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field}")

    # 'paragraph'는 선택적이므로 경고만
    if 'paragraph' not in sample:
        warnings.append("Missing optional field: paragraph")

    # choices가 리스트인지 확인
    if 'choices' in sample and not isinstance(sample['choices'], list):
        errors.append("'choices' must be a list")

    # choices가 비어있지 않은지 확인
    if 'choices' in sample and len(sample['choices']) == 0:
        errors.append("'choices' list is empty")

    # answer가 choices에 포함되는지 확인
    if 'choices' in sample and 'answer' in sample:
        if isinstance(sample['choices'], list) and sample['answer'] not in sample['choices']:
            errors.append(f"Answer '{sample['answer']}' not found in choices")

    # 빈 문자열 확인
    if 'question' in sample and not sample['question'].strip():
        errors.append("Question is empty")

    if 'answer' in sample and not sample['answer'].strip():
        errors.append("Answer is empty")

    return errors, warnings

def load_all_data():
    """모든 데이터 로드 및 통합"""
    print("\n" + "="*80)
    print("[START] CLIcK Dataset Loading")
    print("="*80 + "\n")

    json_files = get_all_json_files()

    all_samples = []
    file_stats = []
    validation_errors = []
    validation_warnings = []

    print("\n[LOAD] Loading files...")
    for i, file_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Loading: {file_path}")

        data, error = download_and_load_file(file_path)

        if error:
            print(f"  [ERROR] Error: {error}")
            validation_errors.append({
                'file': file_path,
                'error': error,
                'type': 'file_load_error'
            })
            continue

        metadata = extract_metadata(file_path)
        num_samples = len(data)
        print(f"  [OK] Loaded {num_samples} samples")

        # 각 샘플에 메타데이터 추가 및 검증
        for idx, sample in enumerate(data):
            # 메타데이터 추가
            sample.update(metadata)

            # 검증
            errors, warnings = validate_sample(sample, idx, file_path)

            if errors:
                validation_errors.append({
                    'file': file_path,
                    'sample_id': sample.get('id', f'index_{idx}'),
                    'errors': errors
                })

            if warnings:
                validation_warnings.append({
                    'file': file_path,
                    'sample_id': sample.get('id', f'index_{idx}'),
                    'warnings': warnings
                })

            all_samples.append(sample)

        file_stats.append({
            'file_path': file_path,
            'category': metadata['category'],
            'subcategory': metadata['subcategory'],
            'topic': metadata['topic'],
            'source': metadata['source'],
            'num_samples': num_samples
        })

    print("\n" + "="*80)
    print("[STATS] Dataset Statistics")
    print("="*80)

    print(f"\n총 샘플 수: {len(all_samples)}")
    print(f"총 파일 수: {len(json_files)}")

    # 카테고리별 통계
    print("\n--- 카테고리별 분포 ---")
    category_counts = defaultdict(int)
    for sample in all_samples:
        category_counts[sample['category']] += 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} samples")

    # 서브카테고리별 통계
    print("\n--- 서브카테고리별 분포 ---")
    subcategory_counts = defaultdict(int)
    for sample in all_samples:
        subcategory_counts[sample['subcategory']] += 1

    for subcategory, count in sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {subcategory}: {count} samples")

    # 출처별 통계
    print("\n--- 출처별 분포 ---")
    source_counts = defaultdict(int)
    for sample in all_samples:
        source_counts[sample['source']] += 1

    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} samples")

    # 선택지 개수 분포
    print("\n--- 선택지 개수 분포 ---")
    choice_counts = defaultdict(int)
    for sample in all_samples:
        if 'choices' in sample and isinstance(sample['choices'], list):
            choice_counts[len(sample['choices'])] += 1

    for num_choices, count in sorted(choice_counts.items()):
        print(f"  {num_choices}개 선택지: {count} samples")

    # paragraph 존재 여부
    print("\n--- Paragraph 존재 여부 ---")
    with_paragraph = sum(1 for s in all_samples if s.get('paragraph', '').strip())
    without_paragraph = len(all_samples) - with_paragraph
    print(f"  Paragraph 있음: {with_paragraph} samples ({with_paragraph/len(all_samples)*100:.1f}%)")
    print(f"  Paragraph 없음: {without_paragraph} samples ({without_paragraph/len(all_samples)*100:.1f}%)")

    # 검증 결과
    print("\n" + "="*80)
    print("[VALIDATE] Validation Results")
    print("="*80)

    if validation_errors:
        print(f"\n[ERROR] Found {len(validation_errors)} validation errors:")
        for i, error_info in enumerate(validation_errors[:10], 1):  # 처음 10개만 표시
            print(f"\n  Error {i}:")
            print(f"    File: {error_info.get('file', 'N/A')}")
            if 'sample_id' in error_info:
                print(f"    Sample ID: {error_info['sample_id']}")
                print(f"    Errors: {', '.join(error_info['errors'])}")
            else:
                print(f"    Error: {error_info.get('error', 'N/A')}")

        if len(validation_errors) > 10:
            print(f"\n  ... and {len(validation_errors) - 10} more errors")
    else:
        print("\n[OK] No validation errors found!")

    if validation_warnings:
        print(f"\n[WARN] Found {len(validation_warnings)} validation warnings:")
        # warnings 요약만 표시
        warning_types = defaultdict(int)
        for warning_info in validation_warnings:
            for warning in warning_info['warnings']:
                warning_types[warning] += 1

        for warning_type, count in warning_types.items():
            print(f"    {warning_type}: {count} occurrences")
    else:
        print("\n[OK] No validation warnings found!")

    # DataFrame으로 파일 통계 저장
    print("\n" + "="*80)
    print("[SAVE] Saving statistics...")
    print("="*80)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # 파일별 통계 저장
    df_files = pd.DataFrame(file_stats)
    stats_path = os.path.join(CACHE_DIR, 'file_statistics.csv')
    df_files.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] File statistics saved to: {stats_path}")

    # 전체 데이터 저장
    data_path = os.path.join(CACHE_DIR, 'all_data.json')
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"[OK] All data saved to: {data_path}")

    # 검증 에러 저장
    if validation_errors:
        errors_path = os.path.join(CACHE_DIR, 'validation_errors.json')
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(validation_errors, f, ensure_ascii=False, indent=2)
        print(f"[OK] Validation errors saved to: {errors_path}")

    print("\n" + "="*80)
    print("[DONE] Dataset loading complete!")
    print("="*80 + "\n")

    return all_samples, file_stats, validation_errors, validation_warnings

if __name__ == "__main__":
    all_samples, file_stats, errors, warnings = load_all_data()

    print(f"\n[SUMMARY]")
    print(f"   Total samples: {len(all_samples)}")
    print(f"   Total files: {len(file_stats)}")
    print(f"   Validation errors: {len(errors)}")
    print(f"   Validation warnings: {len(warnings)}")
    print(f"\n   Data saved to: {CACHE_DIR}/")
