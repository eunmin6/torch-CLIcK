"""
베이스라인 평가 결과 에러 분석 스크립트

이 스크립트는:
1. predictions.jsonl 파일 로드
2. 오답 패턴 분석
3. 카테고리별 약점 파악
4. 개선 방향 도출
5. 분석 결과를 error_analysis.md로 저장
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# 경로 설정
PREDICTIONS_PATH = "./benchmarks/baseline/predictions.jsonl"
METRICS_PATH = "./benchmarks/baseline/overall_metrics.json"
OUTPUT_PATH = "./benchmarks/baseline/error_analysis.md"

def load_predictions():
    """예측 결과 로드"""
    print(f"[INFO] Loading predictions from: {PREDICTIONS_PATH}")
    predictions = []
    with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    print(f"[OK] Loaded {len(predictions)} predictions")
    return predictions

def load_metrics():
    """메트릭 로드"""
    print(f"[INFO] Loading metrics from: {METRICS_PATH}")
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    print(f"[OK] Metrics loaded")
    return metrics

def analyze_errors(predictions):
    """오답 패턴 분석"""
    print("\n[INFO] Analyzing error patterns...")

    # 정답/오답 분리
    correct_preds = [p for p in predictions if p.get('is_correct', False)]
    incorrect_preds = [p for p in predictions if not p.get('is_correct', False)]

    print(f"  Correct: {len(correct_preds)}")
    print(f"  Incorrect: {len(incorrect_preds)}")

    # 에러가 없는 경우
    if len(incorrect_preds) == 0:
        return {
            'total_errors': 0,
            'error_by_category': {},
            'error_by_subcategory': {},
            'error_by_source': {},
            'common_patterns': []
        }

    # 카테고리별 오답 분석
    error_by_category = defaultdict(list)
    for pred in incorrect_preds:
        if 'category' in pred:
            error_by_category[pred['category']].append(pred)

    # 서브카테고리별 오답 분석
    error_by_subcategory = defaultdict(list)
    for pred in incorrect_preds:
        if 'subcategory' in pred:
            error_by_subcategory[pred['subcategory']].append(pred)

    # 출처별 오답 분석
    error_by_source = defaultdict(list)
    for pred in incorrect_preds:
        if 'source' in pred:
            error_by_source[pred['source']].append(pred)

    # 답변이 None인 경우 (추출 실패)
    no_answer_errors = [p for p in incorrect_preds if p.get('predicted_answer') is None]

    # 공통 오답 패턴 찾기
    common_patterns = []

    # 패턴 1: 답변 추출 실패
    if len(no_answer_errors) > 0:
        common_patterns.append({
            'pattern': '답변 추출 실패',
            'count': len(no_answer_errors),
            'percentage': len(no_answer_errors) / len(incorrect_preds) * 100,
            'description': '모델이 생성한 답변에서 선택지를 추출하지 못함'
        })

    # 패턴 2: 선택지 개수별 오답률
    errors_by_choice_count = defaultdict(int)
    for pred in incorrect_preds:
        if 'choices' in pred:
            errors_by_choice_count[len(pred['choices'])] += 1

    return {
        'total_errors': len(incorrect_preds),
        'error_by_category': {k: len(v) for k, v in error_by_category.items()},
        'error_by_subcategory': {k: len(v) for k, v in error_by_subcategory.items()},
        'error_by_source': {k: len(v) for k, v in error_by_source.items()},
        'no_answer_count': len(no_answer_errors),
        'common_patterns': common_patterns,
        'sample_errors': incorrect_preds[:10]  # 처음 10개 오답 샘플
    }

def find_weak_areas(metrics):
    """약점 영역 파악"""
    print("\n[INFO] Identifying weak areas...")

    weak_areas = {
        'categories': [],
        'subcategories': [],
        'sources': []
    }

    # 카테고리별 약점 (정확도 낮은 순)
    if 'by_category' in metrics:
        sorted_categories = sorted(
            metrics['by_category'].items(),
            key=lambda x: x[1]['accuracy']
        )
        weak_areas['categories'] = sorted_categories

    # 서브카테고리별 약점 (정확도 낮은 순, 샘플 수 5개 이상)
    if 'by_subcategory' in metrics:
        sorted_subcategories = sorted(
            [(k, v) for k, v in metrics['by_subcategory'].items() if v['total'] >= 5],
            key=lambda x: x[1]['accuracy']
        )
        weak_areas['subcategories'] = sorted_subcategories[:10]  # 상위 10개

    # 출처별 약점
    if 'by_source' in metrics:
        sorted_sources = sorted(
            metrics['by_source'].items(),
            key=lambda x: x[1]['accuracy']
        )
        weak_areas['sources'] = sorted_sources

    return weak_areas

def generate_recommendations(error_analysis, weak_areas):
    """개선 권장사항 생성"""
    print("\n[INFO] Generating recommendations...")

    recommendations = []

    # 1. 답변 추출 실패가 많은 경우
    if error_analysis['no_answer_count'] > error_analysis['total_errors'] * 0.3:
        recommendations.append({
            'priority': 'HIGH',
            'issue': '답변 추출 실패율 높음',
            'detail': f"{error_analysis['no_answer_count']}개 ({error_analysis['no_answer_count']/error_analysis['total_errors']*100:.1f}%)의 오답이 답변 추출 실패",
            'recommendation': [
                '프롬프트 개선: 더 명확한 답변 형식 요구',
                'Temperature 조정: 더 결정적인 답변 생성',
                '답변 추출 로직 개선: 더 유연한 패턴 매칭'
            ]
        })

    # 2. 특정 카테고리 약점
    if weak_areas['categories']:
        weakest_category, stats = weak_areas['categories'][0]
        if stats['accuracy'] < 50:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'{weakest_category} 카테고리 성능 낮음',
                'detail': f"정확도 {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})",
                'recommendation': [
                    f'{weakest_category} 관련 데이터 증강 고려',
                    'Few-shot learning: 해당 카테고리 예시 프롬프트에 추가',
                    'Fine-tuning 시 해당 카테고리 가중치 증가'
                ]
            })

    # 3. 특정 서브카테고리 약점
    if weak_areas['subcategories']:
        for subcat, stats in weak_areas['subcategories'][:3]:  # 가장 약한 3개
            if stats['accuracy'] < 40:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'issue': f'{subcat} 서브카테고리 성능 매우 낮음',
                    'detail': f"정확도 {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})",
                    'recommendation': [
                        f'{subcat} 관련 추가 학습 데이터 수집 검토',
                        '해당 주제에 특화된 프롬프트 엔지니어링'
                    ]
                })

    # 4. 전반적 성능이 낮은 경우
    # (이 부분은 metrics를 받아서 처리할 수 있음)

    return recommendations

def write_analysis_report(metrics, error_analysis, weak_areas, recommendations):
    """분석 보고서 작성"""
    print(f"\n[INFO] Writing analysis report to: {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        # 제목
        f.write("# 베이스라인 평가 에러 분석 보고서\n\n")
        f.write(f"생성일: {metrics['metadata']['evaluation_date']}\n\n")
        f.write("---\n\n")

        # 1. 전체 요약
        f.write("## 1. 전체 성능 요약\n\n")
        f.write(f"- **전체 정확도**: {metrics['overall']['accuracy']:.2f}% ")
        f.write(f"({metrics['overall']['correct']}/{metrics['overall']['total']})\n")
        f.write(f"- **총 오답 수**: {error_analysis['total_errors']}개\n")
        f.write(f"- **답변 추출 실패**: {error_analysis['no_answer_count']}개\n\n")

        # 2. 카테고리별 성능
        f.write("## 2. 카테고리별 성능\n\n")
        f.write("| 카테고리 | 정확도 | 정답/전체 |\n")
        f.write("|---------|--------|----------|\n")
        for category, stats in sorted(metrics['by_category'].items(), key=lambda x: x[1]['accuracy']):
            f.write(f"| {category} | {stats['accuracy']:.2f}% | {stats['correct']}/{stats['total']} |\n")
        f.write("\n")

        # 3. 약점 영역
        f.write("## 3. 주요 약점 영역\n\n")
        f.write("### 3.1 카테고리별 약점\n\n")
        for category, stats in weak_areas['categories'][:3]:
            f.write(f"- **{category}**: {stats['accuracy']:.2f}% ")
            f.write(f"({stats['correct']}/{stats['total']})\n")
        f.write("\n")

        f.write("### 3.2 서브카테고리별 약점 (정확도 낮은 순, Top 10)\n\n")
        f.write("| 서브카테고리 | 정확도 | 정답/전체 |\n")
        f.write("|------------|--------|----------|\n")
        for subcat, stats in weak_areas['subcategories'][:10]:
            f.write(f"| {subcat} | {stats['accuracy']:.2f}% | {stats['correct']}/{stats['total']} |\n")
        f.write("\n")

        f.write("### 3.3 출처별 성능\n\n")
        f.write("| 출처 | 정확도 | 정답/전체 |\n")
        f.write("|------|--------|----------|\n")
        for source, stats in weak_areas['sources']:
            f.write(f"| {source} | {stats['accuracy']:.2f}% | {stats['correct']}/{stats['total']} |\n")
        f.write("\n")

        # 4. 오답 패턴
        f.write("## 4. 오답 패턴 분석\n\n")
        if error_analysis['common_patterns']:
            for pattern in error_analysis['common_patterns']:
                f.write(f"### {pattern['pattern']}\n\n")
                f.write(f"- **발생 횟수**: {pattern['count']}개\n")
                f.write(f"- **비율**: {pattern['percentage']:.1f}%\n")
                f.write(f"- **설명**: {pattern['description']}\n\n")
        else:
            f.write("특별한 오답 패턴이 발견되지 않았습니다.\n\n")

        # 5. 샘플 오답 사례
        f.write("## 5. 샘플 오답 사례\n\n")
        for i, error in enumerate(error_analysis['sample_errors'][:5], 1):
            f.write(f"### 사례 {i}\n\n")
            f.write(f"- **ID**: {error.get('id', 'N/A')}\n")
            f.write(f"- **카테고리**: {error.get('category', 'N/A')} > {error.get('subcategory', 'N/A')}\n")
            f.write(f"- **출처**: {error.get('source', 'N/A')}\n")
            f.write(f"- **질문**: {error.get('question', 'N/A')[:100]}...\n")
            f.write(f"- **정답**: {error.get('correct_answer', 'N/A')}\n")
            f.write(f"- **예측**: {error.get('predicted_answer', 'None (추출 실패)')}\n")
            if 'generated_text' in error:
                f.write(f"- **생성된 텍스트**: {error['generated_text'][:200]}...\n")
            f.write("\n")

        # 6. 개선 권장사항
        f.write("## 6. 개선 권장사항\n\n")
        if recommendations:
            for rec in recommendations:
                f.write(f"### [{rec['priority']}] {rec['issue']}\n\n")
                f.write(f"**상세**: {rec['detail']}\n\n")
                f.write("**권장사항**:\n")
                for r in rec['recommendation']:
                    f.write(f"- {r}\n")
                f.write("\n")
        else:
            f.write("특별한 개선 권장사항이 없습니다.\n\n")

        # 7. 다음 단계
        f.write("## 7. 다음 단계\n\n")
        f.write("1. Fine-tuning을 통한 전반적 성능 개선\n")
        f.write("2. 약점 카테고리 집중 학습\n")
        f.write("3. 프롬프트 엔지니어링 최적화\n")
        f.write("4. 답변 추출 로직 개선 (필요시)\n")

    print(f"[OK] Analysis report saved")

def main():
    print("="*80)
    print("[START] Baseline Error Analysis")
    print("="*80 + "\n")

    # 파일 존재 확인
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[ERROR] Predictions file not found: {PREDICTIONS_PATH}")
        print("[INFO] Please run baseline evaluation first.")
        return

    if not os.path.exists(METRICS_PATH):
        print(f"[ERROR] Metrics file not found: {METRICS_PATH}")
        print("[INFO] Please run baseline evaluation first.")
        return

    # 1. 데이터 로드
    predictions = load_predictions()
    metrics = load_metrics()

    # 2. 에러 분석
    error_analysis = analyze_errors(predictions)

    # 3. 약점 영역 파악
    weak_areas = find_weak_areas(metrics)

    # 4. 개선 권장사항 생성
    recommendations = generate_recommendations(error_analysis, weak_areas)

    # 5. 보고서 작성
    write_analysis_report(metrics, error_analysis, weak_areas, recommendations)

    print("\n" + "="*80)
    print("[DONE] Error analysis complete!")
    print("="*80)
    print(f"\n분석 보고서: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
