"""
v2 Model Evaluation Variance Analysis

두 평가 결과를 비교하여 편차를 분석합니다.
"""

import json
from datetime import datetime

# ==================== Load Results ====================

def load_results(path):
    """Load evaluation results"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_variance():
    """Compare two evaluations and calculate variance"""

    print("=" * 80)
    print("v2 Model Evaluation Variance Analysis")
    print("=" * 80)
    print()

    # Load both evaluations
    eval1_results = load_results("./benchmarks/finetuned_v2_history/results.json")
    eval1_summary = load_results("./benchmarks/finetuned_v2_history/summary.json")

    eval2_results = load_results("./benchmarks/v2_retest/results.json")
    eval2_summary = load_results("./benchmarks/v2_retest/summary.json")

    # Summary comparison
    print("[1] 전체 정확도 비교")
    print("-" * 80)
    print(f"평가 1 (2025-10-31 16:23:52): {eval1_summary['accuracy']:.2f}% ({eval1_summary['correct']}/{eval1_summary['total_samples']})")
    print(f"평가 2 (2025-11-01 11:11:59): {eval2_summary['accuracy']:.2f}% ({eval2_summary['correct']}/{eval2_summary['total_samples']})")
    print()

    accuracy_diff = abs(eval1_summary['accuracy'] - eval2_summary['accuracy'])
    print(f"정확도 차이: {accuracy_diff:.4f}%p")
    print()

    # Question-by-question comparison
    print("[2] 문항별 정답 일치도 분석")
    print("-" * 80)

    same_correct = 0  # 둘 다 맞음
    same_wrong = 0    # 둘 다 틀림
    diff_answers = 0  # 답이 다름

    diff_details = []

    for i in range(len(eval1_results)):
        q1 = eval1_results[i]
        q2 = eval2_results[i]

        if q1['is_correct'] and q2['is_correct']:
            same_correct += 1
        elif not q1['is_correct'] and not q2['is_correct']:
            same_wrong += 1
        else:
            diff_answers += 1
            diff_details.append({
                'id': q1['id'],
                'question': q1['question'][:60] + "...",
                'eval1_correct': q1['is_correct'],
                'eval1_answer': q1['predicted_answer'],
                'eval2_correct': q2['is_correct'],
                'eval2_answer': q2['predicted_answer'],
                'correct_answer': q1['correct_answer']
            })

    total = len(eval1_results)
    print(f"총 문항 수: {total}")
    print(f"  - 두 평가 모두 정답: {same_correct}개 ({same_correct/total*100:.1f}%)")
    print(f"  - 두 평가 모두 오답: {same_wrong}개 ({same_wrong/total*100:.1f}%)")
    print(f"  - 평가 간 답변 차이: {diff_answers}개 ({diff_answers/total*100:.1f}%)")
    print()

    # Consistency metrics
    consistency = (same_correct + same_wrong) / total * 100
    print(f"[3] 평가 일관성 (Consistency)")
    print("-" * 80)
    print(f"일관성 점수: {consistency:.2f}% ({same_correct + same_wrong}/{total})")
    print()

    if consistency == 100.0:
        print("✓ 완벽한 일관성: 두 평가에서 모든 문항의 정답 여부가 동일합니다.")
    elif consistency >= 95.0:
        print("✓ 매우 높은 일관성: 대부분의 문항에서 동일한 결과를 보입니다.")
    elif consistency >= 90.0:
        print("⚠ 높은 일관성: 일부 문항에서 차이가 발생했습니다.")
    else:
        print("✗ 낮은 일관성: 평가 간 편차가 큽니다.")
    print()

    # Show differences if any
    if diff_answers > 0:
        print("[4] 평가 간 차이가 발생한 문항")
        print("-" * 80)
        for detail in diff_details:
            print(f"\n문항 #{detail['id']}: {detail['question']}")
            print(f"  평가1: {'✓' if detail['eval1_correct'] else '✗'} {detail['eval1_answer']}")
            print(f"  평가2: {'✓' if detail['eval2_correct'] else '✗'} {detail['eval2_answer']}")
            print(f"  정답: {detail['correct_answer']}")
    else:
        print("[4] 평가 간 차이 분석")
        print("-" * 80)
        print("✓ 모든 문항에서 동일한 답변을 생성했습니다.")

    print()
    print("=" * 80)
    print("[결론] v2 모델 평가 안정성 분석")
    print("=" * 80)

    if accuracy_diff == 0 and consistency == 100.0:
        print("✓ 결정론적 모델 (Deterministic Model)")
        print()
        print("v2 모델은 완벽하게 결정론적입니다:")
        print("  - 동일한 입력에 대해 항상 동일한 출력 생성")
        print("  - do_sample=False 설정이 올바르게 작동")
        print("  - 평가 간 편차 0% (완벽한 재현성)")
        print()
        print("이는 다음을 의미합니다:")
        print("  1. 평가 결과를 신뢰할 수 있음")
        print("  2. 재평가 시에도 동일한 결과 보장")
        print("  3. 다른 모델과의 비교가 공정함")
    elif accuracy_diff < 5.0:
        print("✓ 매우 안정적인 모델")
        print(f"  - 정확도 편차: {accuracy_diff:.2f}%p (5% 미만)")
        print(f"  - 일관성: {consistency:.1f}%")
    else:
        print("⚠ 평가 편차 존재")
        print(f"  - 정확도 편차: {accuracy_diff:.2f}%p")
        print(f"  - 일관성: {consistency:.1f}%")
        print()
        print("원인 분석 필요:")
        print("  - do_sample 설정 확인")
        print("  - temperature, top_p, top_k 파라미터 확인")
        print("  - 모델 로딩 방식 확인")

    print()
    print("=" * 80)

    # Save comparison report
    report = {
        "evaluation_1": {
            "timestamp": eval1_summary['timestamp'],
            "accuracy": eval1_summary['accuracy'],
            "correct": eval1_summary['correct']
        },
        "evaluation_2": {
            "timestamp": eval2_summary['timestamp'],
            "accuracy": eval2_summary['accuracy'],
            "correct": eval2_summary['correct']
        },
        "variance_analysis": {
            "accuracy_difference": accuracy_diff,
            "same_correct": same_correct,
            "same_wrong": same_wrong,
            "different_answers": diff_answers,
            "consistency_score": consistency,
            "is_deterministic": (accuracy_diff == 0 and consistency == 100.0)
        },
        "differences": diff_details
    }

    with open("./benchmarks/v2_variance_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Variance report saved to: ./benchmarks/v2_variance_report.json")
    print()

if __name__ == "__main__":
    analyze_variance()
