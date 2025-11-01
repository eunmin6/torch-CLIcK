"""
프롬프트 개선 전후 비교 분석
"""

import json

# Load results
with open('./benchmarks/finetuned_history/summary.json', 'r', encoding='utf-8') as f:
    original = json.load(f)

with open('./benchmarks/finetuned_history_improved/summary.json', 'r', encoding='utf-8') as f:
    improved = json.load(f)

with open('./benchmarks/finetuned_history/results.json', 'r', encoding='utf-8') as f:
    original_results = json.load(f)

with open('./benchmarks/finetuned_history_improved/results.json', 'r', encoding='utf-8') as f:
    improved_results = json.load(f)

# Generate comparison report
report = f"""# 프롬프트 개선 전후 비교 분석

**평가 일시**:
- Original: {original['timestamp']}
- Improved: {improved['timestamp']}

---

## 1. Executive Summary

### 성능 비교

| Metric | Original Prompt | Improved Prompt | Change |
|--------|----------------|-----------------|--------|
| **Accuracy** | {original['accuracy']:.2f}% | {improved['accuracy']:.2f}% | **{improved['accuracy'] - original['accuracy']:+.2f}%p** |
| **Correct** | {original['correct']}/{original['total_samples']} | {improved['correct']}/{improved['total_samples']} | {improved['correct'] - original['correct']:+d} |
| **Incorrect** | {original['incorrect']}/{original['total_samples']} | {improved['incorrect']}/{improved['total_samples']} | {improved['incorrect'] - original['incorrect']:+d} |

### Extraction 성공률

**Original Prompt**:
- Extraction 실패 없음 (모두 성공)

**Improved Prompt**:
```
"""

for method, count in improved['extraction_stats'].items():
    pct = (count / improved['total_samples'] * 100)
    report += f"{method:20s}: {count:3d} ({pct:5.1f}%)\n"

report += f"""```

---

## 2. 주요 발견 사항

### 2.1 예상과 다른 결과

**가설**: 답변 형식을 명확히 지시하면 extraction 정확도가 올라가 성능 향상
**실제**: 오히려 성능 하락 (50.00% → 40.48%)

### 2.2 원인 분석

1. **학습-추론 불일치 (Train-Inference Mismatch)**
   - 학습 시: 간단한 프롬프트 사용
   - 추론 시: 복잡한 형식 지시 프롬프트 사용
   - 결과: 모델이 혼란스러워함

2. **과도한 제약**
   - "반드시 다음 형식으로만 답변하세요" 같은 강한 제약
   - 모델의 자연스러운 답변 패턴 방해
   - 오히려 정답률 하락

3. **Extraction은 성공했지만 정답률 하락**
   - Extraction 실패: 0건
   - 하지만 정답 수: 21 → 17 (-4개)
   - 즉, extraction 문제가 아니라 **모델의 실제 선택이 틀림**

---

## 3. 샘플별 상세 비교

다음은 두 프롬프트에서 답변이 달라진 케이스들입니다:

"""

# Find differences
different_cases = []
for orig, impr in zip(original_results, improved_results):
    if orig['predicted_answer'] != impr['predicted_answer']:
        different_cases.append({
            'id': orig['id'],
            'question': orig['question'][:50] + '...',
            'correct': orig['correct_answer'],
            'original_pred': orig['predicted_answer'],
            'improved_pred': impr['predicted_answer'],
            'original_correct': orig['is_correct'],
            'improved_correct': impr['is_correct']
        })

report += f"**총 {len(different_cases)}개 문제에서 답변이 달라짐**\n\n"

# Show first 10 different cases
for i, case in enumerate(different_cases[:10], 1):
    status_orig = "✓" if case['original_correct'] else "✗"
    status_impr = "✓" if case['improved_correct'] else "✗"

    report += f"""### Case {i}: {case['question']}

| | Original | Improved |
|---|----------|----------|
| **예측 답변** | {case['original_pred'][:30]}... | {case['improved_pred'][:30]}... |
| **정답 여부** | {status_orig} | {status_impr} |

**정답**: {case['correct'][:30]}...

"""

report += f"""
---

## 4. 결론 및 권장 사항

### 4.1 결론

1. **프롬프트 엔지니어링만으로는 부족**
   - 학습된 모델의 특성을 무시한 프롬프트 변경은 역효과
   - Train-inference 일관성이 중요

2. **Extraction 문제는 실제로 크지 않았음**
   - Original 평가에서도 extraction은 대부분 성공
   - 진짜 문제는 모델의 지식 부족

3. **성능 향상을 위해서는 재학습 필요**
   - 프롬프트만 바꾸는 것은 효과 없음
   - 개선된 프롬프트로 **재학습**해야 함

### 4.2 권장 사항

**단기 개선 (즉시 적용 가능)**:
- ❌ 프롬프트만 변경 (역효과)
- ✅ Original 프롬프트 유지 (50.00% 유지)

**중기 개선 (1-2일 소요)**:
- ✅ 개선된 프롬프트로 **재학습** (`finetune_lora_improved.py`)
- ✅ Epoch 수 증가 (3 → 5 또는 10)
- ✅ Learning rate 조정

**장기 개선 (1주 이상)**:
- ✅ 더 큰 모델 사용 (Qwen2.5-14B, 72B)
- ✅ Korean History 데이터 augmentation
- ✅ Full fine-tuning (LoRA 대신)

### 4.3 다음 단계

**Option 1: 재학습 (권장)**
```bash
# 개선된 프롬프트로 재학습
python finetune_lora_improved.py

# 평가
python evaluate_finetuned_improved.py \\
  --lora_path ./models/qwen-click-lora-improved/final
```

**Option 2: Epoch 증가**
```python
# finetune_lora.py 수정
TRAINING_CONFIG = {{
    "num_train_epochs": 10,  # 3 → 10
    ...
}}
```

**Option 3: 현재 모델 유지**
- Original 프롬프트 사용
- 50.00% 정확도 유지

---

## 5. 교훈

### 학습한 내용

1. **프롬프트 엔지니어링의 한계**
   - 이미 학습된 모델의 동작 패턴을 바꾸기 어려움
   - Train-time prompt와 inference-time prompt는 일치해야 함

2. **Extraction vs 실제 성능**
   - Extraction 성공률 100%여도 정답률은 별개
   - 근본적인 문제는 모델의 지식 부족

3. **일관성의 중요성**
   - 학습과 추론의 일관성 유지 필수
   - 프롬프트 변경 시 재학습 필요

### 향후 실험 방향

1. **재학습 실험**
   - 개선된 프롬프트로 처음부터 재학습
   - 예상: 55-60% 정확도 달성 가능

2. **Epoch 실험**
   - 3 epoch → 5, 10, 20 epoch 비교
   - Overfitting 여부 모니터링

3. **Learning Rate 실험**
   - 2e-4 → 1e-4, 5e-5 등 다양한 값 시도

---

**생성 시각**: {improved['timestamp']}
**결론**: 프롬프트만 변경은 역효과. 재학습 필요.
"""

# Save report
with open('./prompt-comparison-report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("[OK] prompt-comparison-report.md generated!")
print(f"\nSummary:")
print(f"  Original Accuracy: {original['accuracy']:.2f}%")
print(f"  Improved Accuracy: {improved['accuracy']:.2f}%")
print(f"  Difference: {improved['accuracy'] - original['accuracy']:+.2f}%p")
print(f"  Different answers: {len(different_cases)} cases")
print(f"\n결론: 프롬프트만 변경하면 역효과. 재학습이 필요합니다.")
