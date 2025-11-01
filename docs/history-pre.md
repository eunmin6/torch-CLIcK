# CLIcK Korean History Fine-tuning Research History

## 프로젝트 개요

CLIcK (Korean L&C Intelligence Benchmark) 데이터셋의 Korean History 서브카테고리를 활용하여 다양한 LLM 모델들의 한국 역사 지식 성능을 측정하고 개선하는 연구를 수행했습니다.

## 연구 기간

2025년 10월 31일 - 2025년 11월 1일

## 실험 개요

### 데이터셋
- **출처**: CLIcK (Korean L&C Intelligence Benchmark)
- **카테고리**: Korean History (한국 역사)
- **샘플 수**:
  - Train: 1,396개
  - Validation: 299개
  - Test: 42개

### 실험 모델

#### 1. Qwen2.5-7B-Instruct
- **브랜치**: `experiment/qwen2.5-7b`
- **베이스라인**: 14.29% (6/42)
- **Fine-tuned 최종 성능**: 52.38% (22/42)

#### 2. Llama3.1-8B-Instruct
- **브랜치**: `experiment/llama3.1-8b`
- **베이스라인**: 2.38% (1/42)
- **Fine-tuned 성능**: 28.57% (12/42)

## 연구 과정

### Phase 1: Qwen2.5-7B 실험 (experiment/qwen2.5-7b)

#### v1: 초기 Fine-tuning
- **방법론**: 기본 LoRA fine-tuning
- **시스템 프롬프트**: "당신은 한국 역사에 정통한 AI 어시스턴트입니다."
- **결과**: 28.57% (12/42)
- **문제점**: 답변 형식이 불안정하고 정확도가 낮음

#### v2: 프롬프트 엔지니어링 개선
- **핵심 변경사항**:
  1. 상세한 형식 지시 추가
  2. System prompt에 명확한 규칙 명시
  3. 답변 형식을 "숫자. 정답내용"으로 통일

- **v2 System Prompt**:
```
당신은 한국 역사에 정통한 AI 어시스턴트입니다.

질문에 답변할 때 다음 규칙을 따르세요:

[객관식 문제] - 선택지가 주어진 경우:
→ 반드시 "숫자. 정답내용" 형식으로만 답변
→ 예시: "3. 팔만대장경", "1. 고구려"

[서술형 문제] - 선택지가 없는 경우:
→ 자유롭게 상세히 설명하여 답변

현재 질문의 유형을 파악하여 적절한 형식으로 답변해주세요.
```

- **결과**: 52.38% (22/42) - **83.3% 성능 향상**
- **분산 분석**: 5회 평가 결과 51.43% ± 1.99%로 안정적

#### v3: 장기 학습 실험 (10 Epochs)
- **목적**: 추가 학습을 통한 성능 향상 가능성 탐색
- **결과**: 45.24% (19/42)
- **결론**: Over-fitting 발생, v2 (3 epochs)가 최적

### Phase 2: Llama3.1-8B 실험 (experiment/llama3.1-8b)

#### 베이스라인 평가
- **결과**: 2.38% (1/42)
- **분석**: Qwen 14.29%에 비해 현저히 낮은 성능
- **원인**: Llama3.1은 영어 중심 학습으로 한국어 이해도가 낮음

#### v2 방법론 적용
- **동일한 프롬프트 엔지니어링 적용**
- **기술적 도전과제**:
  1. Gated Model 인증 문제 해결 (HuggingFace Token)
  2. Data Collator 텐서 형태 불일치 → 수동 패딩 구현
  3. CUDA OOM (37.99 GiB 요청) → Batch size 1 + Gradient Checkpointing
  4. Gradient Checkpointing 호환성 → `enable_input_require_grads()` 추가

- **최종 결과**: 28.57% (12/42)
- **분석**: v2 방법론으로 베이스라인 대비 1,100% 향상했으나, Qwen에 비해서는 절대적 성능 낮음

## 주요 기술적 발견

### 1. 프롬프트 엔지니어링의 중요성
- 명확한 답변 형식 지정이 성능에 결정적 영향
- "숫자. 내용" 형식 강제로 파싱 정확도 향상

### 2. 다국어 사전학습의 중요성
- Qwen2.5-7B (한국어 포함): Baseline 14.29% → Fine-tuned 52.38%
- Llama3.1-8B (영어 중심): Baseline 2.38% → Fine-tuned 28.57%
- 사전학습 단계의 언어 분포가 fine-tuning 효과에 큰 영향

### 3. Over-fitting 방지
- 3 epochs가 10 epochs보다 우수 (52.38% vs 45.24%)
- 작은 데이터셋(1,396 samples)에서는 조기 종료가 효과적

### 4. 메모리 최적화 기법
- Gradient Checkpointing으로 37.99 GiB → 24 GiB 이하로 감소
- Batch size 감소 + Gradient Accumulation으로 effective batch 유지

## LoRA 설정 (공통)

```python
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

## 학습 설정

### Qwen2.5-7B
```python
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}
```

### Llama3.1-8B (메모리 최적화 버전)
```python
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,  # 메모리 절약
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,  # effective batch=16 유지
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}
```

## 최종 성능 비교

| 모델 | 베이스라인 | Fine-tuned (v2) | 향상률 |
|------|-----------|----------------|--------|
| Qwen2.5-7B | 14.29% (6/42) | 52.38% (22/42) | +266.5% |
| Llama3.1-8B | 2.38% (1/42) | 28.57% (12/42) | +1,100% |

## 결론

1. **Qwen2.5-7B가 한국어 태스크에 최적**: 베이스라인과 fine-tuned 모두에서 우수
2. **v2 프롬프트 엔지니어링이 결정적**: 명확한 답변 형식 지정으로 큰 성능 향상
3. **적절한 학습 epoch 중요**: 3 epochs가 10 epochs보다 우수 (over-fitting 방지)
4. **다국어 사전학습의 중요성**: 영어 중심 모델은 한국어 태스크에서 한계

## 향후 연구 방향

1. **더 큰 모델 실험**: Qwen2.5-14B, Llama3.1-70B 등
2. **Data Augmentation**: 한국 역사 데이터 추가 확보
3. **Ensemble 방법**: 여러 모델의 예측 결합
4. **Few-shot Learning**: 추가 fine-tuning 없이 성능 향상
5. **Cross-lingual Transfer**: 영어 역사 데이터 활용

## 코드 저장소

- **Qwen 실험**: `experiment/qwen2.5-7b` 브랜치
- **Llama 실험**: `experiment/llama3.1-8b` 브랜치
- **문서**: `docs/` 디렉토리

## 참고 문서

- `docs/baseline-report.md`: Qwen 베이스라인 평가 보고서
- `docs/finetune-report.md`: Qwen v1 fine-tuning 보고서
- `docs/prompt-comparison-report.md`: v1 vs v2 비교
- `docs/v2-variance-report.md`: v2 분산 분석
- `docs/v3-analysis-report.md`: v3 (10 epochs) 분석
- `docs/model-evolution-report.md`: 전체 모델 진화 과정
- `docs/qwen_vs_llama_comparison.md`: Qwen vs Llama 비교
- `docs/finetune-guide.md`: Fine-tuning 가이드

---

**연구 종료일**: 2025년 11월 1일
