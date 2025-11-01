# torch-CLIcK 프로젝트 정의서

## 프로젝트 개요

HuggingFace의 공개 데이터셋 **EunsuKim/CLIcK**을 기반으로 **gpt-oss:20b** 모델을 학습시키는 프로젝트

## 프로젝트 목적

한국 문화와 언어에 대한 이해도를 높인 언어 모델을 fine-tuning하여, 한국어 객관식 문제 해결 능력을 향상시키는 것

## 데이터셋 정보: CLIcK (Cultural and Linguistic Intelligence in Korean)

### 기본 정보
- **출처**: HuggingFace - EunsuKim/CLIcK
- **형식**: Multiple-choice (객관식)
- **언어**: 한국어 (ko)
- **크기**: 1K-10K 샘플
- **다운로드**: 729회
- **좋아요**: 24개
- **논문**: arXiv:2403.06412

### 데이터 구조
각 샘플은 다음 필드로 구성:
```json
{
  "id": "고유 식별자",
  "paragraph": "지문 (선택적, 비어있을 수 있음)",
  "question": "질문 텍스트",
  "choices": ["선택지1", "선택지2", "선택지3", "선택지4", ...],
  "answer": "정답 텍스트"
}
```

### 데이터 카테고리

#### 1. Culture (문화)
- **Korean Economy** (한국 경제)
  - Economy_KIIP.json
  - Economy_Kedu.json

- **Korean Geography** (한국 지리)
  - Geography_CSAT.json
  - Geography_KIIP.json
  - Geography_Kedu.json

- **Korean History** (한국 역사)
  - History_KHB.json (47개 샘플)
  - History_Kedu.json
  - History_PSE.json

- **Korean Law** (한국 법)
  - Law_KIIP.json
  - Law_PSAT.json

- **Korean Politics** (한국 정치)
  - Politics_KIIP.json
  - Politics_Kedu.json

- **Korean Popular** (한국 대중문화)
  - Popular_KIIP.json
  - Popular_Kedu.json

- **Korean Society** (한국 사회)
  - Society_KIIP.json
  - Society_Kedu.json

- **Korean Tradition** (한국 전통)
  - Tradition_KIIP.json
  - Tradition_Kedu.json

#### 2. Language (언어)
- **Functional** (기능)
  - Functional_CSAT.json
  - Functional_Kedu.json
  - Functional_PSE.json

- **Grammar** (문법)
  - Grammar_CSAT.json (46개 샘플)
  - Grammar_Kedu.json
  - Grammar_TOPIK.json

- **Textual** (텍스트)
  - Textual_CSAT.json
  - Textual_TOPIK.json

### 데이터 출처
- **CSAT**: 대학수학능력시험
- **KIIP**: 사회통합프로그램
- **TOPIK**: 한국어능력시험
- **Kedu**: 한국교육방송공사
- **PSAT**: 공직적격성평가
- **PSE**: 초등교육
- **KHB**: 한국사능력검정시험

## 타겟 모델: gpt-oss-20b

### 기본 정보
- **출처**: OpenAI - openai/gpt-oss-20b
- **모델 페이지**: https://huggingface.co/openai/gpt-oss-20b
- **라이센스**: Apache 2.0 (상업적 사용 가능)
- **월간 다운로드**: 4.7M+

### 아키텍처 및 파라미터
- **총 파라미터 수**: 21 billion (21B)
- **활성 파라미터**: 3.6 billion (Mixture of Experts 아키텍처)
- **정밀도**: BF16 및 U8
- **양자화**: MXFP4 quantization (MoE weights)
- **메모리 요구사항**: 16GB 메모리에서 실행 가능

### 주요 기능
- **강력한 추론 능력**: Chain-of-thought reasoning (low/medium/high 레벨 조절 가능)
- **에이전트 작업**: 복잡한 작업 수행 능력
- **개발자 친화적**: 다양한 개발 사용 사례 지원
- **Function calling**: 구조화된 출력 지원
- **코드 실행**: Python 코드 실행 및 웹 브라우징 기능
- **파인튜닝 가능**: 소비자 하드웨어에서도 fine-tuning 가능

### 기술 요구사항
- **필수 포맷**: OpenAI의 harmony response format 필요 (표준 generation은 작동 불가)
- **Chat Template**: Transformers 라이브러리에서 자동으로 harmony format 적용
- **지원 프레임워크**:
  - Transformers
  - vLLM (OpenAI-compatible API)
  - Ollama (로컬 소비자 하드웨어)
  - LM Studio

### 모델 특성 및 제약사항
- **설계 목적**: gpt-oss-120b의 경량화 버전 (지연시간 민감 애플리케이션용)
- **Chain-of-thought 출력**: 최종 사용자에게 직접 표시하기 위한 용도 아님
- **포맷 의존성**: harmony response format이 필수 - 올바른 포맷 없이는 정상 작동 불가

## 환경 정보

- **작업 디렉토리**: `D:\AI\torch-CLIcK`
- **Python 버전**: 3.14
- **GPU**: CUDA 사용 가능 여부 확인 필요
- **주요 라이브러리**:
  - PyTorch
  - datasets (HuggingFace)
  - huggingface-hub

## 주요 과제

### 1. 데이터 전처리
- [ ] 모든 JSON 파일 통합 및 로드
- [ ] 데이터 품질 검증 (인코딩, 누락 필드 확인)
- [ ] Train/Validation/Test 분할 전략 수립
- [ ] 데이터 포맷 변환 (모델 학습에 적합한 형태)

### 2. 모델 학습 준비
- [ ] gpt-oss:20b 모델 확인 및 로드 방법 조사
- [ ] Fine-tuning 전략 결정
  - LoRA, QLoRA 등 효율적 학습 방법 고려
  - Full fine-tuning vs Parameter-efficient fine-tuning
- [ ] 하이퍼파라미터 설정
- [ ] 평가 메트릭 정의 (Accuracy, F1 등)

### 3. 학습 인프라
- [ ] GPU 리소스 확인 및 최적화
- [ ] 배치 사이즈 및 메모리 관리
- [ ] 체크포인트 및 로깅 시스템
- [ ] 분산 학습 필요성 검토

### 4. 평가 및 검증
- [ ] 벤치마크 설정
- [ ] 카테고리별 성능 분석
- [ ] 오류 분석 및 개선 방향 도출

## 벤치마킹 전략 (매우 중요)

### 학습 전 베이스라인 평가 (Pre-training Benchmark)

#### 목적
Fine-tuning의 효과를 정량적으로 측정하기 위한 기준선 설정

#### 평가 항목
1. **전체 정확도 (Overall Accuracy)**
   - 전체 데이터셋에 대한 정답률
   - 메트릭: Accuracy, F1-Score (macro/weighted)

2. **카테고리별 정확도**
   - **Culture 카테고리**:
     - Korean Economy (경제)
     - Korean Geography (지리)
     - Korean History (역사)
     - Korean Law (법)
     - Korean Politics (정치)
     - Korean Popular (대중문화)
     - Korean Society (사회)
     - Korean Tradition (전통)

   - **Language 카테고리**:
     - Functional (기능)
     - Grammar (문법)
     - Textual (텍스트)

3. **데이터 출처별 정확도**
   - CSAT (대학수학능력시험)
   - KIIP (사회통합프로그램)
   - TOPIK (한국어능력시험)
   - Kedu (한국교육방송공사)
   - PSAT (공직적격성평가)
   - PSE (초등교육)
   - KHB (한국사능력검정시험)

4. **응답 품질 평가**
   - 정확한 선택지 선택 비율
   - 응답 형식 준수율
   - 응답 생성 시간 (latency)
   - 토큰 사용량

5. **한국어 이해도 평가**
   - 지문 이해력 (paragraph가 있는 경우)
   - 문맥 파악 능력
   - 문화적 맥락 이해도

### 학습 후 평가 (Post-training Benchmark)

#### 평가 방법
- 동일한 test set에 대해 학습 전과 동일한 메트릭 측정
- A/B 테스트 방식으로 개선도 비교

#### 성공 기준 (Success Metrics)
- **최소 목표**: 전체 정확도 10% 이상 향상
- **이상적 목표**: 전체 정확도 20% 이상 향상
- **카테고리별**: 모든 카테고리에서 최소 5% 이상 향상
- **특정 약점 개선**: 베이스라인에서 가장 낮은 성능의 카테고리에서 15% 이상 향상

### 평가 데이터 분할

#### Train/Validation/Test Split
- **Train**: 70% (모델 학습용)
- **Validation**: 15% (하이퍼파라미터 튜닝 및 early stopping)
- **Test**: 15% (최종 평가용, 학습 중 절대 사용 금지)

#### Stratified Sampling
- 카테고리별 비율 유지
- 데이터 출처별 비율 유지
- 난이도 분포 유지 (가능한 경우)

### 벤치마크 실행 계획

#### Phase 1: 베이스라인 수립
1. [ ] Test set 분리 및 고정
2. [ ] Pre-trained gpt-oss-20b 모델로 전체 데이터 추론
3. [ ] 모든 평가 메트릭 계산 및 기록
4. [ ] 카테고리별/출처별 상세 분석
5. [ ] 오류 사례 분석 및 패턴 파악

#### Phase 2: Fine-tuning 실행
1. [ ] Training 데이터로 모델 학습
2. [ ] Validation set으로 학습 과정 모니터링
3. [ ] Best checkpoint 선정

#### Phase 3: 학습 후 평가
1. [ ] Fine-tuned 모델로 test set 추론
2. [ ] 동일 메트릭으로 성능 측정
3. [ ] Pre/Post 비교 분석
4. [ ] 개선도 시각화 (그래프, 표)

#### Phase 4: 심층 분석
1. [ ] 개선된 영역 vs 개선되지 않은 영역 분석
2. [ ] 새로 발생한 오류 패턴 파악
3. [ ] Confusion matrix 생성
4. [ ] 정성적 평가 (샘플 케이스 리뷰)

### 벤치마크 결과 문서화

#### 저장할 정보
- 실험 날짜 및 시간
- 모델 버전 및 체크포인트
- 하이퍼파라미터 설정
- 학습 시간 및 리소스 사용량
- 모든 정량적 메트릭
- 샘플 예측 결과 (성공/실패 사례)
- 개선 권장사항

#### 벤치마크 결과 파일 구조
```
benchmarks/
  ├── baseline/
  │   ├── overall_metrics.json
  │   ├── category_metrics.json
  │   ├── source_metrics.json
  │   ├── predictions.jsonl
  │   └── error_analysis.md
  └── finetuned/
      ├── overall_metrics.json
      ├── category_metrics.json
      ├── source_metrics.json
      ├── predictions.jsonl
      ├── error_analysis.md
      └── comparison_report.md
```

## Fine-tuning 전략

### 권장 접근법: Parameter-Efficient Fine-Tuning (PEFT)

#### 이유
- gpt-oss-20b는 21B 파라미터로 매우 큼
- MoE 구조로 3.6B만 활성화되지만 여전히 큰 모델
- 소규모 데이터셋(1K-10K)으로 Full fine-tuning 시 overfitting 위험
- GPU 메모리 효율성 필요

#### 추천 방법: LoRA (Low-Rank Adaptation)

**장점:**
- 원본 모델 가중치 보존
- 학습 파라미터 대폭 감소 (1% 미만)
- 빠른 학습 속도
- 메모리 효율적
- 여러 버전의 adapter를 쉽게 관리 가능

**LoRA 설정:**
- `r` (rank): 8~32 (실험 필요)
- `lora_alpha`: 16~64
- `target_modules`: attention layers (q_proj, v_proj, k_proj, o_proj)
- `lora_dropout`: 0.05~0.1

**대안: QLoRA (Quantized LoRA)**
- GPU 메모리가 부족한 경우
- 4-bit 양자화 + LoRA
- 성능은 LoRA와 유사하나 메모리 사용량 절반

### 데이터 포맷 전략

#### Harmony Response Format 준수

gpt-oss-20b는 OpenAI의 harmony response format이 필수이므로, chat template을 활용:

**포맷 옵션 1: Instruction Following**
```python
messages = [
    {"role": "system", "content": "당신은 한국 문화와 언어에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제를 정확하게 풀어주세요."},
    {"role": "user", "content": f"문제: {question}\n\n선택지:\n{format_choices(choices)}\n\n정답을 선택하고 그 이유를 설명해주세요."},
    {"role": "assistant", "content": f"정답: {answer}\n\n이유: [추론 과정]"}
]
```

**포맷 옵션 2: Direct Answer (권장)**
```python
messages = [
    {"role": "system", "content": "한국 문화와 언어 문제를 풀어주는 전문가입니다."},
    {"role": "user", "content": f"{paragraph}\n\n질문: {question}\n\n{format_choices(choices)}"},
    {"role": "assistant", "content": answer}
]
```

**포맷 옵션 3: Chain-of-Thought (CoT)**
```python
messages = [
    {"role": "system", "content": "단계별로 생각하며 문제를 풀어주세요."},
    {"role": "user", "content": f"{paragraph}\n\n{question}\n\n{format_choices(choices)}\n\n단계별로 분석하여 답을 선택해주세요."},
    {"role": "assistant", "content": f"분석:\n1. [단계 1]\n2. [단계 2]\n\n결론: {answer}"}
]
```

### 하이퍼파라미터 권장 설정

```python
training_args = {
    "learning_rate": 2e-4,  # LoRA에 적합한 learning rate
    "num_train_epochs": 3-5,
    "per_device_train_batch_size": 4,  # GPU 메모리에 따라 조정
    "gradient_accumulation_steps": 4,  # effective batch size = 16
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "evaluation_strategy": "steps",
    "save_total_limit": 3,
    "fp16": True,  # 또는 bf16 (모델이 bf16이므로 bf16 권장)
    "gradient_checkpointing": True,  # 메모리 절약
}
```

### 학습 프로세스

1. **데이터 준비**
   - 모든 JSON 파일 로드 및 통합
   - Stratified split (Train 70% / Val 15% / Test 15%)
   - Harmony format으로 변환
   - Tokenization 및 padding

2. **모델 로드**
   - `AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")`
   - LoRA configuration 적용
   - Chat template 설정 확인

3. **Fine-tuning**
   - Trainer API 사용 (HuggingFace)
   - Validation loss 모니터링
   - Early stopping (patience=3)
   - Best checkpoint 저장

4. **Inference**
   - Chat template을 통한 생성
   - Temperature, top_p 등 generation parameters 조정

## 프로젝트 진행 상황

### 완료된 작업 ✓

#### Phase 1: 데이터 준비 (완료)
- [x] **전체 데이터셋 로드 및 통계 분석** (2025-10-31)
  - 총 1,995개 샘플 로드 (26개 JSON 파일)
  - Culture: 1,345 samples (67.4%)
  - Language: 650 samples (32.6%)
  - 데이터 품질: 에러 0개, 경고 0개
  - 저장 위치: `./data/raw/all_data.json`

- [x] **데이터 품질 검증** (2025-10-31)
  - 필수 필드 검증 (id, question, choices, answer)
  - 선택지-정답 일치성 확인
  - 인코딩 문제 없음

- [x] **Train/Val/Test Split 생성** (2025-10-31)
  - Stratified sampling 사용
  - Train: 1,396 samples (70%)
  - Validation: 299 samples (15%)
  - Test: 300 samples (15%)
  - 카테고리/출처별 비율 유지 확인
  - Random seed: 42
  - 저장 위치: `./data/splits/`

#### Phase 2: 베이스라인 평가 준비 (진행 중)
- [x] **GPU 환경 확인** (2025-10-31)
  - GPU: NVIDIA GeForce RTX 3090 (24GB VRAM)
  - CUDA 사용 가능

- [x] **필수 라이브러리 설치** (2025-10-31)
  - transformers 4.57.1
  - accelerate 1.11.0
  - torch (기존 설치됨)
  - bitsandbytes, sentencepiece

- [x] **Harmony format 데이터 변환 스크립트** (2025-10-31)
  - Direct Answer 방식 구현
  - Chat template 적용
  - 선택지 포맷팅

- [x] **베이스라인 추론 스크립트 작성** (2025-10-31)
  - 파일: `baseline_evaluation.py`
  - 답변 추출 로직 구현
  - 메트릭 계산 및 저장 기능

- [x] **Qwen2.5-7B-Instruct 모델 로드 테스트** (완료 - 2025-10-31)
  - 토크나이저 로드 완료
  - Chat template 확인 완료
  - 모델 로드 성공 (14.19GB GPU 메모리)
  - 한국어 추론 테스트 정상 작동
  - Note: gpt-oss-20b → Llama-3.1-8B → Qwen2.5-7B로 최종 변경

### 진행 예정 작업

#### Phase 2: 베이스라인 평가 (계속)
- [ ] 모델 로드 완료 및 추론 테스트
- [ ] 베이스라인 성능 평가 실행 (300 test samples)
- [ ] 결과 분석 및 저장

#### Phase 3: Fine-tuning (예정)
- [ ] LoRA configuration 설정
- [ ] Training 데이터 준비
- [ ] Fine-tuning 실행
- [ ] Validation 모니터링

#### Phase 4: 평가 및 비교 (예정)
- [ ] Fine-tuned 모델 평가
- [ ] Pre/Post 성능 비교
- [ ] 개선도 분석

## 다음 단계 (우선순위 순)

### 즉시 실행
1. [x] 전체 데이터셋 로드 및 통계 분석 스크립트 작성
2. [x] 데이터 품질 검증 (누락 필드, 인코딩 문제)
3. [x] Train/Val/Test split 생성 및 저장

### 단기 (1-2일) - 현재 단계
4. [x] Qwen2.5-7B-Instruct 모델 로드 테스트 (완료)
5. [x] Chat format 데이터 변환 스크립트
6. [x] 베이스라인 추론 스크립트 작성
7. [ ] 베이스라인 성능 평가 실행 (다음 단계)
8. [ ] 에러 분석 및 약점 파악

### 중기 (3-5일)
9. [ ] LoRA fine-tuning 스크립트 작성
10. [ ] 학습 실행 및 모니터링
11. [ ] 하이퍼파라미터 튜닝

### 장기 (1주 이상)
12. [ ] Fine-tuned 모델 평가
13. [ ] 성능 비교 및 분석
14. [ ] 결과 문서화 및 개선 방향 도출

## 참고사항

- Python 3.14와 일부 라이브러리 호환성 이슈 존재 (datasets 라이브러리 pickle 에러)
- 대안: HuggingFace Hub API를 통한 직접 파일 다운로드 사용
- Windows 환경에서 UTF-8 인코딩 주의 필요

## 개발 중 발생한 문제 및 해결 방법 (Troubleshooting Guide)

### 문제 해결 이력

| # | 문제 상황 | 에러 메시지 | 원인 분석 | 해결 방법 | 날짜 |
|---|----------|-----------|----------|----------|------|
| 1 | **datasets 라이브러리 로드 실패** | `TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given` | Python 3.14와 dill 라이브러리의 pickle 호환성 문제 | HuggingFace Hub API를 사용한 직접 파일 다운로드로 우회. `hf_hub_download()` 함수로 개별 JSON 파일 다운로드 | 2025-10-31 |
| 2 | **Windows 콘솔 출력 에러** | `UnicodeEncodeError: 'cp949' codec can't encode character` | Windows 기본 인코딩(cp949)이 이모지/특수문자 미지원 | 이모지 및 특수문자를 ASCII 텍스트로 교체 (예: ✓ → [OK], ❌ → [ERROR]) | 2025-10-31 |
| 3 | **Python 3.14 torch.compile 미지원** | `RuntimeError: torch.compile is not supported on Python 3.14+` | bitsandbytes 라이브러리가 `@torch.compile` 데코레이터 사용, Python 3.14에서 미지원 | `pip uninstall bitsandbytes -y`로 제거. 모델을 순수 BF16으로 로드 (quantization 없이) | 2025-10-31 |
| 4 | **gpt-oss-20b 모델 로드 실패** | `ModuleNotFoundError: Could not import module 'get_keys_to_not_convert'` | bitsandbytes import 시 torch.compile 에러가 연쇄적으로 발생 | bitsandbytes 제거 후 해결. gpt-oss-20b는 BF16만으로도 충분히 로드 가능 (17.95GB/24GB) | 2025-10-31 |
| 5 | **MoE 레이어 offloading 문제** | `KeyError: 'model.layers.20.mlp.experts.gate_up_proj'` | `device_map="auto"`가 일부 MoE expert 레이어를 CPU/disk로 오프로드했으나, 추론 시 제대로 로드되지 않음 | `device_map="cuda:0"`으로 변경하여 모든 레이어를 GPU에 강제 로드. RTX 3090 24GB로 충분함 | 2025-10-31 |
| 6 | **gpt-oss-20b GPU 메모리 부족** | `CUDA out of memory. Tried to allocate 1.98 GiB... 33.18 GiB is allocated` | MXFP4 quantized 모델이 로드 시 dequantization 과정에서 24GB 이상 메모리 필요. 피크 메모리 사용량 초과 | **모델 변경**: `meta-llama/Meta-Llama-3.1-8B-Instruct`로 전환 시도. 메모리 사용량 ~16GB로 예상 | 2025-10-31 |
| 7 | **Llama-3.1 Gated Model 인증 실패** | `Cannot access gated repo... You must have access to it and be authenticated` | Llama-3.1은 Meta의 gated model이라 HuggingFace 인증 필요. `hf auth login` 실행 필요 | **최종 모델 변경**: `Qwen/Qwen2.5-7B-Instruct`로 전환. Gated 아님, 한국어 성능 우수, 메모리 14.19GB로 안정적 | 2025-10-31 |

### 환경별 주요 이슈 및 권장사항

#### Python 버전 관련
- **Python 3.14 사용 시 주의사항**:
  - `torch.compile` 미지원으로 bitsandbytes 사용 불가
  - 대부분의 quantization 라이브러리 호환성 문제 가능
  - **권장**: 프로덕션 환경에서는 Python 3.11 또는 3.12 사용 고려

#### GPU 메모리 관리
- **gpt-oss-20b 메모리 사용량**:
  - BF16 (quantization 없음): ~18GB
  - 최소 필요 VRAM: 20GB
  - 권장 VRAM: 24GB 이상
- **device_map 설정**:
  - `device_map="auto"`: MoE 모델에서 문제 발생 가능
  - `device_map="cuda:0"`: 안정적이나 전체 모델이 GPU에 로드됨
  - 메모리 부족 시: 더 작은 모델 사용 (예: 7B, 13B)

#### Windows 환경 특이사항
- **콘솔 인코딩**: cp949 기본값, UTF-8 문자 출력 시 에러
  - 해결: ASCII 문자만 사용하거나 환경변수 설정
- **Symlink 미지원**: HuggingFace cache에서 경고 발생
  - 영향: 디스크 공간 더 사용, 기능상 문제 없음
  - 해결 (선택): Developer Mode 활성화 또는 관리자 권한 실행

#### 라이브러리 버전 호환성
```bash
# 검증된 버전 조합 (2025-10-31)
torch==2.9.0+cu129
transformers==4.57.1
accelerate==1.11.0
kernels==0.10.4
# bitsandbytes: Python 3.14에서 제외
```

### 성공적인 환경 구성
```python
# gpt-oss-20b 로드 (Python 3.14 호환)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # MoE 안정성
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
```

### 학습 포인트
1. **Python 버전 선택의 중요성**: 최신 버전이 항상 좋은 것은 아님. 생태계 호환성 고려 필요
2. **Quantization의 트레이드오프**: 메모리 절약 vs 호환성/안정성
3. **MoE 모델의 특수성**: 일반 모델과 다른 메모리 관리 필요
4. **Windows 개발 환경의 제약**: 인코딩, symlink 등 주의사항 존재
