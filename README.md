# torch-CLIcK

Qwen2.5-7B-Instruct 모델을 CLIcK 데이터셋의 Korean History 분야에 Fine-tuning한 프로젝트입니다.

## 프로젝트 개요

**CLIcK** (Cultural and Linguistic Intelligence in Korean)는 한국 문화와 언어에 대한 AI 모델의 이해도를 평가하는 벤치마크입니다. 본 프로젝트는 Qwen2.5-7B-Instruct 모델을 LoRA 기법으로 Fine-tuning하여 Korean History 분야의 성능을 개선합니다.

### 주요 성과

| 모델 | 정확도 | 개선도 |
|------|--------|--------|
| Baseline (Qwen2.5-7B-Instruct) | 14.29% | - |
| **v2 (Fine-tuned, 최종)** | **52.38%** | **+38.09%p** |
| v1 (Fine-tuned) | 50.00% | +35.71%p |
| v3 (Overfitting) | 28.57% | +14.28%p |

## 프로젝트 구조

```
torch-CLIcK/
├── data/                          # 데이터셋
│   ├── raw/                       # 원본 데이터
│   │   ├── all_data.json         # 전체 CLIcK 데이터
│   │   └── file_statistics.csv   # 데이터 통계
│   └── splits/                    # 학습용 분할 데이터
│       ├── train.json            # 학습 데이터 (70%)
│       ├── val.json              # 검증 데이터 (15%)
│       └── test.json             # 테스트 데이터 (15%)
│
├── benchmarks/                    # 평가 결과
│   ├── baseline_history/         # Baseline 모델 결과
│   ├── finetuned_v2_history/     # v2 모델 결과 (최고 성능)
│   ├── finetuned_v3_history/     # v3 모델 결과 (과적합)
│   └── v2_retest/                # v2 재평가 (편차 분석)
│
├── docs/                          # 문서
│   ├── ai-background.md          # AI 학습 개념 설명
│   ├── v2-variance-report.md     # v2 평가 안정성 분석
│   ├── v3-analysis-report.md     # v3 과적합 실패 분석
│   └── ...                       # 기타 리포트
│
├── finetune_lora_improved.py     # v2 학습 스크립트 (최종)
├── finetune_v3_epoch10.py        # v3 학습 스크립트
├── evaluate_v2_history.py        # v2 평가 스크립트
├── create_splits.py              # 데이터 분할 스크립트
└── models/                        # 학습된 모델 (4.7GB, Git 제외)
```

## 주요 기술

### Fine-tuning 방법
- **기법**: LoRA (Low-Rank Adaptation)
- **베이스 모델**: Qwen/Qwen2.5-7B-Instruct
- **학습 파라미터**: 0.53% (38.9M / 7.3B)
- **GPU**: NVIDIA RTX 4090 24GB

### LoRA 설정
```python
LoRA Configuration:
- r: 16
- lora_alpha: 32
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- lora_dropout: 0.05
```

### v2 모델 (최종 모델)
```python
Training Configuration:
- Epochs: 3
- Learning Rate: 2e-4
- Batch Size: 2 (effective: 16 with gradient accumulation)
- LR Scheduler: cosine
- Optimizer: AdamW
- 학습 시간: ~86분
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install torch transformers peft datasets
```

### 2. 데이터 준비

```bash
python create_splits.py
```

데이터셋이 `data/splits/`에 생성됩니다:
- train.json: 1,407개 샘플
- val.json: 302개 샘플
- test.json: 303개 샘플

### 3. 모델 학습

```bash
python finetune_lora_improved.py
```

학습된 모델은 `models/qwen-click-lora-v2/final/`에 저장됩니다.

### 4. 모델 평가

```bash
python evaluate_v2_history.py
```

평가 결과는 `benchmarks/finetuned_v2_history/`에 저장됩니다.

## 모델 성능 분석

### v2 모델: 52.38% (최고 성능)

**강점**:
- ✅ 안정적인 학습 (Train-Val Gap: ~0.07)
- ✅ 완벽한 결정론적 모델 (평가 편차 0%)
- ✅ 효율적인 학습 시간 (86분)

**설정**:
- Epochs: 3
- Learning Rate: 2e-4
- 과적합 없음

상세 분석: [v2-variance-report.md](docs/v2-variance-report.md)

### v3 모델: 28.57% (실패 사례)

**문제점**:
- ❌ 심각한 과적합 (Train-Val Gap: 0.71)
- ❌ Validation Loss 증가 (0.75 → 0.85)
- ❌ v2 대비 -23.81%p 성능 하락

**원인**:
- Epochs 과다 (10 epochs)
- 최적 중단 시점 놓침 (Epoch 5.69)
- 작은 데이터셋 (1,407개) × 10 반복

상세 분석: [v3-analysis-report.md](docs/v3-analysis-report.md)

## 핵심 발견사항

### 1. Train Loss vs Validation Loss

과적합을 방지하려면 **Validation Loss 모니터링**이 필수입니다:

```
좋은 학습 (v2):
Epoch 1: Train Loss 1.20 | Val Loss 1.15 (Gap: 0.05)
Epoch 2: Train Loss 0.95 | Val Loss 0.90 (Gap: 0.05) ✓
Epoch 3: Train Loss 0.75 | Val Loss 0.68 (Gap: 0.07) ✓

나쁜 학습 (v3):
Epoch 5:  Train Loss 0.75 | Val Loss 0.75 (Gap: 0.00) ✓ 최적점
Epoch 7:  Train Loss 0.50 | Val Loss 0.81 (Gap: 0.31) ⚠️
Epoch 10: Train Loss 0.14 | Val Loss 0.85 (Gap: 0.71) ✗ 과적합
```

상세 설명: [ai-background.md](docs/ai-background.md)

### 2. 평가 안정성

v2 모델은 **완벽한 결정론적 모델**입니다:
- 재평가 시 100% 동일한 결과 (52.38%)
- `do_sample=False` 설정으로 재현 가능
- 평가 편차: 0.00%p

### 3. Epoch과 Learning Rate의 균형

| 설정 | Epochs | LR | 결과 |
|------|--------|----|----|
| v1 | 3 | 2e-4 | 50.00% |
| **v2** | **3** | **2e-4** | **52.38%** ✓ |
| v3 | 10 | 1e-4 | 28.57% (과적합) |

**교훈**: More Epochs ≠ Better Performance

## 문서

프로젝트의 모든 분석 문서는 `docs/` 폴더에 있습니다:

- **[ai-background.md](docs/ai-background.md)**: Epoch, Learning Rate, Train/Val Loss 개념 설명
- **[v2-variance-report.md](docs/v2-variance-report.md)**: v2 모델 평가 안정성 분석 (편차 0%)
- **[v3-analysis-report.md](docs/v3-analysis-report.md)**: v3 모델 과적합 실패 사례 분석
- **[baseline-report.md](docs/baseline-report.md)**: Baseline 모델 평가 리포트
- **[finetune-report.md](docs/finetune-report.md)**: Fine-tuning 결과 종합 리포트
- **[model-evolution-report.md](docs/model-evolution-report.md)**: 모델 발전 과정 정리

## 실험 결과 요약

### 데이터셋 통계

```
전체: 2,012개 샘플
- 학습: 1,407개 (70%)
- 검증: 302개 (15%)
- 테스트: 303개 (15%)

Korean History만: 42개 (테스트 세트)
```

### 모델 비교

| 모델 | Epochs | LR | 학습시간 | 정확도 | Train-Val Gap | 상태 |
|------|--------|-------|---------|--------|---------------|------|
| Baseline | - | - | - | 14.29% | - | - |
| v1 | 3 | 2e-4 | ~86분 | 50.00% | ~0.05 | ✓ |
| **v2** | **3** | **2e-4** | **~86분** | **52.38%** | **~0.07** | **✓ 최고** |
| v3 | 10 | 1e-4 | ~287분 | 28.57% | 0.71 | ✗ 과적합 |

## 기술 스택

- **모델**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning**: PEFT (LoRA)
- **프레임워크**: PyTorch, Transformers, PEFT
- **하드웨어**: NVIDIA RTX 3090 24GB
- **데이터셋**: CLIcK (Korean History subset)

## 제한사항

### Git 저장소에 포함되지 않은 파일

- `models/` 폴더 (4.7GB)
  - LoRA 어댑터 파일들
  - Git에서 제외 (.gitignore)
  - 필요시 별도 공유 필요

### 학습 데이터 크기

- Korean History: 42개 샘플 (테스트)
- 전체 학습 데이터: 1,407개 샘플
- 작은 데이터셋으로 인해 과적합 위험 높음

## 성능 개선 전략

현재 v2 모델의 52.38% 정확도를 개선하기 위한 체계적인 접근 방법입니다.

### 1. 데이터 관점 (가장 높은 영향력 ⭐⭐⭐⭐⭐)

#### 데이터 양 증가
```
현재 상황:
- 학습 데이터: 1,407개
- 테스트 데이터: 42개 (통계적으로 매우 작음)

개선 목표:
- 5,000-10,000개 학습 데이터 확보
- 더 많은 Korean History 문제 수집
- 데이터 증강 (질문 패러프레이징)

예상 효과: +5-10%p
```

#### 데이터 품질 개선
- 질문 표현 일관성 확보
- 오답 선택지 품질 개선
- 난이도 분포 균형화

### 2. 프롬프트 엔지니어링 (빠른 개선 ⭐⭐⭐)

#### Few-shot Learning 적용
```python
system_prompt = """당신은 한국 역사에 정통한 AI 어시스턴트입니다.

[예제 1]
질문: 고려 시대 무신정변의 배경은?
선택지:
1. 문신 우대 정책
2. 외적의 침입
3. 경제 위기
정답: 1. 문신 우대 정책

[예제 2]
질문: 조선 세종 때의 대표적 업적은?
선택지:
1. 훈민정음 창제
2. 경국대전 편찬
3. 삼국사기 저술
정답: 1. 훈민정음 창제

이제 다음 문제를 풀어주세요..."""

예상 효과: +2-3%p
구현 난이도: 낮음 (코드 몇 줄)
```

#### Chain-of-Thought 프롬프트
- 단계별 사고 과정 유도
- 역사적 맥락 분석 강조

### 3. 학습 전략 (중간 영향력 ⭐⭐⭐)

#### Ensemble Learning
```
방법: 서로 다른 seed로 학습한 3개 모델
- Majority voting으로 최종 답변 결정
- 안정적인 성능 향상

예상 효과: +3-5%p
추론 시간: 3배 증가 (Trade-off)
```

#### Curriculum Learning
```
학습 순서:
Epoch 1: 쉬운 문제만 (Baseline 정답률 >50%)
Epoch 2: 쉬운+중간 문제 (정답률 >30%)
Epoch 3: 전체 문제

효과: 작은 데이터셋에서 효율적 학습
예상 효과: +2-3%p
```

#### Hard Negative Mining
- Validation에서 자주 틀리는 문제 식별
- 해당 문제에 가중치 부여 또는 반복 학습

### 4. 모델 최적화 (중간 영향력 ⭐⭐)

#### LoRA 하이퍼파라미터 튜닝
```python
현재: r=16, lora_alpha=32

실험 방향:
1. r 증가 (16 → 32, 64)
   - 더 많은 파라미터 학습
   - 표현력 증가 (과적합 주의)

2. target_modules 확장
   - 현재: 7개 모듈
   - 추가: embed_tokens, lm_head

예상 효과: +1-3%p
```

#### Learning Rate 스케줄링 개선
```python
1. Warmup 추가
   warmup_steps = 100
   # 0 → 2e-4 (100 steps) → cosine decay

2. 더 낮은 최종 LR
   # 2e-4 → 1e-5 (현재는 0까지)

예상 효과: +1-2%p
```

### 5. 베이스 모델 변경 (높은 영향력 ⭐⭐⭐⭐)

#### 더 큰 모델
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct (GPU 메모리 제약)

#### 한국어 특화 모델
- SOLAR (Upstage)
- KoGPT
- Polyglot-Ko

**예상 효과**: +5-10%p (하드웨어 제약 고려)

### 개선 로드맵

#### Phase 1: 빠른 적용 (1-2일)
1. **Few-shot 프롬프트 추가**
   - 구현 난이도: 낮음
   - 예상 효과: +2-3%p
   - 목표 정확도: 54-56%

2. **Answer Extraction 개선**
   - 더 많은 패턴 인식
   - 예상 효과: +0.5%p

#### Phase 2: 중기 실험 (1주)
3. **Ensemble (3 models)**
   - seed만 바꿔서 3번 학습
   - 예상 효과: +3-5%p
   - 목표 정확도: 57-60%

4. **Curriculum Learning**
   - 학습 순서 최적화
   - 예상 효과: +2-3%p

#### Phase 3: 장기 개선 (1개월+)
5. **데이터 수집/증강**
   - 5,000개 목표
   - 예상 효과: +5-10%p
   - 목표 정확도: 65-70%

6. **한국어 특화 모델 실험**
   - SOLAR, KoGPT 등
   - 예상 효과: +3-7%p

### 권장 접근 순서

```
1순위: 프롬프트 개선 (Few-shot)
  ↓ 이유: 구현 쉽고 빠른 효과

2순위: Ensemble
  ↓ 이유: 안정적인 성능 향상

3순위: Curriculum Learning
  ↓ 이유: 작은 데이터셋에 효과적

4순위: 데이터 증강
  ↓ 이유: 궁극적으로 가장 효과적 (시간/비용 필요)
```

### 핵심 원칙

**"현재 데이터의 잠재력을 최대한 끌어낸 후, 데이터를 늘린다"**

1. 먼저 프롬프트 최적화 (빠르고 저비용)
2. 다음 학습 전략 개선 (중간 비용)
3. 마지막 데이터 수집 (시간과 비용 높음)

이 접근은 투자 대비 효과(ROI)를 극대화합니다.

### v1 vs v2 코드 차이 분석

**결론: 프롬프트만 변경, 하이퍼파라미터는 100% 동일**

| 항목 | v1 | v2 | 차이 |
|------|----|----|------|
| Epochs | 3 | 3 | 없음 |
| Learning Rate | 2e-4 | 2e-4 | 없음 |
| LoRA Config | r=16, α=32 | r=16, α=32 | 없음 |
| System Prompt | 단순 지시 | 상세한 형식 지시 + 예시 | **있음** |
| 성능 | 50.00% | 52.38% | **+2.38%p** |

**교훈**: 명확한 프롬프트만으로도 2.38%p 향상 가능. Prompt Engineering의 중요성을 보여주는 사례.

## 참고 자료

- [CLIcK Dataset Paper](https://arxiv.org/abs/2403.06412)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 작성되었습니다.

## 연락처

- GitHub: [@eunmin6](https://github.com/eunmin6)
- Repository: [torch-CLIcK](https://github.com/eunmin6/torch-CLIcK)

---

**마지막 업데이트**: 2025-11-01
