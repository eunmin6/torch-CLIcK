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
- **하드웨어**: NVIDIA RTX 4090 24GB
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

## 향후 계획

1. **Early Stopping 구현**
   - Validation Loss 기반 자동 중단
   - 최적 체크포인트 저장

2. **데이터 증강**
   - 더 많은 Korean History 샘플 확보
   - 데이터 다양성 증대

3. **하이퍼파라미터 최적화**
   - Learning Rate 스케줄링 개선
   - Batch Size 실험

4. **모델 경량화**
   - 더 작은 LoRA rank 실험
   - Quantization 적용

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
