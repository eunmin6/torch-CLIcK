# LLM Fine-tuning Toolkit

LLM 모델의 Fine-tuning을 위한 종합 가이드와 도구 모음입니다.

## 목차

1. [HuggingFace와 Ollama 모델 변환](#1-huggingface와-ollama-모델-변환)
2. [데이터셋 제작 구조 및 팁](#2-데이터셋-제작-구조-및-팁)
3. [Fine-tuning VRAM 크기 산출](#3-fine-tuning-vram-크기-산출)

---

## 1. HuggingFace와 Ollama 모델 변환

### 1.1 HuggingFace → Ollama 변환

HuggingFace에서 학습한 모델을 Ollama로 변환하여 로컬에서 빠르게 실행할 수 있습니다.

#### 방법 1: GGUF 변환 후 Ollama 등록

```bash
# 1. llama.cpp 설치
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 2. HuggingFace 모델을 GGUF로 변환
python convert.py /path/to/huggingface/model \
  --outfile model.gguf \
  --outtype q4_0  # 양자화 레벨 (q4_0, q5_0, q8_0 등)

# 3. Ollama Modelfile 생성
cat > Modelfile << EOF
FROM ./model.gguf

TEMPLATE """{{ .System }}
{{ .Prompt }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 4. Ollama에 모델 등록
ollama create my-custom-model -f Modelfile

# 5. 모델 실행
ollama run my-custom-model
```

#### 방법 2: LoRA 어댑터 병합 후 변환

<<<<<<< HEAD
=======
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
>>>>>>> 06dc0681a28decfcfb80fa7a8c70803209ad8a97
```python
# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 베이스 모델과 LoRA 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# LoRA 병합
merged_model = model.merge_and_unload()

# 병합된 모델 저장
merged_model.save_pretrained("./merged-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.save_pretrained("./merged-model")
```

그 후 위의 GGUF 변환 과정을 따릅니다.

#### 양자화 레벨 선택 가이드

| 레벨 | 크기 | 품질 | 추천 용도 |
|------|------|------|----------|
| q2_K | 가장 작음 | 낮음 | 테스트용 |
| q4_0 | 작음 | 중간 | 일반 사용 (권장) |
| q5_0 | 중간 | 좋음 | 품질 중시 |
| q8_0 | 큼 | 매우 좋음 | 최고 품질 |
| f16 | 가장 큼 | 원본 | 벤치마크 |

### 1.2 Ollama → HuggingFace 변환

Ollama 모델을 HuggingFace 형식으로 변환하는 것은 제한적입니다. GGUF는 양자화된 형식이므로 원래의 전체 정밀도로 복원할 수 없습니다.

**권장 워크플로우**: HuggingFace에서 학습 → Ollama로 변환하여 배포

---

## 2. 데이터셋 제작 구조 및 팁

### 2.1 데이터셋 구조

#### 기본 JSON 형식 (객관식 QA)

```json
{
  "id": "unique-id-001",
  "category": "Korean History",
  "subcategory": "Joseon Dynasty",
  "question": "조선 세종 때 창제된 한글의 원래 이름은?",
  "paragraph": "세종대왕은 1443년 훈민정음을 창제하였다. 이는 백성을 가르치는 바른 소리라는 뜻이다.",
  "choices": [
    "훈민정음",
    "한글",
    "언문",
    "정음"
  ],
  "answer": "훈민정음",
  "source": "한국사능력검정시험",
  "difficulty": "medium"
}
```

#### Chat 형식 (대화형)

```json
{
  "id": "chat-001",
  "messages": [
    {
      "role": "system",
      "content": "당신은 한국 역사 전문가입니다."
    },
    {
      "role": "user",
      "content": "고려 시대의 주요 불교 문화재에 대해 설명해주세요."
    },
    {
      "role": "assistant",
      "content": "고려 시대는 불교가 국교였으며, 팔만대장경, 석굴암 등..."
    }
  ]
}
```

### 2.2 데이터셋 분할 (Train/Val/Test)

```python
# create_splits.py
import json
import random
from pathlib import Path

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """데이터를 train/val/test로 분할"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.shuffle(data)
    total = len(data)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

# 사용 예시
with open('all_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train, val, test = split_dataset(data)

# 저장
Path('data/splits').mkdir(parents=True, exist_ok=True)
for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
    with open(f'data/splits/{split_name}.json', 'w', encoding='utf-8') as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)
```

### 2.3 데이터셋 제작 팁

#### ✅ 해야 할 것

1. **충분한 데이터 양 확보**
   - 최소: 1,000개 (기본 학습)
   - 권장: 5,000-10,000개 (안정적 학습)
   - 이상적: 50,000개+ (높은 성능)

2. **데이터 품질 관리**
   ```python
   # 데이터 검증 예시
   def validate_sample(sample):
       required_keys = ['question', 'choices', 'answer']

       # 필수 필드 확인
       for key in required_keys:
           if key not in sample:
               return False, f"Missing key: {key}"

       # 선택지에 정답이 포함되어 있는지 확인
       if sample['answer'] not in sample['choices']:
           return False, "Answer not in choices"

       # 중복 선택지 확인
       if len(sample['choices']) != len(set(sample['choices'])):
           return False, "Duplicate choices"

       return True, "OK"
   ```

3. **균형잡힌 데이터 분포**
   - 카테고리별 균등 분포
   - 난이도별 균등 분포
   - 답변 위치 균등 분포 (항상 1번이 정답이면 안 됨)

4. **Context 길이 관리**
   ```python
   def check_token_length(text, tokenizer, max_length=2048):
       tokens = tokenizer.encode(text)
       if len(tokens) > max_length:
           print(f"Warning: Text too long ({len(tokens)} tokens)")
       return len(tokens)
   ```

#### ❌ 피해야 할 것

1. **데이터 오염 (Data Leakage)**
   - Test set에 Train set과 유사한 문제 포함
   - Validation으로 모델 선택 후 같은 Validation으로 최종 평가

2. **편향된 데이터**
   - 특정 카테고리에 집중
   - 항상 같은 위치에 정답 배치
   - 너무 쉽거나 너무 어려운 문제만

3. **불완전한 데이터**
   - 정답이 선택지에 없음
   - 질문이 모호함
   - Context와 질문이 불일치

### 2.4 데이터 증강 (Data Augmentation)

```python
# 질문 패러프레이징 예시
def paraphrase_question(question):
    """GPT를 사용한 질문 패러프레이징"""
    paraphrases = [
        question,
        question.replace("무엇인가?", "무엇입니까?"),
        question.replace("~은?", "~을 고르시오."),
    ]
    return paraphrases

# Back-translation (한국어 → 영어 → 한국어)
from transformers import pipeline

translator_ko_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
translator_en_ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

def back_translate(text):
    en = translator_ko_en(text)[0]['translation_text']
    ko = translator_en_ko(en)[0]['translation_text']
    return ko
```

---

## 3. Fine-tuning VRAM 크기 산출

### 3.1 기본 공식

Fine-tuning에 필요한 VRAM은 다음 요소로 결정됩니다:

```
총 VRAM = 모델 크기 + 그래디언트 + 옵티마이저 상태 + 활성화 메모리 + 배치 메모리
```

### 3.2 전체 Fine-tuning (Full Fine-tuning)

**공식**:
```
VRAM (GB) ≈ 모델 파라미터 수 (B) × 20
```

**예시**:
- 7B 모델: 7 × 20 = **140 GB** (A100 80GB × 2)
- 13B 모델: 13 × 20 = **260 GB** (A100 80GB × 4)

**상세 계산** (7B 모델):
```
1. 모델 가중치 (FP16):      7B × 2 bytes = 14 GB
2. 그래디언트 (FP16):        7B × 2 bytes = 14 GB
3. 옵티마이저 (AdamW):
   - Momentum:               7B × 4 bytes = 28 GB
   - Variance:               7B × 4 bytes = 28 GB
4. 활성화 메모리:            ~20 GB (배치 크기에 따라)
5. 기타 오버헤드:            ~10 GB
----------------------------------------
총합:                        ~114 GB
```

### 3.3 LoRA Fine-tuning

**공식**:
```
VRAM (GB) ≈ 모델 파라미터 수 (B) × 4 + LoRA 파라미터 비율 × 16
```

**LoRA 파라미터 수 계산**:
```python
def calculate_lora_params(base_params, rank=16, alpha=32, num_layers=32, num_modules=7):
    """
    base_params: 베이스 모델 파라미터 수 (예: 7B)
    rank: LoRA rank (r)
    num_layers: Transformer 레이어 수
    num_modules: 적용할 모듈 수 (q,k,v,o,gate,up,down = 7)
    """
    # 각 모듈의 입력/출력 차원 (7B 모델 기준: 4096)
    hidden_dim = 4096

    # LoRA는 rank를 사용한 저차원 분해
    # 각 모듈: (hidden_dim × rank) + (rank × hidden_dim)
    params_per_module = 2 * hidden_dim * rank

    # 전체 LoRA 파라미터
    total_lora_params = params_per_module * num_modules * num_layers

    # 비율
    ratio = total_lora_params / (base_params * 1e9)

    return total_lora_params, ratio

# 예시: Qwen2.5-7B with LoRA r=16
lora_params, ratio = calculate_lora_params(7, rank=16)
print(f"LoRA 파라미터: {lora_params / 1e6:.1f}M ({ratio * 100:.2f}%)")
# 출력: LoRA 파라미터: 58.7M (0.84%)
```

**VRAM 예시** (7B 모델, LoRA r=16):
```
1. 베이스 모델 (FP16):       7B × 2 bytes = 14 GB
2. LoRA 파라미터 (FP32):     59M × 4 bytes = 0.24 GB
3. 그래디언트 (LoRA만):      59M × 2 bytes = 0.12 GB
4. 옵티마이저 (LoRA만):
   - Momentum:               59M × 4 bytes = 0.24 GB
   - Variance:               59M × 4 bytes = 0.24 GB
5. 활성화 메모리:            ~4 GB
6. 배치 메모리 (batch=2):    ~2 GB
----------------------------------------
총합:                        ~21 GB
```

### 3.4 주요 모델별 VRAM 요구사항 표

| 모델 | 파라미터 | Full Fine-tuning | LoRA (r=16) | LoRA (r=64) |
|------|---------|------------------|-------------|-------------|
| Qwen2.5-1.5B | 1.5B | 30 GB | 8 GB | 10 GB |
| Qwen2.5-7B | 7B | 140 GB | 21 GB | 28 GB |
| Llama3.1-8B | 8B | 160 GB | 24 GB | 32 GB |
| Qwen2.5-14B | 14B | 280 GB | 38 GB | 52 GB |
| Llama3.1-70B | 70B | 1400 GB | 180 GB | 260 GB |

### 3.5 메모리 절약 기법

#### 1. Gradient Checkpointing
```python
# 활성화 메모리를 ~50% 절감 (속도는 ~30% 느려짐)
model.gradient_checkpointing_enable()

# VRAM 절감: ~10-20 GB (7B 모델 기준)
```

#### 2. Batch Size & Gradient Accumulation
```python
# 메모리 절약 설정
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,    # 작은 배치
    "gradient_accumulation_steps": 16,   # 누적으로 effective batch=16 유지
}

# VRAM 절감: batch_size 2→1로 변경 시 ~2-4 GB
```

#### 3. Mixed Precision (BF16/FP16)
```python
# BF16 사용 (FP32 대비 50% 메모리 절감)
TRAINING_CONFIG = {
    "bf16": True,  # Ampere GPU 이상 (RTX 30xx, A100)
    # 또는
    "fp16": True,  # 구형 GPU
}

# VRAM 절감: 모델 크기의 ~50%
```

#### 4. DeepSpeed ZeRO
```python
# deepspeed_config.json
{
  "zero_optimization": {
    "stage": 3,  # Stage 0/1/2/3
    "offload_optimizer": {
      "device": "cpu"  # 옵티마이저를 CPU로
    },
    "offload_param": {
      "device": "cpu"  # 파라미터를 CPU로
    }
  }
}

# VRAM 절감: ZeRO-3 + offload 시 ~70%
# 단, 학습 속도는 크게 느려짐
```

### 3.6 실전 예시: RTX 4090 24GB로 학습 가능한 모델

| 모델 | 방법 | 설정 | 예상 VRAM | 가능 여부 |
|------|------|------|----------|----------|
| Qwen2.5-7B | LoRA | r=16, batch=2 | 21 GB | ✅ 가능 |
| Qwen2.5-7B | LoRA | r=16, batch=1, GC | 16 GB | ✅ 가능 |
| Llama3.1-8B | LoRA | r=16, batch=1, GC | 18 GB | ✅ 가능 |
| Qwen2.5-14B | LoRA | r=16, batch=1, GC | 28 GB | ❌ 불가 (4GB 초과) |
| Qwen2.5-14B | LoRA + ZeRO-3 | offload | ~20 GB | ✅ 가능 (느림) |

**GC = Gradient Checkpointing**

### 3.7 VRAM 계산 스크립트

```python
# vram_calculator.py
def calculate_vram(
    model_size_b,      # 모델 크기 (billions)
    method="lora",     # "full" or "lora"
    lora_rank=16,      # LoRA rank (method="lora"일 때)
    batch_size=2,
    gradient_checkpoint=False,
    precision="fp16"   # "fp32", "fp16", "bf16"
):
    """VRAM 요구량 계산"""

    # Precision에 따른 bytes
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]

    if method == "full":
        # Full fine-tuning
        model_mem = model_size_b * bytes_per_param
        gradient_mem = model_size_b * bytes_per_param
        optimizer_mem = model_size_b * 8  # AdamW: 2 states × 4 bytes

    elif method == "lora":
        # LoRA fine-tuning
        model_mem = model_size_b * 2  # 베이스 모델은 항상 FP16

        # LoRA 파라미터 수 계산 (간단한 추정)
        lora_params_m = (model_size_b * 1000) * 0.01 * (lora_rank / 16)
        lora_mem = lora_params_m * 4 / 1000  # LoRA는 FP32
        gradient_mem = lora_mem
        optimizer_mem = lora_mem * 2  # AdamW: 2 states

    # 활성화 메모리 (배치 크기에 비례)
    activation_mem = model_size_b * 0.5 * batch_size
    if gradient_checkpoint:
        activation_mem *= 0.5  # ~50% 절감

    # 기타 오버헤드
    overhead = 2

    total = model_mem + gradient_mem + optimizer_mem + activation_mem + overhead

    return {
        "model": model_mem,
        "gradient": gradient_mem,
        "optimizer": optimizer_mem,
        "activation": activation_mem,
        "overhead": overhead,
        "total": total
    }

# 예시
result = calculate_vram(
    model_size_b=7,
    method="lora",
    lora_rank=16,
    batch_size=1,
    gradient_checkpoint=True
)

print(f"Total VRAM: {result['total']:.1f} GB")
print(f"  - Model: {result['model']:.1f} GB")
print(f"  - Gradient: {result['gradient']:.1f} GB")
print(f"  - Optimizer: {result['optimizer']:.1f} GB")
print(f"  - Activation: {result['activation']:.1f} GB")
```

---

## 이전 연구 내역

본 프로젝트의 이전 연구 내역은 `docs/history-pre.md`에서 확인할 수 있습니다.

---

**마지막 업데이트**: 2025-11-01
