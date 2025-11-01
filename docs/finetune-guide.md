# Fine-tuning 완벽 가이드

**대상 독자**: AI/머신러닝에 익숙하지 않은 개발자
**목표**: LoRA를 사용한 LLM 파인튜닝의 전체 과정 이해

---

## 목차

1. [기본 개념 이해](#1-기본-개념-이해)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [Fine-tuning 코드 상세 설명](#3-fine-tuning-코드-상세-설명)
4. [학습된 모델 구조](#4-학습된-모델-구조)
5. [모델 사용 방법](#5-모델-사용-방법)
6. [문제 해결 가이드](#6-문제-해결-가이드)

---

## 1. 기본 개념 이해

### 1.1 LLM (Large Language Model)이란?

**간단히 말하면**: 방대한 텍스트 데이터로 학습된 AI 모델입니다.

```
입력: "한국의 수도는?"
LLM 처리 →
출력: "서울입니다."
```

**우리가 사용한 모델**: `Qwen2.5-7B-Instruct`
- **7B**: 70억 개의 파라미터 (모델의 크기)
- **Instruct**: 질문-답변 형식으로 학습된 버전

### 1.2 Fine-tuning이란?

**비유**: 이미 영어를 잘하는 사람에게 의학 용어를 추가로 가르치는 것

```
┌─────────────────┐
│  기본 LLM 모델   │  ← 일반적인 언어 이해 능력
│  (Qwen2.5-7B)   │
└────────┬────────┘
         │
         │ Fine-tuning (추가 학습)
         │
         ↓
┌─────────────────┐
│ 특화된 모델      │  ← 한국 역사/문화에 특화
│ (우리의 모델)    │
└─────────────────┘
```

**왜 필요한가?**
- 기본 모델은 일반적인 지식만 가짐
- 특정 도메인(한국 역사)에서는 성능이 낮음
- Fine-tuning으로 특정 분야의 성능 향상

### 1.3 LoRA (Low-Rank Adaptation)란?

**문제**: 70억 개 파라미터를 모두 다시 학습하면?
- 💾 메모리: 엄청나게 많이 필요 (수백 GB)
- ⏰ 시간: 며칠~몇 주 소요
- 💰 비용: GPU 비용 천문학적

**LoRA의 해결책**: 전체를 학습하지 말고, "작은 어댑터"만 추가로 학습

```
전체 모델 파라미터: 7,600,000,000 개
LoRA 학습 파라미터:    40,300,000 개 (0.53%)
                       ──────────
                       189배 효율적!
```

**비유**:
```
전통적 방법: 집 전체를 리모델링 (시간↑ 비용↑)
LoRA 방법:  작은 확장 공간만 추가 (시간↓ 비용↓)
```

**LoRA 작동 원리**:
```python
# 원래 방식 (Full Fine-tuning)
출력 = 거대한_행렬(입력)  # 7B 파라미터 전부 수정

# LoRA 방식
출력 = 원본_거대한_행렬(입력) + 작은_어댑터(입력)
       └─ 고정 (frozen)          └─ 학습 대상
```

---

## 2. 프로젝트 구조

### 2.1 전체 디렉토리 구조

```
torch-CLIcK/
│
├── data/                          # 데이터셋
│   └── splits/
│       ├── train.json             # 학습 데이터 (1,396 samples)
│       ├── val.json               # 검증 데이터 (299 samples)
│       └── test.json              # 테스트 데이터 (300 samples)
│
├── models/                        # 학습된 모델 저장소
│   └── qwen-click-lora/           # LoRA 어댑터 폴더
│       ├── checkpoint-100/        # 100 step 체크포인트
│       ├── checkpoint-200/        # 200 step 체크포인트
│       ├── checkpoint-264/        # 마지막 step 체크포인트
│       └── final/                 # 최종 모델 ⭐
│           ├── adapter_config.json    # LoRA 설정
│           ├── adapter_model.bin      # LoRA 가중치 (~160MB)
│           ├── tokenizer_config.json  # 토크나이저 설정
│           ├── tokenizer.json         # 토크나이저 데이터
│           └── special_tokens_map.json
│
├── benchmarks/                    # 평가 결과
│   ├── baseline_history/          # 기본 모델 평가
│   │   ├── results.json
│   │   └── summary.json
│   └── finetuned_history/         # 파인튜닝 모델 평가
│       ├── results.json
│       └── summary.json
│
├── finetune_lora.py              # 🔥 핵심: Fine-tuning 스크립트
├── evaluate_finetuned_history.py # 평가 스크립트
├── baseline-report.md            # 기본 모델 평가 리포트
├── finetune-report.md            # 파인튜닝 모델 평가 리포트
└── finetune-guide.md             # 이 문서
```

### 2.2 데이터 형식

**train.json 예시**:
```json
[
  {
    "id": 1,
    "question": "다음 중 고려시대의 문화재는?",
    "choices": [
      "석굴암",
      "첨성대",
      "팔만대장경",
      "석빙고"
    ],
    "answer": "팔만대장경",
    "paragraph": "고려시대에는...",
    "category": "Culture",
    "subcategory": "Korean History",
    "source": "KHB"
  },
  ...
]
```

---

## 3. Fine-tuning 코드 상세 설명

### 3.1 전체 흐름도

```
┌──────────────────┐
│ 1. 데이터 로드    │ train.json, val.json 읽기
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 2. 모델 로드      │ Qwen2.5-7B-Instruct 다운로드
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 3. LoRA 적용     │ 작은 어댑터 추가
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 4. 데이터 변환    │ 질문 → Chat 형식으로 포맷팅
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 5. 학습 실행      │ 3 epoch, ~83분
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 6. 모델 저장      │ LoRA 어댑터만 저장 (~160MB)
└──────────────────┘
```

### 3.2 코드 섹션별 설명

#### 📌 섹션 1: 설정 (Configuration)

```python
# ==================== Configuration ====================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./models/qwen-click-lora"
TRAIN_DATA_PATH = "./data/splits/train.json"
VAL_DATA_PATH = "./data/splits/val.json"

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # LoRA alpha (scaling factor)
    "target_modules": [         # Modules to apply LoRA
        "q_proj",    # Query projection
        "k_proj",    # Key projection
        "v_proj",    # Value projection
        "o_proj",    # Output projection
        "gate_proj", # Gate projection
        "up_proj",   # Up projection
        "down_proj"  # Down projection
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

**각 파라미터 의미**:

| 파라미터 | 의미 | 값 | 설명 |
|---------|------|-----|------|
| `r` | LoRA rank | 16 | 어댑터의 크기 (작을수록 효율↑, 성능↓) |
| `lora_alpha` | Scaling factor | 32 | 어댑터의 영향력 (보통 r의 2배) |
| `target_modules` | 적용 레이어 | 7개 | Transformer의 어텐션/FFN 레이어 |
| `lora_dropout` | Dropout 비율 | 0.05 | 과적합 방지 (5% 뉴런 무작위 비활성화) |

**시각화**:
```
Original Transformer Layer:
┌─────────────────────────────────┐
│  q_proj  k_proj  v_proj  o_proj │ ← 거대한 행렬 (frozen)
└─────────────────────────────────┘
                +
┌─────────────────────────────────┐
│     LoRA adapters (rank=16)      │ ← 작은 행렬 (trainable)
└─────────────────────────────────┘
```

#### 📌 섹션 2: 데이터 처리

```python
def format_question_for_training(sample):
    """
    샘플을 chat format으로 변환
    """
    # 1. 지문 구성
    context = ""
    if sample.get('paragraph', '').strip():
        context = f"지문:\n{sample['paragraph']}\n\n"

    # 2. 선택지 구성
    choices_text = ""
    for i, choice in enumerate(sample['choices'], 1):
        choices_text += f"{i}. {choice}\n"

    # 3. 정답 찾기
    correct_answer = sample['answer']
    answer_idx = None
    for i, choice in enumerate(sample['choices'], 1):
        if choice == correct_answer:
            answer_idx = i
            break

    # 4. Chat 형식으로 변환
    messages = [
        {
            "role": "system",
            "content": "당신은 한국 역사에 정통한 AI 어시스턴트입니다..."
        },
        {
            "role": "user",
            "content": f"{context}질문: {sample['question']}\n\n선택지:\n{choices_text}..."
        },
        {
            "role": "assistant",
            "content": f"{answer_idx}. {correct_answer}"
        }
    ]

    return messages
```

**변환 예시**:

**원본 데이터**:
```json
{
  "question": "고려시대의 문화재는?",
  "choices": ["석굴암", "첨성대", "팔만대장경", "석빙고"],
  "answer": "팔만대장경"
}
```

**변환 후 (Chat 형식)**:
```
<|im_start|>system
당신은 한국 역사에 정통한 AI 어시스턴트입니다...<|im_end|>
<|im_start|>user
질문: 고려시대의 문화재는?

선택지:
1. 석굴암
2. 첨성대
3. 팔만대장경
4. 석빙고

정답을 선택해주세요.<|im_end|>
<|im_start|>assistant
3. 팔만대장경<|im_end|>
```

#### 📌 섹션 3: Dataset 클래스

```python
class CLIcKDataset(TorchDataset):
    """PyTorch Dataset for CLIcK data"""

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []

        for sample in data:
            # 1. Chat 형식으로 변환
            messages = format_question_for_training(sample)

            # 2. 텍스트로 변환 (토크나이저의 chat template 사용)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # 3. 토큰화 (텍스트 → 숫자)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None
            )

            # 4. Labels 설정 (입력 = 정답)
            encoded["labels"] = encoded["input_ids"].copy()

            self.samples.append(encoded)
```

**토큰화 예시**:
```
텍스트: "고려시대의 문화재는?"
  ↓ tokenizer
토큰ID: [42, 1523, 8845, 3421, 9872, ...]
```

#### 📌 섹션 4: Data Collator (배치 패딩)

**문제**: 각 샘플의 길이가 다름
```
Sample 1: [1, 2, 3, 4, 5]           (길이: 5)
Sample 2: [6, 7, 8]                 (길이: 3)
Sample 3: [9, 10, 11, 12, 13, 14]   (길이: 6)
```

**해결**: 배치의 최대 길이에 맞춰 패딩 추가

```python
@dataclass
class DataCollatorForCausalLM:
    """Custom data collator that handles padding for causal LM"""
    tokenizer: Any

    def __call__(self, features):
        # 1. 배치에서 최대 길이 찾기
        max_length = max(len(f["input_ids"]) for f in features)

        # 2. 각 샘플을 max_length로 패딩
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)

            # input_ids 패딩 (pad_token_id 사용)
            padded_input_ids.append(ids + [pad_token_id] * padding_length)

            # labels 패딩 (-100 사용, loss 계산시 무시됨)
            padded_labels.append(lbls + [-100] * padding_length)

            # attention_mask (1=실제 토큰, 0=패딩)
            attention_mask.append([1]*len(ids) + [0]*padding_length)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(padded_labels)
        }
```

**패딩 결과**:
```
Sample 1: [1, 2, 3, 4, 5, 0]        (6으로 패딩)
Sample 2: [6, 7, 8, 0, 0, 0]        (6으로 패딩)
Sample 3: [9, 10, 11, 12, 13, 14]   (이미 6)

Attention Mask:
Sample 1: [1, 1, 1, 1, 1, 0]        (마지막은 패딩)
Sample 2: [1, 1, 1, 0, 0, 0]        (마지막 3개 패딩)
Sample 3: [1, 1, 1, 1, 1, 1]        (패딩 없음)
```

#### 📌 섹션 5: 학습 실행

```python
def main():
    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. 기본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # 메모리 절약 (16bit)
        device_map="auto",            # GPU 자동 할당
        trust_remote_code=True
    )

    # 3. Gradient Checkpointing (메모리 절약)
    model.gradient_checkpointing_enable()

    # 4. LoRA 적용
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # 5. 데이터셋 준비
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)

    # 6. Trainer 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch = 16
        learning_rate=2e-4,
        ...
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 7. 학습 시작!
    trainer.train()

    # 8. 모델 저장
    trainer.save_model(OUTPUT_DIR + "/final")
```

**학습 파라미터 설명**:

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `num_train_epochs` | 3 | 전체 데이터를 3번 반복 학습 |
| `per_device_train_batch_size` | 2 | GPU당 2개 샘플씩 처리 |
| `gradient_accumulation_steps` | 8 | 8번 누적 후 업데이트 (효과적 배치=16) |
| `learning_rate` | 2e-4 | 학습 속도 (0.0002) |
| `lr_scheduler_type` | cosine | 학습률 스케줄러 (점진적 감소) |
| `warmup_ratio` | 0.1 | 처음 10%는 학습률 천천히 증가 |
| `bf16` | True | BFloat16 정밀도 (메모리 절약) |

**학습 진행 과정**:
```
Epoch 1/3:
  Step   0: loss=2.1798
  Step  50: loss=1.4532
  Step 100: loss=1.2866  [Checkpoint 저장]

Epoch 2/3:
  Step 150: loss=1.1023
  Step 200: loss=0.9397  [Checkpoint 저장]

Epoch 3/3:
  Step 250: loss=0.8721
  Step 264: loss=0.8543  [최종 Checkpoint]

Training Complete!
Final validation loss: 1.0242
```

---

## 4. 학습된 모델 구조

### 4.1 폴더 구조 상세

```
models/qwen-click-lora/
│
├── checkpoint-100/              # 100 step 체크포인트
│   ├── adapter_config.json
│   ├── adapter_model.bin        (~160MB)
│   ├── optimizer.pt             (~321MB, Adam optimizer 상태)
│   ├── rng_state.pth            (난수 생성기 상태)
│   ├── scheduler.pt             (Learning rate scheduler 상태)
│   ├── trainer_state.json       (학습 상태 정보)
│   └── training_args.bin
│
├── checkpoint-200/              # 200 step 체크포인트
│   └── (동일한 구조)
│
├── checkpoint-264/              # 최종 step 체크포인트
│   └── (동일한 구조)
│
└── final/                       # ⭐ 실제 사용할 모델
    ├── adapter_config.json      # LoRA 설정 파일
    ├── adapter_model.bin        # ⭐ LoRA 가중치 (~160MB)
    ├── special_tokens_map.json  # 특수 토큰 매핑
    ├── tokenizer_config.json    # 토크나이저 설정
    ├── tokenizer.json           # 토크나이저 어휘
    └── vocab.json               # 어휘 사전
```

### 4.2 파일별 상세 설명

#### adapter_config.json
```json
{
  "alpha": 32,
  "auto_mapping": null,
  "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

**핵심 정보**:
- `base_model_name_or_path`: 원본 모델 (Qwen2.5-7B-Instruct)
- `r`: LoRA rank (16)
- `lora_alpha`: Scaling factor (32)
- `target_modules`: LoRA가 적용된 레이어들

#### adapter_model.bin

**크기**: ~160MB

**내용**: LoRA 어댑터의 가중치 (PyTorch 텐서)

**구조**:
```python
{
  'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': Tensor(16, 3584),
  'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight': Tensor(3584, 16),
  'base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight': Tensor(16, 512),
  ...
}
```

**LoRA 가중치 구조**:
```
Original Weight (3584 x 3584):
┌────────────────────────────┐
│                            │
│    거대한 행렬 (frozen)     │
│                            │
└────────────────────────────┘

LoRA Adapter (Low-Rank):
┌──────┐     ┌────────────────────────────┐
│      │     │                            │
│ A    │  ×  │            B               │  = ΔW (추가 가중치)
│(3584 │     │                     (16)   │
│ × 16)│     │                            │
└──────┘     └────────────────────────────┘

최종 가중치 = Original Weight + ΔW
```

### 4.3 크기 비교

| 구성 요소 | 크기 | 설명 |
|----------|------|------|
| **원본 모델** (Qwen2.5-7B) | ~15GB | 전체 모델 가중치 |
| **LoRA 어댑터** | ~160MB | 우리가 학습한 부분 |
| **비율** | **1.07%** | 원본의 1%만 저장! |

**장점**:
- ✅ 저장 공간 효율적 (15GB → 160MB)
- ✅ 배포 용이 (작은 파일만 공유)
- ✅ 여러 LoRA 어댑터 관리 가능

---

## 5. 모델 사용 방법

### 5.1 기본 사용법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    "./models/qwen-click-lora/final"
)

# 3. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct"
)

# 4. 추론 (Inference)
messages = [
    {"role": "system", "content": "당신은 한국 역사 전문가입니다."},
    {"role": "user", "content": "고려시대의 대표 문화재는?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 5.2 실전 예제: 한국사 퀴즈 봇

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class KoreanHistoryBot:
    def __init__(self, lora_path="./models/qwen-click-lora/final"):
        """한국사 퀴즈 봇 초기화"""
        print("모델 로딩 중...")

        # 기본 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # LoRA 어댑터 로드
        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_path
        )
        self.model.eval()

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct"
        )

        print("✓ 모델 로딩 완료!")

    def ask_question(self, question, choices):
        """
        한국사 문제 풀기

        Args:
            question (str): 질문
            choices (list): 선택지 리스트

        Returns:
            str: 모델의 답변
        """
        # 선택지 포맷팅
        choices_text = "\n".join([
            f"{i}. {choice}"
            for i, choice in enumerate(choices, 1)
        ])

        # 메시지 구성
        messages = [
            {
                "role": "system",
                "content": "당신은 한국 역사에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."
            },
            {
                "role": "user",
                "content": f"질문: {question}\n\n선택지:\n{choices_text}\n\n정답을 선택해주세요."
            }
        ]

        # Chat template 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 토큰화
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # 디코딩
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # assistant 응답만 추출
        if "<|im_start|>assistant\n" in full_response:
            assistant_response = full_response.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>")[0]
        else:
            assistant_response = full_response

        return assistant_response.strip()

# 사용 예시
if __name__ == "__main__":
    # 봇 초기화
    bot = KoreanHistoryBot()

    # 문제 1
    print("=" * 60)
    print("문제 1:")
    answer = bot.ask_question(
        question="다음 중 고려시대의 대표적인 문화재는?",
        choices=[
            "석굴암",
            "첨성대",
            "팔만대장경",
            "석빙고"
        ]
    )
    print(f"모델 답변: {answer}")

    # 문제 2
    print("\n" + "=" * 60)
    print("문제 2:")
    answer = bot.ask_question(
        question="한국전쟁이 발발한 연도는?",
        choices=[
            "1945년",
            "1948년",
            "1950년",
            "1953년"
        ]
    )
    print(f"모델 답변: {answer}")
```

**실행 결과**:
```
모델 로딩 중...
✓ 모델 로딩 완료!
============================================================
문제 1:
모델 답변: 3. 팔만대장경

============================================================
문제 2:
모델 답변: 3. 1950년
```

### 5.3 배치 처리 예제

여러 문제를 한 번에 처리:

```python
def batch_ask_questions(bot, questions_list):
    """
    여러 문제를 배치로 처리

    Args:
        bot: KoreanHistoryBot 인스턴스
        questions_list: [(question, choices), ...] 형식의 리스트

    Returns:
        list: 각 문제에 대한 답변 리스트
    """
    answers = []

    for i, (question, choices) in enumerate(questions_list, 1):
        print(f"처리 중... ({i}/{len(questions_list)})")
        answer = bot.ask_question(question, choices)
        answers.append(answer)

    return answers

# 사용 예시
questions = [
    ("고려시대 문화재는?", ["석굴암", "첨성대", "팔만대장경", "석빙고"]),
    ("한국전쟁 발발 연도는?", ["1945년", "1948년", "1950년", "1953년"]),
    ("조선시대 과거 제도는?", ["골품제", "음서제", "과거제", "호족제"]),
]

bot = KoreanHistoryBot()
answers = batch_ask_questions(bot, questions)

for q, a in zip(questions, answers):
    print(f"Q: {q[0]}")
    print(f"A: {a}\n")
```

### 5.4 REST API 서버 예제

Flask를 사용한 API 서버:

```python
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# 전역 변수로 모델 로드 (서버 시작시 한 번만)
print("서버 시작: 모델 로딩 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "./models/qwen-click-lora/final")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print("✓ 모델 로딩 완료!")

@app.route('/answer', methods=['POST'])
def answer_question():
    """
    POST /answer
    Body: {
        "question": "고려시대 문화재는?",
        "choices": ["석굴암", "첨성대", "팔만대장경", "석빙고"]
    }
    """
    data = request.json
    question = data.get('question')
    choices = data.get('choices')

    if not question or not choices:
        return jsonify({"error": "question과 choices가 필요합니다"}), 400

    # 선택지 포맷팅
    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices, 1)])

    # 메시지 구성
    messages = [
        {"role": "system", "content": "당신은 한국 역사 전문가입니다."},
        {"role": "user", "content": f"질문: {question}\n\n선택지:\n{choices_text}\n\n정답을 선택해주세요."}
    ]

    # 생성
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # assistant 응답 추출
    if "<|im_start|>assistant\n" in response:
        answer = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    else:
        answer = response

    return jsonify({
        "question": question,
        "choices": choices,
        "answer": answer
    })

@app.route('/health', methods=['GET'])
def health():
    """서버 상태 확인"""
    return jsonify({"status": "ok", "model": "qwen-click-lora"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**API 사용 예시**:

```bash
# 서버 시작
python api_server.py

# 다른 터미널에서 요청
curl -X POST http://localhost:5000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "고려시대 문화재는?",
    "choices": ["석굴암", "첨성대", "팔만대장경", "석빙고"]
  }'

# 응답
{
  "question": "고려시대 문화재는?",
  "choices": ["석굴암", "첨성대", "팔만대장경", "석빙고"],
  "answer": "3. 팔만대장경"
}
```

---

## 6. 문제 해결 가이드

### 6.1 일반적인 오류

#### ❌ 오류 1: Out of Memory (OOM)

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**원인**: GPU 메모리 부족

**해결 방법**:

1. **배치 크기 감소**:
```python
# finetune_lora.py 수정
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,  # 2 → 1로 감소
    "gradient_accumulation_steps": 16, # 8 → 16으로 증가 (효과적 배치 유지)
}
```

2. **LoRA rank 감소**:
```python
LORA_CONFIG = {
    "r": 8,  # 16 → 8로 감소
    "lora_alpha": 16,  # rank의 2배 유지
}
```

3. **Gradient Checkpointing 활성화** (이미 적용됨):
```python
model.gradient_checkpointing_enable()
```

#### ❌ 오류 2: Model Not Found

**증상**:
```
OSError: Qwen/Qwen2.5-7B-Instruct does not appear to be a valid model
```

**원인**: 모델이 HuggingFace에서 다운로드되지 않음

**해결 방법**:

1. **인터넷 연결 확인**

2. **수동 다운로드**:
```python
from huggingface_hub import snapshot_download

# 모델 전체 다운로드
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./models/Qwen2.5-7B-Instruct"
)

# 이후 로드시 local path 사용
model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen2.5-7B-Instruct"
)
```

3. **HuggingFace Token 설정** (private 모델인 경우):
```bash
huggingface-cli login
```

#### ❌ 오류 3: Pickle Error (Python 3.14)

**증상**:
```
TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given
```

**원인**: Python 3.14와 HuggingFace datasets 라이브러리 비호환

**해결 방법**:
- 이미 해결됨! `CLIcKDataset(TorchDataset)` 사용
- HuggingFace `datasets` 라이브러리 대신 PyTorch Dataset 사용

#### ❌ 오류 4: LoRA Adapter 로드 실패

**증상**:
```
ValueError: Can't find 'adapter_config.json' at './models/qwen-click-lora/final'
```

**원인**: LoRA 어댑터 경로 문제

**해결 방법**:

1. **경로 확인**:
```python
import os
print(os.path.exists("./models/qwen-click-lora/final/adapter_config.json"))
```

2. **절대 경로 사용**:
```python
import os
lora_path = os.path.abspath("./models/qwen-click-lora/final")
model = PeftModel.from_pretrained(base_model, lora_path)
```

### 6.2 성능 최적화

#### 🚀 최적화 1: Inference 속도 향상

```python
# 1. torch.compile 사용 (PyTorch 2.0+)
model = torch.compile(model)

# 2. Flash Attention 사용
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # Flash Attention 활성화
)

# 3. KV Cache 재사용
past_key_values = None
for i in range(10):  # 여러 문장 생성시
    outputs = model.generate(
        inputs,
        past_key_values=past_key_values,  # 이전 캐시 재사용
        use_cache=True
    )
    past_key_values = outputs.past_key_values
```

#### 🚀 최적화 2: 배치 처리

```python
def batch_generate(model, tokenizer, questions_batch, batch_size=4):
    """여러 질문을 배치로 처리"""
    all_responses = []

    for i in range(0, len(questions_batch), batch_size):
        batch = questions_batch[i:i+batch_size]

        # 배치 토큰화
        texts = [format_question(q) for q in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        # 배치 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        # 디코딩
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend(responses)

    return all_responses
```

#### 🚀 최적화 3: 양자화 (Quantization)

8bit 양자화로 메모리 절약:

```python
from transformers import BitsAndBytesConfig

# 8bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

# 모델 로드시 적용
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(model, lora_path)
```

**효과**:
- 메모리 사용량: 15GB → 7.5GB (50% 감소)
- 속도: 약간 느려질 수 있음 (~10-20%)

### 6.3 디버깅 팁

#### 🔍 팁 1: 학습 과정 모니터링

```python
# finetune_lora.py에 추가
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 출력시 호출"""
        if logs:
            print(f"\n[Step {state.global_step}]")
            print(f"  Loss: {logs.get('loss', 'N/A'):.4f}")
            print(f"  Learning Rate: {logs.get('learning_rate', 'N/A'):.2e}")
            if 'eval_loss' in logs:
                print(f"  Eval Loss: {logs['eval_loss']:.4f}")

# Trainer에 추가
trainer = Trainer(
    ...,
    callbacks=[CustomCallback()]
)
```

#### 🔍 팁 2: 모델 출력 확인

```python
def debug_model_output(model, tokenizer, question):
    """모델 출력 단계별 확인"""
    # 1. 입력 확인
    messages = format_question(question)
    print("=== Messages ===")
    print(messages)

    # 2. 토큰화 확인
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\n=== Formatted Text ===")
    print(text)

    # 3. 토큰 ID 확인
    inputs = tokenizer(text, return_tensors="pt")
    print("\n=== Token IDs ===")
    print(inputs['input_ids'])
    print(f"Length: {len(inputs['input_ids'][0])}")

    # 4. 생성 확인
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'].to(model.device),
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    print("\n=== Generated Token IDs ===")
    print(outputs.sequences[0])

    # 5. 디코딩 확인
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("\n=== Final Response ===")
    print(response)

    return response
```

#### 🔍 팁 3: GPU 메모리 모니터링

```python
import torch

def print_gpu_memory():
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")

# 사용 예시
print_gpu_memory()  # 모델 로드 전
model = load_model()
print_gpu_memory()  # 모델 로드 후
outputs = model.generate(...)
print_gpu_memory()  # 생성 후
```

---

## 7. 부록

### 7.1 용어 사전

| 용어 | 설명 |
|-----|------|
| **LLM** | Large Language Model, 대규모 언어 모델 |
| **Fine-tuning** | 사전 학습된 모델을 특정 태스크에 맞게 추가 학습 |
| **LoRA** | Low-Rank Adaptation, 효율적인 fine-tuning 기법 |
| **Rank** | LoRA에서 어댑터의 크기를 결정하는 파라미터 |
| **Epoch** | 전체 데이터셋을 한 번 학습하는 것 |
| **Batch Size** | 한 번에 처리하는 샘플 수 |
| **Learning Rate** | 학습 속도, 너무 크면 불안정, 너무 작으면 느림 |
| **Tokenizer** | 텍스트를 숫자(토큰 ID)로 변환하는 도구 |
| **Inference** | 학습된 모델로 예측하는 과정 |
| **Checkpoint** | 학습 중간에 저장된 모델 상태 |
| **Gradient Checkpointing** | 메모리 절약을 위한 기법 |
| **BFloat16** | 16bit 부동소수점, 메모리 절약 |

### 7.2 참고 자료

#### 공식 문서
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

#### 논문
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)

#### 튜토리얼
- [Fine-tune LLMs with LoRA (HuggingFace)](https://huggingface.co/blog/lora)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)

### 7.3 자주 묻는 질문 (FAQ)

**Q1: LoRA 어댑터만으로 모델을 실행할 수 있나요?**

A: 아니요. LoRA 어댑터는 기본 모델(Qwen2.5-7B-Instruct)과 함께 사용해야 합니다. 어댑터는 "추가 가중치"일 뿐이므로 원본 모델이 필요합니다.

**Q2: 여러 LoRA 어댑터를 동시에 사용할 수 있나요?**

A: 네, 가능합니다.
```python
# 어댑터 교체
model.unload()  # 기존 어댑터 제거
model = PeftModel.from_pretrained(base_model, "다른_어댑터_경로")
```

**Q3: Fine-tuning을 더 오래 하면 성능이 계속 올라가나요?**

A: 일정 시점 이후에는 과적합(overfitting)이 발생할 수 있습니다. Validation loss를 모니터링하면서 적절한 epoch 수를 찾아야 합니다.

**Q4: LoRA rank를 크게 하면 성능이 더 좋아지나요?**

A: 일반적으로 그렇지만, rank가 너무 크면:
- 메모리 사용량 증가
- 학습 시간 증가
- 과적합 위험 증가

보통 8~64 사이에서 실험합니다.

**Q5: 다른 모델에도 이 코드를 사용할 수 있나요?**

A: 네, 대부분의 HuggingFace 모델에 사용 가능합니다. `MODEL_NAME`만 변경하면 됩니다:
```python
# Llama 3 사용 예시
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Mistral 사용 예시
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
```

**Q6: 학습 데이터를 추가하고 싶습니다. 어떻게 하나요?**

A: `train.json`에 같은 형식으로 데이터를 추가하면 됩니다:
```json
{
  "question": "새로운 질문",
  "choices": ["선택지1", "선택지2", "선택지3", "선택지4"],
  "answer": "정답",
  "paragraph": "지문 (선택사항)",
  "category": "카테고리",
  "subcategory": "서브카테고리",
  "source": "출처"
}
```

---

## 8. 마무리

### 8.1 핵심 요약

이 가이드에서 배운 내용:

✅ **LoRA의 원리**: 전체 모델 대신 작은 어댑터만 학습
✅ **Fine-tuning 과정**: 데이터 준비 → 모델 로드 → 학습 → 저장
✅ **모델 구조**: 기본 모델(15GB) + LoRA 어댑터(160MB)
✅ **사용 방법**: 기본 모델 + 어댑터 로드 → 추론
✅ **실전 활용**: API 서버, 배치 처리, 최적화

### 8.2 다음 단계

이제 다음과 같은 작업을 시도해보세요:

1. **성능 개선**:
   - Epoch 수 증가 (3 → 5 또는 10)
   - Learning rate 조정 (2e-4 → 1e-4)
   - LoRA rank 변경 (16 → 32)

2. **데이터 확장**:
   - 추가 한국사 문제 수집
   - Data augmentation (paraphrasing)

3. **모델 실험**:
   - 더 큰 모델 사용 (Qwen2.5-14B, 72B)
   - 다른 모델 비교 (Llama 3, Mistral)

4. **배포**:
   - REST API 서버 구축
   - 웹 인터페이스 개발
   - Docker 컨테이너화

### 8.3 도움이 필요하면

- GitHub Issues: 문제 보고 및 질문
- HuggingFace Forums: 모델 관련 질문
- PyTorch Forums: 기술적 문제

---

**작성일**: 2025-10-31
**버전**: 1.0
**프로젝트**: torch-CLIcK Fine-tuning

**성과**:
- Baseline: 14.29% → Fine-tuned: 50.00% (3.5배 향상)
- 학습 시간: ~83분
- 모델 크기: ~160MB (LoRA 어댑터)

Happy Fine-tuning! 🚀
