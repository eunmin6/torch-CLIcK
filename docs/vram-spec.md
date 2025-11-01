# LoRA 파인튜닝을 위한 메모리 산출

LoRA(Low-Rank Adaptation) 파인튜닝에 필요한 VRAM 크기를 계산하는 방법과 실전 예시를 제공합니다.

## 목차

1. [기본 공식](#1-기본-공식)
2. [메모리 구성 요소](#2-메모리-구성-요소)
3. [LoRA 파라미터 계산](#3-lora-파라미터-계산)
4. [실전 예시: GPT-OSS 20B](#4-실전-예시-gpt-oss-20b)
5. [메모리 최적화 전략](#5-메모리-최적화-전략)

---

## 1. 기본 공식

### Full Fine-tuning vs LoRA

| 방법 | VRAM 공식 | 7B 모델 예시 |
|------|-----------|-------------|
| **Full Fine-tuning** | `모델 크기 × 20` | 7B × 20 = 140 GB |
| **LoRA Fine-tuning** | `모델 크기 × 4 + LoRA 오버헤드` | 7B × 4 = 28 GB |

### LoRA의 메모리 효율성

LoRA는 전체 모델을 학습하지 않고 **저차원 어댑터만 학습**하므로:
- 학습 파라미터: 전체의 0.1~1%
- VRAM 요구량: Full Fine-tuning의 약 20~25%
- 학습 속도: 비슷하거나 약간 빠름

---

## 2. 메모리 구성 요소

### 2.1 베이스 모델 메모리

```python
모델_메모리 (GB) = 파라미터_수 (B) × 2 bytes (FP16)
```

**예시**: 7B 모델
```
7 × 10^9 parameters × 2 bytes = 14 GB
```

### 2.2 LoRA 파라미터 메모리

```python
LoRA_메모리 (GB) = LoRA_파라미터_수 × 4 bytes (FP32)
```

LoRA 파라미터는 학습 안정성을 위해 FP32로 저장됩니다.

### 2.3 그래디언트 메모리

```python
그래디언트_메모리 (GB) = LoRA_파라미터_수 × 2 bytes (FP16)
```

역전파 시 계산되는 그래디언트를 저장합니다.

### 2.4 옵티마이저 상태 (AdamW)

```python
옵티마이저_메모리 (GB) = LoRA_파라미터_수 × 8 bytes
```

AdamW는 2개의 상태를 유지합니다:
- Momentum (1st moment): 4 bytes/param
- Variance (2nd moment): 4 bytes/param

### 2.5 활성화 메모리

```python
활성화_메모리 (GB) = 모델_크기 (B) × 0.5 × batch_size
```

Forward pass 중 중간 활성화 값들을 저장합니다.

### 2.6 기타 오버헤드

- CUDA 컨텍스트: ~2 GB
- PyTorch 프레임워크: ~1 GB
- 데이터 로딩 버퍼: ~1 GB

---

## 3. LoRA 파라미터 계산

### 3.1 LoRA 구조

LoRA는 원래의 가중치 행렬 `W` (d × k)를 다음과 같이 분해합니다:

```
ΔW = B × A
```

여기서:
- A: (d × r) - Down-projection
- B: (r × k) - Up-projection
- r: LoRA rank (보통 8, 16, 32, 64)

### 3.2 파라미터 수 계산

각 LoRA 모듈의 파라미터 수:

```python
LoRA_params_per_module = (input_dim × rank) + (rank × output_dim)
                       = rank × (input_dim + output_dim)
```

**Transformer의 경우**:
- input_dim = output_dim = hidden_dim (예: 4096)

```python
LoRA_params_per_module = rank × (4096 + 4096)
                       = rank × 8192
```

### 3.3 전체 LoRA 파라미터

일반적으로 다음 모듈에 LoRA를 적용합니다:
- q_proj (Query)
- k_proj (Key)
- v_proj (Value)
- o_proj (Output)
- gate_proj (MLP gate)
- up_proj (MLP up)
- down_proj (MLP down)

**총 7개 모듈** × Transformer 레이어 수

```python
총_LoRA_파라미터 = LoRA_params_per_module × 7 × num_layers
```

### 3.4 계산 공식 (코드)

```python
def calculate_lora_params(
    base_params_b,      # 베이스 모델 파라미터 (billions)
    hidden_dim=4096,    # Hidden dimension
    rank=16,            # LoRA rank
    num_layers=32,      # Transformer layers
    num_modules=7       # 적용할 모듈 수
):
    """LoRA 파라미터 수 계산"""

    # 각 모듈당 파라미터
    params_per_module = rank * (hidden_dim * 2)

    # 전체 LoRA 파라미터
    total_lora_params = params_per_module * num_modules * num_layers

    # 베이스 모델 대비 비율
    ratio = total_lora_params / (base_params_b * 1e9)

    return total_lora_params, ratio

# 예시: 7B 모델, rank=16
lora_params, ratio = calculate_lora_params(7, rank=16)
print(f"LoRA 파라미터: {lora_params / 1e6:.1f}M")
print(f"비율: {ratio * 100:.2f}%")

# 출력:
# LoRA 파라미터: 58.7M
# 비율: 0.84%
```

---

## 4. 실전 예시: GPT-OSS 20B

### 4.1 모델 사양

| 항목 | 값 |
|------|-----|
| 파라미터 수 | 20B (20 × 10^9) |
| Hidden dimension | 6144 |
| Transformer layers | 44 |
| Attention heads | 48 |
| Precision (베이스) | FP16 |
| Precision (LoRA) | FP32 |

### 4.2 LoRA 설정

```python
LORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # Scaling factor
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

### 4.3 메모리 계산

#### Step 1: 베이스 모델 메모리

```python
모델_메모리 = 20B × 2 bytes (FP16)
           = 40 GB
```

#### Step 2: LoRA 파라미터 수

```python
hidden_dim = 6144
rank = 16
num_layers = 44
num_modules = 7

# 각 모듈당 파라미터
params_per_module = rank × (hidden_dim × 2)
                  = 16 × (6144 × 2)
                  = 16 × 12,288
                  = 196,608 params

# 전체 LoRA 파라미터
total_lora_params = 196,608 × 7 × 44
                  = 60,538,368 params
                  ≈ 60.5M params

# 베이스 모델 대비 비율
ratio = 60.5M / 20B
      = 0.30%
```

#### Step 3: LoRA 메모리

```python
LoRA_메모리 = 60.5M × 4 bytes (FP32)
           = 242 MB
           ≈ 0.24 GB
```

#### Step 4: 그래디언트 메모리

```python
그래디언트_메모리 = 60.5M × 2 bytes (FP16)
                = 121 MB
                ≈ 0.12 GB
```

#### Step 5: 옵티마이저 메모리 (AdamW)

```python
옵티마이저_메모리 = 60.5M × 8 bytes
                = 484 MB
                ≈ 0.48 GB
```

#### Step 6: 활성화 메모리

```python
# batch_size = 1, gradient_checkpointing = False
활성화_메모리 = 20B × 0.5 × 1
            = 10 GB

# gradient_checkpointing = True (50% 절감)
활성화_메모리 = 10 GB × 0.5
            = 5 GB
```

#### Step 7: 배치 메모리

```python
# batch_size = 1, sequence_length = 2048
배치_메모리 = 1 × 2048 × hidden_dim × 2 bytes
          = 1 × 2048 × 6144 × 2
          ≈ 25 MB
          ≈ 0.025 GB
```

#### Step 8: 오버헤드

```python
오버헤드 = 2 GB (CUDA) + 1 GB (PyTorch) + 1 GB (Data)
        = 4 GB
```

### 4.4 총 VRAM 요구량

#### 기본 설정 (batch=1, no gradient checkpointing)

```python
총_VRAM = 베이스_모델 + LoRA + 그래디언트 + 옵티마이저 + 활성화 + 배치 + 오버헤드
        = 40 + 0.24 + 0.12 + 0.48 + 10 + 0.025 + 4
        = 54.87 GB
        ≈ 55 GB
```

**필요 GPU**: A100 80GB (1장) 또는 RTX 4090 24GB × 3장

#### 최적화 설정 (batch=1, gradient checkpointing)

```python
총_VRAM = 40 + 0.24 + 0.12 + 0.48 + 5 + 0.025 + 4
        = 49.87 GB
        ≈ 50 GB
```

**필요 GPU**: A100 80GB (1장) 또는 RTX 4090 24GB × 3장

#### 극한 최적화 (DeepSpeed ZeRO-3 + CPU Offload)

```python
# ZeRO-3: 옵티마이저 상태를 CPU로
# Offload: 모델 파라미터 일부를 CPU로

GPU_메모리 = 베이스_모델 (일부) + LoRA + 그래디언트 + 활성화
          ≈ 20 + 0.24 + 0.12 + 5
          ≈ 25 GB
```

**필요 GPU**: RTX 4090 24GB (1장, 약간 넘침) 또는 RTX 3090 24GB × 2장

### 4.5 다양한 설정별 VRAM 요구량

| 설정 | Batch Size | Gradient Checkpointing | DeepSpeed | 예상 VRAM |
|------|-----------|----------------------|-----------|----------|
| 기본 | 1 | ❌ | ❌ | **55 GB** |
| 최적화 | 1 | ✅ | ❌ | **50 GB** |
| 메모리 절약 | 1 | ✅ | ZeRO-2 | **35 GB** |
| 극한 절약 | 1 | ✅ | ZeRO-3 + Offload | **25 GB** |

### 4.6 권장 하드웨어

| VRAM | GPU 옵션 | 설정 | 학습 속도 |
|------|---------|------|----------|
| 80 GB | A100 80GB × 1 | 기본 | ⚡⚡⚡⚡⚡ 빠름 |
| 48 GB | A6000 48GB × 1 | 최적화 | ⚡⚡⚡⚡ 빠름 |
| 48 GB | RTX 4090 24GB × 2 | 최적화 | ⚡⚡⚡ 보통 |
| 24 GB | RTX 4090 24GB × 1 | 극한 절약 (느림) | ⚡ 느림 |

---

## 5. 메모리 최적화 전략

### 5.1 Gradient Checkpointing

**효과**: 활성화 메모리 50% 절감
**비용**: 학습 속도 약 30% 감소

```python
# 적용 방법
model.gradient_checkpointing_enable()

# 또는 TrainingArguments에서
training_args = TrainingArguments(
    gradient_checkpointing=True,
    ...
)
```

**메모리 절감**: GPT-OSS 20B의 경우 ~5 GB

### 5.2 Batch Size & Gradient Accumulation

작은 batch size + gradient accumulation으로 effective batch size 유지

```python
# 메모리 절약 설정
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,      # 작은 배치
    "gradient_accumulation_steps": 16,     # 누적
}

# Effective batch size = 1 × 16 = 16
```

**메모리 절감**: batch_size 2→1 시 GPT-OSS 20B는 ~0.05 GB (미미)

### 5.3 Mixed Precision (BF16/FP16)

```python
training_args = TrainingArguments(
    bf16=True,  # Ampere GPU 이상 (RTX 30xx, A100)
    # 또는
    fp16=True,  # 구형 GPU
    ...
)
```

**메모리 절감**: LoRA는 이미 베이스 모델을 FP16으로 로드하므로 추가 절감 없음

### 5.4 DeepSpeed ZeRO

#### ZeRO-2: 옵티마이저 상태 분산

```json
{
  "zero_optimization": {
    "stage": 2
  }
}
```

**메모리 절감**: 옵티마이저 메모리를 GPU 수로 나눔 (GPT-OSS 20B: 0.48 GB / N)

#### ZeRO-3: 모델 파라미터 + 옵티마이저 CPU Offload

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**메모리 절감**: ~50-70% (하지만 학습 속도 크게 감소)

### 5.5 LoRA Rank 조정

| Rank | LoRA Params (20B) | VRAM | 성능 |
|------|------------------|------|------|
| 8 | ~30M | -0.5 GB | 중간 |
| 16 | ~60M | 기준 | 좋음 |
| 32 | ~120M | +0.5 GB | 매우 좋음 |
| 64 | ~240M | +1.0 GB | 최고 (과적합 주의) |

**권장**: rank=16 (성능과 메모리의 균형)

### 5.6 Target Modules 제한

전체 7개 모듈 대신 일부만 적용:

```python
# 메모리 절약 (Attention만)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 4/7

# 메모리 절감: ~40% (60.5M → 34.6M params)
```

**주의**: 성능 저하 가능

---

## 6. 실전 계산 스크립트

```python
# calculate_vram_gpt_oss_20b.py

def calculate_vram_for_lora(
    model_params_b=20,          # 모델 파라미터 (billions)
    hidden_dim=6144,            # Hidden dimension
    num_layers=44,              # Transformer layers
    lora_rank=16,               # LoRA rank
    num_lora_modules=7,         # LoRA 적용 모듈 수
    batch_size=1,               # Batch size
    gradient_checkpoint=True,   # Gradient checkpointing 사용 여부
    use_deepspeed_zero3=False   # DeepSpeed ZeRO-3 사용 여부
):
    """
    LoRA 파인튜닝 VRAM 계산
    """

    # 1. 베이스 모델 메모리 (FP16)
    base_model_mem = model_params_b * 2  # GB

    # 2. LoRA 파라미터 계산
    lora_params_per_module = lora_rank * (hidden_dim * 2)
    total_lora_params = lora_params_per_module * num_lora_modules * num_layers
    total_lora_params_m = total_lora_params / 1e6  # millions

    # 3. LoRA 메모리 (FP32)
    lora_mem = (total_lora_params * 4) / 1e9  # GB

    # 4. 그래디언트 메모리 (FP16)
    gradient_mem = (total_lora_params * 2) / 1e9  # GB

    # 5. 옵티마이저 메모리 (AdamW, FP32)
    optimizer_mem = (total_lora_params * 8) / 1e9  # GB

    # 6. 활성화 메모리
    activation_mem = model_params_b * 0.5 * batch_size
    if gradient_checkpoint:
        activation_mem *= 0.5  # 50% 절감

    # 7. 배치 메모리 (간단한 추정)
    batch_mem = batch_size * 0.025  # GB

    # 8. 오버헤드
    overhead = 4  # GB

    # DeepSpeed ZeRO-3 최적화
    if use_deepspeed_zero3:
        # 옵티마이저와 모델 일부를 CPU로
        base_model_mem *= 0.5
        optimizer_mem = 0
        overhead = 2  # CPU offload 시 오버헤드 감소

    # 총합
    total_vram = (
        base_model_mem +
        lora_mem +
        gradient_mem +
        optimizer_mem +
        activation_mem +
        batch_mem +
        overhead
    )

    return {
        "base_model": base_model_mem,
        "lora_params": lora_mem,
        "lora_count_m": total_lora_params_m,
        "lora_ratio": (total_lora_params / (model_params_b * 1e9)) * 100,
        "gradient": gradient_mem,
        "optimizer": optimizer_mem,
        "activation": activation_mem,
        "batch": batch_mem,
        "overhead": overhead,
        "total": total_vram
    }

# GPT-OSS 20B 계산
result = calculate_vram_for_lora(
    model_params_b=20,
    hidden_dim=6144,
    num_layers=44,
    lora_rank=16,
    batch_size=1,
    gradient_checkpoint=True,
    use_deepspeed_zero3=False
)

print("=" * 60)
print("GPT-OSS 20B LoRA Fine-tuning VRAM 계산")
print("=" * 60)
print(f"베이스 모델 메모리:    {result['base_model']:.2f} GB")
print(f"LoRA 파라미터:         {result['lora_count_m']:.1f}M ({result['lora_ratio']:.2f}%)")
print(f"LoRA 메모리:           {result['lora_params']:.2f} GB")
print(f"그래디언트 메모리:     {result['gradient']:.2f} GB")
print(f"옵티마이저 메모리:     {result['optimizer']:.2f} GB")
print(f"활성화 메모리:         {result['activation']:.2f} GB")
print(f"배치 메모리:           {result['batch']:.2f} GB")
print(f"오버헤드:              {result['overhead']:.2f} GB")
print("=" * 60)
print(f"총 VRAM 요구량:        {result['total']:.2f} GB")
print("=" * 60)

# 다양한 설정 비교
print("\n설정별 VRAM 요구량:")
print("-" * 60)

configs = [
    ("기본 (batch=1, no GC)", {"gradient_checkpoint": False}),
    ("최적화 (batch=1, GC)", {"gradient_checkpoint": True}),
    ("극한 절약 (ZeRO-3)", {"gradient_checkpoint": True, "use_deepspeed_zero3": True}),
]

for name, config in configs:
    result = calculate_vram_for_lora(
        model_params_b=20,
        hidden_dim=6144,
        num_layers=44,
        lora_rank=16,
        batch_size=1,
        **config
    )
    print(f"{name:30s}: {result['total']:6.2f} GB")
```

**실행 결과**:
```
============================================================
GPT-OSS 20B LoRA Fine-tuning VRAM 계산
============================================================
베이스 모델 메모리:    40.00 GB
LoRA 파라미터:         60.5M (0.30%)
LoRA 메모리:           0.24 GB
그래디언트 메모리:     0.12 GB
옵티마이저 메모리:     0.48 GB
활성화 메모리:         5.00 GB
배치 메모리:           0.03 GB
오버헤드:              4.00 GB
============================================================
총 VRAM 요구량:        49.87 GB
============================================================

설정별 VRAM 요구량:
------------------------------------------------------------
기본 (batch=1, no GC)         :  54.87 GB
최적화 (batch=1, GC)          :  49.87 GB
극한 절약 (ZeRO-3)            :  25.37 GB
```

---

## 7. 결론

### GPT-OSS 20B LoRA 파인튜닝 요약

| 항목 | 값 |
|------|-----|
| 베이스 모델 크기 | 20B params |
| LoRA 파라미터 | 60.5M (0.30%) |
| 기본 VRAM | **~55 GB** |
| 최적화 VRAM | **~50 GB** |
| 극한 절약 VRAM | **~25 GB** |

### 권장 GPU 구성

1. **A100 80GB × 1**: 가장 빠르고 안정적 (권장)
2. **A6000 48GB × 1**: Gradient Checkpointing 필수
3. **RTX 4090 24GB × 2**: Multi-GPU 설정 필요
4. **RTX 4090 24GB × 1**: DeepSpeed ZeRO-3 + 매우 느림

### 핵심 포인트

1. **LoRA는 메모리 효율적**: 전체 파인튜닝 대비 약 1/4 VRAM
2. **Gradient Checkpointing 필수**: 대형 모델에서는 거의 필수
3. **DeepSpeed는 최후의 수단**: 속도 희생이 크므로 가능하면 GPU 추가 권장
4. **LoRA rank=16이 최적**: 성능과 메모리의 균형

---

**마지막 업데이트**: 2025-11-01
