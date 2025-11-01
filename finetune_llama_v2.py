"""
Llama3.1:8B LoRA Fine-tuning (v2 방식 적용)

Qwen v2에서 성공한 프롬프트 엔지니어링을 Llama3.1에 적용:
1. 상세한 형식 지시 ("반드시 '숫자. 내용' 형식으로만 답변")
2. System prompt에 형식 예시 포함
3. 모든 답변을 "숫자. 내용" 형식으로 통일
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset as TorchDataset
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List

# ==================== Configuration ====================

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./models/llama-click-lora-v2"
TRAIN_DATA_PATH = "./data/splits/train.json"
VAL_DATA_PATH = "./data/splits/val.json"

# LoRA Configuration (Qwen v2와 동일)
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
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

# Training Configuration (메모리 절약 버전)
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,  # 2->1 (메모리 절약)
    "per_device_eval_batch_size": 1,   # 2->1 (메모리 절약)
    "gradient_accumulation_steps": 16,  # 8->16 (effective batch=16 유지)
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "fp16": False,
    "bf16": True,
    "optim": "adamw_torch",
    "max_grad_norm": 1.0,
    # gradient_checkpointing은 model.gradient_checkpointing_enable()로 처리
}

# ==================== Data Processing ====================

def load_dataset(path):
    """Load and parse JSON dataset"""
    print(f"[INFO] Loading dataset from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] Loaded {len(data)} samples")
    return data

def format_question_for_training(sample):
    """
    v2 프롬프트 방식: 답변을 "숫자. 내용" 형식으로 강제
    """
    # Build context
    context = ""
    if sample.get('paragraph', '').strip():
        context = f"지문:\n{sample['paragraph']}\n\n"

    # Build choices
    choices_text = ""
    for i, choice in enumerate(sample['choices'], 1):
        choices_text += f"{i}. {choice}\n"

    # Find the correct answer index
    correct_answer = sample['answer']
    answer_idx = None
    for i, choice in enumerate(sample['choices'], 1):
        if choice == correct_answer:
            answer_idx = i
            break

    if answer_idx is None:
        # Fallback: try to find partial match
        for i, choice in enumerate(sample['choices'], 1):
            if correct_answer in choice or choice in correct_answer:
                answer_idx = i
                break

    # v2 System prompt (Qwen과 동일)
    system_prompt = """당신은 한국 역사에 정통한 AI 어시스턴트입니다.

질문에 답변할 때 다음 규칙을 따르세요:

[객관식 문제] - 선택지가 주어진 경우:
→ 반드시 "숫자. 정답내용" 형식으로만 답변
→ 예시: "3. 팔만대장경", "1. 고구려"

[서술형 문제] - 선택지가 없는 경우:
→ 자유롭게 상세히 설명하여 답변

현재 질문의 유형을 파악하여 적절한 형식으로 답변해주세요."""

    # User message
    user_message = f"{context}질문: {sample['question']}\n\n선택지:\n{choices_text}\n정답을 선택해주세요."

    # Assistant answer (Qwen과 동일한 형식)
    assistant_answer = f"{answer_idx}. {sample['answer']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_answer}
    ]

    return messages

class CLIcKDataset(TorchDataset):
    """Custom dataset for CLIcK data"""

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = format_question_for_training(sample)

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False
        )

        # Make sure labels is a list, not a tensor
        tokenized["labels"] = list(tokenized["input_ids"])

        return tokenized

@dataclass
class DataCollatorForCLIcK:
    """Data collator for CLIcK dataset"""
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels separately
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # Find max length
        max_length = max(len(ids) for ids in input_ids)

        # Pad manually
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, labs, mask in zip(input_ids, labels, attention_mask):
            padding_length = max_length - len(ids)

            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_labels.append(labs + [-100] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long)
        }

# ==================== Main Training ====================

def main():
    print("=" * 80)
    print("[START] Llama3.1:8B LoRA Fine-tuning (v2 방식)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load datasets
    train_data = load_dataset(TRAIN_DATA_PATH)
    val_data = load_dataset(VAL_DATA_PATH)

    # HuggingFace access token for gated models
    import os
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set your token as environment variable

    # 2. Load tokenizer and model
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[OK] Tokenizer loaded\n")

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    print("[OK] Base model loaded\n")

    # 3. Apply LoRA
    print("[INFO] Applying LoRA configuration...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] LoRA applied")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%\n")

    # 4. Create datasets
    print("[INFO] Creating datasets...")
    train_dataset = CLIcKDataset(train_data, tokenizer)
    val_dataset = CLIcKDataset(val_data, tokenizer)
    print(f"[OK] Datasets created")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples\n")

    # 5. Create data collator
    data_collator = DataCollatorForCLIcK(tokenizer=tokenizer)

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    # 7. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 8. Train
    print("[INFO] Starting training...")
    print("=" * 80)
    trainer.train()
    print("=" * 80)
    print("[OK] Training complete\n")

    # 9. Save final model
    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    print(f"[INFO] Saving final model to: {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print("[OK] Final model saved\n")

    # 10. Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": f"{100 * trainable_params / total_params:.2f}%",
        "training_completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    info_path = os.path.join(final_output_dir, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE] Llama3.1:8B LoRA Fine-tuning Complete!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: {final_output_dir}")

if __name__ == "__main__":
    main()
