"""
Qwen2.5-7B-Instruct LoRA Fine-tuning on CLIcK Dataset

This script:
1. Loads full train/val datasets
2. Formats as chat conversations
3. Fine-tunes using LoRA (Parameter-Efficient Fine-Tuning)
4. Saves the fine-tuned adapter
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset as TorchDataset
import json
import os
from datetime import datetime

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

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,  # Effective batch size = 2 * 8 = 16
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
    Format a sample into chat format for training.
    Returns the full conversation including the assistant's answer.
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

    # System prompt based on category
    if sample.get('subcategory') == 'Korean History':
        system_prompt = "당신은 한국 역사에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."
    elif sample.get('category') == 'Culture':
        system_prompt = "당신은 한국 문화에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."
    else:
        system_prompt = "당신은 한국어와 한국 문화에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."

    # Build messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"{context}질문: {sample['question']}\n\n선택지:\n{choices_text}\n정답을 선택해주세요."
        },
        {
            "role": "assistant",
            "content": f"{answer_idx}. {correct_answer}" if answer_idx else correct_answer
        }
    ]

    return messages

class CLIcKDataset(TorchDataset):
    """PyTorch Dataset for CLIcK data"""

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []

        print(f"[INFO] Tokenizing {len(data)} samples...")
        for sample in data:
            messages = format_question_for_training(sample)

            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None
            )

            # For causal LM, labels are the same as input_ids
            encoded["labels"] = encoded["input_ids"].copy()

            self.samples.append(encoded)

        print(f"[OK] Tokenized {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def prepare_dataset(data, tokenizer):
    """Convert raw data into tokenized format for training"""
    return CLIcKDataset(data, tokenizer)

# ==================== Main Training ====================

def main():
    print("="*80)
    print("[START] LoRA Fine-tuning on CLIcK Dataset")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load tokenizer
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[OK] Tokenizer loaded\n")

    # 2. Load model
    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("[OK] Model loaded\n")

    # 3. Enable gradient checkpointing for memory efficiency
    print("[INFO] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()

    # 4. Apply LoRA
    print("[INFO] Applying LoRA configuration...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] LoRA applied")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}\n")

    # 5. Load and prepare datasets
    print("[INFO] Loading training data...")
    train_data = load_dataset(TRAIN_DATA_PATH)
    val_data = load_dataset(VAL_DATA_PATH)

    print("\n[INFO] Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    print(f"[OK] Train dataset: {len(train_dataset)} samples")
    print(f"[OK] Val dataset: {len(val_dataset)} samples\n")

    # 6. Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )

    # 7. Setup data collator with padding
    from transformers import DefaultDataCollator
    from dataclasses import dataclass
    from typing import Any, Dict, List

    @dataclass
    class DataCollatorForCausalLM:
        """Custom data collator that handles padding for causal LM"""
        tokenizer: Any

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # Extract input_ids and labels
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]

            # Get max length in batch
            max_length = max(len(ids) for ids in input_ids)

            # Pad sequences
            padded_input_ids = []
            padded_labels = []
            attention_mask = []

            for ids, lbls in zip(input_ids, labels):
                padding_length = max_length - len(ids)

                # Pad input_ids with pad_token_id
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)

                # Pad labels with -100 (ignore index)
                padded_labels.append(lbls + [-100] * padding_length)

                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask.append([1] * len(ids) + [0] * padding_length)

            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long)
            }

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # 8. Create trainer
    print("[INFO] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    print("[OK] Trainer initialized\n")

    # 9. Start training
    print("="*80)
    print("[TRAIN] Starting fine-tuning...")
    print("="*80)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Total steps: ~{len(train_dataset) // (TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']) * TRAINING_CONFIG['num_train_epochs']}")
    print("="*80 + "\n")

    trainer.train()

    # 10. Save final model
    print("\n" + "="*80)
    print("[SAVE] Saving fine-tuned model...")
    print("="*80)

    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"[OK] Model saved to: {final_output_dir}")

    # 11. Print training summary
    print("\n" + "="*80)
    print("[DONE] Fine-tuning completed!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: {final_output_dir}")
    print("\nNext steps:")
    print("  1. Run evaluation on test set")
    print("  2. Compare with baseline performance")
    print("="*80)

if __name__ == "__main__":
    main()
