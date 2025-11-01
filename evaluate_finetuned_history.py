"""
Fine-tuned Model Evaluation on Korean History Subset

This script:
1. Loads the base Qwen2.5-7B-Instruct model
2. Loads the fine-tuned LoRA adapter
3. Evaluates on Korean History test samples only
4. Saves detailed results for report generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from datetime import datetime

# ==================== Configuration ====================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER_PATH = "./models/qwen-click-lora/final"
TEST_DATA_PATH = "./data/splits/test.json"
OUTPUT_DIR = "./benchmarks/finetuned_history"

# ==================== Data Loading ====================

def load_test_data():
    """Test set 로드 및 Korean History 필터링"""
    print(f"[INFO] Loading test data from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Korean History만 필터링
    history_data = [sample for sample in data if sample.get('subcategory') == 'Korean History']
    print(f"[OK] Loaded {len(history_data)} Korean History samples (filtered from {len(data)} total)")
    return history_data

# ==================== Question Formatting ====================

def format_question(sample):
    """샘플을 chat format으로 변환"""
    # Build context
    context = ""
    if sample.get('paragraph', '').strip():
        context = f"지문:\n{sample['paragraph']}\n\n"

    # Build choices
    choices_text = ""
    for i, choice in enumerate(sample['choices'], 1):
        choices_text += f"{i}. {choice}\n"

    # System prompt
    system_prompt = "당신은 한국 역사에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."

    # Build messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"{context}질문: {sample['question']}\n\n선택지:\n{choices_text}\n정답을 선택해주세요."
        }
    ]

    return messages

# ==================== Model Inference ====================

def generate_answer(model, tokenizer, messages):
    """모델로부터 답변 생성"""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    if "<|im_start|>assistant\n" in full_response:
        assistant_response = full_response.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0]
    else:
        # Fallback: try to extract after the last user message
        user_content = messages[-1]['content']
        if user_content in full_response:
            assistant_response = full_response.split(user_content)[-1].strip()
        else:
            assistant_response = full_response

    return assistant_response.strip(), full_response

# ==================== Answer Extraction ====================

def extract_answer(response, choices):
    """응답에서 답변 추출"""
    response_lower = response.lower()

    # Method 1: Look for "답:" or "정답:"
    for prefix in ["답:", "정답:", "답 :", "정답 :"]:
        if prefix in response_lower:
            after_prefix = response_lower.split(prefix, 1)[1].strip()
            # Check if it starts with a number
            for i, choice in enumerate(choices, 1):
                if after_prefix.startswith(str(i)):
                    return choice

    # Method 2: Look for number at the start
    response_stripped = response.strip()
    if response_stripped and response_stripped[0].isdigit():
        num = int(response_stripped[0])
        if 1 <= num <= len(choices):
            return choices[num - 1]

    # Method 3: Find exact choice match
    for choice in choices:
        if choice in response:
            return choice

    # Method 4: Find partial match
    for choice in choices:
        # Remove common punctuation for comparison
        choice_clean = choice.replace(".", "").replace(",", "").strip()
        if choice_clean in response:
            return choice

    return None

# ==================== Evaluation ====================

def evaluate_model(model, tokenizer, test_data):
    """모델 평가 실행"""
    results = []
    correct = 0

    print(f"\n[INFO] Starting evaluation on {len(test_data)} samples...")
    print("=" * 80)

    for idx, sample in enumerate(test_data, 1):
        # Format question
        messages = format_question(sample)

        # Generate answer
        assistant_response, full_response = generate_answer(model, tokenizer, messages)

        # Extract predicted answer
        predicted_answer = extract_answer(assistant_response, sample['choices'])
        correct_answer = sample['answer']

        # Check correctness
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct += 1

        # Store result
        result = {
            "id": idx,
            "question": sample['question'],
            "paragraph": sample.get('paragraph', ''),
            "choices": sample['choices'],
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "assistant_response": assistant_response,
            "full_response": full_response,
            "is_correct": is_correct,
            "source": sample.get('source', 'Unknown'),
            "category": sample.get('category', ''),
            "subcategory": sample.get('subcategory', '')
        }
        results.append(result)

        # Progress update
        if idx % 10 == 0 or idx == len(test_data):
            current_acc = (correct / idx) * 100
            print(f"[Progress] {idx}/{len(test_data)} | Correct: {correct} | Accuracy: {current_acc:.2f}%")

    print("=" * 80)

    # Calculate final accuracy
    accuracy = (correct / len(test_data)) * 100 if test_data else 0

    summary = {
        "total_samples": len(test_data),
        "correct": correct,
        "incorrect": len(test_data) - correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return results, summary

# ==================== Main ====================

def main():
    print("=" * 80)
    print("[START] Fine-tuned Model Evaluation - Korean History")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load tokenizer
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("[OK] Tokenizer loaded\n")

    # 2. Load base model
    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("[OK] Base model loaded\n")

    # 3. Load LoRA adapter
    print(f"[INFO] Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    model.eval()
    print("[OK] LoRA adapter loaded\n")

    # 4. Load test data
    test_data = load_test_data()

    # 5. Run evaluation
    print("\n[EVAL] Running evaluation...")
    results, summary = evaluate_model(model, tokenizer, test_data)

    # 6. Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] Results saved to: {results_path}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary saved to: {summary_path}")

    # 7. Print summary
    print("\n" + "=" * 80)
    print("[SUMMARY] Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Correct: {summary['correct']}")
    print(f"Incorrect: {summary['incorrect']}")
    print(f"Accuracy: {summary['accuracy']:.2f}%")
    print("=" * 80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
