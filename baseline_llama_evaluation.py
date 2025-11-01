"""
Llama3.1:8B Baseline Evaluation

Llama3.1:8B 모델의 Korean History 초기 성능을 측정합니다.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from datetime import datetime

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEST_DATA_PATH = "./data/splits/test.json"
OUTPUT_DIR = "./benchmarks/baseline_llama"

def load_test_data():
    """Test set 로드 및 Korean History 필터링"""
    print(f"[INFO] Loading test data from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    history_data = [sample for sample in data if sample.get('subcategory') == 'Korean History']
    print(f"[OK] Loaded {len(history_data)} Korean History samples")
    return history_data

def format_question(sample):
    """질문을 Llama chat format으로 변환"""
    context = ""
    if sample.get('paragraph', '').strip():
        context = f"지문:\n{sample['paragraph']}\n\n"

    choices_text = ""
    for i, choice in enumerate(sample['choices'], 1):
        choices_text += f"{i}. {choice}\n"

    system_prompt = "당신은 한국 역사에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}질문: {sample['question']}\n\n선택지:\n{choices_text}\n정답을 선택해주세요."}
    ]

    return messages

def generate_answer(model, tokenizer, messages):
    """모델로 답변 생성"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        if "<|eot_id|>" in assistant_response:
            assistant_response = assistant_response.split("<|eot_id|>")[0]
    else:
        user_content = messages[-1]['content']
        if user_content in full_response:
            assistant_response = full_response.split(user_content)[-1].strip()
        else:
            assistant_response = full_response

    return assistant_response.strip(), full_response

def extract_answer(response, choices):
    """응답에서 답변 추출"""
    response_lower = response.lower()

    # 1. "답:" or "정답:"
    for prefix in ["답:", "정답:", "답 :", "정답 :"]:
        if prefix in response_lower:
            after_prefix = response_lower.split(prefix, 1)[1].strip()
            for i, choice in enumerate(choices, 1):
                if after_prefix.startswith(str(i)):
                    return choice

    # 2. 시작이 숫자
    response_stripped = response.strip()
    if response_stripped and response_stripped[0].isdigit():
        num = int(response_stripped[0])
        if 1 <= num <= len(choices):
            return choices[num - 1]

    # 3. 정확한 매칭
    for choice in choices:
        if choice in response:
            return choice

    # 4. 부분 매칭
    for choice in choices:
        choice_clean = choice.replace(".", "").replace(",", "").strip()
        if choice_clean in response:
            return choice

    return None

def evaluate_model(model, tokenizer, test_data):
    """모델 평가"""
    results = []
    correct = 0

    print(f"\n[INFO] Starting evaluation on {len(test_data)} samples...")
    print("=" * 80)

    for idx, sample in enumerate(test_data, 1):
        messages = format_question(sample)
        assistant_response, full_response = generate_answer(model, tokenizer, messages)
        predicted_answer = extract_answer(assistant_response, sample['choices'])
        correct_answer = sample['answer']
        is_correct = (predicted_answer == correct_answer)

        if is_correct:
            correct += 1

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

        if idx % 10 == 0 or idx == len(test_data):
            current_acc = (correct / idx) * 100
            print(f"[Progress] {idx}/{len(test_data)} | Correct: {correct} | Accuracy: {current_acc:.2f}%")

    print("=" * 80)

    accuracy = (correct / len(test_data)) * 100 if test_data else 0

    summary = {
        "model": MODEL_NAME,
        "total_samples": len(test_data),
        "correct": correct,
        "incorrect": len(test_data) - correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return results, summary

def main():
    print("=" * 80)
    print("[START] Llama3.1:8B Baseline Evaluation - Korean History")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # HuggingFace access token for gated models
    import os
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set your token as environment variable

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    print("[OK] Tokenizer loaded\n")

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    model.eval()
    print("[OK] Base model loaded\n")

    test_data = load_test_data()

    print("\n[EVAL] Running evaluation...")
    results, summary = evaluate_model(model, tokenizer, test_data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] Results saved to: {results_path}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("[SUMMARY] Llama3.1:8B Baseline Results")
    print("=" * 80)
    print(f"Model: {summary['model']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Correct: {summary['correct']}")
    print(f"Incorrect: {summary['incorrect']}")
    print(f"Accuracy: {summary['accuracy']:.2f}%")
    print("=" * 80)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
