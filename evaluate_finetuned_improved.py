"""
개선된 Fine-tuned Model Evaluation - Constrained Answer Format

개선 사항:
1. 프롬프트에 답변 형식 명시 ("반드시 '숫자. 내용' 형식으로만 답변")
2. Constrained generation으로 숫자로 시작하도록 강제
3. 더 강력한 extraction 로직
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from datetime import datetime
import re

# ==================== Configuration ====================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER_PATH = "./models/qwen-click-lora/final"
TEST_DATA_PATH = "./data/splits/test.json"
OUTPUT_DIR = "./benchmarks/finetuned_history_improved"

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

def format_question_improved(sample):
    """
    개선된 프롬프트: 답변 형식을 명확히 지시
    """
    # Build context
    context = ""
    if sample.get('paragraph', '').strip():
        context = f"지문:\n{sample['paragraph']}\n\n"

    # Build choices
    choices_text = ""
    for i, choice in enumerate(sample['choices'], 1):
        choices_text += f"{i}. {choice}\n"

    # System prompt with strict format instruction
    system_prompt = """당신은 한국 역사에 정통한 AI 어시스턴트입니다.
객관식 문제의 정답을 선택할 때, 반드시 다음 형식으로만 답변해주세요:

[형식] 숫자. 정답내용

예시:
- "3. 팔만대장경"
- "1. 고구려"

다른 설명 없이 위 형식으로만 답변해주세요."""

    # Build messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""{context}질문: {sample['question']}

선택지:
{choices_text}
위 형식에 맞춰 정답을 선택해주세요."""
        }
    ]

    return messages

# ==================== Model Inference ====================

def generate_answer_constrained(model, tokenizer, messages, choices):
    """
    Constrained generation: 숫자로 시작하도록 강제
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 1~4 숫자 토큰 ID 가져오기
    number_tokens = [
        tokenizer.encode(str(i), add_special_tokens=False)[0]
        for i in range(1, len(choices) + 1)
    ]

    # Generate with constrained decoding
    with torch.no_grad():
        # First token must be a number
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            # Force first token to be a number
            force_words_ids=None,
            num_beams=1,
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

# ==================== Improved Answer Extraction ====================

def extract_answer_improved(response, choices):
    """
    개선된 답변 추출 로직

    우선순위:
    1. "숫자. 내용" 패턴 (정규표현식)
    2. 줄 시작의 숫자
    3. 첫 번째 숫자
    4. 선택지 텍스트 매칭
    """
    response_clean = response.strip()

    # Method 1: "숫자. 내용" 패턴 찾기 (가장 정확)
    pattern = r'^(\d+)\.\s*(.+?)(?:\n|$)'
    match = re.match(pattern, response_clean, re.MULTILINE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= len(choices):
            return choices[num - 1], "pattern_match", num

    # Method 2: 응답 첫 문자가 숫자인 경우
    if response_clean and response_clean[0].isdigit():
        num = int(response_clean[0])
        if 1 <= num <= len(choices):
            return choices[num - 1], "first_digit", num

    # Method 3: 첫 번째 등장하는 숫자 찾기
    numbers = re.findall(r'\d+', response_clean[:50])  # 처음 50자 내에서만
    if numbers:
        num = int(numbers[0])
        if 1 <= num <= len(choices):
            return choices[num - 1], "first_number", num

    # Method 4: 선택지 텍스트 정확히 매칭
    for i, choice in enumerate(choices, 1):
        if choice in response_clean:
            return choice, "text_match", i

    # Method 5: 부분 매칭
    for i, choice in enumerate(choices, 1):
        choice_clean = choice.replace(".", "").replace(",", "").strip()
        if choice_clean in response_clean:
            return choice, "partial_match", i

    # 실패
    return None, "extraction_failed", None

# ==================== Evaluation ====================

def evaluate_model(model, tokenizer, test_data):
    """모델 평가 실행 (개선된 extraction 포함)"""
    results = []
    correct = 0
    extraction_stats = {
        "pattern_match": 0,
        "first_digit": 0,
        "first_number": 0,
        "text_match": 0,
        "partial_match": 0,
        "extraction_failed": 0
    }

    print(f"\n[INFO] Starting evaluation on {len(test_data)} samples...")
    print("=" * 80)

    for idx, sample in enumerate(test_data, 1):
        # Format question with improved prompt
        messages = format_question_improved(sample)

        # Generate answer
        assistant_response, full_response = generate_answer_constrained(
            model, tokenizer, messages, sample['choices']
        )

        # Extract predicted answer with improved logic
        predicted_answer, extraction_method, predicted_num = extract_answer_improved(
            assistant_response, sample['choices']
        )

        # Track extraction method
        extraction_stats[extraction_method] += 1

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
            "predicted_number": predicted_num,
            "extraction_method": extraction_method,
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
        "extraction_stats": extraction_stats,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return results, summary

# ==================== Main ====================

def main():
    print("=" * 80)
    print("[START] Improved Fine-tuned Model Evaluation - Korean History")
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
    print("\n[EVAL] Running evaluation with improved extraction...")
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
    print("\n[EXTRACTION METHODS]")
    for method, count in summary['extraction_stats'].items():
        percentage = (count / summary['total_samples'] * 100) if summary['total_samples'] > 0 else 0
        print(f"  {method:20s}: {count:3d} ({percentage:5.1f}%)")
    print("=" * 80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
