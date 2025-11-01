"""
Qwen2.5-7B-Instruct 베이스라인 평가 스크립트

이 스크립트는:
1. Test set 로드
2. Chat format으로 변환
3. 모델로 추론 실행
4. 정확도 및 카테고리별 성능 측정
5. 결과를 benchmarks/baseline/에 저장

Note: gpt-oss-20b에서 변경됨 (GPU 메모리 제약으로 인해)
Note2: Llama-3.1에서 변경됨 (gated model 인증 이슈)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import datetime

# PyTorch CUDA 메모리 관리 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 경로 설정
TEST_DATA_PATH = "./data/splits/test.json"
OUTPUT_DIR = "./benchmarks/baseline"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 생성 파라미터
GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.3,  # 낮은 temperature로 더 일관된 답변
    "do_sample": True,
    "top_p": 0.9,
}

def load_test_data():
    """Test set 로드"""
    print(f"[INFO] Loading test data from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] Loaded {len(data)} test samples")
    return data

def format_question(sample):
    """샘플을 harmony format으로 변환 (Direct Answer 방식)"""
    # paragraph가 있으면 포함
    context = ""
    if sample.get('paragraph', '').strip():
        context = f"지문:\n{sample['paragraph']}\n\n"

    # 선택지 포맷팅
    choices_text = ""
    for i, choice in enumerate(sample['choices'], 1):
        choices_text += f"{i}. {choice}\n"

    # 메시지 구성
    messages = [
        {
            "role": "system",
            "content": "당신은 한국 문화와 언어에 정통한 AI 어시스턴트입니다. 주어진 객관식 문제의 정답을 선택해주세요."
        },
        {
            "role": "user",
            "content": f"{context}질문: {sample['question']}\n\n선택지:\n{choices_text}\n정답을 선택해주세요."
        }
    ]

    return messages

def extract_answer(generated_text, choices):
    """생성된 텍스트에서 답변 추출"""
    generated_lower = generated_text.lower()

    # 방법 1: 선택지 텍스트가 직접 포함되어 있는지 확인
    for choice in choices:
        if choice in generated_text:
            return choice

    # 방법 2: 선택지의 핵심 키워드가 포함되어 있는지 확인
    for choice in choices:
        # 선택지를 단어로 분리하여 매칭
        choice_words = choice.split()
        if len(choice_words) > 2:
            # 긴 선택지의 경우 주요 부분 매칭
            key_words = choice_words[:3]  # 처음 3단어
            if all(word in generated_text for word in key_words):
                return choice

    # 방법 3: 번호 매칭 (1, 2, 3, 4 등)
    for i, choice in enumerate(choices, 1):
        patterns = [f"{i}.", f"{i})", f"({i})", f"번: {i}", f"정답: {i}"]
        for pattern in patterns:
            if pattern in generated_text[:200]:  # 앞부분만 검사
                return choice

    # 매칭 실패 시 None 반환
    return None

def evaluate_model(model, tokenizer, test_data):
    """모델 평가 실행"""
    print("\n" + "="*80)
    print("[EVAL] Starting baseline evaluation")
    print("="*80 + "\n")

    results = []
    correct = 0
    total = 0

    # 카테고리별 통계
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    subcategory_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    start_time = time.time()

    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        # Harmony format으로 변환
        messages = format_question(sample)

        try:
            # Chat template 적용
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 토크나이징
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            # 생성
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **GENERATION_CONFIG,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 입력 부분 제거 (생성된 부분만 추출)
            if input_text in generated_text:
                generated_answer = generated_text[len(input_text):].strip()
            else:
                generated_answer = generated_text

            # 답변 추출
            predicted_answer = extract_answer(generated_answer, sample['choices'])
            correct_answer = sample['answer']

            # 정확도 체크
            is_correct = (predicted_answer == correct_answer)

            if is_correct:
                correct += 1
                category_stats[sample['category']]["correct"] += 1
                subcategory_stats[sample['subcategory']]["correct"] += 1
                source_stats[sample['source']]["correct"] += 1

            total += 1
            category_stats[sample['category']]["total"] += 1
            subcategory_stats[sample['subcategory']]["total"] += 1
            source_stats[sample['source']]["total"] += 1

            # 결과 저장
            result = {
                "id": sample['id'],
                "category": sample['category'],
                "subcategory": sample['subcategory'],
                "source": sample['source'],
                "question": sample['question'],
                "choices": sample['choices'],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "generated_text": generated_answer[:500],  # 처음 500자만 저장
                "is_correct": is_correct
            }
            results.append(result)

        except Exception as e:
            print(f"\n[ERROR] Failed to process sample {idx}: {e}")
            result = {
                "id": sample.get('id', f'sample_{idx}'),
                "error": str(e),
                "is_correct": False
            }
            results.append(result)
            total += 1

        # 중간 진행상황 출력 (매 50개마다)
        if (idx + 1) % 50 == 0:
            current_acc = correct / total * 100
            print(f"\n[INFO] Progress: {idx+1}/{len(test_data)} - Current Accuracy: {current_acc:.2f}%")

    elapsed_time = time.time() - start_time

    # 최종 메트릭 계산
    overall_accuracy = correct / total * 100 if total > 0 else 0

    metrics = {
        "overall": {
            "accuracy": overall_accuracy,
            "correct": correct,
            "total": total
        },
        "by_category": {},
        "by_subcategory": {},
        "by_source": {},
        "metadata": {
            "model_name": MODEL_NAME,
            "test_samples": len(test_data),
            "evaluation_date": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "generation_config": GENERATION_CONFIG
        }
    }

    # 카테고리별 정확도
    for category, stats in category_stats.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        metrics["by_category"][category] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    # 서브카테고리별 정확도
    for subcategory, stats in subcategory_stats.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        metrics["by_subcategory"][subcategory] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    # 출처별 정확도
    for source, stats in source_stats.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        metrics["by_source"][source] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    return results, metrics

def save_results(results, metrics):
    """결과 저장"""
    print("\n" + "="*80)
    print("[SAVE] Saving evaluation results")
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 전체 메트릭 저장
    metrics_path = os.path.join(OUTPUT_DIR, "overall_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Overall metrics saved to: {metrics_path}")

    # 카테고리별 메트릭 저장
    category_metrics_path = os.path.join(OUTPUT_DIR, "category_metrics.json")
    with open(category_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics["by_category"], f, ensure_ascii=False, indent=2)
    print(f"[OK] Category metrics saved to: {category_metrics_path}")

    # 출처별 메트릭 저장
    source_metrics_path = os.path.join(OUTPUT_DIR, "source_metrics.json")
    with open(source_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics["by_source"], f, ensure_ascii=False, indent=2)
    print(f"[OK] Source metrics saved to: {source_metrics_path}")

    # 예측 결과 저장 (JSONL 형식)
    predictions_path = os.path.join(OUTPUT_DIR, "predictions.jsonl")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"[OK] Predictions saved to: {predictions_path}")

    print(f"\n[INFO] All results saved to: {OUTPUT_DIR}/")

def print_summary(metrics):
    """결과 요약 출력"""
    print("\n" + "="*80)
    print("[SUMMARY] Baseline Evaluation Results")
    print("="*80)

    print(f"\n전체 정확도: {metrics['overall']['accuracy']:.2f}% ({metrics['overall']['correct']}/{metrics['overall']['total']})")

    print("\n--- 카테고리별 정확도 ---")
    for category, stats in sorted(metrics["by_category"].items()):
        print(f"  {category:<20}: {stats['accuracy']:>6.2f}% ({stats['correct']}/{stats['total']})")

    print("\n--- 서브카테고리별 정확도 (상위 10개) ---")
    sorted_subcats = sorted(metrics["by_subcategory"].items(), key=lambda x: x[1]['total'], reverse=True)[:10]
    for subcategory, stats in sorted_subcats:
        print(f"  {subcategory:<25}: {stats['accuracy']:>6.2f}% ({stats['correct']}/{stats['total']})")

    print("\n--- 출처별 정확도 ---")
    for source, stats in sorted(metrics["by_source"].items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {source:<15}: {stats['accuracy']:>6.2f}% ({stats['correct']}/{stats['total']})")

    elapsed_min = metrics['metadata']['elapsed_time_seconds'] / 60
    print(f"\n실행 시간: {elapsed_min:.2f}분")

def main():
    print("="*80)
    print("[START] Baseline Evaluation for gpt-oss-20b")
    print("="*80 + "\n")

    # 1. Test 데이터 로드
    test_data = load_test_data()

    # 2. 모델 및 토크나이저 로드
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("[OK] Tokenizer loaded")

    print("\n[INFO] Loading model (this may take a few minutes)...")
    print("[INFO] Using BF16 with all layers on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Force all layers to GPU
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()  # 평가 모드
    print("[OK] Model loaded")

    # 3. 평가 실행
    results, metrics = evaluate_model(model, tokenizer, test_data)

    # 4. 결과 저장
    save_results(results, metrics)

    # 5. 요약 출력
    print_summary(metrics)

    print("\n" + "="*80)
    print("[DONE] Baseline evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
