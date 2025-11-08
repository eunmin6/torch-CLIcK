"""
BM25 RAG 모델 평가 스크립트
"""

import json
import os
from datetime import datetime
from rag_bm25 import BM25RAGKoreanHistoryQA
from tqdm import tqdm


def evaluate_bm25_rag():
    """BM25 RAG 모델 평가"""
    print("="*60)
    print("BM25 RAG 모델 평가 시작")
    print("="*60)

    # 1. 모델 로드
    print("\n모델 초기화 중...")
    qa_system = BM25RAGKoreanHistoryQA()

    # 2. 테스트 데이터 로드
    print("\n테스트 데이터 로드 중...")
    test_file = "./data/test_korean_history.json"

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"[OK] {len(test_data)}개 테스트 문제 로드")

    # 3. 평가
    print(f"\n평가 시작... (총 {len(test_data)}개 문제)")
    correct = 0
    total = len(test_data)
    results = []

    for sample in tqdm(test_data, desc="평가 중"):
        # RAG 답변 생성
        predicted, response, retrieved_docs = qa_system.answer_with_rag(
            question=sample["question"],
            choices=sample["choices"],
            k=3
        )

        # 정답 확인
        is_correct = (predicted == sample["answer"])
        correct += is_correct

        # 결과 저장
        results.append({
            "id": sample["id"],
            "question": sample["question"],
            "choices": sample["choices"],
            "predicted": predicted,
            "actual": sample["answer"],
            "correct": is_correct,
            "response": response,
            "retrieved_docs": [
                {
                    "text": doc["text"][:200],
                    "score": doc["score"],
                    "source": doc["metadata"].get("source", "")
                }
                for doc in retrieved_docs
            ]
        })

    # 4. 결과 계산
    accuracy = correct / total * 100

    print("\n" + "="*60)
    print("평가 완료!")
    print("="*60)
    print(f"정확도: {accuracy:.2f}% ({correct}/{total})")
    print(f"정답: {correct}개")
    print(f"오답: {total - correct}개")

    # 5. 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./benchmarks/rag_bm25_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 상세 결과 저장
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "model": "BM25 RAG + Qwen2.5-7B-Instruct (LoRA v2)",
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {results_file}")

    # 6. 요약 리포트 생성
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("BM25 RAG 평가 결과 요약\n")
        f.write("="*60 + "\n\n")
        f.write(f"평가 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"모델: BM25 RAG + Qwen2.5-7B-Instruct (LoRA v2)\n")
        f.write(f"테스트 문제 수: {total}개\n\n")
        f.write(f"정확도: {accuracy:.2f}%\n")
        f.write(f"정답 수: {correct}개\n")
        f.write(f"오답 수: {total - correct}개\n\n")
        f.write("="*60 + "\n")
        f.write("정답 문제 목록\n")
        f.write("="*60 + "\n\n")

        for r in results:
            if r["correct"]:
                f.write(f"[ID: {r['id']}] {r['question'][:50]}...\n")
                f.write(f"  정답: {r['actual']}\n\n")

        f.write("\n" + "="*60 + "\n")
        f.write("오답 문제 목록\n")
        f.write("="*60 + "\n\n")

        for r in results:
            if not r["correct"]:
                f.write(f"[ID: {r['id']}] {r['question'][:50]}...\n")
                f.write(f"  예측: {r['predicted']}\n")
                f.write(f"  정답: {r['actual']}\n\n")

    print(f"요약 저장: {summary_file}")

    # 7. v2 모델과 비교
    print("\n" + "="*60)
    print("v2 모델과 비교")
    print("="*60)

    v2_accuracy = 52.38
    v2_correct = 22

    improvement = accuracy - v2_accuracy
    additional_correct = correct - v2_correct

    print(f"v2 (Fine-tuned only): {v2_accuracy:.2f}% ({v2_correct}/{total})")
    print(f"BM25 RAG:             {accuracy:.2f}% ({correct}/{total})")
    print(f"개선:                 {improvement:+.2f}%p ({additional_correct:+d}개)")

    # 비교 결과 저장
    comparison_file = os.path.join(output_dir, "comparison.txt")
    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("v2 모델 vs BM25 RAG 비교\n")
        f.write("="*60 + "\n\n")
        f.write(f"v2 (Fine-tuned only):\n")
        f.write(f"  정확도: {v2_accuracy:.2f}%\n")
        f.write(f"  정답 수: {v2_correct}/{total}\n\n")
        f.write(f"BM25 RAG:\n")
        f.write(f"  정확도: {accuracy:.2f}%\n")
        f.write(f"  정답 수: {correct}/{total}\n\n")
        f.write(f"개선:\n")
        f.write(f"  정확도: {improvement:+.2f}%p\n")
        f.write(f"  정답 수: {additional_correct:+d}개\n\n")

    print(f"비교 저장: {comparison_file}")

    return accuracy, correct, total


if __name__ == "__main__":
    evaluate_bm25_rag()
