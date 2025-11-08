"""
간소화된 RAG 기반 한국사 QA 시스템

BM25 검색 + Fine-tuned 모델 결합
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from rank_bm25 import BM25Okapi


class BM25RAGKoreanHistoryQA:
    def __init__(self,
                 base_model_path="Qwen/Qwen2.5-7B-Instruct",
                 lora_path="./models/qwen-click-lora-v2/final",
                 knowledge_base_path="./data/knowledge_base/processed_documents.json"):
        """
        BM25 기반 RAG QA 시스템 초기화

        Args:
            base_model_path: 베이스 모델 경로
            lora_path: LoRA 어댑터 경로
            knowledge_base_path: 전처리된 문서 경로
        """
        print("="*60)
        print("BM25 RAG Korean History QA 시스템 초기화")
        print("="*60)

        # 1. Fine-tuned 모델 로드
        print("\n[1/2] Fine-tuned 모델 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 메모리 효율적으로 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # bfloat16 대신 float16
            device_map="auto",
            low_cpu_mem_usage=True,  # 메모리 효율성
            max_memory={0: "20GB", "cpu": "30GB"}  # GPU/CPU 메모리 제한
        )

        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        print("[OK] Fine-tuned 모델 로드 완료")

        # 2. 지식 베이스 로드 및 BM25 인덱스 구축
        print("\n[2/2] 지식 베이스 로드 중...")
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # 문서 텍스트와 토큰화
        self.doc_texts = [doc["text"] for doc in self.documents]

        # 간단한 한글 토크나이저 (공백 및 특수문자 기준)
        tokenized_corpus = [self._tokenize(text) for text in self.doc_texts]

        # BM25 인덱스 구축
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[OK] 지식 베이스 로드 완료: {len(self.documents)}개 문서")

        print("\n" + "="*60)
        print("초기화 완료!")
        print("="*60)

    def _tokenize(self, text):
        """간단한 한글 토크나이저"""
        # 한글, 영문, 숫자만 남기고 나머지는 공백으로
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        # 공백으로 분리
        tokens = text.lower().split()
        return tokens

    def retrieve_context(self, question, k=3):
        """BM25를 사용하여 질문과 관련된 문서 검색"""
        # 질문 토크나이징
        tokenized_query = self._tokenize(question)

        # BM25 점수 계산
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Top-K 인덱스 가져오기
        import numpy as np
        top_k_indices = np.argsort(doc_scores)[::-1][:k]

        # 결과 수집
        retrieved_docs = []
        for idx in top_k_indices:
            retrieved_docs.append({
                "text": self.doc_texts[idx],
                "score": float(doc_scores[idx]),
                "metadata": self.documents[idx]["metadata"]
            })

        return retrieved_docs

    def format_context(self, retrieved_docs):
        """검색된 문서를 컨텍스트 문자열로 변환"""
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            source = doc["metadata"].get("source", "unknown")
            title = doc["metadata"].get("title", "")
            text = doc["text"]

            context_parts.append(f"[참고자료 {i}] (출처: {source})\n{text}")

        return "\n\n".join(context_parts)

    def answer_with_rag(self, question, choices, k=3):
        """RAG 기반 답변 생성"""
        # 1. 관련 문서 검색
        retrieved_docs = self.retrieve_context(question, k=k)
        context = self.format_context(retrieved_docs)

        # 2. 선택지 포맷팅
        choices_text = "\n".join([
            f"{i}. {choice}"
            for i, choice in enumerate(choices, 1)
        ])

        # 3. 프롬프트 구성 (Context 포함)
        messages = [
            {
                "role": "system",
                "content": "당신은 한국 역사 전문가입니다. 주어진 참고 자료를 바탕으로 정확하게 답변하세요."
            },
            {
                "role": "user",
                "content": f"""다음 참고 자료를 바탕으로 문제를 풀어주세요:

[참고 자료]
{context}

[질문]
{question}

[선택지]
{choices_text}

정답을 "숫자. 내용" 형식으로 답변하세요."""
            }
        ]

        # 4. 생성
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 5. 답변 추출
        answer = self.extract_answer(response, choices)

        return answer, response, retrieved_docs

    def extract_answer(self, response, choices):
        """응답에서 답변 추출"""
        # 패턴 1: "숫자. 내용" 형식
        pattern1 = r'^(\d+)\.\s*(.+)$'
        match = re.search(pattern1, response.strip(), re.MULTILINE)
        if match:
            num = int(match.group(1))
            if 1 <= num <= len(choices):
                return choices[num - 1]

        # 패턴 2: 선택지 내용이 응답에 포함됨
        for i, choice in enumerate(choices):
            if choice in response:
                return choice

        # 실패 시 첫 번째 선택지 반환
        return choices[0] if choices else ""


def demo():
    """데모 실행"""
    print("\n" + "="*60)
    print("BM25 RAG QA 시스템 데모")
    print("="*60)

    # 시스템 초기화
    qa_system = BM25RAGKoreanHistoryQA()

    # 테스트 질문
    test_question = {
        "question": "고려시대의 대표적인 문화재는?",
        "choices": ["석굴암", "첨성대", "팔만대장경", "석빙고"]
    }

    print(f"\n질문: {test_question['question']}")
    print(f"선택지: {test_question['choices']}")

    # RAG 답변
    answer, response, retrieved_docs = qa_system.answer_with_rag(
        test_question["question"],
        test_question["choices"],
        k=3
    )

    print(f"\n=== 검색된 문서 (BM25) ===")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n[{i}] (score: {doc['score']:.4f})")
        print(doc["text"][:200] + "...")

    print(f"\n=== 모델 응답 ===")
    print(response)

    print(f"\n=== 추출된 답변 ===")
    print(answer)


if __name__ == "__main__":
    demo()
