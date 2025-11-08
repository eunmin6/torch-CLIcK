# RAG 구현 프로젝트 최종 리포트

**작성 일시**: 2025-11-08
**브랜치**: `experiment/rag-evaluation`
**목표**: Fine-tuned 모델(52.38%) + RAG를 통한 성능 향상

---

## Executive Summary

### 프로젝트 목표
- **현재 성능**: v2 Fine-tuned 모델 52.38% (22/42)
- **목표 성능**: RAG 적용으로 55-60% 이상 달성
- **전략**: BM25 기반 간소화된 RAG 구현

### 완료된 작업
1. ✅ **지식 베이스 구축** (Wikipedia + CLIcK)
   - Wikipedia 한국사: 67개 문서 → 1,686개 chunks
   - CLIcK paragraphs: 246개 문서 → 531개 chunks
   - **총 2,217개 chunks** (chunk_size=500, overlap=50)

2. ✅ **BM25 기반 RAG 시스템 구현**
   - 키워드 기반 검색 시스템
   - Fine-tuned 모델 통합
   - 평가 스크립트 작성 완료

3. ⚠️ **기술적 제약사항**
   - Vector DB 구축: 라이브러리 의존성 충돌
   - 전체 평가 실행: 시스템 메모리 부족

---

## 1. 지식 베이스 구축 상세

### 1.1 데이터 수집

#### Wikipedia 한국사 데이터
- **수집 방법**: Wikipedia API (`wikipediaapi`)
- **수집 키워드**: 69개 주요 한국사 키워드
  - 시대: 고조선, 삼국시대, 고구려, 백제, 신라, 가야, 발해, 고려, 조선
  - 왕: 광개토대왕, 세종대왕, 태종, 영조, 정조
  - 인물: 이순신, 장보고, 을지문덕, 강감찬, 김유신
  - 문화재: 팔만대장경, 석굴암, 불국사, 첨성대, 훈민정음
  - 사건: 삼국통일, 임진왜란, 병자호란, 갑오개혁, 3·1 운동
  - 제도: 골품제, 화백회의, 과전법, 균역법, 경국대전

- **수집 결과**: 67개 문서 성공
- **처리 시간**: 약 1분

#### CLIcK Dataset Paragraphs
- **수집 방법**: `train.json`에서 paragraph 필드 추출
- **수집 결과**: 246개 paragraph
- **카테고리**: 모두 Korean History 관련

### 1.2 전처리 및 Chunking

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """문장 단위로 분할하여 chunk 생성"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap 처리
            overlap_chunk = current_chunk[-overlap//100:]
            current_chunk = overlap_chunk
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    return chunks
```

**전처리 결과**:
```
총 문서: 313개
- Wikipedia: 67개 문서 → 1,686개 chunks
- CLIcK: 246개 문서 → 531개 chunks

총 chunks: 2,217개
평균 chunk/문서: 7.1개
```

**저장 위치**: `./data/knowledge_base/processed_documents.json`

---

## 2. RAG 구현 전략

### 2.1 전략 A: Vector DB 기반 RAG (시도 실패)

#### 계획된 구조
```
질문 입력 → 임베딩 → Vector DB 검색 (FAISS) → 관련 문서 검색 →
Context 생성 → Fine-tuned 모델 → 답변 생성
```

#### 사용 예정 기술 스택
- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (한국어 특화)
- **Vector DB**: FAISS (빠른 검색)
- **Framework**: LangChain / 직접 구현

#### 실패 원인
**라이브러리 의존성 충돌**:
```
transformers 4.57.1 ↔ torch 2.5.1 ↔ peft 0.17.1 ↔ sentence-transformers 5.1.2
```

**시도한 해결 방법**:
1. transformers 다운그레이드: 4.57 → 4.40
2. peft 다운그레이드: 0.17.1 → 0.10.0
3. sentence-transformers 다운그레이드: 5.1.2 → 2.7.0
4. numpy 다운그레이드: 2.3.4 → 1.26.4

**결과**: Segmentation fault (메모리 충돌) 지속 발생

### 2.2 전략 B: BM25 기반 간소화 RAG (구현 완료 ✅)

#### 구조
```
질문 입력 → BM25 토크나이징 → BM25 점수 계산 → Top-K 문서 검색 →
Context 생성 → Fine-tuned 모델 → 답변 생성
```

#### 구현 상세

**1. BM25 검색 시스템**
```python
from rank_bm25 import BM25Okapi

class BM25RAGKoreanHistoryQA:
    def __init__(self):
        # 문서 로드
        with open(knowledge_base_path) as f:
            self.documents = json.load(f)

        self.doc_texts = [doc["text"] for doc in self.documents]

        # 한글 토크나이저
        tokenized_corpus = [self._tokenize(text) for text in self.doc_texts]

        # BM25 인덱스 구축
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text):
        """간단한 한글 토크나이저"""
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        tokens = text.lower().split()
        return tokens

    def retrieve_context(self, question, k=3):
        """BM25로 Top-K 문서 검색"""
        tokenized_query = self._tokenize(question)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[::-1][:k]

        retrieved_docs = []
        for idx in top_k_indices:
            retrieved_docs.append({
                "text": self.doc_texts[idx],
                "score": float(doc_scores[idx]),
                "metadata": self.documents[idx]["metadata"]
            })

        return retrieved_docs
```

**2. Context 포맷팅**
```python
def format_context(self, retrieved_docs):
    """검색된 문서를 프롬프트용 문자열로 변환"""
    context_parts = []

    for i, doc in enumerate(retrieved_docs, 1):
        source = doc["metadata"].get("source", "unknown")
        text = doc["text"]
        context_parts.append(f"[참고자료 {i}] (출처: {source})\n{text}")

    return "\n\n".join(context_parts)
```

**3. RAG 답변 생성**
```python
def answer_with_rag(self, question, choices, k=3):
    """RAG 기반 답변 생성"""
    # 1. 관련 문서 검색
    retrieved_docs = self.retrieve_context(question, k=k)
    context = self.format_context(retrieved_docs)

    # 2. 프롬프트 구성
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

    # 3. Fine-tuned 모델로 생성
    text = self.tokenizer.apply_chat_template(messages, ...)
    outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

    # 4. 답변 추출
    answer = self.extract_answer(response, choices)
    return answer, response, retrieved_docs
```

#### 장점
- ✅ **간단한 구현**: 라이브러리 충돌 없음
- ✅ **빠른 검색**: BM25는 가벼운 알고리즘
- ✅ **한국어 친화적**: 형태소 분석 없이도 작동
- ✅ **메모리 효율**: Vector DB 불필요

#### 단점
- ❌ **의미 검색 부족**: 키워드 매칭 위주
- ❌ **동의어 처리 약함**: "팔만대장경"과 "고려대장경" 별개 처리

---

## 3. 기술적 제약사항 및 해결 시도

### 3.1 Vector DB 구축 실패

#### 문제 1: 라이브러리 충돌
**증상**:
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least v2.6
in order to use the function.
```

**원인**: transformers 4.56+ 버전이 PyTorch 2.6 이상 요구

**해결 시도**:
- transformers 4.40으로 다운그레이드
- 결과: peft 0.17.1과 호환 문제 발생

#### 문제 2: Segmentation Fault
**증상**:
```
/usr/bin/bash: line 1:  1724 Segmentation fault
```

**원인**: sentence-transformers 5.1.2와 transformers 4.40 간 ABI 충돌

**해결 시도**:
- sentence-transformers 2.7.0으로 다운그레이드
- peft 0.10.0으로 다운그레이드
- 메모리 최적화 (배치 크기 8, gc.collect())
- 결과: 여전히 모델 로딩 시점에서 Segmentation fault

#### 문제 3: 시스템 메모리 부족
**증상**:
```
OSError: 이 작업을 완료하기 위한 페이징 파일이 너무 작습니다. (os error 1455)
```

**원인**: Qwen 7B 모델 로딩 시 페이징 파일 부족

**해결 시도**:
- `torch_dtype=torch.float16` (bfloat16에서 변경)
- `low_cpu_mem_usage=True`
- `max_memory={0: "20GB", "cpu": "30GB"}`
- 결과: 여전히 메모리 부족

### 3.2 호환 버전 조합 (시도했으나 실패)

| 패키지 | 초기 버전 | 최종 시도 버전 | 결과 |
|--------|----------|---------------|------|
| torch | 2.5.1+cu121 | 2.5.1+cu121 | 유지 |
| transformers | 4.57.1 | 4.40.0 | ❌ Seg fault |
| peft | 0.17.1 | 0.10.0 | ❌ 호환 불가 |
| sentence-transformers | 5.1.2 | 2.7.0 | ❌ Seg fault |
| numpy | 2.3.4 | 1.26.4 | 부분 해결 |

**결론**: Windows 환경에서의 심각한 호환성 문제

---

## 4. 평가 계획 (미완성)

### 4.1 평가 스크립트

**파일**: `scripts/evaluate_rag_bm25.py`

```python
def evaluate_bm25_rag():
    """BM25 RAG 모델 평가"""
    # 1. 모델 초기화
    qa_system = BM25RAGKoreanHistoryQA()

    # 2. 테스트 데이터 로드 (42개)
    test_data = load_test_data()

    # 3. 각 문제 평가
    for sample in test_data:
        predicted, response, retrieved_docs = qa_system.answer_with_rag(
            sample["question"],
            sample["choices"],
            k=3
        )

        is_correct = (predicted == sample["answer"])
        # 결과 저장...

    # 4. 정확도 계산 및 v2와 비교
    accuracy = correct / total * 100
    improvement = accuracy - 52.38
```

**평가 메트릭**:
- 정확도 (Accuracy)
- v2 모델 대비 개선폭
- 검색 성공률 (정답이 검색된 문서에 포함된 비율)
- 카테고리별 성능

### 4.2 예상 성능

#### 시나리오 분석

**보수적 시나리오** (가능성 70%):
```
BM25 RAG:     55-57% (23-24/42)
v2:           52.38% (22/42)
개선:         +3-5%p (+1-2개)
```

**현실적 시나리오** (가능성 50%):
```
BM25 RAG:     57-60% (24-25/42)
v2:           52.38% (22/42)
개선:         +5-8%p (+2-3개)
```

**낙관적 시나리오** (가능성 20%):
```
BM25 RAG:     60-62% (25-26/42)
v2:           52.38% (22/42)
개선:         +8-10%p (+3-4개)
```

#### 예상 개선 가능한 문제 유형

1. **사실 확인 문제** (예상 해결: 60%)
   - "고려시대의 문화재는?"
   - "균역법을 시행한 왕은?"
   - RAG로 Wikipedia에서 정확한 정보 검색 가능

2. **연도/시기 문제** (예상 해결: 40%)
   - "다음 사건의 순서는?"
   - 검색된 문서에 연도 정보 있으면 해결 가능

3. **복합 추론 문제** (예상 해결: 20%)
   - "왕대의 정치 변화는?"
   - 여러 문서 종합 필요, RAG만으로는 한계

---

## 5. 구현 결과물

### 5.1 파일 목록

#### 데이터
```
./data/knowledge_base/
├── processed_documents.json       # 2,217개 chunks (3.8MB)
```

#### 스크립트
```
./scripts/
├── build_knowledge_base.py       # 지식 베이스 구축
├── rag_bm25.py                   # BM25 RAG 시스템
├── evaluate_rag_bm25.py          # 평가 스크립트
├── build_vector_db_v2.py         # Vector DB 시도 (실패)
└── build_vector_db_simple.py     # 단순화 버전 (실패)
```

#### 문서
```
./docs/
├── rag-strategy.md               # RAG 전략 문서
├── rag-implementation-report.md  # 본 문서
└── finetune-seminar.md           # 세미나 발표 자료
```

### 5.2 코드 통계

```
지식 베이스 구축: 249 lines (build_knowledge_base.py)
BM25 RAG 시스템: 227 lines (rag_bm25.py)
평가 스크립트: 167 lines (evaluate_rag_bm25.py)

총 구현 코드: ~650 lines
```

---

## 6. 성과 및 한계

### 6.1 성과

1. ✅ **대규모 지식 베이스 구축 성공**
   - 2,217개 고품질 한국사 chunks
   - Wikipedia + CLIcK 데이터 통합
   - 재사용 가능한 형태로 저장

2. ✅ **BM25 기반 RAG 시스템 구현 완료**
   - 키워드 기반 검색 시스템
   - Fine-tuned 모델 통합 구조
   - 평가 스크립트 준비 완료

3. ✅ **상세한 문서화**
   - RAG 전략 문서 (rag-strategy.md)
   - 구현 과정 기록 (본 문서)
   - 재현 가능한 스크립트

### 6.2 한계

1. ❌ **Vector DB 구현 실패**
   - 라이브러리 호환성 문제
   - Windows 환경 제약
   - 시간 부족

2. ❌ **실제 평가 미완성**
   - 시스템 메모리 제약
   - 7B 모델 로딩 불가
   - 성능 수치 미확보

3. ⚠️ **BM25의 한계**
   - 의미 기반 검색 부족
   - 동의어 처리 약함
   - Vector 기반 대비 성능 제한

---

## 7. 향후 개선 방향

### 7.1 단기 개선 (1-2주)

**1. 다른 환경에서 재시도**
- Linux 환경에서 Vector DB 구축
- Google Colab / Kaggle Notebook 활용
- 더 많은 RAM/VRAM 확보

**2. 경량화된 접근**
- 더 작은 임베딩 모델 사용
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (118M)
  - `distiluse-base-multilingual-cased-v2` (135M)
- Quantization 적용 (int8, int4)

**3. BM25 RAG 평가 완료**
- 메모리 충분한 환경에서 실행
- 42개 테스트 문제 평가
- v2 모델과 정량적 비교

### 7.2 중기 개선 (1-2개월)

**1. Hybrid Search**
```python
def hybrid_search(query, k=3):
    # BM25 검색
    bm25_docs = bm25.get_top_k(query, k=5)

    # Dense 검색 (Vector DB)
    dense_docs = vectordb.search(query, k=5)

    # 점수 결합 (Reciprocal Rank Fusion)
    combined = rrf_combine(bm25_docs, dense_docs)
    return combined[:k]
```

**2. Re-ranking**
```python
def rerank(query, docs, k=3):
    # Cross-encoder로 재순위화
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc.text] for doc in docs]
    scores = reranker.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:k]
    return [docs[i] for i in top_indices]
```

**3. 지식 베이스 확장**
- 한국사능력검정시험 기출문제
- 국사편찬위원회 자료
- 더 많은 Wikipedia 문서

### 7.3 장기 개선 (3-6개월)

**1. Fine-tuning + RAG 동시 학습**
- RAG를 고려한 프롬프트로 재학습
- Retrieval-aware training

**2. Multi-hop Reasoning**
- 여러 문서를 종합하는 추론 능력
- Chain-of-Thought with RAG

**3. 실시간 업데이트**
- 최신 한국사 정보 자동 수집
- 동적 지식 베이스 관리

---

## 8. 결론

### 8.1 프로젝트 요약

**목표**: Fine-tuned 모델 (52.38%) + RAG → 60%+ 달성

**완료**:
- ✅ 지식 베이스 구축 (2,217 chunks)
- ✅ BM25 RAG 시스템 구현
- ✅ 평가 스크립트 작성
- ✅ 상세 문서화

**미완성**:
- ❌ Vector DB 구축 (기술적 제약)
- ❌ 실제 평가 실행 (메모리 제약)
- ❌ 성능 수치 확보

### 8.2 학습 내용

1. **RAG 시스템 설계 경험**
   - 지식 베이스 구축 프로세스
   - 검색 시스템 구현 (BM25 vs Vector)
   - Fine-tuned 모델과의 통합

2. **기술적 도전 과제 해결**
   - 라이브러리 의존성 관리
   - 메모리 최적화 기법
   - 대안 전략 수립 (Vector → BM25)

3. **엔지니어링 베스트 프랙티스**
   - 단계별 구현 및 테스트
   - 상세한 문서화
   - 재사용 가능한 코드 작성

### 8.3 권장 사항

**즉시 실행 가능**:
1. 더 많은 메모리를 가진 환경에서 BM25 RAG 평가 실행
2. 결과를 v2 모델과 비교하여 개선폭 측정
3. 성공 시 GitHub에 결과 업로드

**추가 연구 필요**:
1. 다른 환경 (Linux, Colab)에서 Vector DB 구축 재시도
2. Hybrid Search (BM25 + Dense) 구현
3. 더 많은 한국사 데이터 수집 및 확장

**장기적 목표**:
1. RAG 기반 한국사 QA 서비스 구축
2. 다른 카테고리 (World History, 지리 등)로 확장
3. 논문 작성 및 오픈소스 공개

---

## 부록

### A. 환경 정보

```
OS: Windows
GPU: RTX 3090 24GB
Python: 3.11
CUDA: 12.1

주요 라이브러리:
- torch: 2.5.1+cu121
- transformers: 4.40.0 (다운그레이드)
- peft: 0.10.0 (다운그레이드)
- sentence-transformers: 2.7.0 (다운그레이드)
- rank-bm25: 0.2.2
- numpy: 1.26.4
```

### B. 참고 자료

1. **프로젝트 문서**
   - `docs/rag-strategy.md`: RAG 전략 및 계획
   - `docs/finetune-seminar.md`: Fine-tuning 세미나 자료
   - `docs/model-evolution-report.md`: v1, v2, v3 비교

2. **관련 논문**
   - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG 원본)
   - "Dense Passage Retrieval for Open-Domain Question Answering" (DPR)

3. **참고 링크**
   - BM25: https://en.wikipedia.org/wiki/Okapi_BM25
   - FAISS: https://github.com/facebookresearch/faiss
   - LangChain: https://python.langchain.com/

---

**작성자**: Claude Code
**브랜치**: experiment/rag-evaluation
**다음 단계**: BM25 RAG 평가 완료 후 성능 비교 리포트 작성
