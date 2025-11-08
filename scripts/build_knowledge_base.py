"""
지식 베이스 구축 스크립트

Phase 1: Wikipedia 한국사 데이터 수집 + CLIcK paragraph 추출
"""

import json
import os
from typing import List, Dict
import wikipediaapi
from tqdm import tqdm

def collect_wikipedia_data() -> List[Dict]:
    """Wikipedia 한국사 관련 문서 수집"""
    print("\n=== Wikipedia 한국사 데이터 수집 ===")

    # Wikipedia API 초기화 (한국어)
    wiki = wikipediaapi.Wikipedia(
        language='ko',
        user_agent='KoreanHistoryQA/1.0 (educational purpose)'
    )

    # 한국사 주요 키워드
    topics = [
        # 시대
        "한국사", "한국의 역사",
        "고조선", "삼국시대", "고구려", "백제", "신라", "가야",
        "남북국시대", "발해", "통일신라",
        "고려", "조선",

        # 주요 왕
        "광개토대왕", "세종대왕", "태종 (조선)", "영조", "정조",
        "고종 (조선)", "순종 (조선)",

        # 주요 인물
        "이순신", "장보고", "신사임당", "을지문덕", "강감찬",
        "김유신", "최영", "정도전", "김부식",

        # 문화재
        "팔만대장경", "석굴암", "불국사", "첨성대", "금관총",
        "훈민정음", "직지심체요절", "무령왕릉",

        # 주요 사건
        "삼국통일", "고려의 건국", "조선의 건국",
        "임진왜란", "병자호란", "갑오개혁", "3·1 운동",

        # 제도/법률
        "골품제", "화백회의", "과전법", "균역법", "경국대전", "속대전",
        "호패법", "대동법",

        # 문화/종교
        "고려청자", "조선백자", "불교", "성리학", "실학",
        "서원", "향교", "과거제도",

        # 경제
        "시전", "보부상", "상평통보", "활구",

        # 기타
        "훈민정음", "세종대왕", "한글", "측우기", "앙부일구"
    ]

    documents = []

    for topic in tqdm(topics, desc="Wikipedia 수집"):
        try:
            page = wiki.page(topic)

            if page.exists():
                # 문서가 너무 짧으면 스킵 (100자 미만)
                if len(page.text) < 100:
                    continue

                documents.append({
                    "title": topic,
                    "content": page.text,
                    "url": page.fullurl,
                    "source": "wikipedia"
                })

        except Exception as e:
            print(f"오류 발생 ({topic}): {e}")
            continue

    print(f"[OK] Wikipedia 문서 수집 완료: {len(documents)}개")
    return documents


def collect_click_paragraphs() -> List[Dict]:
    """CLIcK 데이터셋의 paragraph 추출"""
    print("\n=== CLIcK Paragraph 추출 ===")

    documents = []

    # train.json에서 paragraph 추출
    train_path = "./data/splits/train.json"

    if not os.path.exists(train_path):
        print(f"경고: {train_path} 파일을 찾을 수 없습니다.")
        return documents

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    for sample in tqdm(train_data, desc="CLIcK 추출"):
        paragraph = sample.get("paragraph", "").strip()

        if paragraph:
            documents.append({
                "title": f"CLIcK-{sample['id']}",
                "content": paragraph,
                "category": sample.get("subcategory", ""),
                "source": "click"
            })

    print(f"[OK] CLIcK paragraph 추출 완료: {len(documents)}개")
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """텍스트를 chunk로 분할"""
    # 문장 단위로 분할
    sentences = text.replace('. ', '.|').replace('.\n', '.|').split('|')

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence)

        # chunk_size를 초과하면 새 chunk 시작
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # overlap 처리: 마지막 몇 문장 유지
            overlap_chars = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                if overlap_chars + len(s) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_chars += len(s)
                else:
                    break

            current_chunk = overlap_chunk
            current_length = overlap_chars

        current_chunk.append(sentence)
        current_length += sentence_length

    # 마지막 chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def preprocess_documents(documents: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """문서 전처리 및 chunking"""
    print(f"\n=== 문서 전처리 (chunk_size={chunk_size}, overlap={overlap}) ===")

    processed = []

    for doc in tqdm(documents, desc="전처리"):
        content = doc["content"]

        # Chunking
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

        # 각 chunk를 별도 문서로 저장
        for i, chunk in enumerate(chunks):
            processed.append({
                "text": chunk,
                "metadata": {
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", ""),
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "category": doc.get("category", ""),
                    "url": doc.get("url", "")
                }
            })

    print(f"[OK] 전처리 완료: {len(processed)}개 chunks")
    return processed


def save_processed_data(processed_docs: List[Dict], output_path: str):
    """전처리된 데이터 저장"""
    print(f"\n=== 데이터 저장: {output_path} ===")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)

    print(f"[OK] 저장 완료: {len(processed_docs)}개 문서")


def main():
    """메인 실행 함수"""
    print("="*60)
    print("지식 베이스 구축 시작")
    print("="*60)

    # 1. Wikipedia 데이터 수집
    wiki_docs = collect_wikipedia_data()

    # 2. CLIcK paragraph 추출
    click_docs = collect_click_paragraphs()

    # 3. 통합
    all_docs = wiki_docs + click_docs
    print(f"\n총 수집된 문서: {len(all_docs)}개")
    print(f"  - Wikipedia: {len(wiki_docs)}개")
    print(f"  - CLIcK: {len(click_docs)}개")

    # 4. 전처리 및 chunking
    processed_docs = preprocess_documents(all_docs, chunk_size=500, overlap=50)

    # 5. 저장
    output_path = "./data/knowledge_base/processed_documents.json"
    save_processed_data(processed_docs, output_path)

    # 6. 통계 출력
    print("\n" + "="*60)
    print("지식 베이스 구축 완료!")
    print("="*60)
    print(f"총 문서 수: {len(all_docs)}개")
    print(f"총 chunk 수: {len(processed_docs)}개")
    print(f"평균 chunk/문서: {len(processed_docs)/len(all_docs):.1f}")

    # Source별 통계
    wiki_chunks = sum(1 for d in processed_docs if d["metadata"]["source"] == "wikipedia")
    click_chunks = sum(1 for d in processed_docs if d["metadata"]["source"] == "click")
    print(f"\nChunk 분포:")
    print(f"  - Wikipedia: {wiki_chunks}개")
    print(f"  - CLIcK: {click_chunks}개")

    print(f"\n다음 단계: Vector DB 구축")
    print(f"실행: python scripts/build_vector_db.py")


if __name__ == "__main__":
    main()
