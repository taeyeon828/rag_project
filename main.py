import os, csv
# =====================================
# 프로젝트 경로 설정
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Vector DB
DB_PATH_RAW = os.path.join(BASE_DIR, "db", "chroma_raw")
DB_PATH_PROCESSED = os.path.join(BASE_DIR, "db", "chroma_processed")

# Raw inputs
PDF_DIR = os.path.join(DATA_DIR, "raw", "pdf")
CSV_DIR = os.path.join(DATA_DIR, "raw", "csv")
IMG_DIR = os.path.join(DATA_DIR, "raw", "images")

# Processed outputs
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_TEXT_DIR = os.path.join(PROCESSED_DIR, "text")
PROCESSED_CHUNKS_DIR = os.path.join(PROCESSED_DIR, "chunks")
CHUNKS_JSONL_PATH = os.path.join(PROCESSED_CHUNKS_DIR, "chunks.jsonl")



import streamlit as st
import pandas as pd
import numpy as np
import cv2
import json
import numpy as np
import re
from glob import glob
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import defaultdict
from pathlib import Path
from db_agent.db_agent import get_db_context
from prompts import build_prompt
prompt = build_prompt() 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

__all__ = ["retrieve_context", "ask_rag", "llm"]

def analyze_docs(docs, name=""):
    lengths = [len(d.page_content) for d in docs]
    total_chunks = len(docs)
    short_chunks = sum(l < 20 for l in lengths)
    avg_length = np.mean(lengths) if lengths else 0

    texts = [d.page_content.strip() for d in docs]
    duplicates = len(texts) - len(set(texts))

    print(f"\n📊 [{name}] 분석 결과")
    print("-" * 30)
    print(f"전체 청크 수: {total_chunks}")
    print(f"20자 이하 청크 수: {short_chunks}")
    print(f"평균 청크 길이: {avg_length:.2f}")
    print(f"중복 청크 수: {duplicates}")


# Gemini 설정
# Mac: 터미널에서 export GEMINI_API_KEY="내가 발급받은 API 키"
# Windows: PowerShell에서 $env:GEMINI_API_KEY= “내가 발급받은 API 키"


# =====================================
# 데이터 처리 함수
# =====================================
def load_processed_pdf_chunks(chunks_jsonl_path: str) -> list[Document]:
    docs: list[Document] = []

    with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                continue

            # 전처리에서 만든 메타를 최대한 유지 + 네 메타 키와 호환되게 보강
            meta = {k: v for k, v in rec.items() if k not in ("text",)}

            meta.setdefault("source_type", "pdf")
            meta.setdefault("source_file", meta.get("source") or meta.get("source_file"))
            meta.setdefault("source_path", meta.get("source_path"))  # 없으면 None
            # page가 있으면 사람이 보는 페이지(page_human)도 만들기
            if "page" in meta and isinstance(meta["page"], int):
                meta["page_human"] = meta["page"] + 1

            docs.append(Document(page_content=text, metadata=meta))

    return docs

rep = json.loads(Path("data/processed/report/quality_report.json").read_text(encoding="utf-8"))
for f in rep["files"]:
    print(f["file"], "pages=", f["pages"], "chunks=", f["chunks"], "empty_pages=", f["empty_pages"])

# CSV 데이터 전처리 
def clean_csv_value(v: str) -> str:
    v = str(v)
    v = v.replace("\ufeff", "")          # BOM 제거
    v = v.strip()                        # 앞뒤 공백 제거
    v = re.sub(r"\s+", " ", v)           # 연속 공백 정리
    return v

def load_all_csvs(csv_dir: str):
    docs = []
    seen = set()  # 중복 row 제거용 

    for csv_path in glob(os.path.join(csv_dir, "*.csv")):
        with open(csv_path, "r", encoding="cp949", errors="ignore") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):

                # (1) row 값 정제: 공백/개행/중복 공백 등
                cleaned_row = {}
                for k, v in row.items():
                    if not k:
                        continue
                    if v is None:
                        continue
                    vv = clean_csv_value(v)
                    if vv:  # 빈 값 제거
                        cleaned_row[k] = vv

                # 행 -> 텍스트
                lines = [f"[{k}] {v}" for k, v in cleaned_row.items()]
                text = "\n".join(lines).strip()
                if not text:
                    continue

                # (2) 중복 row 제거 (텍스트가 완전히 동일하면 스킵)
                sig = (os.path.basename(csv_path), text)
                if sig in seen:
                    continue
                seen.add(sig)

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source_type": "csv",
                        "source_file": os.path.basename(csv_path),
                        "row": idx,
                    }
                ))
    return docs



# 이미지 텍스트화 
def _imread_unicode(path: str):
    """
    한글/특수문자 경로에서도 안전하게 이미지 읽기
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def load_all_images(img_dir: str) -> list[Document]:
    image_docs: list[Document] = []
    import easyocr
    reader = easyocr.Reader(["ko", "en"])

    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        img_paths.extend(glob(os.path.join(img_dir, ext)))

    for img_path in img_paths:
        img = _imread_unicode(img_path)
        
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        texts = reader.readtext(img, detail=0)
        ocr_text = "\n".join(texts).strip()

        if not ocr_text:
            continue

        image_docs.append(
            Document(
                page_content=ocr_text,
                metadata={
                    "source_type": "image",
                    "source_file": os.path.basename(img_path),
                    "source_path": img_path,
                },
            )
        )
    return image_docs


# =====================================
# [Step 1] 데이터 읽기 
# =====================================
pdf_docs = load_processed_pdf_chunks(CHUNKS_JSONL_PATH)
csv_docs = load_all_csvs(CSV_DIR)
image_docs = load_all_images(IMG_DIR)

all_docs = pdf_docs + csv_docs + image_docs
analyze_docs(all_docs, "전 데이터 통합")

print(sorted(set(d.metadata.get("source_file") for d in pdf_docs))[:20])

# =====================================
# [Step 2] 텍스트 분할
# =====================================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=90)

pdf_chunks = pdf_docs
other_docs = csv_docs + image_docs
other_chunks = text_splitter.split_documents(other_docs)

chunks = pdf_chunks + other_chunks

analyze_docs(chunks, "전처리 ON")

print("pdf_docs:", len(pdf_docs))
print("csv_docs:", len(csv_docs))

print("all_docs:", len(all_docs))


# =====================================
# [Step 3] 임베딩 &  DB 만들기
# =====================================
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "cloud")

@st.cache_resource(show_spinner="벡터 DB 로드 중...")
def load_vectorstore():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ✅ Cloud에선 '반드시' 기존 DB만 로드 (생성 금지)
    if DEPLOY_MODE == "cloud":
        if not (os.path.exists(DB_PATH_PROCESSED) and os.listdir(DB_PATH_PROCESSED)):
            raise RuntimeError(
                f"[Cloud] Vector DB not found at {DB_PATH_PROCESSED}. "
                "로컬에서 전처리/임베딩 후 db 폴더를 GitHub에 포함시켜야 합니다."
            )
        return Chroma(
            persist_directory=DB_PATH_PROCESSED,
            embedding_function=embeddings,
            collection_name="smart_factory_processed",
        )

    # ✅ 로컬에서는 없으면 생성 허용
    if os.path.exists(DB_PATH_PROCESSED) and os.listdir(DB_PATH_PROCESSED):
        return Chroma(
            persist_directory=DB_PATH_PROCESSED,
            embedding_function=embeddings,
            collection_name="smart_factory_processed",
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,  # 로컬에서만 생성
        embedding=embeddings,
        collection_name="smart_factory_processed",
        persist_directory=DB_PATH_PROCESSED,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.persist()
    return vectorstore

print("\n 질문을 입력하세요.")



# =====================================
# [Step 4] 질의 처리 및 RAG 응답 생성 
# =====================================
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 각 답변마다 출처 달기 
def format_source(meta: dict) -> str:
    stype = meta.get("source_type")

    if stype == "pdf":
        return f"{meta.get('source_file')} p.{meta.get('page_human', meta.get('page'))}"
    if stype == "csv":
        return f"{meta.get('source_file')} row.{meta.get('row')}"
    if stype == "sql":
        return f"{meta.get('source_file')} / {meta.get('table')} row.{meta.get('row')}"
    if stype == "image":
        return f"{meta.get('source_file')} (image)"

    # 혹시 누락 대비
    return f"{meta.get('source_file', 'unknown')}"

def make_context(use_docs: list[Document]) -> str:
    parts = []

    for d in use_docs:
        src = format_source(d.metadata)
        parts.append(
            f"{d.page_content}\n(출처: {src})"
        )

    return "\n\n---\n\n".join(parts)


def retrieve_context(query: str, k: int= 7):
    vectorstore = load_vectorstore()
    pairs = vectorstore.similarity_search_with_score(query, k=k)
    return pairs 


def build_prompt_pdf(query: str, context: str, profile: dict | None = None) -> str:
    profile = profile or {}
    industry = profile.get("industry", "미입력")
    size = profile.get("size", "미입력")
    pain = profile.get("pain", "미입력")
    process = profile.get("process", "미입력")

    return f"""
    너는 중소기업을 위한 스마트공장 도입 컨설턴트 AI야.
    다음 CONTEXT에는 문서 검색 결과와 DB 조회 결과가 포함되어 있다.
    DB 조회 결과가 존재하면 이를 우선적으로 활용하시오.
    답변을 작성할 때, 불릿(-) 또는 번호(1., 2., 3.) 형식으로 작성하라. 
    사례를 설명할 때, [회사명] [본문] 의 형식으로 작성하라. 
    컨텍스트에 해당 정보가 없을 경우에만 “제공된 데이터에서 해당 정보를 찾을 수 없습니다”라고 답해라.

    [사용자 기업 정보]
    - 업종: {industry}
    - 규모: {size}
    - 주요 고민: {pain}
    - 공정 특징: {process}

    답변을 마친 후, 아래 조건을 만족하는 "다음에 할 수 있는 질문 예시"를 2~4개 제안해.

    [질문 제안 규칙]
    - 반드시 제공된 발췌문(PDF)에 근거해 제안할 것
    - 새로운 정보나 문서에 없는 내용을 가정하지 말 것
    - 사용자가 스마트공장을 처음 도입하는 중소기업 사장이라는 전제를 유지할 것
    - 질문 유형은 서로 다르게 구성할 것

    [질문 유형 예시]
    (1) 개념 이해 질문
    (2) 사례 질문
    (3) 도입 절차/방법 질문
    (4) 의사결정/컨설팅 질문

    [출처 표기 규칙 - 매우 중요]
    - 답변의 모든 문장 끝에는 반드시 출처를 붙여라.
    - 출처 형식: "(출처: 파일명 p.쪽)" 또는 "(출처: 파일명 row.행)"
    - 출처 문구는 CONTEXT 안에 있는 "(출처: ...)" 문자열을 그대로 복사해서 붙여라.
    - CONTEXT에 없는 내용은 추측하지 말고 "제공된 데이터에서 해당 정보를 찾을 수 없습니다."라고 답한다.

    질문: {query}
    발췌문: {context}
""".strip()

def build_prompt_csv(query: str, context: str) -> str:
    return f"""
    너는 CSV 근거만으로 답한다.
    CONTEXT에 있는 사실만 사용하고, 정의/효과/역할/추측/일반상식 설명은 절대 하지 마라.

    [출력 규칙]
    1) CONTEXT에 확인된 기술 키워드를 추출하라.
    2) 질문에 명시된 업종(예: 식품 제조업)에 해당하는 정보만 추출하라. 업종 문자열이 정확히 일치하거나 의미상 동일한 경우만 허용한다.
    3) 동일/유사 기술은 3~5개 카테고리로 묶어 재정리하라 (새 정보 추가 금지).
    4) 각 카테고리 끝에, 해당 키워드가 등장한 출처를 "(출처: ... row.xxx)" 형식으로 표기하라.
    5) (선택) 용어의 일반적 설명이 필요하면 “일반 설명:”으로 표시하고, CSV에 근거가 없음을 명시하라.
    6) CONTEXT에 없는 내용은 추가하지 말고 "제공된 데이터에서 해당 정보를 찾을 수 없습니다."라고 답한다. 
    7) 마지막에는 Markdown 표 형식으로 요약하라. (열 구성: 회사명 | 적용 업종 | 제공 기술)
       - 제공 기술은 쉼표(,)로 구분된 키워드만 작성한다.
       - 출처는 표 아래에 한 출로 정리한다.

    답변을 마친 후, 아래 조건을 만족하는 "다음에 할 수 있는 질문 예시"를 2~4개 제안해.

    [질문 제안 규칙]
    - 반드시 제공된 발췌문(CSV)에 근거해 제안할 것
    - 새로운 기술이나 개념을 추가하지 말 것
    - 사용자가 스마트공장을 처음 도입하는 중소기업 사장이라는 전제를 유지할 것
    - 질문은 데이터 조회/비교/필터링 중심으로 구성할 것

    [질문 유형 예시]
    (1) 특정 기술 보유 기업 조회 질문
    (2) 특정 업종 대상 공급 기업 조회 질문 
    (3) 스마트 공장 도입한 기업의 기술 조회 질문 
    (4) 의사결정/컨설팅 질문


    질문: {query}
    발췌문: {context}
""".strip()


def pick_mode(pairs, query: str):
    """
    1) 질문 의도가 '공급기업/제공기술/업종'이면 CSV 우선
    2) 그 외에는 기존처럼 score 기반으로 PDF/CSV 선택
    """
    q = (query or "").lower()

    # 1) CSV가 우선인 질문 패턴 (너가 겪는 케이스 방지용)
    csv_intent_terms = ["공급", "공급기업", "공급 기업", "기업", "업종", "제공", "전문 기술", "전문기술", "키워드"]
    if any(t in q for t in csv_intent_terms):
        # 검색 결과에 csv 문서가 실제로 포함되어 있을 때만 csv 강제
        if any(d.metadata.get("source_type") == "csv" for d, _ in pairs):
            return "csv"

    # 2) PDF가 우선인 질문 패턴 (사례/도입 절차 등)
    pdf_intent_terms = ["사례", "성공", "도입", "절차", "단계", "어떻게", "방법", "효과", "개념"]
    if any(t in q for t in pdf_intent_terms):
        if any(d.metadata.get("source_type") == "pdf" for d, _ in pairs):
            return "pdf"

    # 3) fallback: score 기반(기존 로직 유지)
    scores = defaultdict(list)
    for d, s in pairs:
        t = d.metadata.get("source_type", "unknown")
        if t in ("csv", "pdf"):
            scores[t].append(s)

    if not scores:
        return "pdf"

    avg = {t: sum(v) / len(v) for t, v in scores.items()}
    return min(avg, key=avg.get)  # distance 낮을수록 유사하다고 가정

def ask_rag(query: str, pairs: list, profile: dict | None = None, db_context: str = ""):
    profile = profile or {}

    # pairs는 [(Document, score), ...]
    # 1) 모드 결정
    mode = pick_mode(pairs, query)

    # 2) 타입별로 문서 필터링
    use_docs = [d for d, _ in pairs if d.metadata.get("source_type") == mode]

    # 3) fallback: 선택된 타입이 비었으면 전체 사용
    if not use_docs:
        use_docs = [d for d, _ in pairs]

    # 4) context 생성 (출처 포함)
    context = make_context(use_docs)
    db_ctx = ""
    try:
        db_result = get_db_context(query, llm)
        st.session_state["db_result"] = db_result
        print("✅ DB_AGENT error:", db_result.get("error"))
        print("✅ DB_AGENT sql:", db_result.get("sql"))
        print("✅ DB_AGENT rows:", len(db_result.get("rows", [])))

        if db_result.get("error") is None and db_result.get("db_context_text"):
            db_ctx = (
                "[DB 조회 결과]\n"
                + db_result["db_context_text"]
                + "\n\n[SQL]\n"
                + db_result.get("sql", "")
                + "\n(출처: PostgreSQL ragdb)"
        )
    except Exception as e:
    # 디버깅용 (필요하면 print)
       print("DB_AGENT_ERROR:", e)
    
    if db_ctx:
        context += "\n\n" + db_ctx



    # 5) 프롬프트 선택
    if db_ctx:  # ✅ DB가 있으면 PDF 프롬프트로 강제 (CSV 규칙 충돌 방지)
        prompt = build_prompt_pdf(query, context, profile)
        print("✅ DB_CONTEXT_HEAD:", (db_context[:200] + "..."))
    else:
        if mode == "csv":
            prompt = build_prompt_csv(query, context)
        else:
            prompt = build_prompt_pdf(query, context, profile)

    print("DB_CTX_LEN:", len(db_ctx))

    # 6) LLM 호출
    response = llm.invoke(prompt)

    return response.content, context, mode
