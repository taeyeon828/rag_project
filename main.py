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

# Preprocess
REPORT_DIR = os.path.join(PROCESSED_DIR, "report")
QUALITY_REPORT_PATH = os.path.join(REPORT_DIR, "quality_report.json")
RAW_PDF_DIR = Path("data/raw/pdf")
OUT_TEXT_DIR = Path("data/processed/text")
OUT_CHUNKS_DIR = Path("data/processed/chunks")
OUT_REPORT_DIR = Path("data/processed/report")



import streamlit as st
import numpy as np
import cv2
import json
import re
from glob import glob
from datetime import datetime
from langchain_core.documents import Document
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


# Gemini 설정
# Mac: 터미널에서 export GEMINI_API_KEY="내가 발급받은 API 키"
# Windows: PowerShell에서 $env:GEMINI_API_KEY= “내가 발급받은 API 키"


# =====================================
# 데이터 처리 함수
# =====================================
def preprocess_pdfs_if_needed():
    os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_CHUNKS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    if os.path.exists(CHUNKS_JSONL_PATH) and os.path.exists(QUALITY_REPORT_PATH):
        return
    
    def clean_text(text: str) -> str:
        if not text:
            return ""
        
        text = text.replace("\ufeff", "")         
        text = text.replace("\xa0", " ")        
        text = re.sub(r"\s+", " ", text)           
        text = text.strip()          
        return text
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=90,
    separators=["\n\n", "\n", ". ", "다. ", " "]
    )
    
    chunks_path = OUT_CHUNKS_DIR / "chunks.jsonl"
    report = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "files": []
    }
    
    with chunks_path.open("w", encoding="utf-8") as fout:
        for pdf_path in RAW_PDF_DIR.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            file_stat = {
                "file": pdf_path.name,
                "pages": len(pages),
                "empty_pages": 0,
                "chunks": 0
        }
            
            file_text_dir = OUT_TEXT_DIR / pdf_path.stem
            file_text_dir.mkdir(parents=True, exist_ok=True)

            cleaned_pages = []
            for i, doc in enumerate(pages, start=1):
                raw = doc.page_content or ""
                cleaned = clean_text(raw)
            
                if len(cleaned) < 10:
                    file_stat["empty_pages"] += 1

            (file_text_dir / f"page_{i:03d}.txt").write_text(cleaned, encoding="utf-8")
            cleaned_pages.append((i, cleaned))

            for page_no, text in cleaned_pages:
                if not text:
                    continue
                docs = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "source_type": "pdf",
                        "source": pdf_path.name,
                        "page": page_no,
                }]
            )
                for j, d in enumerate(docs, start=1):
                    rec = {
                        "chunk_id": f"{pdf_path.stem}_p{page_no:03d}_c{j:03d}",
                        "text": d.page_content,
                        **d.metadata,
                        "char_len": len(d.page_content)
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                file_stat["chunks"] += 1

        report["files"].append(file_stat)

    (OUT_REPORT_DIR / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
        )



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

            meta = {k: v for k, v in rec.items() if k not in ("text",)}

            meta.setdefault("source_type", "pdf")
            meta.setdefault("source_file", meta.get("source") or meta.get("source_file"))
            meta.setdefault("source_path", meta.get("source_path")) 
            if "page" in meta and isinstance(meta["page"], int):
                meta["page_human"] = meta["page"]

            docs.append(Document(page_content=text, metadata=meta))

    return docs

rep = json.loads(Path("data/processed/report/quality_report.json").read_text(encoding="utf-8"))
for f in rep["files"]:
    print(f["file"], "pages=", f["pages"], "chunks=", f["chunks"], "empty_pages=", f["empty_pages"])

# CSV 데이터 전처리 
def clean_csv_value(v: str) -> str:
    v = str(v)
    v = v.replace("\ufeff", "")          
    v = v.strip()                        
    v = re.sub(r"\s+", " ", v)           
    return v

def load_all_csvs(csv_dir: str):
    docs = []
    seen = set() 

    for csv_path in glob(os.path.join(csv_dir, "*.csv")):
        with open(csv_path, "r", encoding="cp949", errors="ignore") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):

                # (1) row 값 정제
                cleaned_row = {}
                for k, v in row.items():
                    if not k:
                        continue
                    if v is None:
                        continue
                    vv = clean_csv_value(v)
                    if vv:  
                        cleaned_row[k] = vv

                lines = [f"[{k}] {v}" for k, v in cleaned_row.items()]
                text = "\n".join(lines).strip()
                if not text:
                    continue

                # (2) 중복 row 제거 
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
preprocess_pdfs_if_needed()

pdf_docs = load_processed_pdf_chunks(CHUNKS_JSONL_PATH)
csv_docs = load_all_csvs(CSV_DIR)
image_docs = load_all_images(IMG_DIR)

all_docs = pdf_docs + csv_docs + image_docs


# =====================================
# [Step 2] 텍스트 분할
# =====================================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=90)

pdf_chunks = pdf_docs
other_docs = csv_docs + image_docs
other_chunks = text_splitter.split_documents(other_docs)

chunks = pdf_chunks + other_chunks

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

    if os.path.exists(DB_PATH_PROCESSED) and os.listdir(DB_PATH_PROCESSED):
        return Chroma(
            persist_directory=DB_PATH_PROCESSED,
            embedding_function=embeddings,
            collection_name="smart_factory_processed",
        )

    vectorstore = Chroma.from_documents(
        documents=chunks, 
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
    너는 중소기업을 위한 스마트공장 도입 가이드 AI야.
    답변은 위 기업 정보에 맞춰 우선순위/적용 포인트를 조정하되, 사실 근거는 제공된 문서 발췌문만 사용한다.
    근거가 부족하면 '문서에서 해당 내용을 찾지 못했다' 또는 '제공된 통계에서 확인되지 않았다.'고 말해.
    사용자는 스마트공장을 처음 도입하려는 중소기업 사장이라고 가정해. 

    [사용자 기업 정보]
    - 업종: {industry}
    - 규모: {size}
    - 주요 고민: {pain}
    - 공정 특징: {process}

    1. 질문이 ‘판단/의사결정’을 요구할 경우
       - 결론 요약 → 왜냐하면(근거) → 현장 적용 포인트

    2. 질문이 ‘사례/이해’를 요구할 경우:  
       - 도입 배경 → 무엇을 어떻게 바꿨는지 → 결과 → 마지막에 반드시 다음 단락을 추가한다: '만약 우리 공장에 적용한다면 지금 당장 생각해볼 점'

    3. 질문이 '개념 설명'을 요구할 경우: 
       - 개념의 정의 → 스마트공장/제조 현장에서의 역할 → 도입 시 기대할 수 있는 효과

    4. 도입 배경을 현장 상황 중심으로 구체적으로 설명한다
       - 누가, 어디서, 어떤 작업 중 어떤 문제가 반복됐는지
       - 기술 설명보다 ‘구조· 프로세스 변화’를 우선 설명한다.

    [출처 표기 규칙]
    - 각 문장 또는 각 불릿의 끝에 반드시 출처를 괄호로 표기한다.
    - 출처 형식은 다음과 같이 작성한다: (출처: 파일명) 또는 (출처: 파일명, p.페이지번호)
    - DB 조회 결과를 사용한 문장은 다음과 같이 표기한다: (출처: DB 조회 결과)
    - 출처를 확인할 수 없는 내용은 작성하지 않는다.
    - 하나의 문장에 여러 근거가 있으면 가장 직접적인 출처 1개만 표기한다.

    [답변 형식]
    - 실제 컨설팅 답변처럼 실용적으로 작성 
    - 필요 시 불릿 또는 번호 사용 가능 
    - 출처를 직접 꾸며내지 말 것 
    - 모든 문장 끝 또는 모든 불릿 끝에 반드시 출처 표기 

    답변을 마친 후, 아래 조건을 만족하는 "다음에 할 수 있는 질문 예시"를 2~4개 제안해.

    [질문 제안 규칙]
    - 반드시 제공된 발췌문(PDF)에 근거해 제안할 것
    - 새로운 정보나 문서에 없는 내용을 가정하지 말 것
    - 질문 유형은 서로 다르게 구성할 것

    [질문 유형 예시]
    (1) 개념 이해 질문
    (2) 동종업계 사례 질문
    (3) 도입 절차/방법 질문
    (4) 의사결정/컨설팅 질문

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

def is_db_query(query: str) -> bool:
    q = (query or "").lower()

    db_terms = [
        "db", "데이터베이스", "테이블", "컬럼", "행", "조회", "목록",
        "건수", "개수", "몇 개", "몇건", "몇 건",
        "평균", "합계", "최대", "최소", "순위", "상위", "하위",
        "라인별", "설비별", "공정별",
        "생산량", "불량률", "가동률", "재고", "수율"
    ]

    return any(term in q for term in db_terms)

def build_prompt_db(query: str, db_context: str) -> str:
    return f"""
    너는 데이터베이스 조회 결과를 근거로 답변하는 시스템이다.
    
    [출력 규칙]
    1. 아래 DB 조회 결과만 사용하라.
    2. PDF, CSV, 일반 상식, 추측을 추가하지 마라.
    3. DB 결과에 없는 내용은 "DB 조회 결과에서 확인되지 않았습니다."라고 답하라.
    4. 숫자, 항목명, 비교 결과를 명확히 제시하라.
    5. 목록 조회는 불릿으로, 집계 결과는 숫자를 먼저 제시하라.
    6. 마지막 문장에 반드시 "(출처: DB 조회 결과)"를 붙여라.
    7. 문서 출처(pdf, csv 등)는 절대 쓰지 마라.
    
    [DB 조회 결과]
    {db_context}
    
    질문: {query}
""".strip()

def pick_mode(pairs, query: str):
    """
    1) 질문 의도가 '공급기업/제공기술/업종'이면 CSV 우선
    2) 그 외에는 기존처럼 score 기반으로 PDF/CSV 선택
    """
    q = (query or "").lower()

    csv_intent_terms = ["공급", "공급기업", "공급 기업", "기업", "업종", "제공", "전문 기술", "전문기술", "키워드"]
    if any(t in q for t in csv_intent_terms):
        if any(d.metadata.get("source_type") == "csv" for d, _ in pairs):
            return "csv"

    pdf_intent_terms = ["사례", "성공", "도입", "절차", "단계", "어떻게", "방법", "효과", "개념"]
    if any(t in q for t in pdf_intent_terms):
        if any(d.metadata.get("source_type") == "pdf" for d, _ in pairs):
            return "pdf"

    scores = defaultdict(list)
    for d, s in pairs:
        t = d.metadata.get("source_type", "unknown")
        if t in ("csv", "pdf"):
            scores[t].append(s)

    if not scores:
        return "pdf"

    avg = {t: sum(v) / len(v) for t, v in scores.items()}
    return min(avg, key=avg.get) 

def ask_rag(
    query: str,
    pairs: list,
    profile: dict | None = None,
    db_context: str = "",
    source_mode: str | None = None,
):
    profile = profile or {}
    db_ctx = db_context or ""

    if source_mode is None:
        if db_ctx and is_db_query(query):
            source_mode = "db"
        else:
            source_mode = pick_mode(pairs, query)  # pdf or csv

    if source_mode == "db":
        if db_ctx:
            prompt = build_prompt_db(query, db_ctx)
            response = llm.invoke(prompt)
            return response.content, db_ctx, "db"
        else:
            return "DB 조회 결과를 찾지 못했습니다.", "", "db"

    use_docs = [d for d, _ in pairs if d.metadata.get("source_type") == source_mode]

    if not use_docs:
        use_docs = [d for d, _ in pairs]

    context = make_context(use_docs)

    if source_mode == "csv":
        prompt = build_prompt_csv(query, context)
    else:
        prompt = build_prompt_pdf(query, context, profile)

    response = llm.invoke(prompt)
    return response.content, context, source_mode
