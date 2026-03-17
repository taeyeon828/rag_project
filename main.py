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

    [답변 규칙]
    - 질문이 ‘판단/의사결정’을 요구할 경우: 결론 요약 → 왜냐하면(근거) → 현장 적용 포인트
    - 질문이 ‘사례/이해’를 요구할 경우:  도입 배경 → 무엇을 어떻게 바꿨는지 → 결과 → 마지막에 반드시 다음 단락을 추가한다: '만약 우리 공장에 적용한다면 지금 당장 생각해볼 점'
    - 질문이 '개념 설명'을 요구할 경우: 개념의 정의 → 스마트공장/제조 현장에서의 역할 → 도입 시 기대할 수 있는 효과
    - 도입 배경을 현장 상황 중심으로 구체적으로 설명한다(누가, 어디서, 어떤 작업 중 어떤 문제가 반복됐는지) - ‘구조· 프로세스 변화’를 우선 설명하라

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

def ask_rag(query: str, pairs: list, profile: dict | None = None, db_context: str = ""):
    profile = profile or {}

    mode = pick_mode(pairs, query)
    use_docs = [d for d, _ in pairs if d.metadata.get("source_type") == mode]

    if not use_docs:
        use_docs = [d for d, _ in pairs]

    context = make_context(use_docs)
    db_ctx = ""
    try:
        db_result = get_db_context(query, llm)
        st.session_state["db_result"] = db_result
   
        if db_result.get("error") is None and db_result.get("db_context_text"):
            db_ctx = (
                "[DB 조회 결과]\n"
                + db_result["db_context_text"]
                + "\n\n[SQL]\n"
                + db_result.get("sql", "")
                + "\n(출처: PostgreSQL ragdb)"
        )
    except Exception as e:
       print("DB_AGENT_ERROR:", e)
    
    if db_ctx:
        context += "\n\n" + db_ctx



    # 프롬프트 선택
    if db_ctx:  
        prompt = build_prompt_pdf(query, context, profile)
    else:
        if mode == "csv":
            prompt = build_prompt_csv(query, context)
        else:
            prompt = build_prompt_pdf(query, context, profile)

    print("DB_CTX_LEN:", len(db_ctx))

    # LLM 호출
    response = llm.invoke(prompt)

    return response.content, context, mode
