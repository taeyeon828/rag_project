import re
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

DOCS_DIR = Path("data")


def _simple_score(query: str, text: str) -> int:
    q = re.findall(r"[0-9A-Za-z가-힣]+", (query or "").lower())
    t = (text or "").lower()
    return sum(1 for tok in q if tok in t)


@st.cache_data
def _load_pdf_texts():
    from pypdf import PdfReader

    pdf_texts = []
    for p in DOCS_DIR.rglob("*.pdf"):
        try:
            reader = PdfReader(str(p))
            pages = []
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages.append((page_idx, page_text))

            if pages:
                full_text = "\n".join(text for _, text in pages)
                pdf_texts.append(
                    {
                        "source_type": "pdf",
                        "source": str(p),
                        "text": full_text[:12000],
                    }
                )
        except Exception:
            pass
    return pdf_texts


@st.cache_data
def _load_csv_texts(max_rows: int = 200):
    csv_texts = []
    for p in DOCS_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(p)
            df_head = df.head(max_rows)
            csv_texts.append(
                {
                    "source_type": "csv",
                    "source": str(p),
                    "text": df_head.to_csv(index=False),
                }
            )
        except Exception:
            pass
    return csv_texts


def retrieve_context(user_query: str, top_k: int = 2) -> list[dict]:
    candidates = []

    for item in _load_pdf_texts():
        score = _simple_score(user_query, item["text"])
        candidates.append({**item, "score": score})

    for item in _load_csv_texts():
        score = _simple_score(user_query, item["text"])
        candidates.append({**item, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)

    results = []
    for item in candidates[:top_k]:
        if item["score"] <= 0:
            continue
        results.append(
            {
                "source_type": item["source_type"],
                "source": item["source"],
                "text": item["text"][:1200],
            }
        )
    return results


def pick_mode(query: str, pairs: list[dict]) -> str:
    q = (query or "").lower()

    csv_terms = ["공급기업", "공급 기업", "업종", "전문기술", "제공 기술", "키워드", "기업"]
    pdf_terms = ["사례", "도입", "절차", "단계", "방법", "효과", "개념", "설명"]

    if any(term in q for term in csv_terms):
        if any(item.get("source_type") == "csv" for item in pairs):
            return "csv"

    if any(term in q for term in pdf_terms):
        if any(item.get("source_type") == "pdf" for item in pairs):
            return "pdf"

    if any(item.get("source_type") == "pdf" for item in pairs):
        return "pdf"
    if any(item.get("source_type") == "csv" for item in pairs):
        return "csv"
    return "pdf"


def make_context(items: list[dict]) -> str:
    parts = []
    for item in items:
        src = item.get("source", "unknown")
        text = item.get("text", "").strip()
        if text:
            parts.append(f"{text}\n(출처: {src})")
    return "\n\n---\n\n".join(parts)


def build_prompt_db(query: str, db_context: str) -> str:
    return f"""
    너는 데이터베이스 조회 결과만을 근거로 답변하는 시스템이다.
    
    [중요 규칙]
    1. 아래 DB 조회 결과만 사용하라.
    2. PDF, CSV, 일반 상식, 추측을 추가하지 마라.
    3. DB 결과에 없는 내용은 "DB 조회 결과에서 확인되지 않았습니다."라고 답하라.
    4. 숫자, 항목명, 비교 결과를 명확히 제시하라.
    5. 목록 조회는 불릿으로, 집계 결과는 숫자를 먼저 제시하라.
    6. 마지막 문장에 반드시 "(출처: DB 조회 결과)"를 붙여라.

    [DB 조회 결과]
    {db_context}

    질문: {query}
""".strip()


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
    
    너는 CSV 데이터에 근거해서만 답하는 시스템이다.
    
    [출력 규칙]
    1. 아래 CSV 발췌문에서 확인되는 정보만 사용하라.
    2. 공급기업, 업종, 제공 기술, 키워드를 중심으로 정리하라.
    3. 없는 정보는 추측하지 말고 "제공된 CSV 데이터에서 확인되지 않았습니다."라고 답하라.
    4. 가능하면 표나 불릿으로 정리하라.
    5. 마지막에는 사용한 데이터 출처를 간단히 언급하라.
    
    [CSV 발췌문]
    {context}

질문: {query}
""".strip()


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )


def ask_rag(
    query: str,
    pairs: list,
    profile: dict | None = None,
    db_context: str = "",
    source_mode: str | None = None,
):
    profile = profile or {}
    db_ctx = db_context or ""
    llm = get_llm()

    if source_mode is None:
        if db_ctx:
            source_mode = "db"
        else:
            source_mode = pick_mode(query, pairs)

    if source_mode == "db":
        if db_ctx:
            prompt = build_prompt_db(query, db_ctx)
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                print("LLM_DB_ERROR:", str(e))
                return "DB 조회 결과는 확인했지만 답변 생성 중 오류가 발생했습니다."
        return "DB 조회 결과를 찾지 못했습니다."

    use_docs = [item for item in pairs if item.get("source_type") == source_mode]
    if not use_docs:
        use_docs = pairs

    context = make_context(use_docs)

    if source_mode == "csv":
        prompt = build_prompt_csv(query, context)
    else:
        prompt = build_prompt_pdf(query, context, profile)

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print("LLM_INVOKE_ERROR:", str(e))
        return "현재 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."