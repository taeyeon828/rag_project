import os

DEPLOY_MODE = os.getenv("DEPLOY_MODE", "cloud") 
if DEPLOY_MODE == "cloud":
    from main_cloud import ask_rag, retrieve_context
else:
    from main import ask_rag, retrieve_context 

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from db_agent.db_agent import get_db_context
from dotenv import load_dotenv
if DEPLOY_MODE != "cloud":
    load_dotenv()

@st.cache_resource
def get_engine():
    from sqlalchemy import create_engine
    return create_engine(st.secrets["DB_URL"])

engine = get_engine()


st.set_page_config(page_title="스마트공장 도입 가이드 시스템", layout="wide")
st.title("스마트공장 도입 가이드 시스템")

from sqlalchemy import text

with engine.connect() as conn:
    row = conn.execute(text("SELECT current_database(), current_schema(), current_user")).fetchone()
    st.write("DEBUG - current_database:", row[0])
    st.write("DEBUG - current_schema:", row[1])
    st.write("DEBUG - current_user:", row[2])


# =====================================
# 초기화
# =====================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "started" not in st.session_state:
    st.session_state["started"] = False

# 기업 프로필
if "profile" not in st.session_state:
    st.session_state["profile"] = {}

if "db_context" not in st.session_state:
    st.session_state["db_context"] = ""

# =====================================
# 사용자 정보 입력
# =====================================
def company_intake_form():
    st.subheader("컨설팅 시작 전, 기업 정보를 입력해주세요")

    with st.form("company_form"):
        industry = st.selectbox("업종", ["식품 제조업", "금속 제조업", "통신장비 제조업", "기타"])
        size = st.selectbox("기업 규모", ["10인 미만", "10~49인", "50~99인", "100인 이상"])
        revenue = st.text_input("매출(선택)", placeholder="예: 30억 / 미기재")
        pain = st.multiselect("가장 큰 고민", ["인력 부족", "품질/불량", "위생/추적성", "납기", "원가", "기타"])
        process = st.text_area("현재 생산 방식/공정 특징(선택)", placeholder="예: 수작업 위주, 배합-가공-포장...")

        submitted = st.form_submit_button("저장하고 시작하기")

    if submitted:
        st.session_state["profile"] = {
            "industry": industry,
            "size": size,
            "revenue": revenue.strip(),
            "pain": pain,
            "process": process.strip(),
        }
        st.session_state["started"] = True
        st.rerun()


if not st.session_state["started"]:
    company_intake_form()
    st.stop()

def decide_source_mode(query: str, db_ctx: str) -> str:
    q = (query or "").lower()

    db_terms = [
        "db", "데이터베이스", "테이블", "컬럼", "행", "조회", "목록",
        "건수", "개수", "몇 개", "몇건", "몇 건",
        "평균", "합계", "최대", "최소", "순위", "상위", "하위",
        "라인별", "설비별", "공정별",
        "생산량", "불량률", "가동률", "재고", "수율"
    ]
    pdf_terms = ["사례", "도입", "절차", "단계", "방법", "효과", "개념", "설명"]
    csv_terms = ["공급기업", "공급 기업", "제공 기술", "전문기술", "업종", "키워드"]

    if any(term in q for term in db_terms) and db_ctx:
        return "db"
    if any(term in q for term in csv_terms):
        return "csv"
    if any(term in q for term in pdf_terms):
        return "pdf"

    return "pdf"

# =====================================
# 이전 메시지 출력
# =====================================
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =====================================
# 채팅 입력창
# =====================================
user_text = st.chat_input("질문을 입력하세요")

if user_text:
    # 1) 사용자 메시지 저장 + 출력
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) RAG 검색 + 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 생성 중..."):
            pairs = retrieve_context(user_text)

            from langchain_google_genai import ChatGoogleGenerativeAI
            @st.cache_resource
            def get_llm():
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=st.secrets["GEMINI_API_KEY"],
                    temperature=0.2,
                    )
            llm = get_llm()

            db_result = get_db_context(user_text, llm, engine)
            st.write("DEBUG - db_result:", db_result)
            st.write("DEBUG - db_result error:", db_result.get("error"))
            st.write("DEBUG - db_result sql:", db_result.get("sql"))

            st.session_state["db_result"] = db_result
            db_ctx = ""
            if db_result.get("error") is None and db_result.get("db_context_text"):
                db_ctx = db_result["db_context_text"]
                
            source_mode = decide_source_mode(user_text, db_ctx)
            st.write("DEBUG - db_ctx:", db_ctx)
            st.write("DEBUG - db_ctx length:", len(db_ctx))
            st.write("DEBUG - source_mode:", source_mode)
            answer = ask_rag(
                user_text,
                pairs,
                profile=st.session_state.get("profile", {}),
                db_context=db_ctx,
                source_mode=source_mode,
)
    
        st.markdown(answer)
        
        if "db_result" in st.session_state:
            db_result = st.session_state["db_result"]
            if db_result.get("sql"):
                with st.expander("사용된 SQL 보기"):
                    st.code(db_result["sql"], language="sql")

        # 3) 근거(발췌문/출처) 출력
        with st.expander("근거(발췌문/출처) 보기"):
            if isinstance(pairs, list) and len(pairs) > 0:
                for i, d in enumerate(pairs, start=1):
                    if isinstance(d, tuple):
                        d = d[0]
                    
                    if hasattr(d, "metadata") and hasattr(d, "page_content"):
                        meta = d.metadata or {}
                        source = meta.get("source_file", meta.get("source", "unknown"))
                        st.markdown(f"**[{i}]** {source}")
                        st.write(d.page_content[:800])
                    
                    elif isinstance(d, dict):
                        source = d.get("source", "unknown")
                        st.markdown(f"**[{i}]** {source}")
                        st.write(d.get("text", "")[:800])
                        
                    else:
                        st.write(f"(지원하지 않는 타입: {type(d)})")
            else:
                st.write("(문서 검색 결과 없음)")


    st.session_state["messages"].append({"role": "assistant", "content": answer})