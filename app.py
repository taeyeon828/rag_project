import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from main import ask_rag, retrieve_context
from db_agent.db_agent import get_db_context
from main import retrieve_context, ask_rag, llm
from dotenv import load_dotenv
load_dotenv()

if "engine" not in st.session_state:
    st.session_state["engine"] = create_engine(os.getenv("DB_URL"))

engine = st.session_state["engine"]


st.set_page_config(page_title="스마트공장 챗봇", layout="wide")
st.title("스마트공장 도입 컨설턴트 AI")


# =====================================
# 초기화
# =====================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "started" not in st.session_state:
    st.session_state["started"] = False

# 기업 프로필(세션 저장)
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


# company_intake_form() 호출
if not st.session_state["started"]:
    company_intake_form()
    st.stop()

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

            # ✅ DB Agent 실행
            db_result = get_db_context(user_text, llm, engine)
            db_ctx = ""
            if db_result["error"] is None and db_result["db_context_text"]:
                db_ctx = db_result["db_context_text"]
                
            answer, context, mode = ask_rag(
                user_text,
                pairs,
                profile=st.session_state.get("profile", {}),
                db_context=db_ctx
    )
        st.markdown(answer)
        
        if "db_result" in st.session_state:
            db_result = st.session_state["db_result"]
            if db_result.get("sql"):
                with st.expander("사용된 SQL 보기"):
                    st.code(db_result["sql"], language="sql")

        # 3) 근거(발췌문/출처) 출력
        with st.expander("근거(발췌문/출처) 보기"):
            for i, d in enumerate(pairs, start=1):
                if isinstance(d, tuple):
                    d = d[0]
                meta = d.metadata or {}
                source = meta.get("source_file", meta.get("source", "unknown"))
                st.markdown(f"**[{i}]** {source}")
                st.write(d.page_content[:800])

    st.session_state["messages"].append({"role": "assistant", "content": answer})