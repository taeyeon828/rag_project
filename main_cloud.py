import os
from typing import List, Dict, Any

def retrieve_context(user_query: str) -> str:
    # Cloud 데모에서는 문서 벡터 검색을 잠깐 끄고, DB 컨텍스트만 사용(또는 빈 문자열)
    return ""

def ask_rag(user_query: str, docs_or_context, profile: dict | None = None) -> str:
    """
    docs_or_context: app.py에서 넘기는 retrieved_context 또는 docs
    """
    # Gemini 키 가져오기
    import streamlit as st
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )

    context = docs_or_context if isinstance(docs_or_context, str) else ""
    prompt = f"""너는 스마트공장 컨설팅 AI이다.
아래 컨텍스트를 참고하여 질문에 답하라.
- 모르면 모른다고 말하라.
- 답변 끝에 '근거' 섹션에 컨텍스트 요약을 3줄 이내로 첨부하라.

[질문]
{user_query}

[컨텍스트]
{context}
"""
    return llm.invoke(prompt).content