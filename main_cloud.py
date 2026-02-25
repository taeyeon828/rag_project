import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def retrieve_context(user_query: str):
    return ""  # 지금은 문서 RAG 끔

def ask_rag(user_query: str, pairs, profile=None, db_context: str = "") -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )

    prompt = f"""
너는 스마트공장 컨설턴트 AI다.
아래 DB 컨텍스트를 참고해 질문에 답하라.
모르면 모른다고 말해라.

[기업 프로필]
{profile}

[DB 컨텍스트]
{db_context}

[질문]
{user_query}
"""
    return llm.invoke(prompt).content