import os
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import build_prompt 

DOCS_DIR = Path("data") 

def _simple_score(query: str, text: str) -> int:
    q = re.findall(r"[0-9A-Za-z가-힣]+", query.lower())
    t = text.lower()
    return sum(1 for tok in q if tok in t)

@st.cache_data
def _load_pdf_texts():
    # 가벼운 PDF 텍스트 로딩
    from pypdf import PdfReader
    pdf_texts = []
    for p in DOCS_DIR.rglob("*.pdf"):
        try:
            reader = PdfReader(str(p))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if text.strip():
                pdf_texts.append((str(p), text))
        except Exception:
            pass
    return pdf_texts

@st.cache_data
def _load_csv_texts(max_rows=200):
    csv_texts = []
    for p in DOCS_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(p)
            df_head = df.head(max_rows)
            csv_texts.append((str(p), df_head.to_csv(index=False)))
        except Exception:
            pass
    return csv_texts

def retrieve_context(user_query: str, top_k: int = 3) -> list[dict]:
    # ✅ “벡터 검색” 대신 “키워드 점수”로 상위 문서 뽑기
    candidates = []
    for src, text in _load_pdf_texts():
        candidates.append((src, text, _simple_score(user_query, text)))
    for src, text in _load_csv_texts():
        candidates.append((src, text, _simple_score(user_query, text)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    results = []
    for src, text, score in candidates[:top_k]:
        if score <= 0:
            continue
        # 너무 길면 잘라서 컨텍스트로
        results.append({"source": src, "snippet": text[:2000]})
    return results

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )

def ask_rag(user_query: str, pairs, profile=None, db_context: str = "") -> str:
    llm = get_llm()

    doc_ctx = ""
    if isinstance(pairs, list) and pairs:
        doc_ctx = "\n\n".join([f"[{p['source']}]\n{p['snippet']}" for p in pairs])

    context = f"""[DB 컨텍스트]
{db_context}

[문서 컨텍스트]
{doc_ctx}
"""

    prompt = build_prompt(query=user_query, context=context, profile=profile)


    return llm.invoke(prompt).content