import os
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy

from langchain_community.chat_models import ChatOllama
from main import load_vectorstore, ask_rag


# -----------------------------
# 1) Ollama 평가자(critic) LLM
# -----------------------------
def get_judge_llm():
    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )


# -----------------------------------------
# 2) contexts를 "리스트" 형태로 준비 
# -----------------------------------------
def retrieve_contexts_list(query: str, k: int = 5) -> list[str]:
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=k)

    # Faithfulness는 꾸민 헤더(출처: …)보다 "순수 텍스트"가 더 안정적이라
    # d.page_content만 넣는 걸 추천
    return [d.page_content for d in docs]


# -----------------------------
# 3) 평가 실행
# -----------------------------
def main():
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError('GEMINI_API_KEY가 없습니다. export GEMINI_API_KEY="..." 먼저 해줘.')

    judge_llm = get_judge_llm()

    questions = [
        "식품 제조 중소기업이 스마트공장을 처음 도입할 때 우선 고려할 기술은?",
        "식품 업계에서 위생 관리를 위해 적용된 스마트공장 사례가 있으면 설명해줘.",
        "인력 부족 문제를 해결하기 위한 스마트공장 도입 전략을 단계별로 알려줘.",
    ]

    rows = []
    for q in questions:
        contexts = retrieve_contexts_list(q, k=5)
        answer = ask_rag(q, "\n\n".join(contexts))
        rows.append(
            {
                "question": q,
                "contexts": contexts,  
                "answer": answer,
            }
        )

    dataset = Dataset.from_list(rows)

    # Faithfulness + ResponseRelevancy
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge_llm),
            ResponseRelevancy(llm=judge_llm),
        ],
        llm=judge_llm,
        raise_exceptions=False,
    )

    print("\n=== RAGAS RESULT (평균) ===")
    print(result)

    print("\n=== RAGAS RESULT (문항별) ===")
    print(result.to_pandas())


if __name__ == "__main__":
    main()


