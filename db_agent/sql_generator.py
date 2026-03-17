import json
from typing import Dict, Any

SYSTEM_PROMPT = """You are a database analyst.
Generate ONE safe SQL query for PostgreSQL.

Rules:
- Output MUST be valid JSON (no markdown, no extra text).
- Only SELECT is allowed.
- Use only allowed tables.
- Always include LIMIT <= 200.
- Use correct table/column names from the schema.

Return JSON with keys:
- sql: string
- tables_used: list of strings
- reasoning: string (short)
"""

def build_prompt(user_question: str, schema_context: str, allowed_tables: list[str]) -> str:
    return f"""{SYSTEM_PROMPT}

Allowed tables: {allowed_tables}

Schema:
{schema_context}

User question:
{user_question}
"""

def llm_generate_sql(llm, prompt: str) -> Dict[str, Any]:
    resp = llm.invoke(prompt)
    text = resp.content.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    return json.loads(text)