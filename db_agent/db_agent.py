from typing import Dict, Any
import json
import os
from db_agent.executor import run_sql
from db_agent.guard import enforce_policy
from db_agent.schema import build_schema_context
from db_agent.sql_generator import build_prompt, llm_generate_sql
from sqlalchemy import create_engine

ALLOWED_TABLES = {
  "batch_production",
  "production_daily",
  "qc_inspection",
  "kpi_production_line_daily",
  "kpi_product_quality_daily",
}

def format_rows_as_text(rows: list[dict], max_rows: int = 20) -> str:
    if not rows:
        return "DB query returned 0 rows."
    rows = rows[:max_rows]
    lines = []
    for r in rows:
        lines.append(", ".join([f"{k}={v}" for k, v in r.items()]))
    return "\n".join(lines)

def get_db_context(user_question: str, llm, engine=None, max_retry: int = 2) -> Dict[str, Any]:
    if engine is None:
        db_url = os.getenv("DB_URL")
        if not db_url:
            return {"sql": "", "rows": [], "db_context_text": "", "error": "DB_URL is not set in environment."}
        engine = create_engine(db_url)

    last_err = None
    schema_context = build_schema_context(engine, ALLOWED_TABLES)

    for _ in range(max_retry + 1):
        try:
            prompt = build_prompt(user_question, schema_context, sorted(ALLOWED_TABLES))
            out = llm_generate_sql(llm, prompt)

            raw_sql = out.get("sql", "")
            safe_sql = enforce_policy(raw_sql, allowed_tables=ALLOWED_TABLES, max_limit=200)

            rows = run_sql(safe_sql)

            return {
                "sql": safe_sql,
                "rows": rows,
                "db_context_text": format_rows_as_text(rows),
                "error": None,
            }
        except Exception as e:
            last_err = str(e)
            continue

    return {"sql": "", "rows": [], "db_context_text": "", "error": last_err}