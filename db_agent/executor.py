import os
from sqlalchemy import create_engine, text

_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        db_url = os.getenv("DB_URL")
        if not db_url:
            raise RuntimeError("DB_URL is not set in environment.")
        _ENGINE = create_engine(db_url, pool_pre_ping=True)
    return _ENGINE

def run_sql(sql: str, params: dict | None = None, engine=None) -> list[dict]:
    params = params or {}
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]