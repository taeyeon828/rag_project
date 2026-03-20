from typing import Iterable
from sqlalchemy import text

def build_schema_context(engine, allowed_tables: Iterable[str]) -> str:
    allowed = sorted(set(allowed_tables))
    if not allowed:
        return "No tables are allowed."
    
    cols_sql = text("""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name = ANY(:tables)
    ORDER BY table_name, ordinal_position
    """)

    cols = conn.execute(cols_sql, {"tables": allowed}).mappings().all()

    # PostgreSQL: PK 
    pk_sql = text("""
    SELECT
      tc.table_name,
      kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    WHERE tc.table_schema='public'
      AND tc.constraint_type='PRIMARY KEY'
      AND tc.table_name = ANY(:tables)
    ORDER BY tc.table_name, kcu.ordinal_position;
    """)

    with engine.begin() as conn:
        cols = conn.execute(cols_sql, {"tables": allowed}).mappings().all()
        pks  = conn.execute(pk_sql,  {"tables": allowed}).mappings().all()

    by_table = {}
    for r in cols:
        by_table.setdefault(r["table_name"], []).append((r["column_name"], r["data_type"]))

    pk_by_table = {}
    for r in pks:
        pk_by_table.setdefault(r["table_name"], []).append(r["column_name"])

    lines = []
    lines.append("You can query the following PostgreSQL tables/views (read-only).")
    for i, t in enumerate(allowed, 1):
        lines.append(f"\n{i}) {t}")
        for col, dtype in by_table.get(t, []):
            lines.append(f"- {col} ({dtype})")
        if t in pk_by_table:
            lines.append(f"Primary Key: ({', '.join(pk_by_table[t])})")
    return "\n".join(lines)