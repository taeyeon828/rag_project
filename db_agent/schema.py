from typing import Iterable
from sqlalchemy import inspect

def build_schema_context(engine, allowed_tables: Iterable[str]) -> str:
    allowed = sorted(set(allowed_tables))
    if not allowed:
        return "No tables are allowed."

    inspector = inspect(engine)

    lines = []
    lines.append("You can query the following PostgreSQL tables/views (read-only).")

    for i, table_name in enumerate(allowed, 1):
        lines.append(f"\n{i}) {table_name}")

        try:
            columns = inspector.get_columns(table_name, schema="public")
            pk = inspector.get_pk_constraint(table_name, schema="public")
        except Exception as e:
            lines.append(f"- [schema read error] {e}")
            continue

        if not columns:
            lines.append("- [warning] table not found or no visible columns")
            continue

        for col in columns:
            col_name = col.get("name", "unknown")
            col_type = str(col.get("type", "unknown"))
            lines.append(f"- {col_name} ({col_type})")

        pk_cols = pk.get("constrained_columns") if pk else None
        if pk_cols:
            lines.append(f"Primary Key: ({', '.join(pk_cols)})")

    return "\n".join(lines)