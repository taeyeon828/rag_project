import re

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b", re.I)

def enforce_policy(sql: str, allowed_tables: set[str], max_limit: int = 200) -> str:
    s = (sql or "").strip().rstrip(";")

    if not re.match(r"(?is)^\s*select\b", s):
        raise ValueError("Only SELECT statements are allowed.")

    if FORBIDDEN.search(s):
        raise ValueError("Forbidden keyword detected.")


    tokens = re.findall(r"(?is)\b(from|join)\s+([a-zA-Z0-9_\.]+)", s)
    used = {t[1].split(".")[-1] for t in tokens}
    if used and not used.issubset(allowed_tables):
        raise ValueError(f"Disallowed tables used: {sorted(used - allowed_tables)}")

 
    if re.search(r"(?is)\blimit\b", s) is None:
        s += f" LIMIT {max_limit}"
    else:
        s = re.sub(
            r"(?is)\blimit\s+(\d+)",
            lambda m: f"LIMIT {min(int(m.group(1)), max_limit)}",
            s
        )

    return s