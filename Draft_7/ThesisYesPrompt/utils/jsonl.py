import json
import os


def load_docs_jsonl(path: str, field: str) -> dict:
    """Load a JSONL file → dict keyed by doc_id with value = field."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                out[r["doc_id"]] = str(r.get(field) or "")
            except Exception:
                pass
    return out
