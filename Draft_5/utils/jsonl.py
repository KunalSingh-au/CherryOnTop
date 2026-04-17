"""Load doc_id → field maps from JSONL (extracted / translated docs)."""

from __future__ import annotations

import json
import os


def load_docs_jsonl(path: str, field: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["doc_id"]] = str(r.get(field) or "")
    return out
