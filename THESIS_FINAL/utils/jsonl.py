"""
utils/jsonl.py — Helpers for reading and writing JSONL data files.

JSONL (JSON Lines) is used for all extracted and translated document stores
because it is streamable and does not require loading everything into memory.
"""

import json
import os


def load_docs_jsonl(path: str, field: str) -> dict:
    """
    Load a JSONL file into a dict keyed by doc_id.

    Parameters
    ----------
    path  : Path to the .jsonl file.
    field : The field to extract as the value (e.g. "answer_en", "answer_hi").

    Returns
    -------
    dict mapping doc_id → field value (str).
    Returns an empty dict if the file does not exist yet (safe during setup).
    """
    if not os.path.exists(path):
        return {}

    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                out[record["doc_id"]] = str(record.get(field) or "")
            except (json.JSONDecodeError, KeyError):
                pass   # Skip malformed lines silently
    return out


def write_jsonl(records: list, path: str) -> None:
    """
    Write a list of dicts to a JSONL file.

    Creates parent directories automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
