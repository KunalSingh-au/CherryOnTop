# utils/extract.py — PDF reading and Q/A parsing

import re
import pdfplumber


def read_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        return f"PDF_ERROR: {e}"


def parse_english_qa(text: str) -> tuple:
    """Returns (question, answer) from a Lok Sabha English PDF."""
    q = re.search(
        r"Will the (?:Minister|Ministry|Government|Chairman|Speaker)",
        text, re.I
    )
    a = re.search(r"\nANSWER\s*\n", text, re.I)
    if not q or not a:
        return "", ""

    question = text[q.start():a.start()].strip()

    answer_raw = text[a.end():].strip()
    lines = answer_raw.split("\n")
    start = 0
    for i, line in enumerate(lines):
        l = line.strip()
        if re.match(r"^MINISTER|^SECRETARY|^\(.*\)$", l, re.I) or not l:
            start = i + 1
        else:
            break
    answer = "\n".join(lines[start:])

    cut = re.search(r"\nAnnexure\b|\n\*{3,}", answer, re.I)
    if cut:
        answer = answer[:cut.start()]

    return question.strip(), answer.strip()


def split_sentences(text: str, min_len: int = 30) -> list:
    """Split English text into sentences, strip sub-labels, deduplicate."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    result = []
    for s in parts:
        s = re.sub(r"^\([a-zA-Z]\)\s*", "", s.strip()).strip()
        if len(s) < min_len:
            continue
        key = re.sub(r"\s+", "", s.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        result.append(s)
    return result
