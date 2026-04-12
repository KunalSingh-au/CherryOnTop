# utils/extract.py
# Extract clean English Q and A text from Lok Sabha PDF files.

import re
import pdfplumber


def read_pdf(path: str) -> str:
    """Extract all text from a PDF."""
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return ""


def parse_loksabha_english(text: str) -> tuple[str, str]:
    """
    Parse a Lok Sabha English PDF into (question, answer).

    Question: from 'Will the Minister...' to ANSWER divider.
    Answer:   from after minister name lines to Annexure/*****
    """
    # Question start
    q = re.search(
        r"Will the (?:Minister|Ministry|Government|Chairman|Speaker)",
        text, re.I
    )
    # Answer divider
    a = re.search(r"\nANSWER\s*\n", text, re.I)

    if not q or not a:
        return "", ""

    question = text[q.start():a.start()].strip()

    # Strip minister name/title lines from answer block
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

    # Stop before Annexure / *****
    cut = re.search(r"\nAnnexure\b|\n\*{3,}", answer, re.I)
    if cut:
        answer = answer[:cut.start()]

    return question.strip(), answer.strip()


def split_sentences(text: str, min_len: int = 30) -> list[str]:
    """
    Split English text into individual sentences.
    Strips sub-labels (a) (b) etc. and deduplicates.
    """
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
