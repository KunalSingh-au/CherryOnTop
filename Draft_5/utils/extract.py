# PDF extraction — English and Hindi Lok Sabha–style Q&A blocks.

from __future__ import annotations

import re
import pdfplumber


def read_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        return f"PDF_ERROR: {e}"


def parse_english_qa(text: str) -> tuple[str, str]:
    """(question, answer) from English Lok Sabha PDF."""
    q = re.search(
        r"Will the (?:Minister|Ministry|Government|Chairman|Speaker)",
        text,
        re.I,
    )
    a = re.search(r"\nANSWER\s*\n", text, re.I)
    if not q or not a:
        return "", ""

    question = text[q.start() : a.start()].strip()
    answer_raw = text[a.end() :].strip()
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
        answer = answer[: cut.start()]
    return question.strip(), answer.strip()


def parse_hindi_qa(text: str) -> tuple[str, str]:
    """(question, answer) from Hindi official PDF — tolerant patterns."""
    if not text or text.startswith("PDF_ERROR"):
        return "", ""

    # Answer markers (Hindi / mixed)
    a = re.search(
        r"(?:^|\n)\s*(?:उत्तर|UTTAR|Answer)\s*[:：]?\s*\n",
        text,
        re.I | re.M,
    )
    if not a:
        a = re.search(r"\nANSWER\s*\n", text, re.I)

    # Question: often starts after star header or "क्या सरकार"
    q_start = re.search(
        r"(?:क्या सरकार|क्या माननीय|श्रीमती|श्री\s|whether\s+the)",
        text,
        re.I,
    )
    if a and q_start and q_start.start() < a.start():
        question = text[q_start.start() : a.start()].strip()
        answer_raw = text[a.end() :].strip()
    elif a:
        question = text[: a.start()].strip()[-4000:]
        answer_raw = text[a.end() :].strip()
    else:
        return "", ""

    lines = answer_raw.split("\n")
    start = 0
    for i, line in enumerate(lines):
        l = line.strip()
        if re.match(r"^मंत्री|^सचिव|^\(.*\)$", l) or not l:
            start = i + 1
        else:
            break
    answer = "\n".join(lines[start:])
    cut = re.search(r"\n(?:Annexure|अनुबंध|Annex)\b|\n\*{3,}", answer, re.I)
    if cut:
        answer = answer[: cut.start()]
    return question.strip(), answer.strip()
