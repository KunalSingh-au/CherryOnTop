"""
utils/extract.py — PDF text extraction and parliamentary document parsing.

Two parsers are provided:
  - parse_english_qa()  for English Lok Sabha PDFs
  - parse_hindi_qa()    for official Hindi Lok Sabha PDFs

Both return (question_text, answer_text).  If the expected headers are not
found, the full page text is returned as the answer (safe fallback).
"""

import re
import pdfplumber


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT READER
# ─────────────────────────────────────────────────────────────────────────────

def read_pdf(path: str) -> str:
    """
    Extract raw text from a PDF using pdfplumber (layout-aware).

    Falls back gracefully and returns a PDF_ERROR string if the file cannot
    be opened, so downstream code never crashes on a bad PDF.

    ── TUNABLE: If Hindi PDFs produce garbled text, try replacing pdfplumber
       with PyMuPDF: `import fitz; doc = fitz.open(path)` ──────────────────
    """
    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        return f"PDF_ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# ENGLISH PARLIAMENTARY DOCUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_english_qa(text: str) -> tuple:
    """
    Split a Lok Sabha English QA document into (question, answer) parts.

    Looks for:
      - Question start: "Will the Minister/Government ..."
      - Answer start:   "ANSWER" header on its own line

    Returns full text as answer if structure is not found (safe fallback).
    """
    q_match = re.search(
        r"Will the (?:Minister|Ministry|Government|Chairman|Speaker)", text, re.I
    )
    a_match = re.search(r"\nANSWER\s*\n", text, re.I)

    if not q_match or not a_match:
        # Fallback: treat entire text as answer (still usable as context)
        return "", text.strip()

    question = text[q_match.start(): a_match.start()].strip()
    answer   = text[a_match.end():].strip()

    # Trim at Annexure section — usually data tables not needed for QA
    cut = re.search(r"\nAnnexure\b|\n\*{3,}", answer, re.I)
    if cut:
        answer = answer[: cut.start()].strip()

    return question, answer


# ─────────────────────────────────────────────────────────────────────────────
# HINDI PARLIAMENTARY DOCUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_hindi_qa(text: str) -> tuple:
    """
    Split a Lok Sabha Hindi QA document into (question, answer) parts.

    Looks for:
      - Answer header: "उत्तर" (Uttar) or "UTTAR" on its own line
      - Question start: common honorific phrases

    Returns full text as answer if structure is not found.
    """
    # Match answer header — handles Unicode and romanised variants
    a_match = re.search(
        r"(?:^|\n)\s*(?:उत्तर|UTTAR|Answer)\s*[:：]?\s*\n",
        text,
        re.I | re.M,
    )
    if not a_match:
        # Secondary: English ANSWER header in a Hindi doc
        a_match = re.search(r"\nANSWER\s*\n", text, re.I)

    q_match = re.search(
        r"(?:क्या सरकार|क्या माननीय|श्रीमती|श्री\s|whether\s+the)",
        text,
        re.I,
    )

    if a_match and q_match and q_match.start() < a_match.start():
        question = text[q_match.start(): a_match.start()].strip()
        answer   = text[a_match.end():].strip()
        return question, answer

    # Fallback: full text as answer
    return "", text.strip()
