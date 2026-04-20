import re
import pdfplumber


def read_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        return f"PDF_ERROR: {e}"


def parse_english_qa(text: str) -> tuple:
    q = re.search(r"Will the (?:Minister|Ministry|Government|Chairman|Speaker)", text, re.I)
    a = re.search(r"\nANSWER\s*\n", text, re.I)
    if not q or not a:
        return "", text.strip()   # fallback: return full text
    question = text[q.start(): a.start()].strip()
    answer   = text[a.end():].strip()
    # Cut at Annexure
    cut = re.search(r"\nAnnexure\b|\n\*{3,}", answer, re.I)
    if cut:
        answer = answer[:cut.start()]
    return question, answer.strip()


def parse_hindi_qa(text: str) -> tuple:
    a = re.search(r"(?:^|\n)\s*(?:उत्तर|UTTAR|Answer)\s*[:：]?\s*\n", text, re.I | re.M)
    if not a:
        a = re.search(r"\nANSWER\s*\n", text, re.I)
    q_start = re.search(
        r"(?:क्या सरकार|क्या माननीय|श्रीमती|श्री\s|whether\s+the)", text, re.I
    )
    if a and q_start and q_start.start() < a.start():
        return text[q_start.start(): a.start()].strip(), text[a.end():].strip()
    return "", text.strip()   # fallback: full text
