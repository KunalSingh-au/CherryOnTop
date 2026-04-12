import os
import re
import csv
import time
import requests
import pdfplumber
from pathlib import Path

# ================= CONFIG =================
API_KEY = "sk_y9ase7u7_W4hR4pKttAd0wOY3n8DKZ7cW"
ROOT_FOLDER = "/Users/kunalsingh/Documents/Thesis/Draft_2/raw_pdfs"
OUTPUT_FILE = "v2_loksabha_output.csv"

MAX_FILES = 2
SLEEP_SEC = 1.0
MAX_CHARS = 900
INCLUDE_ANNEXURE = False
# ==========================================


# ================= TRANSLATION =================
def translate(text, src, tgt):
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""

    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={
                "api-subscription-key": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "input": text,
                "source_language_code": src,
                "target_language_code": tgt,
                "mode": "formal"
            },
            timeout=60
        )
        d = r.json()
        return d.get("translated_text", f"API_ERROR: {d}")

    except Exception as e:
        return f"NETWORK_ERROR: {e}"


def en_to_hi(text):
    time.sleep(SLEEP_SEC)
    return translate(text, "en-IN", "hi-IN")


def hi_to_en(text):
    time.sleep(SLEEP_SEC)
    return translate(text, "hi-IN", "en-IN")


# ================= FILE DISCOVERY =================
def get_pdf_files(root):
    files = []
    for path in Path(root).rglob("*.pdf"):
        ministry = path.parent.name
        fid = path.stem
        files.append((ministry, fid, str(path)))
    return sorted(files)


# ================= PDF EXTRACTION =================
def clean_pdf_text(text):
    if not text:
        return ""

    text = text.replace("￾", "-")
    text = text.replace("†", "")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # fix broken hyphen line wraps
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()


def read_pdf(path):
    pages = []

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2)
                if txt:
                    pages.append(txt)

    except Exception as e:
        print(f"PDF ERROR {path}: {e}")
        return ""

    return clean_pdf_text("\n".join(pages))


# ================= Q/A PARSER =================
def parse_loksabha_pdf(text):
    """
    Robust parser for Lok Sabha Q&A PDFs.
    """
    answer_match = re.search(r"\nANSWER\s*\n", text, re.I)
    if not answer_match:
        return "", ""

    pre_answer = text[:answer_match.start()].strip()
    answer_raw = text[answer_match.end():].strip()

    # Find question start
    q_match = re.search(
        r"Will the Minister.*?state:",
        pre_answer,
        re.I | re.S
    )

    question = pre_answer[q_match.start():].strip() if q_match else pre_answer

    # remove minister signature block
    answer_lines = answer_raw.split("\n")
    cleaned = []
    skip_header = True

    for line in answer_lines:
        line = line.strip()

        if skip_header:
            if re.search(r"^\([A-Z .]+\)$", line):
                continue
            if re.search(r"^THE MINISTER", line, re.I):
                continue
            skip_header = False

        cleaned.append(line)

    answer = "\n".join(cleaned)

    if not INCLUDE_ANNEXURE:
        cut = re.search(r"\nAnnexure\b|\n\*{3,}", answer, re.I)
        if cut:
            answer = answer[:cut.start()]

    return question.strip(), answer.strip()


# ================= SENTENCE SPLITTER =================
def split_sentences(text):
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()

    # protect abbreviations
    text = text.replace("State/UT", "StateUT")
    text = text.replace("Govt.", "Govt")

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)

    results = []
    seen = set()

    for s in sentences:
        s = s.strip()

        s = re.sub(r"^\([a-zA-Z]\)\s*", "", s)
        s = re.sub(r"^\([a-zA-Z]\)\s*&\s*\([a-zA-Z]\)\s*", "", s)

        if len(s) < 25:
            continue

        key = re.sub(r"\W+", "", s.lower())[:100]
        if key in seen:
            continue
        seen.add(key)

        results.append(s.replace("StateUT", "State/UT"))

    return results


# ================= MAIN =================
def run():
    files = get_pdf_files(ROOT_FOLDER)

    if MAX_FILES:
        files = files[:MAX_FILES]

    print(f"Found {len(files)} PDFs")

    cols = [
        "ministry",
        "file_name",
        "section",
        "sentence_num",
        "eng_original",
        "hindi_sarvam",
        "eng_back"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for ministry, fid, path in files:
            print(f"\n--- {ministry} | {fid} ---")

            text = read_pdf(path)
            question, answer = parse_loksabha_pdf(text)

            if not answer:
                print("Skipping: answer not found")
                continue

            for section, block in [("Question", question), ("Answer", answer)]:
                sentences = split_sentences(block)

                for i, sent in enumerate(sentences, 1):
                    print(f"{section} {i}: {sent[:80]}")

                    hi = en_to_hi(sent)
                    back = hi_to_en(hi) if "ERROR" not in hi else ""

                    writer.writerow({
                        "ministry": ministry,
                        "file_name": fid,
                        "section": section,
                        "sentence_num": i,
                        "eng_original": sent,
                        "hindi_sarvam": hi,
                        "eng_back": back
                    })
                    f.flush()

    print(f"\nDONE → {OUTPUT_FILE}")


if __name__ == "__main__":
    run()