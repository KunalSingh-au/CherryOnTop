"""
Lok Sabha Translation Pipeline — Option A
==========================================
English sentence → Sarvam Hindi → back to English

For each sentence extracted from English PDFs:
  - eng_original   : sentence from the PDF
  - hindi_sarvam   : Sarvam translation (EN → HI)
  - eng_back       : Sarvam back-translation (HI → EN)

Change MAX_FILES to control how many files to run.
"""

import os, re, csv, time, requests, pdfplumber

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY     = "sk_y9ase7u7_W4hR4pKttAd0wOY3n8DKZ7cW"
ENG_FOLDER  = "LS_Eng"
OUTPUT_FILE = "loksabha_output.csv"
MAX_FILES   = None      # ← change this number
SLEEP_SEC   = 1.0
MAX_CHARS   = 900     # Sarvam hard limit is 1000
# ─────────────────────────────────────────────────────────────────────────────


# ── SARVAM ────────────────────────────────────────────────────────────────────

def translate(text, src, tgt):
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"api-subscription-key": API_KEY,
                     "Content-Type": "application/json"},
            json={"input": text,
                  "source_language_code": src,
                  "target_language_code": tgt,
                  "mode": "formal"},
            timeout=30
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


# ── PDF EXTRACTION ────────────────────────────────────────────────────────────

def read_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        print(f"  PDF error {path}: {e}")
        return ""


# ── PARSING ───────────────────────────────────────────────────────────────────

def parse_english(text):
    """
    Returns (question, answer) from English Lok Sabha PDF.
    Handles two question formats:
      1. "Will the Minister..."
      2. "Will the Minister of STATE/MINISTRY..."
    """
    # Question start — find first question-like line
    q = re.search(
        r'Will the (?:Minister|Ministry|Government|Chairman|Speaker)',
        text, re.I
    )
    # Some files start question with question number directly
    if not q:
        q = re.search(r'\n\d{4}[.\s]+[A-Z]', text)

    # Answer divider
    a = re.search(r'\nANSWER\s*\n', text, re.I)

    if not q or not a:
        return "", ""

    question = text[q.start():a.start()].strip()

    # Strip minister name lines from answer
    answer_raw = text[a.end():].strip()
    lines = answer_raw.split('\n')
    start = 0
    for i, line in enumerate(lines):
        l = line.strip()
        if re.match(r'^MINISTER|^SECRETARY|^\(.*\)$', l, re.I) or not l:
            start = i + 1
        else:
            break
    answer = '\n'.join(lines[start:])

    # Cut before Annexure / *****
    cut = re.search(r'\nAnnexure\b|\n\*{3,}', answer, re.I)
    if cut:
        answer = answer[:cut.start()]

    return question.strip(), answer.strip()


# ── SENTENCE SPLITTING ────────────────────────────────────────────────────────

def split_sentences(text):
    """
    Split English text into individual sentences.
    Removes sub-labels (a) (b) etc.
    Deduplicates.
    """
    if not text:
        return []

    # Flatten to single line
    text = re.sub(r'\s+', ' ', text).strip()

    # Split on sentence endings
    parts = re.split(r'(?<=[.!?])\s+', text)

    seen = set()
    result = []

    for s in parts:
        s = s.strip()

        # Skip short fragments
        if len(s) < 30:
            continue

        # Strip leading sub-labels like (a) (b) (c) to (g)
        s = re.sub(r'^\([a-zA-Z]\)\s*', '', s).strip()
        if len(s) < 30:
            continue

        # Deduplicate
        key = re.sub(r'\s+', '', s.lower())[:80]
        if key in seen:
            continue
        seen.add(key)

        result.append(s)

    return result


# ── FILE DISCOVERY ────────────────────────────────────────────────────────────

def get_files():
    if not os.path.isdir(ENG_FOLDER):
        print(f"ERROR: folder '{ENG_FOLDER}' not found")
        return []
    files = []
    for fname in sorted(os.listdir(ENG_FOLDER)):
        if fname.lower().endswith('.pdf'):
            fid = re.sub(r'_en\.pdf$|\.pdf$', '', fname, flags=re.I)
            files.append((fid, os.path.join(ENG_FOLDER, fname)))
    return files


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print(f"\nLok Sabha Translation Pipeline")
    print(f"  Folder    : {ENG_FOLDER}")
    print(f"  Max files : {MAX_FILES}")
    print(f"  Output    : {OUTPUT_FILE}\n")

    files = get_files()
    if not files:
        return
    files = files[:MAX_FILES] if MAX_FILES else files
    print(f"Found {len(files)} file(s)\n")

    cols = ["file_name", "section", "sentence_num",
            "eng_original", "hindi_sarvam", "eng_back"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for fid, path in files:
            print(f"{'─'*50}")
            print(f"  {fid}")

            text = read_pdf(path)
            question, answer = parse_english(text)

            if not question and not answer:
                print(f"  WARNING: could not parse Q/A — skipping")
                continue

            for section, block in [("Question", question), ("Answer", answer)]:
                sentences = split_sentences(block)
                print(f"  [{section}] {len(sentences)} sentences")

                if not sentences:
                    print(f"  No sentences — skipping")
                    continue

                for i, sent in enumerate(sentences):
                    print(f"  {i+1}/{len(sentences)}: {sent[:60]}...")

                    hindi  = en_to_hi(sent)
                    back   = hi_to_en(hindi) if "ERROR" not in hindi else ""

                    writer.writerow({
                        "file_name":    fid,
                        "section":      section,
                        "sentence_num": i + 1,
                        "eng_original": sent,
                        "hindi_sarvam": hindi,
                        "eng_back":     back,
                    })
                    f.flush()

    print(f"\nDone → {OUTPUT_FILE}")


if __name__ == "__main__":
    run()