# utils/translate.py — Sarvam and GPT-4o translation + QA

import time, requests, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (SARVAM_KEY, OPENAI_KEY, SARVAM_CHAT_MODEL,
                    OPENAI_CHAT_MODEL, MAX_CHARS, SLEEP_SARVAM, SLEEP_OPENAI)


# ── Sarvam translate ───────────────────────────────────────────────────────────

def sarvam_translate(text: str, src: str, tgt: str) -> str:
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"api-subscription-key": SARVAM_KEY,
                     "Content-Type": "application/json"},
            json={"input": text, "source_language_code": src,
                  "target_language_code": tgt, "mode": "formal"},
            timeout=30
        )
        time.sleep(SLEEP_SARVAM)
        d = r.json()
        return d.get("translated_text", f"SARVAM_ERROR: {d}")
    except Exception as e:
        return f"SARVAM_ERROR: {e}"

def sarvam_en_to_hi(text): return sarvam_translate(text, "en-IN", "hi-IN")
def sarvam_hi_to_en(text): return sarvam_translate(text, "hi-IN", "en-IN")


# ── GPT-4o translate ───────────────────────────────────────────────────────────

def gpt_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    import openai
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    try:
        client = openai.OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content":
                f"Translate the following {src_lang} text to {tgt_lang}. "
                f"Return only the translation.\n\n{text}"}],
            temperature=0
        )
        time.sleep(SLEEP_OPENAI)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT_ERROR: {e}"

def gpt_en_to_hi(text): return gpt_translate(text, "English", "Hindi")
def gpt_hi_to_en(text): return gpt_translate(text, "Hindi", "English")


# ── Sarvam QA ─────────────────────────────────────────────────────────────────

def sarvam_qa(context: str, question_hi: str) -> str:
    """Ask Hindi question over a context (English or Hindi). Returns Hindi answer."""
    prompt = (
        "नीचे दिए गए दस्तावेज़ को पढ़ें और प्रश्न का उत्तर हिंदी में दें।\n\n"
        f"दस्तावेज़:\n{context[:2000]}\n\n"
        f"प्रश्न: {question_hi}\n\nउत्तर:"
    )
    try:
        r = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"api-subscription-key": SARVAM_KEY,
                     "Content-Type": "application/json"},
            json={"model": SARVAM_CHAT_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 200, "temperature": 0},
            timeout=30
        )
        time.sleep(SLEEP_SARVAM)
        d = r.json()
        return d["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"SARVAM_QA_ERROR: {e}"


# ── GPT-4o QA ──────────────────────────────────────────────────────────────────

def gpt_qa(context: str, question_hi: str) -> str:
    """Ask Hindi question over a context using GPT-4o. Returns Hindi answer."""
    import openai
    prompt = (
        "Read the following document and answer the question in Hindi.\n\n"
        f"Document:\n{context[:3000]}\n\n"
        f"Question: {question_hi}\n\nAnswer in Hindi:"
    )
    try:
        client = openai.OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=200
        )
        time.sleep(SLEEP_OPENAI)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT_QA_ERROR: {e}"


# ── Translate long text in sentences ──────────────────────────────────────────

def translate_long_text(text: str, fn) -> str:
    """Split text at sentence boundaries, translate each chunk, rejoin."""
    import re
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    results = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
        # If still too long, hard-truncate
        results.append(fn(sent[:MAX_CHARS]))
    return " ".join(results)
