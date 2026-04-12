# utils/translate.py
# All translation functions in one place.
# Sarvam API, OpenAI GPT-4o, and IndicTrans2 (GPU).

import time
import requests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    SARVAM_KEY, OPENAI_KEY,
    SARVAM_CHAT_MODEL, OPENAI_CHAT_MODEL,
    MAX_CHARS, SLEEP_SARVAM, SLEEP_OPENAI,
    INDICTRANS2_MODEL
)

# ── Sarvam translate API ──────────────────────────────────────────────────────

def sarvam_translate(text: str, src: str, tgt: str) -> str:
    """
    Translate using Sarvam translate API.
    src/tgt: 'en-IN' or 'hi-IN'
    """
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"api-subscription-key": SARVAM_KEY,
                     "Content-Type": "application/json"},
            json={"input": text,
                  "source_language_code": src,
                  "target_language_code": tgt,
                  "mode": "formal"},
            timeout=30
        )
        d = r.json()
        time.sleep(SLEEP_SARVAM)
        return d.get("translated_text", f"SARVAM_ERROR: {d}")
    except Exception as e:
        return f"SARVAM_ERROR: {e}"


def sarvam_en_to_hi(text: str) -> str:
    return sarvam_translate(text, "en-IN", "hi-IN")

def sarvam_hi_to_en(text: str) -> str:
    return sarvam_translate(text, "hi-IN", "en-IN")


# ── OpenAI GPT-4o ─────────────────────────────────────────────────────────────

def gpt_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate using GPT-4o."""
    import openai
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    try:
        client = openai.OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Translate the following {src_lang} text to {tgt_lang}. "
                    f"Return only the translation, nothing else.\n\n{text}"
                )
            }],
            temperature=0
        )
        time.sleep(SLEEP_OPENAI)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT_ERROR: {e}"


def gpt_en_to_hi(text: str) -> str:
    return gpt_translate(text, "English", "Hindi")

def gpt_hi_to_en(text: str) -> str:
    return gpt_translate(text, "Hindi", "English")


# ── IndicTrans2 (local GPU) ───────────────────────────────────────────────────

_indictrans_model = None
_indictrans_tokenizer = None

def _load_indictrans():
    """Load IndicTrans2 model once and cache it."""
    global _indictrans_model, _indictrans_tokenizer
    if _indictrans_model is not None:
        return

    print("Loading IndicTrans2 model (first time — may take 2-3 minutes)...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    _indictrans_tokenizer = AutoTokenizer.from_pretrained(
        INDICTRANS2_MODEL, trust_remote_code=True
    )
    _indictrans_model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS2_MODEL, trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _indictrans_model = _indictrans_model.to(device)
    _indictrans_model.eval()
    print(f"IndicTrans2 loaded on {device}")


def indictrans_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate using local IndicTrans2 model.
    src_lang / tgt_lang use IndicTrans2 codes:
      English → 'eng_Latn'
      Hindi   → 'hin_Deva'
    """
    import torch
    _load_indictrans()

    inputs = _indictrans_tokenizer(
        text,
        src_lang=src_lang,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    device = next(_indictrans_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_tokens = _indictrans_model.generate(
            **inputs,
            forced_bos_token_id=_indictrans_tokenizer.lang_code_to_id[tgt_lang],
            max_length=512,
            num_beams=4
        )
    return _indictrans_tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True
    )[0]


def indictrans_en_to_hi(text: str) -> str:
    return indictrans_translate(text, "eng_Latn", "hin_Deva")

def indictrans_hi_to_en(text: str) -> str:
    return indictrans_translate(text, "hin_Deva", "eng_Latn")


# ── Sarvam QA (chat) ──────────────────────────────────────────────────────────

def sarvam_qa(context: str, question_hi: str) -> str:
    """
    Ask a Hindi question against a context (English or Hindi).
    Returns Hindi answer.
    """
    prompt = (
        "नीचे दिए गए दस्तावेज़ को पढ़ें और प्रश्न का उत्तर हिंदी में दें।\n\n"
        f"दस्तावेज़:\n{context[:2000]}\n\n"
        f"प्रश्न: {question_hi}\n\n"
        "उत्तर:"
    )
    try:
        r = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"api-subscription-key": SARVAM_KEY,
                     "Content-Type": "application/json"},
            json={
                "model": SARVAM_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0
            },
            timeout=30
        )
        d = r.json()
        time.sleep(SLEEP_SARVAM)
        return d["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"SARVAM_QA_ERROR: {e}"


def gpt_qa(context: str, question_hi: str) -> str:
    """Ask a Hindi question against a context using GPT-4o. Returns Hindi answer."""
    import openai
    prompt = (
        "Read the following document and answer the question in Hindi.\n\n"
        f"Document:\n{context[:3000]}\n\n"
        f"Question: {question_hi}\n\n"
        "Answer in Hindi:"
    )
    try:
        client = openai.OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        time.sleep(SLEEP_OPENAI)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT_QA_ERROR: {e}"
