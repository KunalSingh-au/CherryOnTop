# Unified Hindi QA over four providers.

from __future__ import annotations

import re
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    SARVAM_KEY,
    SARVAM_CHAT_MODEL,
    SLEEP_SARVAM,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    SLEEP_GEMINI,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    SLEEP_MISTRAL,
    GROQ_API_KEY,
    LLAMA_GROQ_MODEL,
    SLEEP_GROQ,
    MAX_CONTEXT_CHARS,
    MAX_ANSWER_TOKENS,
)

import requests


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>|<redacted_thinking>.*?</redacted_thinking>",
        "",
        text,
        flags=re.DOTALL | re.I,
    ).strip()


def build_user_prompt(context: str, question_hi: str, condition: str) -> str:
    ctx = (context or "")[:MAX_CONTEXT_CHARS]
    if condition == "C3":
        return (
            "कोई आधिकारिक दस्तावेज़ प्रदान नहीं किया गया है। "
            "अपने सामान्य ज्ञान के आधार पर नीचे दिए प्रश्न का उत्तर केवल हिंदी में दें। "
            "यदि आप निश्चित नहीं हैं तो कहें कि आप निश्चित नहीं हैं।\n\n"
            f"प्रश्न: {question_hi}\n\nउत्तर:"
        )
    return (
        "नीचे दिए गए दस्तावेज़ को पढ़ें और प्रश्न का उत्तर केवल हिंदी में दें। "
        "दस्तावेज़ में न मिले तो स्पष्ट रूप से कहें कि दस्तावेज़ में यह जानकारी उपलब्ध नहीं है।\n\n"
        f"दस्तावेज़:\n{ctx}\n\n"
        f"प्रश्न: {question_hi}\n\nउत्तर:"
    )


def qa_sarvam(context: str, question_hi: str, condition: str) -> str:
    prompt = build_user_prompt(context, question_hi, condition)
    try:
        r = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "api-subscription-key": SARVAM_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": SARVAM_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_ANSWER_TOKENS,
                "temperature": 0,
            },
            timeout=120,
        )
        time.sleep(SLEEP_SARVAM)
        d = r.json()
        return _strip_thinking(d["choices"][0]["message"]["content"])
    except Exception as e:
        return f"SARVAM_QA_ERROR: {e}"


def qa_gemini(context: str, question_hi: str, condition: str) -> str:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = build_user_prompt(context, question_hi, condition)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": MAX_ANSWER_TOKENS,
            },
        )
        time.sleep(SLEEP_GEMINI)
        return _strip_thinking(resp.text or "")
    except Exception as e:
        return f"GEMINI_QA_ERROR: {e}"


def qa_mistral(context: str, question_hi: str, condition: str) -> str:
    try:
        from mistralai import Mistral

        client = Mistral(api_key=MISTRAL_API_KEY)
        prompt = build_user_prompt(context, question_hi, condition)
        out = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=MAX_ANSWER_TOKENS,
        )
        time.sleep(SLEEP_MISTRAL)
        return _strip_thinking(out.choices[0].message.content or "")
    except Exception as e:
        return f"MISTRAL_QA_ERROR: {e}"


def qa_llama_groq(context: str, question_hi: str, condition: str) -> str:
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_user_prompt(context, question_hi, condition)
        out = client.chat.completions.create(
            model=LLAMA_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=MAX_ANSWER_TOKENS,
        )
        time.sleep(SLEEP_GROQ)
        return _strip_thinking(out.choices[0].message.content or "")
    except Exception as e:
        return f"LLAMA_GROQ_ERROR: {e}"


QA_FUNCS = {
    "sarvam": qa_sarvam,
    "gemini": qa_gemini,
    "mistral": qa_mistral,
    "llama": qa_llama_groq,
}


def run_qa(model: str, context: str, question_hi: str, condition: str) -> str:
    fn = QA_FUNCS.get(model)
    if not fn:
        return f"UNKNOWN_MODEL: {model}"
    return fn(context, question_hi, condition)


def sarvam_translate(text: str, src: str, tgt: str) -> str:
    text = (text or "").strip()[:MAX_CONTEXT_CHARS]
    if not text:
        return ""
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={
                "api-subscription-key": SARVAM_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input": text[:900],
                "source_language_code": src,
                "target_language_code": tgt,
                "mode": "formal",
            },
            timeout=60,
        )
        time.sleep(SLEEP_SARVAM)
        d = r.json()
        return d.get("translated_text", f"SARVAM_ERROR: {d}")
    except Exception as e:
        return f"SARVAM_ERROR: {e}"


def sarvam_hi_to_en(text: str) -> str:
    return sarvam_translate(text, "hi-IN", "en-IN")


def sarvam_en_to_hi(text: str) -> str:
    return sarvam_translate(text, "en-IN", "hi-IN")


def translate_long_text(text: str, fn, max_chunk: int = 900) -> str:
    import re as _re

    text = _re.sub(r"\s+", " ", (text or "")).strip()
    parts = _re.split(r"(?<=[.!?।])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(fn(p[:max_chunk]))
    return " ".join(out)
