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
    GEMINI_MAX_RETRIES,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    SLEEP_MISTRAL,
    GROQ_API_KEY,
    LLAMA_GROQ_MODEL,
    SLEEP_GROQ,
    GROQ_MAX_RETRIES,
    MAX_CONTEXT_CHARS,
    MAX_ANSWER_TOKENS,
    SARVAM_TRANSLATE_MAX_INPUT,
)

import requests


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>|<redacted_thinking>.*?</redacted_thinking>",
        "",
        text,
        flags=re.DOTALL | re.I,
    ).strip()


_HINDI_ONLY = (
    "**भाषा:** उत्तर पूर्णतः देवनागरी हिंदी में लिखें। "
    "पूरा उत्तर अंग्रेज़ी में न लिखें; केवल आवश्यक संक्षिप्त नाम/संकेत अंग्रेज़ी में रह सकते हैं।\n\n"
)


def build_user_prompt(context: str, question_hi: str, condition: str) -> str:
    ctx = (context or "")[:MAX_CONTEXT_CHARS]
    if condition == "C3":
        return (
            _HINDI_ONLY
            "कोई आधिकारिक दस्तावेज़ प्रदान नहीं किया गया है। "
            "अपने सामान्य ज्ञान के आधार पर नीचे दिए प्रश्न का उत्तर हिंदी में दें। "
            "यदि आप निश्चित नहीं हैं तो हिंदी में कहें कि आप निश्चित नहीं हैं।\n\n"
            f"प्रश्न: {question_hi}\n\nउत्तर:"
        )
    return (
        _HINDI_ONLY
        "नीचे दिए गए दस्तावेज़ को पढ़ें और प्रश्न का उत्तर हिंदी में दें। "
        "दस्तावेज़ में न मिले तो हिंदी में स्पष्ट लिखें कि दस्तावेज़ में यह जानकारी उपलब्ध नहीं है।\n\n"
        f"दस्तावेज़:\n{ctx}\n\n"
        f"प्रश्न: {question_hi}\n\nउत्तर:"
    )


def _parse_retry_wait_seconds(err_msg: str) -> float | None:
    m = re.search(r"retry in ([0-9.]+)\s*s", err_msg, re.I)
    if m:
        return float(m.group(1)) + 2.0
    m = re.search(r"try again in (\d+)m([0-9.]+)s?", err_msg, re.I)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 2.0
    return None


def _is_rate_limit_error(err_msg: str) -> bool:
    e = err_msg.lower()
    return (
        "429" in err_msg
        or "quota" in e
        or "rate limit" in e
        or "resource exhausted" in e
        or "too many requests" in e
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
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = build_user_prompt(context, question_hi, condition)
    last_err: Exception | None = None
    for attempt in range(max(1, GEMINI_MAX_RETRIES)):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": MAX_ANSWER_TOKENS,
                },
            )
            time.sleep(SLEEP_GEMINI)
            try:
                text = resp.text or ""
            except ValueError:
                text = ""
            return _strip_thinking(text)
        except Exception as e:
            last_err = e
            err_s = str(e)
            if not _is_rate_limit_error(err_s) or attempt >= GEMINI_MAX_RETRIES - 1:
                break
            wait = _parse_retry_wait_seconds(err_s) or min(120.0, 12.0 * (attempt + 1))
            time.sleep(wait)
    return f"GEMINI_QA_ERROR: {last_err}"


def qa_mistral(context: str, question_hi: str, condition: str) -> str:
    try:
        try:
            from mistralai.client import Mistral
        except ImportError:
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
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    prompt = build_user_prompt(context, question_hi, condition)
    last_err: Exception | None = None
    for attempt in range(max(1, GROQ_MAX_RETRIES)):
        try:
            out = client.chat.completions.create(
                model=LLAMA_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=MAX_ANSWER_TOKENS,
            )
            time.sleep(SLEEP_GROQ)
            return _strip_thinking(out.choices[0].message.content or "")
        except Exception as e:
            last_err = e
            err_s = str(e)
            if not _is_rate_limit_error(err_s) or attempt >= GROQ_MAX_RETRIES - 1:
                break
            wait = _parse_retry_wait_seconds(err_s) or min(180.0, 20.0 * (attempt + 1))
            time.sleep(wait)
    return f"LLAMA_GROQ_ERROR: {last_err}"


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
                "input": text[:SARVAM_TRANSLATE_MAX_INPUT],
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


def translate_long_text(text: str, fn, max_chunk: int | None = None) -> str:
    if max_chunk is None:
        max_chunk = SARVAM_TRANSLATE_MAX_INPUT
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
