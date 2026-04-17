#!/usr/bin/env python3
"""
fix_failed_runs.py
==================
Drop this file in your project root (next to config.py) and run:

    python fix_failed_runs.py

It finds every ERROR row in outputs/runs/all_runs.csv,
re-runs those calls with the fixes below, and overwrites the CSV.

Fixes applied:
  - Mistral: switched from broken `mistralai` package → Groq API (same key)
  - Gemini: exponential backoff on 429, longer sleep between calls
  - Llama:  retry with longer sleep on 429

After this finishes, re-run your evaluation:
    python scripts/04_evaluate.py
    python scripts/05_analyze.py
"""

import os, sys, time, re, json, random
import pandas as pd
import requests

# ── Config — reads your existing env vars or config.py ──────────────────────
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import (
        SARVAM_KEY, GEMINI_KEY, GROQ_KEY,
        MAX_CONTEXT_CHARS,
    )
    # Some Cursor codebases use GOOGLE_API_KEY / GROQ_API_KEY
    GEMINI_KEY = GEMINI_KEY or os.getenv("GOOGLE_API_KEY", "")
    GROQ_KEY   = GROQ_KEY   or os.getenv("GROQ_API_KEY",  "")
except ImportError:
    SARVAM_KEY = os.getenv("SARVAM_KEY", "")
    GEMINI_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_KEY", ""))
    GROQ_KEY   = os.getenv("GROQ_API_KEY",  os.getenv("GROQ_KEY", ""))
    MAX_CONTEXT_CHARS = 3000

RUNS_CSV = "outputs/runs/all_runs.csv"

# ── System prompt (same for all models — fair comparison) ────────────────────
SYSTEM = (
    "आप एक सरकारी दस्तावेज़ विश्लेषक हैं। "
    "केवल दिए गए दस्तावेज़ के आधार पर प्रश्न का उत्तर हिंदी में दें। "
    "उत्तर संक्षिप्त और तथ्यात्मक होना चाहिए। "
    "केवल उत्तर दें — कोई अभिवादन या भूमिका नहीं।"
)

SYSTEM_NO_DOC = (
    "आप एक सरकारी नीति विशेषज्ञ हैं। "
    "प्रश्न का उत्तर अपने ज्ञान के आधार पर हिंदी में दें। "
    "उत्तर संक्षिप्त और तथ्यात्मक होना चाहिए। "
    "केवल उत्तर दें।"
)


def build_messages(context, question_hi):
    system = SYSTEM if context else SYSTEM_NO_DOC
    user_content = (
        f"दस्तावेज़:\n{str(context)[:MAX_CONTEXT_CHARS]}\n\nप्रश्न: {question_hi}\n\nउत्तर:"
        if context else
        f"प्रश्न: {question_hi}\n\nउत्तर:"
    )
    return system, user_content


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ── Gemini with exponential backoff ─────────────────────────────────────────
def call_gemini(context, question_hi, retries=5):
    if not GEMINI_KEY:
        return "GEMINI_ERROR: no key"
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
    except ImportError:
        return "GEMINI_ERROR: google-generativeai not installed"

    system, user_content = build_messages(context, question_hi)
    prompt = f"{system}\n\n{user_content}"

    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp  = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0, max_output_tokens=300
                ),
            )
            time.sleep(5)   # 5s = safe under 15 req/min free tier
            return clean(resp.text.strip())
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = (2 ** attempt) * 10 + random.uniform(0, 5)
                print(f"    Gemini 429 — waiting {wait:.0f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                return f"GEMINI_ERROR: {err}"
    return "GEMINI_ERROR: max retries exceeded"


# ── Mistral via Groq (avoids broken mistralai package) ───────────────────────
def call_mistral(context, question_hi, retries=4):
    if not GROQ_KEY:
        return "MISTRAL_ERROR: no GROQ_KEY"
    try:
        from groq import Groq
    except ImportError:
        return "MISTRAL_ERROR: groq package not installed — pip install groq"

    system, user_content = build_messages(context, question_hi)
    client = Groq(api_key=GROQ_KEY)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="mistral-saba-24b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0, max_tokens=300,
            )
            time.sleep(3)
            return clean(resp.choices[0].message.content.strip())
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (2 ** attempt) * 8 + random.uniform(0, 3)
                print(f"    Mistral/Groq 429 — waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                return f"MISTRAL_ERROR: {err}"
    return "MISTRAL_ERROR: max retries exceeded"


# ── Llama via Groq with retry ─────────────────────────────────────────────────
def call_llama(context, question_hi, retries=4):
    if not GROQ_KEY:
        return "LLAMA_ERROR: no GROQ_KEY"
    try:
        from groq import Groq
    except ImportError:
        return "LLAMA_ERROR: groq package not installed"

    system, user_content = build_messages(context, question_hi)
    client = Groq(api_key=GROQ_KEY)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0, max_tokens=300,
            )
            time.sleep(3)
            return clean(resp.choices[0].message.content.strip())
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (2 ** attempt) * 8 + random.uniform(0, 3)
                print(f"    Llama/Groq 429 — waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                return f"LLAMA_ERROR: {err}"
    return "LLAMA_ERROR: max retries exceeded"


MODEL_FN = {
    "gemini":  call_gemini,
    "mistral": call_mistral,
    "llama":   call_llama,
}

# ── Load full contexts from extracted doc files ───────────────────────────────
def load_jsonl(path):
    """Load a JSONL file into a dict keyed by doc_id."""
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                result[r["doc_id"]] = r
            except Exception:
                pass
    return result


def find_context_files():
    """Try several possible paths that Cursor's codebase might use."""
    candidates = {
        "en":     ["data/extracted/english_docs.jsonl",
                   "data/extracted/docs_en.jsonl"],
        "sarvam": ["data/translated/sarvam_hi.jsonl",
                   "data/translated/sarvam_translated.jsonl",
                   "data/translated/docs_sarvam_hi.jsonl"],
        "hi":     ["data/extracted/hindi_docs.jsonl",
                   "data/extracted/docs_hi.jsonl",
                   "data/extracted/official_hindi_docs.jsonl"],
    }
    found = {}
    for key, paths in candidates.items():
        for p in paths:
            if os.path.exists(p):
                found[key] = p
                break
    return found


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(RUNS_CSV):
        print(f"ERROR: {RUNS_CSV} not found. Run from your project root.")
        sys.exit(1)

    df = pd.read_csv(RUNS_CSV)
    print(f"Loaded {len(df)} rows from {RUNS_CSV}")

    # Load full-context doc files
    ctx_files = find_context_files()
    print(f"\nContext files found: {list(ctx_files.keys())}")

    en_docs     = load_jsonl(ctx_files.get("en", ""))
    sarvam_docs = load_jsonl(ctx_files.get("sarvam", ""))
    hi_docs     = load_jsonl(ctx_files.get("hi", ""))

    print(f"  English docs:  {len(en_docs)}")
    print(f"  Sarvam Hindi:  {len(sarvam_docs)}")
    print(f"  Official Hindi:{len(hi_docs)}")

    if not en_docs:
        print("\nWARNING: No extracted English docs found.")
        print("Make sure you've run: python scripts/01_extract_docs.py")
        print("Continuing — C3 (no-doc) rows will still work.")

    def get_context(condition, doc_id):
        if condition == "C1":
            d = en_docs.get(doc_id, {})
            return d.get("text") or d.get("answer_en") or d.get("answer_text")
        elif condition == "C2":
            d = sarvam_docs.get(doc_id, {})
            return d.get("hindi_text") or d.get("text") or d.get("answer_hi")
        elif condition == "C3":
            return None
        elif condition == "C4":
            d = hi_docs.get(doc_id, {})
            return d.get("text") or d.get("answer_hi")
        return None

    # Find error rows for the 3 broken models
    is_error = df.answer_hi.str.contains("_ERROR:", na=False) | df.answer_hi.isna()
    todo = df[is_error & df.model.isin(["gemini", "mistral", "llama"])].copy()

    print(f"\nError rows to fix: {len(todo)}")
    print(todo.groupby(["model", "condition"]).size().unstack(fill_value=0).to_string())

    # Check keys
    print("\nKey check:")
    print(f"  GEMINI_KEY:  {'SET' if GEMINI_KEY else 'MISSING — gemini will be skipped'}")
    print(f"  GROQ_KEY:    {'SET' if GROQ_KEY   else 'MISSING — mistral+llama will be skipped'}")

    if not GEMINI_KEY and not GROQ_KEY:
        print("\nNo keys set. Export GOOGLE_API_KEY and GROQ_API_KEY then re-run.")
        sys.exit(1)

    # Filter to only models we have keys for
    can_run = []
    if GEMINI_KEY: can_run.append("gemini")
    if GROQ_KEY:   can_run += ["mistral", "llama"]
    todo = todo[todo.model.isin(can_run)]
    print(f"\nWill fix {len(todo)} rows for: {can_run}")
    print("This will take a while due to rate limits. Safe to Ctrl+C and re-run.\n")

    fixed = 0
    skipped = 0

    for idx, row in todo.iterrows():
        model     = row["model"]
        condition = row["condition"]
        question  = row.get("question_hi_used") or row.get("question_hi", "")

        # Get FULL context (not the 200-char preview)
        context = get_context(condition, row["doc_id"])
        if context is None and condition != "C3":
            # fallback to context_preview — better than nothing
            preview = row.get("context_preview", "")
            context = str(preview) if pd.notna(preview) and str(preview) != "nan" else None

        fn = MODEL_FN[model]
        print(f"  [{fixed+1}/{len(todo)}] {model} {condition} {row['doc_id']} {row.get('question_id','')}",
              end=" ... ", flush=True)

        answer = fn(context, question)
        is_new_error = "_ERROR:" in answer

        if is_new_error:
            print(f"STILL ERROR: {answer[:60]}")
            skipped += 1
        else:
            df.at[idx, "answer_hi"] = answer
            fixed += 1
            print(f"OK → {answer[:60]}")

        # Save after every 10 fixes
        if (fixed + skipped) % 10 == 0:
            df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig")
            print(f"    (checkpoint saved — {fixed} fixed, {skipped} still errors)")

    # Final save
    df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"Done. Fixed: {fixed} | Still errors: {skipped}")
    print(f"Saved → {RUNS_CSV}")

    # Summary of remaining errors
    remaining = df[df.answer_hi.str.contains("_ERROR:", na=False)]
    if len(remaining):
        print(f"\nRemaining errors ({len(remaining)}):")
        print(remaining.groupby(["model","condition"]).size().to_string())
    else:
        print("\n✓ No errors remaining!")

    print("\nNext steps:")
    print("  python scripts/04_evaluate.py")
    print("  python scripts/05_analyze.py")
