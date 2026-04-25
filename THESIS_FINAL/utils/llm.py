"""
utils/llm.py — Prompt construction, answer cleaning, and back-translation.

KEY DESIGN: build_qa_prompt() takes a `use_prompt` boolean.
  use_prompt=True  → adds Hindi task instructions (the "Prompted" condition)
  use_prompt=False → sends only context + question (the "No-Prompt" condition)

This eliminates the need for two separate codebases.
The flag is passed down from the CLI argument --prompts / --no-prompts.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CONTEXT_CHAR_LIMIT, SARVAM_TRANSLATE_HF


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER — unified, flag-controlled
# ─────────────────────────────────────────────────────────────────────────────

# Hindi task instructions used in the "Prompted" condition.
# ── TUNABLE: Edit the numbered instructions below to change model behaviour ───
_HINDI_INSTRUCTIONS = (
    "**निर्देश:**\n"
    "1. उत्तर पूर्णतः देवनागरी हिंदी में लिखें।\n"
    "2. उत्तर सटीक और सीधा होना चाहिए (100 शब्दों से कम)।\n"
    "3. यदि उत्तर दस्तावेज़ में नहीं है, तो केवल "
    "'जानकारी उपलब्ध नहीं है' लिखें।"
)


def build_qa_prompt(
    question_hi: str,
    context: str,
    condition: str,
    use_prompt: bool = True,
) -> str:
    """
    Build the user-turn string passed to the LLM.

    Parameters
    ----------
    question_hi : Hindi question string from gold_qa_master.csv
    context     : Document text (full-doc or RAG chunks).  Empty for C3.
    condition   : Condition code (C1–C9).  Only C3 gets special treatment.
    use_prompt  : If True, prepend Hindi task instructions.
                  If False, send bare context + question (no-prompt baseline).

    Returns
    -------
    str : Formatted user-turn string, ready for tokenizer.apply_chat_template()
    """
    # Truncate context to character limit defined in config.py
    ctx_clean = str(context).strip()[:CONTEXT_CHAR_LIMIT] if context else ""

    # ── C3: Closed-book (no context) ──────────────────────────────────────────
    if condition == "C3":
        if use_prompt:
            return (
                "आप एक विशेषज्ञ सहायक हैं। "
                "बिना किसी संदर्भ के निम्नलिखित प्रश्न का उत्तर दें:\n\n"
                f"प्रश्न: {question_hi.strip()}\n\n"
                f"{_HINDI_INSTRUCTIONS}"
            )
        else:
            # Bare question only — absolute zero-shot floor
            return question_hi.strip()

    # ── All other conditions: context + question ───────────────────────────────
    if use_prompt:
        doc_section = ctx_clean if ctx_clean else "दस्तावेज़ उपलब्ध नहीं है।"
        return (
            "आप एक विशेषज्ञ सहायक हैं। "
            "दिए गए 'संदर्भ दस्तावेज़' के आधार पर प्रश्न का उत्तर दें।\n\n"
            f"=== संदर्भ दस्तावेज़ ===\n{doc_section}\n"
            "====================\n\n"
            f"प्रश्न: {question_hi.strip()}\n\n"
            f"{_HINDI_INSTRUCTIONS}"
        )
    else:
        # No-prompt: raw context then raw question, no wrapper text
        return f"{ctx_clean}\n\n{question_hi.strip()}"


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER CLEANER — strips <think> reasoning blocks from Sarvam / Qwen outputs
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer_from_output(text: str) -> str:
    """
    Remove chain-of-thought reasoning blocks that some models generate.

    Sarvam-30B and Qwen-2.5 may wrap their reasoning in <think>...</think>
    before the actual answer.  This function strips those blocks so that
    only the final answer text reaches the evaluator.
    """
    if not isinstance(text, str):
        return str(text)
    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed <think> block that runs to end of string
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# BACK-TRANSLATION — Hindi answer → English for metric computation
# ─────────────────────────────────────────────────────────────────────────────

# Module-level singletons so the translation model loads once per process
_bt_llm = None
_bt_tok = None


def _load_back_translator():
    """Lazy-load Sarvam-Translate via vLLM (runs on GPU 0 only, 30% VRAM)."""
    global _bt_llm, _bt_tok
    if _bt_llm is not None:
        return
    from vllm import LLM
    from transformers import AutoTokenizer

    print("  [back-translate] Loading Sarvam-Translate …", flush=True)
    _bt_tok = AutoTokenizer.from_pretrained(SARVAM_TRANSLATE_HF)
    _bt_llm = LLM(
        model=SARVAM_TRANSLATE_HF,
        tensor_parallel_size=1,      # Back-translator runs on a single GPU
        gpu_memory_utilization=0.30, # Leaves headroom for BERTScore model
    )
    print("  [back-translate] Ready.", flush=True)


def hi_to_en_local(text: str) -> str:
    """
    Translate a Hindi string to English using Sarvam-Translate (local, GPU).

    Input is capped at 1500 characters to stay within the translation model's
    context window.  The back-translated string is used only for metric
    computation — it is never shown to users.

    Returns an empty string for empty inputs.
    """
    if not text or not str(text).strip():
        return ""

    _load_back_translator()
    from vllm import SamplingParams

    msgs = [
        {"role": "system", "content": "Translate the text below to English."},
        {"role": "user",   "content": str(text)[:1500]},
    ]
    formatted = _bt_tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    outputs = _bt_llm.generate(
        [formatted],
        SamplingParams(temperature=0.0, max_tokens=512),
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text.strip()


# Alias kept for any legacy import
hi_to_en_local_sarvam = hi_to_en_local
