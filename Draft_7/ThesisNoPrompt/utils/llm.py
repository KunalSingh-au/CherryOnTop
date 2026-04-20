import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Minimalist Prompt Builder (Zero Instructions) ─────────────────────────────
def build_qa_prompt(question_hi: str, context: str, condition: str) -> str:
    """
    Build user-turn prompt WITHOUT any system instructions or constraints.
    Just the raw document context followed by the raw Hindi question.
    """
    if condition == "C3":
        # Zero-shot hallucination floor (No Context)
        return question_hi.strip()
    else:
        # Pass the raw context followed by the raw question
        ctx = str(context)[:3500] if context else ""
        return f"{ctx}\n\n{question_hi.strip()}"

# ── Answer extractor (Handles Sarvam <think> tags if they appear) ─────────────
def extract_answer_from_output(text: str) -> str:
    if not isinstance(text, str): return text
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

# ── Local Sarvam Back-Translation (For Evaluation later) ──────────────────────
_sarvam_bt_llm = None
_sarvam_bt_tokenizer = None

def _load_sarvam_bt():
    global _sarvam_bt_llm, _sarvam_bt_tokenizer
    if _sarvam_bt_llm is None:
        from vllm import LLM
        from transformers import AutoTokenizer
        print("  [back-translate] Loading local Sarvam Translate via vLLM...")
        model_id = "sarvamai/sarvam-translate"
        _sarvam_bt_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _sarvam_bt_llm = LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.3)

def hi_to_en_local_sarvam(text: str) -> str:
    if not text or not text.strip():
        return ""
    _load_sarvam_bt()
    from vllm import SamplingParams
    msgs = [
        {"role": "system", "content": "Translate the text below to English."},
        {"role": "user", "content": text[:1500]}
    ]
    formatted = _sarvam_bt_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    outputs = _sarvam_bt_llm.generate([formatted], SamplingParams(temperature=0.0, max_tokens=512), use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


# ── COMPATIBILITY ALIAS ──
# This fixes the ImportError in scripts/04_evaluate.py
hi_to_en_local = hi_to_en_local_sarvam