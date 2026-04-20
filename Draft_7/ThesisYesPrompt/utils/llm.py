import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Prompt Builder WITH Instructions ──────────────────────────────────────────
def build_qa_prompt(question_hi: str, context: str, condition: str) -> str:
    """
    Build user-turn prompt WITH system instructions and constraints.
    Provides the necessary scaffolding for specialized Indic models.
    """
    instructions = (
        "**निर्देश:**\n"
        "1. उत्तर पूर्णतः देवनागरी हिंदी में लिखें।\n"
        "2. उत्तर सटीक और सीधा होना चाहिए (100 शब्दों से कम)।\n"
        "3. यदि उत्तर दस्तावेज़ में नहीं है, तो केवल 'जानकारी उपलब्ध नहीं है' लिखें।"
    )
    
    if condition == "C3":
        return (
            "आप एक विशेषज्ञ सहायक हैं। बिना किसी संदर्भ के निम्नलिखित प्रश्न का उत्तर दें:\n\n"
            f"प्रश्न: {question_hi.strip()}\n\n"
            f"{instructions}"
        )
    else:
        ctx = str(context)[:3500] if context else "दस्तावेज़ उपलब्ध नहीं है।"
        return (
            "आप एक विशेषज्ञ सहायक हैं। दिए गए 'संदर्भ दस्तावेज़' के आधार पर प्रश्न का उत्तर दें।\n\n"
            f"=== संदर्भ दस्तावेज़ ===\n{ctx}\n====================\n\n"
            f"प्रश्न: {question_hi.strip()}\n\n"
            f"{instructions}"
        )

# ── Answer extractor ─────────────────────────────────────────────────────────
def extract_answer_from_output(text: str) -> str:
    if not isinstance(text, str): return text
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

# ── Local Sarvam Back-Translation ────────────────────────────────────────────
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
    
    formatted = _sarvam_bt_tokenizer.apply_chat_template(
        msgs, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    outputs = _sarvam_bt_llm.generate(
        [formatted], 
        SamplingParams(temperature=0.0, max_tokens=512), 
        use_tqdm=False
    )
    
    return outputs[0].outputs[0].text.strip()

# ── COMPATIBILITY ALIAS ──
# This fixes the ImportError in scripts/04_evaluate.py
hi_to_en_local = hi_to_en_local_sarvam