import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_qa_prompt(question_hi: str, context: str, condition: str) -> str:
    """
    Build user-turn prompt. Identical across all models for fair comparison.
    """
    instructions = (
        "**भाषा:** उत्तर पूर्णतः देवनागरी हिंदी में लिखें। "
        "केवल आवश्यक संक्षिप्त नाम/संख्याएँ मूल रूप में रखी जा सकती हैं।\n"
        "**संक्षिप्तता:** उत्तर सटीक और सीधा होना चाहिए (अधिकतम 100 शब्द)।\n"
    )
    if condition == "C3":
        return (
            "आप एक विशेषज्ञ सहायक हैं। निम्नलिखित प्रश्न का उत्तर दें:\n\n"
            f"प्रश्न: {question_hi}\n\nनिर्देश:\n{instructions}"
        )
    else:
        ctx = str(context)[:3500] if context else ""
        return (
            "आप एक विशेषज्ञ सहायक हैं। दिए गए 'संदर्भ दस्तावेज़' के आधार पर "
            "प्रश्न का उत्तर दें। यदि उत्तर दस्तावेज़ में नहीं है, तो "
            "'जानकारी उपलब्ध नहीं है' कहें।\n\n"
            f"=== संदर्भ दस्तावेज़ ===\n{ctx}\n====================\n\n"
            f"प्रश्न: {question_hi}\n\nनिर्देश:\n{instructions}"
        )


# ── Answer extractor (handles Sarvam-30B <think> blocks) ──────────────────────
def extract_answer_from_output(text: str) -> str:
    text = str(text).strip()
    if "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if len(after) > 3:
            return after
    if "<think>" in text:
        for pat in [
            r"\*\*Final Answer[:\*\s]+\"?([^\n\"]{5,400})",
            r"Final Answer[:\s]+\"?([^\n\"]{5,400})",
            r"\*\*उत्तर[:\*\s]+([^\n]{5,400})",
            r"हिंदी में उत्तर[:\s]+([^\n]{5,400})",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip().strip('"').strip("*").strip()
                if len(ans) > 4:
                    return ans
        meta = ["Constraint", "Instruction", "Task", "Persona", "Rule", "Step"]
        for line in reversed(re.findall(r"[^\n]*[\u0900-\u097F][^\n]{5,}", text)):
            line = line.strip().strip("*").strip('"')
            if len(line) > 10 and not any(w in line for w in meta):
                return line
    return text[:400]


# ── Local back-translation: IndicTrans2 hi→en ─────────────────────────────────
# Singleton — loads once per process, reused for all 2500 rows
_bt_model = None
_bt_tok   = None
_bt_ip    = None


def _load_bt():
    global _bt_model, _bt_tok, _bt_ip
    if _bt_model is not None:
        return
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # hi→en model (note: different from en→indic used in 02_translate_docs.py)
    model_id = os.getenv("INDICTRANS2_HI_EN", "ai4bharat/indictrans2-indic-en-1B")
    print(f"  [back-translate] Loading {model_id} ...")
    _bt_tok   = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _bt_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True
    ).to("cuda:0")
    _bt_model.eval()
    try:
        from IndicTransToolkit import IndicProcessor
        _bt_ip = IndicProcessor(inference=True)
    except ImportError:
        _bt_ip = None
    print("  [back-translate] Ready")


def hi_to_en_local(text: str) -> str:
    """Back-translate a Hindi string to English using local IndicTrans2. No API."""
    import torch
    if not text or not text.strip():
        return ""
    text = text.strip()[:1500]
    try:
        _load_bt()
        if _bt_ip:
            batch  = _bt_ip.preprocess_batch([text], src_lang="hin_Deva", tgt_lang="eng_Latn")
            inputs = _bt_tok(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to("cuda:0")
        else:
            inputs = _bt_tok(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to("cuda:0")
        with torch.no_grad():
            out_ids = _bt_model.generate(**inputs, max_new_tokens=512, use_cache=False)
        decoded = _bt_tok.batch_decode(out_ids, skip_special_tokens=True)
        return (_bt_ip.postprocess_batch(decoded, lang="eng_Latn")[0]
                if _bt_ip else decoded[0])
    except Exception as e:
        return f"BT_ERROR: {e}"


# ── Public alias (04_evaluate.py calls this) ──────────────────────────────────
def sarvam_hi_to_en(text: str) -> str:
    """Back-translate Hindi → English locally (no API key needed)."""
    return hi_to_en_local(text)
