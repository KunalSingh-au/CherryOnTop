"""
scripts/02_translate_docs.py — Translate English docs to Hindi (C2 and C5).

What it does:
  - Reads data/extracted/english_docs.jsonl
  - Translates each document to Hindi using two engines:
      Sarvam-Translate  → data/translated/sarvam_hi.jsonl       (C2 context)
      IndicTrans2       → data/translated/indictrans2_hi.jsonl   (C5 context)
  - Use --engine to run one or both

Run once.  Translations take ~30–60 min per engine on a single GPU.

Usage:
    python scripts/02_translate_docs.py --engine sarvam
    python scripts/02_translate_docs.py --engine indictrans2
    python scripts/02_translate_docs.py --engine both        ← runs both
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    EXTRACTED_EN, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS,
    SARVAM_TRANSLATE_HF, INDICTRANS2_HF,
)


# ─────────────────────────────────────────────────────────────────────────────
# SARVAM-TRANSLATE (via vLLM)
# ─────────────────────────────────────────────────────────────────────────────

def run_sarvam_translation(docs: list) -> list:
    """
    Batch-translate a list of English document dicts to Hindi using
    Sarvam-Translate served by vLLM.

    ── TUNABLE: Increase max_tokens (currently 2048) for very long documents ──
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading {SARVAM_TRANSLATE_HF} on GPU 0 …")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_TRANSLATE_HF)
    llm = LLM(
        model=SARVAM_TRANSLATE_HF,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.60,
    )

    prompts = []
    for doc in docs:
        text = doc.get("answer_en", "")[:4000]  # Cap to avoid context overflow
        msgs = [
            {"role": "system", "content": "Translate the text below to Hindi. "
                                          "Preserve all numbers, scheme names, "
                                          "and institutional acronyms verbatim."},
            {"role": "user", "content": text},
        ]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2048))

    # Free VRAM before returning so IndicTrans2 can load if needed
    del llm
    torch.cuda.empty_cache()

    return [out.outputs[0].text.strip() for out in outputs]


# ─────────────────────────────────────────────────────────────────────────────
# INDICTRANS2 (via HuggingFace + IndicTransToolkit)
# ─────────────────────────────────────────────────────────────────────────────

def run_indictrans2_translation(docs: list) -> list:
    """
    Translate English documents to Hindi using IndicTrans2 (en-indic-1B).

    Uses the official AI4Bharat IndicTransToolkit preprocessor/postprocessor
    which handles Devanagari normalisation correctly.

    ── TUNABLE: max_length (1024) controls max input tokens per segment ────────
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    print(f"Loading {INDICTRANS2_HF} on GPU 0 …")
    tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_HF, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS2_HF, trust_remote_code=True
    ).to("cuda:0")

    ip = IndicProcessor(inference=True)
    results = []

    for i, doc in enumerate(docs):
        text = doc.get("answer_en", "")[:4000]

        if not text.strip():
            results.append("")
            continue

        try:
            # Step 1: Normalize and tag for source/target languages
            batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="hin_Deva")

            # Step 2: Tokenize
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,    # ── TUNABLE ───────────────────────────────
            ).to("cuda:0")

            # Step 3: Generate (use_cache=False prevents transformer version clash)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=False)

            # Step 4: Decode and post-process
            raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated = ip.postprocess_batch(raw, lang="hin_Deva")[0]
            results.append(translated)

        except Exception as e:
            print(f"  [WARN] IndicTrans2 failed for doc {i}: {e}")
            results.append("")

    del model
    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Translate English docs to Hindi.")
    parser.add_argument(
        "--engine",
        choices=["sarvam", "indictrans2", "both"],
        default="both",
        help="Which translation engine to run (default: both)",
    )
    args = parser.parse_args()

    # Load extracted English documents
    if not os.path.exists(EXTRACTED_EN):
        print(f"Error: {EXTRACTED_EN} not found. Run 01_extract_docs.py first.")
        sys.exit(1)

    with open(EXTRACTED_EN, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(docs)} English documents.")

    os.makedirs(os.path.dirname(TRANSLATED_SARVAM), exist_ok=True)

    # ── Sarvam-Translate ───────────────────────────────────────────────────────
    if args.engine in ("sarvam", "both"):
        print("\n── Running Sarvam-Translate ──")
        translations = run_sarvam_translation(docs)
        with open(TRANSLATED_SARVAM, "w", encoding="utf-8") as f:
            for doc, hi in zip(docs, translations):
                f.write(json.dumps(
                    {"doc_id": doc["doc_id"], "ministry": doc["ministry"], "answer_hi": hi},
                    ensure_ascii=False,
                ) + "\n")
        print(f"  Saved → {TRANSLATED_SARVAM}")

    # ── IndicTrans2 ────────────────────────────────────────────────────────────
    if args.engine in ("indictrans2", "both"):
        print("\n── Running IndicTrans2 ──")
        translations = run_indictrans2_translation(docs)
        with open(TRANSLATED_INDICTRANS, "w", encoding="utf-8") as f:
            for doc, hi in zip(docs, translations):
                f.write(json.dumps(
                    {"doc_id": doc["doc_id"], "ministry": doc["ministry"], "answer_hi": hi},
                    ensure_ascii=False,
                ) + "\n")
        print(f"  Saved → {TRANSLATED_INDICTRANS}")

    print("\n✓ Translation complete.")


if __name__ == "__main__":
    main()
