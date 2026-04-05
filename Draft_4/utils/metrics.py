# utils/metrics.py — all evaluation metrics

import re
from typing import Optional


def chrf_score(hypothesis: str, reference: str) -> float:
    try:
        import sacrebleu
        return sacrebleu.sentence_chrf(hypothesis, [reference]).score
    except:
        return 0.0


def bertscore_batch(hypotheses: list, references: list) -> list:
    try:
        from bert_score import score
        _, _, F = score(
            [h if h else " " for h in hypotheses],
            [r if r else " " for r in references],
            lang="en", verbose=False
        )
        return [round(float(f), 4) for f in F]
    except:
        return [0.0] * len(hypotheses)


def ne_preservation(original: str, back: str) -> Optional[float]:
    """Fraction of named entities from original that survive in back-translation."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        orig_ents = set(e.text.lower() for e in nlp(original).ents)
        back_ents = set(e.text.lower() for e in nlp(back).ents)
        if not orig_ents:
            return None
        return round(len(orig_ents & back_ents) / len(orig_ents), 3)
    except:
        return None


def numeric_fidelity(original: str, back: str) -> Optional[float]:
    """Fraction of numbers from original that survive in back-translation."""
    def extract_nums(text):
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    orig_nums = extract_nums(original)
    if not orig_nums:
        return None
    return round(len(orig_nums & extract_nums(back)) / len(orig_nums), 3)


def acronym_fidelity(original: str, back: str) -> Optional[float]:
    """Fraction of acronyms from original that survive in back-translation."""
    orig_acr = set(re.findall(r"\b[A-Z]{2,}\b", original))
    if not orig_acr:
        return None
    back_acr = set(re.findall(r"\b[A-Z]{2,}\b", back))
    return round(len(orig_acr & back_acr) / len(orig_acr), 3)


def keyword_match(gold_answer_en: str, back_translated_answer: str,
                  keywords_str: str = "") -> bool:
    """
    Check if any expected keyword appears in the back-translated answer.
    keywords_str: pipe-separated alternatives e.g. "nine|9|नौ"
    Falls back to substring match against gold_answer_en if no keywords.
    """
    text = back_translated_answer.lower()
    if keywords_str:
        keywords = [k.strip().lower() for k in keywords_str.split("|")]
        return any(k in text for k in keywords if k)
    # Fallback: check if gold answer words appear
    gold_words = set(gold_answer_en.lower().split())
    return len(gold_words & set(text.split())) / max(len(gold_words), 1) >= 0.5


def semantic_score_gpt(gold_en: str, generated_en: str, openai_key: str) -> dict:
    """
    GPT-4o judges semantic equivalence on 0-5 scale.
    5 = identical meaning, 0 = completely wrong.
    """
    import re, json
    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        prompt = f"""You are evaluating a generated answer against a gold answer for Indian government QA.

Gold answer: {gold_en}
Generated answer: {generated_en}

Rate semantic equivalence 0-5:
5 = All facts correct, identical meaning
4 = Correct meaning, minor wording difference  
3 = Mostly correct, one small detail off
2 = Partially correct, key fact wrong or missing
1 = Wrong with some overlap
0 = Completely wrong or empty

Return JSON only: {{"score": <0-5>, "reasoning": "<one sentence>"}}"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=80
        )
        text = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        return json.loads(text)
    except Exception as e:
        return {"score": -1, "reasoning": str(e)}
