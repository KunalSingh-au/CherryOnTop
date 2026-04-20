import json
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXTRACTED_EN, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS, SARVAM_TRANSLATE_HF, INDICTRANS2_HF

def run_sarvam_vllm(docs):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    print("Loading Sarvam Translate on GPU 0...")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_TRANSLATE_HF)
    llm = LLM(model=SARVAM_TRANSLATE_HF, tensor_parallel_size=1, gpu_memory_utilization=0.6)
    
    prompts = []
    for doc in docs:
        text = doc.get("answer_en", "")[:4000] # Cap length just in case
        msgs = [{"role": "system", "content": "Translate the text below to Hindi."},
                {"role": "user", "content": text}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2048))
    del llm # Free VRAM
    torch.cuda.empty_cache()
    return [out.outputs[0].text.strip() for out in outputs]

def run_indictrans2(docs):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor
    
    print("Loading IndicTrans2 on GPU 0...")
    tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_HF, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(INDICTRANS2_HF, trust_remote_code=True).to("cuda:0")
    
    # Initialize the official AI4Bharat processor
    ip = IndicProcessor(inference=True)
    
    results = []
    for doc in docs:
        text = doc.get("answer_en", "")[:4000]
        
        # Skip empty strings to prevent processing errors
        if not text.strip():
            results.append("")
            continue
            
        # 1. Preprocess: Normalizes text and adds proper tags
        batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="hin_Deva")
        
        # 2. Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda:0")
        
        # 3. Generate Translation (use_cache=False fixes the transformers version clash!)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=False)
            
        # 4. Decode and Post-process (cleans up spacing and native characters)
        hi_text_raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_hi_text = ip.postprocess_batch(hi_text_raw, lang="hin_Deva")[0]
        
        results.append(final_hi_text)
        
    del model
    torch.cuda.empty_cache()
    return results

def main():
    with open(EXTRACTED_EN, "r") as f: 
        docs = [json.loads(line) for line in f]
    
    # --- COMMENTED OUT BECAUSE SARVAM IS ALREADY DONE! ---
    # os.makedirs(os.path.dirname(TRANSLATED_SARVAM), exist_ok=True)
    # sarvam_hi = run_sarvam_vllm(docs)
    # with open(TRANSLATED_SARVAM, "w", encoding="utf-8") as f:
    #     for doc, hi in zip(docs, sarvam_hi):
    #         f.write(json.dumps({"doc_id": doc["doc_id"], "answer_hi": hi}, ensure_ascii=False) + "\n")
            
    # 2. IndicTrans2 Translation (C5)
    os.makedirs(os.path.dirname(TRANSLATED_INDICTRANS), exist_ok=True)
    indic_hi = run_indictrans2(docs)
    with open(TRANSLATED_INDICTRANS, "w", encoding="utf-8") as f:
        for doc, hi in zip(docs, indic_hi):
            f.write(json.dumps({"doc_id": doc["doc_id"], "answer_hi": hi}, ensure_ascii=False) + "\n")

    print("Success: Local translations complete.")

if __name__ == "__main__": main()