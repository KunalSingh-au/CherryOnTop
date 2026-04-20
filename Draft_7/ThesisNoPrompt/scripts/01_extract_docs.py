import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PDF_DIR_EN, PDF_DIR_HI, MINISTRY_MAP, EXTRACTED_EN, EXTRACTED_HI
from utils.extract import read_pdf, parse_english_qa, parse_hindi_qa

def extract_tree(base_dir, lang):
    records = []
    for sub, min_label in MINISTRY_MAP.items():
        folder = os.path.join(base_dir, sub)
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.endswith(".pdf"):
                doc_id = f.replace(".pdf", "")
                text = read_pdf(os.path.join(folder, f))
                q, a = parse_english_qa(text) if lang == "en" else parse_hindi_qa(text)
                records.append({"doc_id": doc_id, "ministry": min_label, f"answer_{lang}": a})
    return records

def main():
    os.makedirs(os.path.dirname(EXTRACTED_EN), exist_ok=True)
    
    en_recs = extract_tree(PDF_DIR_EN, "en")
    with open(EXTRACTED_EN, "w", encoding="utf-8") as f:
        for r in en_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    hi_recs = extract_tree(PDF_DIR_HI, "hi")
    with open(EXTRACTED_HI, "w", encoding="utf-8") as f:
        for r in hi_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
    print(f"Extracted {len(en_recs)} EN docs and {len(hi_recs)} HI docs.")

if __name__ == "__main__": main()