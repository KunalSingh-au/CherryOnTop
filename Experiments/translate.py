import requests
import csv
import time

API_KEY = "sk_rksnmkuo_wAFfihZCARFnqDCExLM6slF7"

def translate(text):
    response = requests.post(
        "https://api.sarvam.ai/translate",
        headers={
            "api-subscription-key": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "input": text,
            "source_language_code": "en-IN",
            "target_language_code": "hi-IN",
            "mode": "formal"
        }
    )
    data = response.json()
    return data.get("translated_text", "ERROR")

# Read sentences
rows = []
with open("sentences.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Translate each sentence
for row in rows:
    print(f"Translating: {row['sentence']}")
    row["hindi_sarvam"] = translate(row["sentence"])
    time.sleep(1)  # be polite to the API

# Write results
fieldnames = ["id", "occupation", "sentence", "hindi_sarvam"]
with open("translations.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nDone. Open translations.csv to see results.")