from pathlib import Path
import pandas as pd
from langdetect import detect
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent

manifest = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "image_manifest.parquet"
)

print(f"Total records: {len(manifest)}")

sample = manifest.sample(5000, random_state=42)

languages = []

for title in sample["title"].fillna(""):

    try:
        lang = detect(str(title))
    except:
        lang = "unknown"

    languages.append(lang)

counter = Counter(languages)

print("\nLanguage Distribution (5000 Sample)\n")

for lang, count in counter.most_common():
    print(f"{lang:10} : {count}")