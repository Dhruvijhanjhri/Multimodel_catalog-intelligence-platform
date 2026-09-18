import numpy as np
import pandas as pd
import faiss


EMBEDDINGS_FILE = "embeddings/train_text_embeddings.npy"
METADATA_FILE = "embeddings/embedding_metadata.parquet"


embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
metadata = pd.read_parquet(METADATA_FILE)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

scores, indices = index.search(embeddings, 2)

candidate_count = 0
exact_title_matches = 0
different_title_candidates = 0

for i in range(len(metadata)):
    score = float(scores[i][1])
    j = int(indices[i][1])

    if score > 0.90:
        candidate_count += 1

        title_a = str(metadata.iloc[i]["title"]).strip().lower()
        title_b = str(metadata.iloc[j]["title"]).strip().lower()

        if title_a == title_b:
            exact_title_matches += 1
        else:
            different_title_candidates += 1


print("Duplicate candidates:", candidate_count)
print("Exact title matches:", exact_title_matches)
print("Different title candidates:", different_title_candidates)