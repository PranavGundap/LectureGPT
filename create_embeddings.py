import os
import json
import requests
import pandas as pd
import joblib

JSON_FOLDER = "jsons"

OUTPUT_FILE = "embeddings.joblib"


def create_embedding(text_list):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    return response.json()["embeddings"]


all_chunks = []

chunk_id = 0


for file in os.listdir(JSON_FOLDER):

    if not file.endswith(".json"):
        continue

    path = os.path.join(JSON_FOLDER, file)

    with open(path, encoding="utf-8") as f:

        data = json.load(f)


    texts = [c["text"] for c in data["chunks"]]

    print("Embedding:", file)


    embeddings = create_embedding(texts)


    for i, chunk in enumerate(data["chunks"]):

        all_chunks.append({

            "chunk_id": chunk_id,

            "source": data["source"],

            "start": chunk["start"],

            "end": chunk["end"],

            "text": chunk["text"],

            "embedding": embeddings[i]

        })

        chunk_id += 1


df = pd.DataFrame(all_chunks)

joblib.dump(df, OUTPUT_FILE)

print("Embeddings saved to", OUTPUT_FILE)