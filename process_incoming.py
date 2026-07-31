import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"

# -------- EMBEDDING FUNCTION --------
def create_embedding(text_list):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return response.json()["embeddings"]

# -------- LLM FUNCTION (STABLE FOR 8GB) --------
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2:1.5b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 200,
                "temperature": 0.3
            }
        }
    )

    data = response.json()

    if "response" not in data:
        print("LLM error:", data)
        raise Exception("LLM failed")

    return data["response"]

# -------- LOAD EMBEDDINGS --------
df = joblib.load("embeddings.joblib")
print("Chunks loaded:", len(df))

chat_history = ""

print("\nType 'exit' to stop.\n")

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        break

    # -------- EMBED QUESTION --------
    question_embedding = create_embedding([user_question])[0]

    # -------- SIMILARITY SEARCH --------
    similarities = []
    for emb in df["embedding"]:
        sim = cosine_similarity([emb], [question_embedding])[0][0]
        similarities.append(sim)

    similarities = np.array(similarities)
    top_results = 2
    max_indx = similarities.argsort()[::-1][:top_results]
    new_df = df.loc[max_indx]

    # -------- BUILD SMALL CONTEXT --------
    context_text = ""

    for _, row in new_df.iterrows():
        short_text = row["text"][:250]
        start_time = format_time(row['start'])
        end_time = format_time(row['end'])
        context_text += f"""
Video: {row['source']}
Time: {start_time} - {end_time}
Text: {short_text}
"""

    # -------- BUILD FINAL PROMPT --------
    prompt = f"""
You are an AI teaching assistant.

Use ONLY the information from the transcript below.
Do NOT guess.
If the topic is not clearly mentioned, say it is not found.

Transcript:
{context_text}

Student Question:
{user_question}

Instructions:
- Clearly state the video name.
- Clearly state timestamps.
- Do not include unrelated videos.
- Keep answer concise.
"""

    answer = ask_llm(prompt)

    print("\nAssistant:", answer, "\n")

    chat_history += f"\nUser: {user_question}\nAssistant: {answer}\n"