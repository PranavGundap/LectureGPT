import os
import subprocess
import json
import glob
import shutil
import yt_dlp
import whisper
import time
import requests
import joblib
import numpy as np
import pandas as pd
import faiss
from pipeline.chunker import merge_segments
from rank_bm25 import BM25Okapi

print("Loaded main_pipeline from:", __file__)

# ===============================
# CONFIG
# ===============================
AUDIO_FOLDER = "audio"
JSON_FOLDER = "data/jsons"
EMBED_FILE = "data/embeddings.joblib"
WHISPER_MODEL = "medium"

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

# ===============================
# TIME FORMAT
# ===============================
def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"

def merge_adjacent_chunks(context_df):

    if len(context_df) <= 1:
        return context_df

    context_df = context_df.sort_values(
        ["source", "start"]
    ).reset_index(drop=True)

    merged = []

    current = context_df.iloc[0].copy()

    for i in range(1, len(context_df)):

        row = context_df.iloc[i]

        # Merge only if:
        # 1. Same video
        # 2. Gap <= 2 sec

        gap = row["start"] - current["end"]

        if (
            row["source"] == current["source"]
            and gap <= 2
        ):

            current["end"] = row["end"]

            current["text"] = (
                current["text"] + " " + row["text"]
            )

        else:

            merged.append(current)

            current = row.copy()

    merged.append(current)

    return pd.DataFrame(merged)

# ===============================
# DOWNLOAD FROM YOUTUBE
# ===============================
def download_youtube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{AUDIO_FOLDER}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        # Get video information
        info = ydl.extract_info(url, download=False)

        video_title = info.get("title", "YouTube Lecture")

        # Download video
        ydl.download([url])
    print("RETURNING:", video_title, f"{video_title}.mp3")
    audio_file = os.path.basename(
        ydl.prepare_filename(info)
    ).replace(".webm", ".mp3").replace(".m4a", ".mp3")

    print("TITLE:", video_title)
    print("AUDIO:", audio_file)

    return (video_title, audio_file)
    

def convert_video_to_mp3(video_path):
    """
    Convert any supported video (mp4/avi/mkv/mov) to mp3
    and save it inside AUDIO_FOLDER.
    """

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    mp3_path = os.path.join(AUDIO_FOLDER, base_name + ".mp3")

    # Already converted
    if os.path.exists(mp3_path):
        return mp3_path

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        mp3_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    # Delete original video after successful conversion
    if os.path.exists(mp3_path) and os.path.exists(video_path):
        os.remove(video_path)

    return mp3_path

# ===============================
# TRANSCRIBE AUDIO FILES
# ===============================
def transcribe_all():
    print("Loading Whisper model...")
    model = whisper.load_model(WHISPER_MODEL)

    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3"))
    print("Found audio files:", audio_files)

    for audio in audio_files:
        json_name = os.path.basename(audio) + ".json"
        json_path = os.path.join(JSON_FOLDER, json_name)

        if os.path.exists(json_path):
            continue 

        print("Transcribing:", audio)

        result = model.transcribe(
            audio,
            language="en",
            fp16=False
        )

        chunks = merge_segments(result["segments"])

        data = {
            "source": os.path.basename(audio),
            "chunks": chunks
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

# ===============================
# EMBEDDING
# ===============================
def create_embedding(text_list):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return response.json()["embeddings"]

def update_embeddings():
    existing_df = None
    processed_sources = set()

    if os.path.exists(EMBED_FILE):
        existing_df = joblib.load(EMBED_FILE)
        processed_sources = set(existing_df["source"].unique())

    json_files = glob.glob(f"{JSON_FOLDER}/*.json")
    new_records = []

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            content = json.load(f)

        source = content["source"]

        if source in processed_sources:
            continue

        print("Embedding:", source)

        texts = [c["text"] for c in content["chunks"]]
        embeddings = create_embedding(texts)

        for i, chunk in enumerate(content["chunks"]):
            new_records.append({
                "source": source,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "embedding": embeddings[i]
            })

    if new_records:
        new_df = pd.DataFrame(new_records)

        if existing_df is not None:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        joblib.dump(final_df, EMBED_FILE)
        print("Embeddings updated.")
    else:
        print("No new files to embed.")

def rebuild_embeddings():
    print("Rebuilding embeddings...")

    json_files = glob.glob(f"{JSON_FOLDER}/*.json")
    records = []

    for jf in json_files:

        with open(jf, encoding="utf-8") as f:
            content = json.load(f)

        source = content["source"]

        print("Embedding:", source)

        texts = [chunk["text"] for chunk in content["chunks"]]
        embeddings = create_embedding(texts)

        for i, chunk in enumerate(content["chunks"]):
            records.append({
                "source": source,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "embedding": embeddings[i]
            })

    if len(records) == 0:

        if os.path.exists(EMBED_FILE):
            os.remove(EMBED_FILE)

        print("No lectures remaining.")
        return

    df = pd.DataFrame(records)

    joblib.dump(df, EMBED_FILE)

    print("Embeddings rebuilt successfully.")

def delete_lecture(source):

    # -----------------------------
    # Delete Audio File
    # -----------------------------
    audio_path = os.path.join(AUDIO_FOLDER, source)

    if os.path.exists(audio_path):
        os.remove(audio_path)
        print("Deleted audio:", audio_path)

    # -----------------------------
    # Delete JSON
    # -----------------------------
    json_path = os.path.join(
        JSON_FOLDER,
        source + ".json"
    )

    if os.path.exists(json_path):
        os.remove(json_path)
        print("Deleted json:", json_path)

    # -----------------------------
    # Rebuild Embeddings
    # -----------------------------
    rebuild_embeddings()
# ===============================
# LLM
# ===============================
def ask_llm(prompt, num_predict=800, json_mode=False):

    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": 4096
        }
    }

    # 👇 Force JSON output when needed
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload
    )

    print("Status Code:", response.status_code)

    data = response.json()

    if "response" not in data:
        raise Exception(f"Ollama Error: {data}")

    return data["response"]
def parse_answer(answer):

    import re

    answer = re.sub(r"<[^>]*>", "", answer)
    answer = answer.replace("<tool_call>", "")
    answer = answer.replace("user", "")
    answer = answer.strip()

    sections = {
        "definition": "",
        "explanation": "",
        "advantages": "",
        "disadvantages": "",
        "applications": "",
        "interview": ""
    }

    current = None

    for line in answer.splitlines():
        line = line.strip()
        line_lower = line.lower().strip()

        line_lower = (
            line_lower
            .replace("📘", "")
            .replace("📖", "")
            .strip()
        )

        if line_lower.startswith("definition"):
            current = "definition"
            continue

        elif line_lower.startswith("explanation"):
            current = "explanation"
            continue

        elif line_lower.startswith("advantages"):
            current = "advantages"
            continue

        elif line_lower.startswith("disadvantages"):
            current = "disadvantages"
            continue

        elif (
            line_lower.startswith("applications")
            or line_lower.startswith("real-world applications")
        ):
            current = "applications"
            continue

        elif line_lower.startswith("interview"):
            current = "interview"
            continue

        if current:
            import re

            
            # Remove HTML tags
            line = re.sub(r"<[^>]+>", "", line)

            # Remove escaped HTML
            line = (
                line.replace("&lt;", "<")
                    .replace("&gt;", ">")
            )

            # Remove tags again
            line = re.sub(r"<[^>]+>", "", line)

            # Remove common junk
            junk = {
                "",
                "</div>",
                "<div>",
                "div",
                "/div",
                "<br>",
                "</br>"
            }

            line = line.strip()

            if line.lower() in junk:
                continue

            sections[current] += line + "\n"

    # Clean all sections AFTER parsing
    for key in sections:
        sections[key] = (
            sections[key]
            .replace("</div>", "")
            .replace("<div>", "")
            .replace("&lt;/div&gt;", "")
            .replace("&lt;div&gt;", "")
            .replace("/div", "")
            .strip()
        )

    if sections[key] == "":
        sections[key] = "Not available"

    return sections

    for key in sections:
        if not sections[key].strip():
            sections[key] = "Not available"

import re

def clean_llm_output(text):

    text = re.sub(r"<[^>]+>", "", text)

    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = re.sub(r"<[^>]+>", "", text)

    text = text.replace("```", "")

    text = text.replace("<tool_call>", "")
    text = text.replace("</tool_call>", "")

    text = re.sub(r"</[a-zA-Z0-9]+>", "", text)
    text = re.sub(r"<[a-zA-Z0-9]+>", "", text)

    return text.strip()


# ===============================
# CHATBOT
# ===============================
def start_chatbot():
    df = joblib.load(EMBED_FILE)

    t1 = time.time()

    # ----------------------------
    # Build FAISS Index
    # ----------------------------

    embeddings = np.array(
        df["embedding"].tolist(),
        dtype=np.float32
    )

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)
    # ----------------------------
    # BM25 Index (created once)
    # ----------------------------

    tokenized_corpus = [
        text.lower().split()
        for text in df["text"]
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    print("🔍 Index Build Time:", round(time.time() - t1, 2), "sec")

    print("Chunks loaded:", len(df))
    print("\nType 'exit' to stop.\n")
 
def ask_question(question, selected_source=None):
    print("ask_question called. selected_source =", selected_source)

    t0 = time.time()

    df = joblib.load(EMBED_FILE)

    print("📂 Load Time:", round(time.time() - t0, 2), "sec")

    print("=" * 60)
    print("Selected Source:", selected_source)
    print("Available Sources:")
    print(df["source"].unique())
    print("=" * 60)

    # Search only in selected lecture
    if selected_source is not None:
        df = df[df["source"] == selected_source].reset_index(drop=True)

        print("Rows after filter:", len(df))   # <-- YEH LINE ADD KARO

        if df.empty:
            return {
                "answer": "No data found for the selected lecture.",
                "definition": "",
                "explanation": "",
                "advantages": "",
                "disadvantages": "",
                "applications": "",
                "interview": "",
                "summary": "",
                "source": "Unknown",
                "timestamp": "--"
            }
    # ----------------------------
    # Build FAISS Index
    # ----------------------------

    embeddings = np.array(
        df["embedding"].tolist(),
        dtype=np.float32
    )
    t1 = time.time()
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    # ----------------------------
    # BM25
    # ----------------------------

    tokenized_corpus = [
        text.lower().split()
        for text in df["text"]
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # ----------------------------
    # Embedding Search
    # ----------------------------

    question_embedding = np.array(
        [create_embedding([question])[0]],
        dtype=np.float32
    )

    faiss.normalize_L2(question_embedding)

    scores, ids = index.search(
        question_embedding,
        len(df)
    )

    embedding_scores = np.zeros(len(df))

    for score, idx in zip(scores[0], ids[0]):
        embedding_scores[idx] = score

    # ----------------------------
    # BM25 Search
    # ----------------------------

    query_tokens = question.lower().split()

    bm25_scores = np.array(
        bm25.get_scores(query_tokens),
        dtype=np.float32
    )

    # Normalize BM25 scores

    if bm25_scores.max() != 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # ----------------------------
    # Hybrid Score
    # ----------------------------

    final_scores = (
        0.70 * embedding_scores
        +
        0.30 * bm25_scores
    )
    # ----------------------------
    # Definition Question Boost
    # ----------------------------

    question_lower = question.lower().strip()

    if question_lower.startswith("what is") or question_lower.startswith("define"):

        definition_patterns = [
            " is a ",
            " is an ",
            " refers to ",
            " is the ",
            " means "
        ]

        for i, row in df.iterrows():

            text = row["text"].lower()

            # Boost chunks that look like definitions
            if any(pattern in text for pattern in definition_patterns):
                final_scores[i] += 0.15

            # Extra boost if the exact concept appears
            concept = question_lower.replace("what is", "").replace("define", "").strip()

            if concept and concept in text:
                final_scores[i] += 0.20

    if question.lower().startswith(("what is", "define")):
        TOP_K = 5
    else:
        TOP_K = 3

    indices = final_scores.argsort()[::-1][:TOP_K]


    print("\n===== TOP SCORES =====")

    for idx in indices:
        print("\nScore:", final_scores[idx])
        print(
            "Time:",
            format_time(df.loc[idx]["start"]),
            "-",
            format_time(df.loc[idx]["end"])
        )
        print(
            "Text:",
            df.loc[idx]["text"][:150]
        )


    context_df = df.loc[indices]
    context_df = merge_adjacent_chunks(context_df)
    print("\nMerged Chunks :", len(context_df))
    print("\n" + "=" * 70)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 70)

    for i, row in context_df.iterrows():

        print(f"\nChunk {i+1}")

        print("Video :", row["source"])

        print(
            "Time  :",
            format_time(row["start"]),
            "-",
            format_time(row["end"])
        )

        print("\nTranscript:")

        print(row["text"])

        print("-" * 70)

    # ----------------------------
    # Best Retrieved Chunk
    # ----------------------------

    # ----------------------------
# Best Retrieved Chunk
# ----------------------------

    concept = (
        question_lower
        .replace("what is", "")
        .replace("define", "")
        .replace("explain", "")
        .replace("tell me about", "")
        .strip()
    )

    matched_rows = df[
        df["text"]
        .str.lower()
        .str.contains(
            concept,
            na=False
        )
    ]

    if len(matched_rows) > 0:
        best_row = matched_rows.iloc[0]

    else:
        best_row = df.loc[indices[0]]

    source_video = best_row["source"]

    timestamp = (
        f"{format_time(best_row['start'])}"
        f" - "
        f"{format_time(best_row['end'])}"
    )

    print("\n🤖 Generating answer from retrieved lecture content...\n")

    print(f"📺 Source Video : {source_video}")
    print(f"⏰ Timestamp    : {timestamp}\n")

    context_text = ""

    for i, (_, row) in enumerate(context_df.iterrows(), start=1):

        start_time = format_time(row["start"])
        end_time = format_time(row["end"])

        context_text += row["text"] + "\n\n"

    prompt = f"""
    You are VidMind AI, an intelligent RAG-based Teaching Assistant.

    Your job is to answer the user's question using the provided lecture transcript.

    Rules:

    - Use the lecture transcript as the PRIMARY source.
    - Explain in simple language.
    - Do NOT copy long sentences from the transcript.
    - You may use your own general knowledge ONLY for:
    - Advantages
    - Disadvantages
    - Applications
    - Interview Questions
    - Never contradict the lecture.
    - If the answer is not present in the transcript, reply ONLY:
    I could not find the answer in the uploaded lecture.

    Return ONLY plain text.

    Return EXACTLY in this format.

    Definition:
    (1-2 sentences)

    Explanation:
    (4-6 sentences)

    Advantages:
    • Point 1
    • Point 2
    • Point 3

    Disadvantages:
    • Point 1
    • Point 2
    • Point 3

    Applications:
    • Point 1
    • Point 2
    • Point 3

    Interview Questions:
    1.
    2.
    3.
    4.
    5.

    Lecture Transcript:
    {context_text}

    User Question:
    {question}
    """

    summary_prompt = f"""
    You are VidMind AI.

    Generate a professional lecture summary.

    Rules:

    - Use the lecture transcript as the PRIMARY source.
    - Use your own general knowledge only to improve explanations.
    - Never contradict the transcript.
    - Explain in very simple language.
    - Cover every important topic discussed.
    - Mention key concepts.
    - Around 200-300 words.
    - Return ONLY the summary.
    - Do not use HTML.
    - Do not use Markdown.
    - Do not use XML.
    - Do not use code blocks.

    Lecture Transcript:

    {context_text}
    """
    t2 = time.time()

    answer = ask_llm(prompt)
    answer = clean_llm_output(answer)


    print("🤖 LLM Time:", round(time.time() - t2, 2), "sec")
    print(answer)
    print("="*50)
    print("="*50)

    summary_prompt = f"""
    You are an AI Teaching Assistant.

    Read the following lecture transcript and generate a lecture summary.

    Transcript:
    {context_text}

    Instructions:
    - Use the transcript as the primary source.
    - Use your own general knowledge only to improve explanations where appropriate.
    - Do not contradict the lecture.
    - Cover all important topics.
    - Explain in simple language.
    - Write around 200-300 words.

    Return ONLY the summary.
"""

    summary = ask_llm(summary_prompt)
    summary = clean_llm_output(summary)
    sections = parse_answer(answer)

    return {

        "answer": answer,

        "definition": sections["definition"],

        "explanation": sections["explanation"],

        "advantages": sections["advantages"],

        "disadvantages": sections["disadvantages"],

        "applications": sections["applications"],

        "interview": sections["interview"],

        "summary": summary,

        "source": source_video,

        "timestamp": timestamp,

        "context_text": context_text
    }

from quiz_engine import parse_quiz

def generate_quiz(context_text):

    quiz_prompt = f"""
You are VidMind AI.

Read the lecture transcript carefully.

Generate EXACTLY 10 multiple choice questions.

Rules:

- Use ONLY lecture content.
- Do not ask outside the lecture.
- Each question must have exactly four options.
- Use your own general knowledge only if the lecture does not provide sufficient information.

Return EXACTLY in this format.

Question 1:
What is Machine Learning?

A. Database
B. Artificial Intelligence Technique
C. Operating System
D. Compiler

Answer: B

Explanation:
Machine Learning is a branch of Artificial Intelligence.

----------------------------------------

Question 2:

...

Lecture Transcript:

{context_text}
"""

    response = ask_llm(
        quiz_prompt,
        num_predict=1200
    )

    response = clean_llm_output(response)

    print("\n================ QUIZ =================\n")
    print(response)
    print("\n=======================================\n")

    quiz = parse_quiz(response)

    return quiz

# ===============================
# MAIN MENU
# ===============================
if __name__ == "__main__":

    print("1. Process YouTube URL")
    print("2. Add Local Video File")
    print("3. Reprocess Existing Audio Files")
    print("4. Start Chatbot")

    choice = input("Choose option: ")

    if choice == "1":
        url = input("Enter YouTube URL or Playlist: ")
        download_youtube(url)
        transcribe_all()
        update_embeddings()
        start_chatbot()

    elif choice == "2":
        path = input("Enter local video file path: ")
        shutil.copy(path, AUDIO_FOLDER)
        transcribe_all()
        update_embeddings()
        start_chatbot()

    elif choice == "3":

        transcribe_all()

        update_embeddings()

        print("\n✅ Existing audio files processed successfully.\n")

    elif choice == "4":

        while True:

            question = input("You: ")

            if question.lower() == "exit":
                break

            result = ask_question(question)

            print(result["answer"])
            print("\n" + "─"*60)