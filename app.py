import streamlit as st
import os
import joblib
import numpy as np
import requests
import shutil
import yt_dlp
import whisper
import glob
import json
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================
DATA_FOLDER = "data"
AUDIO_FOLDER = os.path.join(DATA_FOLDER, "audios")
JSON_FOLDER = os.path.join(DATA_FOLDER, "jsons")
EMBED_FILE = os.path.join(DATA_FOLDER, "embeddings.joblib")
WHISPER_MODEL = "medium"

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

st.set_page_config(page_title="VidMind AI", layout="wide")

# =====================================
# STYLING
# =====================================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #00FFD1;
}
.creator {
    font-size: 20px;
    font-weight: bold;
    color: #FFD700;
}
.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #00FFD1;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown(
    '<div class="big-title">🎓 VIDMIND AI</div>',
    unsafe_allow_html=True
)

st.caption("AI Teaching Assistant powered by RAG + Ollama")
st.markdown("---")
st.caption("Developed by Pranav Gundap")


# =====================================
# UTIL FUNCTIONS
# =====================================
def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds//60:02d}:{seconds%60:02d}"

def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": [text]}
    )
    return r.json()["embeddings"][0]

def ask_llm(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2:1.5b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 150, "temperature": 0.3}
        }
    )
    return r.json()["response"]

def load_embeddings():
    if os.path.exists(EMBED_FILE):
        return joblib.load(EMBED_FILE)
    return None

# =====================================
# SIDEBAR
# =====================================
menu = st.sidebar.radio(
    "Navigation",
    ["📺 Upload Video", "🔗 Add YouTube URL", "💬 Ask Questions", "📂 Library"]
)

# =====================================
# VIDEO UPLOAD
# =====================================
if menu == "📺 Upload Video":

    st.markdown('<div class="section-title">Upload Recorded Video</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "mkv"])

    if uploaded_file:
        save_path = os.path.join(AUDIO_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Video uploaded successfully.")
        st.info("Now process using terminal → python main_pipeline.py")

# =====================================
# YOUTUBE URL
# =====================================
elif menu == "🔗 Add YouTube URL":

    st.markdown('<div class="section-title">Process YouTube Video</div>', unsafe_allow_html=True)

    url = st.text_input("Enter YouTube URL")

    if st.button("Download & Process"):
        st.info("Run in terminal: python main_pipeline.py → Option 1")

# =====================================
# CHAT
# =====================================
elif menu == "💬 Ask Questions":

    st.markdown('<div class="section-title">Ask Questions About Your Videos</div>', unsafe_allow_html=True)

    df = load_embeddings()

    if df is None:
        st.error("No videos processed yet.")
    else:
        question = st.text_input("Enter your question")

        if st.button("Get Answer"):

            with st.spinner("Searching..."):

                q_embed = create_embedding(question)

                embeddings = np.array(
                    df["embedding"].tolist(),
                    dtype=np.float32
                )

                q_embed = np.array(q_embed, dtype=np.float32)

                sims = cosine_similarity(
                    embeddings,
                    q_embed.reshape(1, -1)
                ).flatten()

                top_idx = sims.argsort()[::-1][:2]
                context_df = df.loc[top_idx]
                context = "\n\n".join(context_df["text"].tolist())
                answer_prompt = f"""
                You are VidMind AI, an AI Teaching Assistant.

                Use ONLY the transcript context below to answer the user's question.

                If the answer is not present in the context, say:

                "I couldn't find the answer in the processed lecture."

                Explain in simple English.

                Give the answer in bullet points whenever possible.

                Question:
                {question}

                Transcript Context:
                {context}
                """
                answer = ask_llm(answer_prompt)
                
                st.markdown("## 📌 Results")
                st.markdown("## 🤖 AI Answer")

                with st.chat_message("assistant"):
                 st.write(answer)

                st.markdown("---")

                for rank, idx in enumerate(top_idx):

                    row = df.loc[idx]
                    score = sims[idx] * 100

                    start_time = format_time(row["start"])
                    end_time = format_time(row["end"])

                    youtube_link = (
                        f"https://www.youtube.com/results?"
                        f"search_query={row['source']}"
                    )
                    score = sims[idx] * 100

                    with st.container(border=True):

                        st.markdown("---")

                    if rank == 0:
                        st.subheader("🥇 Best Match")
                    else:
                        st.subheader("🥈 Second Match")
                    title = (
                        row["source"]
                        .replace(".mp3", "")
                        .replace("Python Full Course", "")
                    )

                    st.markdown(f"### {title}")

                    col1 = st.container()

                    with col1:

                        st.caption(
                        f"⏰ {start_time} → {end_time}   |   ⭐ {score:.1f}% Match"
                        )

                        st.markdown("📝 **Transcript Preview:**")

                        preview = row["text"].strip()

                        if len(preview) > 220:
                            preview = preview[:220] + "..."

                        st.info(preview)

                        with st.expander("📄 View Full Transcript"):
                            st.write(row["text"])

                        st.link_button("▶ Open Video", youtube_link)

                    

                # SUMMARY BUTTON
                if st.button("Generate Summary"):
                    combined_text = " ".join(context_df["text"].tolist())[:800]

                    prompt = f"""
Summarize the following transcript section clearly:

{combined_text}
"""

                    summary = ask_llm(prompt)
                    st.markdown("### 🧠 Summary")
                    st.write(summary)

# =====================================
# LIBRARY
# =====================================
elif menu == "📂 Library":

    st.markdown('<div class="section-title">Processed Videos</div>', unsafe_allow_html=True)

    df = load_embeddings()

    if df is None:
        st.warning("No videos processed yet.")
    else:
        videos = df["source"].unique()
        for v in videos:
            st.markdown(f"• {v}")

            