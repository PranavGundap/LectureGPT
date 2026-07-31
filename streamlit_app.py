import streamlit as st
import os
import json
import shutil
from auth_ui import show_auth
import uuid
from datetime import datetime
AUDIO_FOLDER = "audio"

from main_pipeline import (
    ask_question,
    convert_video_to_mp3,
    generate_quiz,
    download_youtube,
    transcribe_all,
    update_embeddings,
    delete_lecture
)
import pyperclip
import time
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from quiz_ui import show_quiz

if "selected_lecture" not in st.session_state:
    st.session_state.selected_lecture = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "question" not in st.session_state:
    st.session_state.question = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "source_video" not in st.session_state:
    st.session_state.source_video = "Waiting..."

if "timestamp" not in st.session_state:
    st.session_state.timestamp = "--"

if "context" not in st.session_state:
    st.session_state.context = ""

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

# ================= QUIZ STATES =================

if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

LIBRARY_FILE = "lecture_library.json"


def load_library():
    if not os.path.exists(LIBRARY_FILE):
        return []

    with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_library():
    try:
        with open(LIBRARY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

st.set_page_config(
    page_title="LectureGPT",
    page_icon="🧠",
    layout="wide"
)

# Authentication
if not show_auth():
    st.stop()

def save_to_library(title, source, source_type, youtube_url=None):
    print("Lecture Saved:", title)

    library = load_library()

    # Duplicate check
    for item in library:
        if item.get("source") == source:
            return

    lecture = {
        "id": str(uuid.uuid4()),
        "title": title,
        "source": source,
        "type": source_type,
        "youtube_url": youtube_url,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    library.append(lecture)

    with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=4, ensure_ascii=False)

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

import re

def clean_title(title):

    title = os.path.splitext(title)[0]

    patterns = [
        r"vidssave\.com",
        r"\b144P\b",
        r"\b240P\b",
        r"\b360P\b",
        r"\b480P\b",
        r"\b720P\b",
        r"\b1080P\b",
        r"\b2160P\b"
    ]

    for pattern in patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)

    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title)

    return title.strip()

def create_pdf(
    question,
    definition,
    explanation,
    advantages,
    limitations,
    applications,
    interview,
    source,
    timestamp
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("<b><font size='24'>🎓 LectureGPT AI - Smart Learning Notes</font></b>", styles["Title"])
    )

    story.append(Paragraph("<br/>", styles["BodyText"]))

    sections = [
        ("Question", question),
        ("Definition", definition),
        ("Explanation", explanation),
        ("Advantages", advantages),
        ("Limitations", limitations),
        ("Applications", applications),
        ("Interview Questions", interview),
        ("Source Video", source),
        ("Timestamp", timestamp),
    ]

    for title, content in sections:

        story.append(
            Paragraph(f"<b>{title}</b>", styles["Heading2"])
        )

        story.append(
            Paragraph(
                str(content).replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )

        story.append(
            Paragraph("<br/>", styles["BodyText"])
        )

    doc.build(story)

    buffer.seek(0)

    return buffer

def card(title, content):
    import re

    clean_content = re.sub(r"<[^>]+>", "", content)
    clean_content = (
        clean_content
        .replace("</div>", "")
        .replace("<div>", "")
        .replace("\r", "")
        .replace("&lt;/div&gt;", "")
        .replace("&lt;div&gt;", "")
        .strip()
    )

    lines = []

    for line in clean_content.split("\n"):

        if line.strip().startswith("-"):

            lines.append(
                f"➡️ {line.strip()[1:].strip()}"
            )

        elif line.strip().startswith(tuple(str(i) for i in range(1,10))):

            lines.append(
                f"🔹 {line}"
            )

        else:

            lines.append(line)

    clean_content = "<br>".join(lines)

    st.markdown(
        f"""
       <div class="card">
            <h4>{title}</h4>
            <div class="card-content">
                {clean_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def info_card(icon, title, value):
    st.markdown(
        f"""
        <div class="info-card">
            <h4>{icon} {title}</h4>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def clickable_info_card(icon, title, text, url=None):

    note = ""

    if title == "Timestamp":
        note = """
        <p style="
            margin-top:8px;
            font-size:12px;
            color:#9CA3AF;
        ">
        (Approx. • Try ±30 sec)
        </p>
        """

    if url:
        value = f'<a href="{url}" target="_blank" style="color:#60A5FA;text-decoration:none;">🔗 {text}</a>'
    else:
        value = text

    st.markdown(
        f"""
        <div class="info-card">
            <h4>{icon} {title}</h4>
            <p>{value}</p>
            {note}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🎓 LectureGPT</div>
        <div class="hero-subtitle">
            AI-Powered Lecture Understanding Assistant
        </div>
        <div class="hero-features">
        🔍 Search • 📝 Summarize • 🧠 Quiz • 💬 Chat
        </div>
        <div class="hero-author">
            Made with ❤️ by Pranav Gundap
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state.selected_lecture is not None:

    library = get_library()

    current = next(
        (
            x
            for x in library
            if x["source"] == st.session_state.selected_lecture
        ),
        None
    )

    if current:
        st.success(f"📚 Current Lecture : {clean_title(current['title'])}")
with st.container(border=True):
    st.markdown("### 📚 Import Lecture")

    st.markdown("#### 📺 YouTube Lecture")

    youtube_url = st.text_input(
        "",
        placeholder="Paste YouTube URL...",
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#8B93A7;
            margin:8px 0 12px 0;
            font-size:14px;
            font-weight:600;
        ">
            ─────── OR ───────
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("#### 📁 Local Video")

    uploaded_video = st.file_uploader(
        "",
        type=["mp4","avi","mkv","mov"],
        label_visibility="collapsed"
    )

    process_video = st.button(
        "🚀 Process Lecture",
        use_container_width=True
    )


st.sidebar.title("🎓 LectureGPT")
# ==========================
# User Profile
# ==========================

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Profile")

email = st.session_state.get("user", "Unknown User")

# Email ke first part se simple name banana
name = email.split("@")[0].replace(".", " ").replace("_", " ").title()

st.sidebar.write(f"**Name:** {name}")
st.sidebar.write(f"**Email:** {email}")

st.sidebar.markdown("---")
if "selected_lecture" not in st.session_state:
    st.session_state.selected_lecture = None

if st.sidebar.button("➕ New Chat"):

    st.session_state.question = ""

    st.session_state.last_result = None

    st.session_state.context = ""

    st.session_state.quiz = None

    st.session_state.summary = ""

    st.rerun()

st.sidebar.markdown("---")

st.sidebar.subheader("📚 Lecture Library")

library = get_library()

st.sidebar.caption(f"{len(library)} Lecture(s)")

library = get_library()

if len(library) == 0:
    st.sidebar.info("No lectures processed yet.")
else:
    for lecture in reversed(library):

        icon = "📺" if lecture["type"] == "YouTube" else "📹"

        if st.session_state.selected_lecture == lecture["source"]:
            label = f"🟢 {icon} {clean_title(lecture['title'])}"
        else:
            label = f"{icon} {clean_title(lecture['title'])}"

        col1, col2 = st.sidebar.columns([5,1])

        with col1:

            st.markdown('<div class="library-item">', unsafe_allow_html=True)

            clicked = st.button(
                label,
                key=f"lecture_{lecture['id']}",
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

            if clicked:

                st.session_state.selected_lecture = lecture["source"]

                st.rerun()

        with col2:

            st.markdown('<div class="delete-btn">', unsafe_allow_html=True)

            delete_clicked = st.button(
                "🗑",
                key=f"delete_{lecture['id']}",
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

            if delete_clicked:

                delete_lecture(lecture["source"])

                library.remove(lecture)

                with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
                    json.dump(
                        library,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

                if st.session_state.selected_lecture == lecture["source"]:
                    st.session_state.selected_lecture = None

                st.success("Lecture deleted successfully!")

                st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Chat History")


if len(st.session_state.chat_history) == 0:
    st.sidebar.info("No chats yet.\n\nAsk your first question!")
else:
   for lecture, chats in reversed(list(st.session_state.chat_history.items())):

        st.sidebar.markdown(f"### 📚 {clean_title(lecture)}")

        for chat in reversed(chats):
            st.sidebar.markdown(
                f"""
    <div style="
    background:#1B1F2B;
    padding:10px;
    margin-bottom:8px;
    border-radius:10px;
    border:1px solid #2d3142;
    font-size:14px;
    ">
    💬 {chat}
    </div>
    """,
                unsafe_allow_html=True
            )

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear History"):
    st.session_state.chat_history = {}
    st.rerun()

# ==========================
# Logout
# ==========================

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout", use_container_width=True):

    st.session_state.logged_in = False
    st.session_state.user = ""

    st.session_state.selected_lecture = None
    st.session_state.question = ""
    st.session_state.last_result = None
    st.session_state.context = ""
    st.session_state.quiz = None
    st.session_state.summary = ""
    st.session_state.source_video = "Waiting..."
    st.session_state.timestamp = "--"
    st.session_state.chat_history = {}

    st.rerun()

st.sidebar.markdown("---")

st.sidebar.caption("Powered By")

st.sidebar.write("Whisper")
st.sidebar.write("BGE-M3")
st.sidebar.write("FAISS")
st.sidebar.write("BM25")
st.sidebar.write("Gemma3")

col1, col2 = st.columns([6, 1])

with col1:
    question = st.text_input(
        "",
        key="question",
        placeholder="🔍 Ask anything about your lecture...",
        label_visibility="collapsed"
    )

with col2:
    ask = st.button(
        "🚀 Ask",
        use_container_width=True,
        key="ask_button"
    )

if process_video:

    # --------------------------
    # Validation
    # --------------------------

    if youtube_url.strip() != "" and uploaded_video is not None:
        st.warning("Please use either a YouTube URL OR upload a local video.")
        st.stop()

    if youtube_url.strip() == "" and uploaded_video is None:
        st.warning("Please enter a YouTube URL or upload a video.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # --------------------------
    # YouTube Processing
    # --------------------------

    if youtube_url.strip() != "":

        status.info("📥 Downloading YouTube video...")
        result = download_youtube(youtube_url)

        video_title = youtube_url
        audio_file = youtube_url

        if isinstance(result, dict):
            video_title = result.get("title", video_title)
            audio_file = result.get("audio_file", result.get("source", audio_file))

        elif isinstance(result, (tuple, list)):
            if len(result) >= 1 and result[0]:
                video_title = result[0]

            if len(result) >= 2 and result[1]:
                audio_file = result[1]
                audio_file = audio_file.replace(".mp4", ".mp3")

        elif isinstance(result, str) and result:
            audio_file = result

        progress.progress(30)

    # --------------------------
    # Local Video Processing
    # --------------------------

    else:

        os.makedirs(AUDIO_FOLDER, exist_ok=True)

        video_path = os.path.join(
            AUDIO_FOLDER,
            uploaded_video.name
        )

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        status.info("🎵 Extracting Audio...")
        audio_path = convert_video_to_mp3(video_path)

        progress.progress(30)

        video_title = uploaded_video.name
        audio_file = os.path.basename(audio_path)

    # --------------------------
    # Common Processing
    # --------------------------

    status.info("🎤 Transcribing Lecture...")
    transcribe_all()
    progress.progress(70)

    status.info("🧠 Creating Embeddings...")
    update_embeddings()
    progress.progress(100)

    # --------------------------
    # Save Lecture
    # --------------------------

    if youtube_url.strip() != "":

        save_to_library(
            title=video_title,
            source=audio_file,
            source_type="YouTube",
            youtube_url=youtube_url
        )

        processed_source = audio_file

    else:

        save_to_library(
            title=uploaded_video.name,
            source=audio_file,
            source_type="Local"
        )

        processed_source = audio_file

    st.toast("✅ Lecture processed successfully!")
    time.sleep(2)
    st.session_state.selected_lecture = processed_source
    st.rerun()

# Default values


source_video = st.session_state.source_video
timestamp = st.session_state.timestamp
response_time = 0.00
youtube_url = None
timestamp_url = None


definition = "Waiting for answer..."
explanation = "Waiting for answer..."
advantages = "Waiting for answer..."
disadvantages = "Waiting for answer..."
applications = "Waiting for answer..."
interview = "Waiting for answer..."
summary = "Waiting for summary..."
full_answer = ""


if st.session_state.last_result:

    full_answer = st.session_state.last_result["answer"]

    definition = st.session_state.last_result["definition"]

    explanation = st.session_state.last_result["explanation"]

    advantages = st.session_state.last_result["advantages"]

    disadvantages = st.session_state.last_result["disadvantages"]

    applications = st.session_state.last_result["applications"]

    interview = st.session_state.last_result["interview"]
    
    summary = st.session_state.last_result["summary"]

if ask:

    start = time.time()

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    lecture = st.session_state.selected_lecture or "General"

    if lecture not in st.session_state.chat_history:
        st.session_state.chat_history[lecture] = []

    if question not in st.session_state.chat_history[lecture]:
        st.session_state.chat_history[lecture].append(question)

    status = st.empty()

    status.info("🧠 Reading Transcript...")

    time.sleep(0.5)

    status.info("🔍 Running Hybrid Search...")

    time.sleep(0.5)

    status.info("⚡ Retrieving Relevant Chunks...")

    time.sleep(0.5)

    status.info("🤖 Generating Answer...")

    result = ask_question(
    question,
    st.session_state.selected_lecture
    )

    st.session_state.last_result = result

    st.session_state.context = result["context_text"]

    st.session_state.summary = result["summary"]

    full_answer = result["answer"]
    end = time.time()

    response_time = round(end - start, 2)

    status.success("✅ Done")

    time.sleep(0.8)

    status.empty()

    st.session_state.source_video = result["source"]
    st.session_state.timestamp = result["timestamp"]

    source_video = st.session_state.source_video
    timestamp = st.session_state.timestamp

    youtube_url = None

    library = get_library()

    for lecture in library:
        if lecture["source"] == source_video:
            youtube_url = lecture.get("youtube_url")
            break

    timestamp_url = None

    if youtube_url and timestamp != "--":
        try:
            start_time = timestamp.split("-")[0].strip()

            mins, secs = start_time.split(":")
            total_seconds = int(mins) * 60 + int(secs)

            timestamp_url = f"{youtube_url}&t={total_seconds}s"

        except:
            timestamp_url = youtube_url


col1, col2 = st.columns(2)

with col1:
    if youtube_url:

        clickable_info_card(
            "📺",
            "Source Video",
            clean_title(source_video),
            youtube_url
        )

    else:

        info_card(
            "📹",
            "Uploaded Video",
            clean_title(source_video)
        )

with col2:
    clickable_info_card(
        "⏰",
        "Timestamp",
        timestamp,
        timestamp_url
    )


st.markdown(
    f"""
<div style="
display:flex;
gap:15px;
margin-top:10px;
margin-bottom:20px;
">

<div style="
background:#1B1F2B;
padding:10px 18px;
border-radius:12px;
border:1px solid #2d3142;
">
🟢 <b>Gemma3 Online</b>
</div>

<div style="
background:#1B1F2B;
padding:10px 18px;
border-radius:12px;
border:1px solid #2d3142;
">
⚡ <b>FAISS + BM25</b>
</div>

<div style="
background:#1B1F2B;
padding:10px 18px;
border-radius:12px;
border:1px solid #2d3142;
">
📄 <b>3 Chunks</b>
</div>

<div style="
background:#1B1F2B;
padding:10px 18px;
border-radius:12px;
border:1px solid #2d3142;
">
⏱ <b>{response_time} sec</b>
</div>

<div style="
background:#1B1F2B;
padding:10px 18px;
border-radius:12px;
border:1px solid #2d3142;
">
🧠 <b>BGE-M3</b>
</div>

</div>
""",
    unsafe_allow_html=True
)


    # ==========================================
# DEFINITION + EXPLANATION + ADVANTAGES
# ==========================================

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    card(
        "📘 Definition",
        st.session_state.last_result["definition"]
        if st.session_state.last_result
        else "Waiting for answer..."
    )

with col2:
    card(
        "📖 Explanation",
        st.session_state.last_result["explanation"]
        if st.session_state.last_result
        else "Waiting for answer..."
    )

with col3:
    card(
        "✅ Advantages",
        st.session_state.last_result["advantages"]
        if st.session_state.last_result
        else "Waiting for answer..."
    )



# New Row
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    card(
        "❌ Limitations",
        st.session_state.last_result["disadvantages"]
        if st.session_state.last_result
        else "Waiting for answer..."
    )

with col2:
    card(
        "💼 Applications",
        st.session_state.last_result["applications"]
        if st.session_state.last_result
        else "Waiting for answer..."
    )

with col3:
    card(
        "🎯 Interview Qs",
        st.session_state.last_result["interview"]
        if st.session_state.last_result
        else "Waiting for answer..."
    )

# ==========================
# Lecture Summary
# ==========================

st.markdown("---")

card(
    "📝 Lecture Summary",
    st.session_state.last_result["summary"]
    if st.session_state.last_result
    else "Waiting for summary..."
)
if st.button("📝 Generate Quiz", use_container_width=True):

    if st.session_state.context == "":

        st.warning("Please ask a question first.")

    else:

        with st.spinner("Generating Quiz..."):

            st.session_state.quiz = generate_quiz(
                st.session_state.context
            )

            # Reset Quiz State
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_submitted = False
            st.session_state.selected_option = None

        st.success("Quiz Generated!")

if st.session_state.quiz is not None:
    show_quiz()
    
col1, col2, col3, col4 = st.columns([1,1,1,3])

with col1:

    if st.button("📋 Copy", use_container_width=True):

        pyperclip.copy(
            st.session_state.last_result["answer"]
            if st.session_state.last_result
            else ""
        )

        st.success("Answer copied!")

with col2:

    st.download_button(
        "📄 TXT",
        st.session_state.last_result["answer"]
        if st.session_state.last_result
        else "",
        file_name="LectureGPT_Notes.txt",
        mime="text/plain",
        use_container_width=True,
    )

with col3:

    pdf = create_pdf(
    question,
    st.session_state.last_result["definition"]
    if st.session_state.last_result else "",

    st.session_state.last_result["explanation"]
    if st.session_state.last_result else "",

    st.session_state.last_result["advantages"]
    if st.session_state.last_result else "",

    st.session_state.last_result["disadvantages"]
    if st.session_state.last_result else "",

    st.session_state.last_result["applications"]
    if st.session_state.last_result else "",

    st.session_state.last_result["interview"]
    if st.session_state.last_result else "",

    clean_title(source_video),
    timestamp
)

    st.download_button(
        "📄 PDF",
        pdf,
        file_name="LectureGPT_Notes.pdf",
        mime="application/pdf",
        use_container_width=True
    )
