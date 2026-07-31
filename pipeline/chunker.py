from typing import List

def merge_segments(
    segments: List[dict],
    max_words: int = 60,
    overlap: int = 0
):
    """
    Merge Whisper segments into semantic chunks based on word count.
    """

    chunks = []

    current_words = []
    current_start = None
    current_end = None

    for segment in segments:

        text = segment["text"].strip()

        if not text:
            continue

        words = text.split()

        if current_start is None:
            current_start = segment["start"]

        current_words.extend(words)
        current_end = segment["end"]

        if len(current_words) >= max_words:

            chunk_words = current_words[:max_words]

            chunks.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(chunk_words)
            })

            # overlap
            current_words = current_words[max_words - overlap:]

            current_start = segment["start"]

    if current_words:

        chunks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_words)
        })

    return chunks