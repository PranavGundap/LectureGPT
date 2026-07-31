import re

from streamlit import text

def parse_quiz(text):
    questions = []
    print("RAW QUIZ TEXT:")
    print(text)
    pattern = re.findall(
        r"Question\s*\d+:\s*(.*?)\n"
        r"A\.\s*(.*?)\n"
        r"B\.\s*(.*?)\n"
        r"C\.\s*(.*?)\n"
        r"D\.\s*(.*?)\n\n"
        r"Answer:\s*([ABCD])\n\n"
        r"Explanation:\s*(.*?)(?=\n-+\n\nQuestion|\nQuestion\s*\d+:|\Z)",
        text,
        re.DOTALL
    )

    for q in pattern:

        questions.append({
            "question": q[0].strip(),
            "options": [
                "A. " + q[1].strip(),
                "B. " + q[2].strip(),
                "C. " + q[3].strip(),
                "D. " + q[4].strip(),
            ],
            "answer": q[5].strip(),
            "explanation": q[6].strip()
        })
    print(f"Questions Parsed: {len(questions)}")
    return questions