
import streamlit as st

def show_quiz():
    quiz = st.session_state.get("quiz")
    if not quiz:
        return

    idx = st.session_state.quiz_index
    total = len(quiz)

    if idx >= total:

        score = st.session_state.quiz_score
        accuracy = (score / total) * 100

        st.balloons()

        st.markdown("## 🏆 Quiz Completed!")

        st.metric("🎯 Score", f"{score} / {total}")
        st.metric("📊 Accuracy", f"{accuracy:.0f}%")

        if accuracy >= 90:
            st.success("🌟 Outstanding Performance!")
        elif accuracy >= 75:
            st.success("🎉 Excellent Work!")
        elif accuracy >= 50:
            st.info("👍 Good Job! Keep Practicing.")
        else:
            st.warning("📚 Practice More. You'll Improve!")

        st.divider()

        if st.button("🔄 Retry Quiz"):

            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_submitted = False
            st.session_state.selected_option = None

            st.rerun()

        return

    q = quiz[idx]
    # Progress Bar
    progress = (idx + 1) / total

    st.progress(progress)

    st.caption(f"Progress : {idx + 1} of {total} Questions")
    st.markdown(f"## 📝 Question {idx+1}/{total}")
    st.markdown(f"### {q['question']}")

    choice = st.radio(
        "Choose your answer",
        q["options"],
        key=f"quiz_radio_{idx}"
    )

    if not st.session_state.quiz_submitted:
        if st.button("✅ Submit Answer", use_container_width=True):
            st.session_state.selected_option = choice
            if choice.split(".")[0].strip() == q["answer"]:
                st.session_state.quiz_score += 1
            st.session_state.quiz_submitted = True
            st.rerun()
    else:
        selected = (
            st.session_state.selected_option.split(".")[0].strip()
            if st.session_state.selected_option
            else ""
        )
        if selected == q["answer"]:
            st.success("✅ Correct Answer!")
        else:
            st.error("❌ Wrong Answer!")

        correct_option = next(
            opt for opt in q["options"]
            if opt.startswith(q["answer"])
        )

        st.success(f"✅ Correct Answer: {correct_option}")
        st.markdown("### 📖 Explanation")
        st.write(q["explanation"])

        if st.button("➡️ Next Question", use_container_width=True):
            st.session_state.quiz_index += 1
            st.session_state.quiz_submitted = False
            st.session_state.selected_option = None
            st.rerun()
