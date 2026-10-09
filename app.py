
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS = [
    "python", "java", "c++", "javascript", "sql",
    "html", "css", "react", "machine learning",
    "deep learning", "artificial intelligence",
    "data analysis", "data science", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "natural language processing", "excel", "power bi",
    "tableau", "communication", "problem solving",
    "teamwork", "leadership", "git", "github",
    "docker", "flask", "django", "streamlit",
    "statistics", "data visualization", "computer vision",
    "cloud computing", "data structures", "algorithms"
]


def extract_text(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception:
        return ""
    return text.lower()


def extract_skills(text):
    found = []
    for skill in SKILLS:
        pattern = (
            r"(?<!\w)"
            + re.escape(skill).replace(r"\ ", r"\s+")
            + r"(?!\w)"
        )
        if re.search(pattern, text.lower()):
            found.append(skill)
    return sorted(set(found))


def text_similarity(resume, job):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform([resume, job])
        score = cosine_similarity(
            matrix[0:1], matrix[1:2]
        )[0][0]
        return float(score * 100)
    except ValueError:
        return 0.0


def screen_resume(filename, resume_text, job_description):
    job_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)

    matched = sorted(set(job_skills) & set(resume_skills))
    missing = sorted(set(job_skills) - set(resume_skills))

    similarity = text_similarity(resume_text, job_description)

    if job_skills:
        coverage = len(matched) / len(job_skills) * 100
        final_score = 0.60 * similarity + 0.40 * coverage
    else:
        coverage = None
        final_score = similarity

    return {
        "Resume": filename,
        "Final Match (%)": round(final_score, 2),
        "Text Similarity (%)": round(similarity, 2),
        "Skill Coverage (%)": (
            round(coverage, 2) if coverage is not None else "N/A"
        ),
        "Matched Skills": ", ".join(matched) or "None detected",
        "Skills Not Found": ", ".join(missing) or "None"
    }


# ---------------- Dashboard ----------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Based Resume Screening System")
st.write("Compare PDF resumes with a job description.")

job_description = st.text_area(
    "Paste the job description",
    height=180,
    placeholder="Enter the required skills and job responsibilities..."
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Screen Resumes", type="primary"):
    if not job_description.strip():
        st.warning("Please enter a job description.")

    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")

    else:
        results = []
        progress = st.progress(0)

        for index, pdf in enumerate(uploaded_files):
            resume_text = extract_text(pdf)

            if resume_text.strip():
                results.append(
                    screen_resume(
                        pdf.name, resume_text, job_description
                    )
                )
            progress.progress((index + 1) / len(uploaded_files))

        if results:
            results.sort(
                key=lambda item: item["Final Match (%)"],
                reverse=True
            )

            df = pd.DataFrame(results)

            st.subheader("Screening Results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Resumes analyzed", len(results))
            col2.metric(
                "Highest text similarity",
                f"{max(r['Text Similarity (%)'] for r in results):.2f}%"
            )
            col3.metric(
                "Recognized job skills",
                len(extract_skills(job_description))
            )

            st.download_button(
                "Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="screening_results.csv",
                mime="text/csv"
            )

            st.info(
                "Scores indicate text and skill overlap only. "
                "They do not establish a candidate's qualifications "
                "or determine whether to hire them."
            )
        else:
            st.error(
                "No readable text was found in the uploaded PDFs. "
                "Scanned resumes may require OCR."
            )