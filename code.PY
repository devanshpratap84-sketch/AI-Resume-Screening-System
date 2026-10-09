
import re
import csv
from pathlib import Path

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------
# CONFIGURATION
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESUME_FOLDER = BASE_DIR / "resumes"
OUTPUT_FILE = BASE_DIR / "screening_results.csv"

# Skills that the system can recognize
SKILLS = [
    "python", "java", "c++", "c", "javascript",
    "sql", "html", "css", "react", "node.js",
    "machine learning", "deep learning",
    "artificial intelligence", "data analysis",
    "data science", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "nlp", "natural language processing",
    "excel", "power bi", "tableau",
    "communication", "problem solving",
    "teamwork", "leadership",
    "git", "github", "docker",
    "flask", "django", "streamlit",
    "statistics", "data visualization",
    "computer vision", "cloud computing"
]


# ------------------------------------------
# EXTRACT TEXT FROM PDF
# ------------------------------------------

def extract_resume_text(pdf_path):
    text = ""

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + " "

    except Exception as error:
        print(f"Could not read {pdf_path.name}: {error}")

    return text.lower()


# ------------------------------------------
# EXTRACT SKILLS FROM TEXT
# ------------------------------------------

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        # Match whole words, allowing flexible spaces
        # in multi-word skills.
        pattern = (
            r"(?<!\w)"
            + re.escape(skill).replace(r"\ ", r"\s+")
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(set(found_skills))


# ------------------------------------------
# CALCULATE TEXT SIMILARITY
# ------------------------------------------

def calculate_similarity(resume_text, job_description):
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    try:
        documents = [resume_text, job_description]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(score * 100)

    except ValueError:
        # For example, if there are no usable words.
        return 0.0


# ------------------------------------------
# SCREEN RESUMES
# ------------------------------------------

def screen_resumes(job_description):
    if not RESUME_FOLDER.exists():
        print("\nError: Create a 'resumes' folder first.")
        return

    pdf_files = sorted(RESUME_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("\nNo PDF resumes found in the resumes folder.")
        return

    job_skills = extract_skills(job_description)
    results = []

    print("\nAnalyzing resumes. Please wait...\n")

    for pdf_path in pdf_files:
        resume_text = extract_resume_text(pdf_path)

        if not resume_text.strip():
            print(f"Skipping {pdf_path.name}: no readable text.")
            continue

        resume_skills = extract_skills(resume_text)

        matched_skills = sorted(
            set(job_skills) & set(resume_skills)
        )

        missing_skills = sorted(
            set(job_skills) - set(resume_skills)
        )

        text_score = calculate_similarity(
            resume_text,
            job_description
        )

        # Skill coverage is only calculated when the
        # job description contains recognized skills.
        if job_skills:
            skill_score = (
                len(matched_skills) / len(job_skills)
            ) * 100

            final_score = (
                0.60 * text_score
                + 0.40 * skill_score
            )
        else:
            skill_score = None
            final_score = text_score

        results.append({
            "Resume": pdf_path.name,
            "Final Match (%)": round(final_score, 2),
            "Text Similarity (%)": round(text_score, 2),
            "Skill Coverage (%)": (
                round(skill_score, 2)
                if skill_score is not None else "N/A"
            ),
            "Matched Skills": ", ".join(matched_skills),
            "Skills Not Found": ", ".join(missing_skills)
        })

    if not results:
        print("No readable PDF resumes could be analyzed.")
        return

    # Sort by score, highest first.
    results.sort(
        key=lambda item: item["Final Match (%)"],
        reverse=True
    )

    print("=" * 70)
    print("       AI-BASED RESUME SCREENING RESULTS")
    print("=" * 70)

    if job_skills:
        print("Skills identified in job description:")
        print(", ".join(job_skills))
    else:
        print("No skills from the built-in skill list were found.")

    print("\n")

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['Resume']}")
        print(f"   Final Match: {result['Final Match (%)']}%")
        print(
            f"   Text Similarity: "
            f"{result['Text Similarity (%)']}%"
        )
        print(
            f"   Skill Coverage: "
            f"{result['Skill Coverage (%)']}"
            + (
                "%"
                if result["Skill Coverage (%)"] != "N/A"
                else ""
            )
        )
        print(
            f"   Matched Skills: "
            f"{result['Matched Skills'] or 'None detected'}"
        )
        print(
            f"   Skills Not Found: "
            f"{result['Skills Not Found'] or 'None'}"
        )
        print("-" * 70)

    # Save results as a CSV file.
    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_FILE}")


# ------------------------------------------
# MAIN PROGRAM
# ------------------------------------------

if __name__ == "__main__":
    print("\nAI-Based Resume Screening System")
    print("--------------------------------")

    job_description = input(
        "\nEnter the job description:\n"
    ).strip()

    if not job_description:
        print("Error: Job description cannot be empty.")
    else:
        screen_resumes(job_description)