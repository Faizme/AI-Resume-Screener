import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set up local NLTK data path
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure necessary NLTK resources are available
nltk_dependencies = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]
for path, dep in nltk_dependencies:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(dep, download_dir=NLTK_DATA_PATH)

stop_words = set(stopwords.words("english"))

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Screening", layout="wide")

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in pdf.pages])  # Handle None cases
        return text.strip() if text.strip() else "No readable text found."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and word.isalnum()]
    return " ".join(words)

# Function to rank resumes
def rank_resumes(job_description, resumes):
    try:
        processed_resumes = [preprocess_text(resume) for resume in resumes]
        processed_job_desc = preprocess_text(job_description)
        documents = [processed_job_desc] + processed_resumes
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
        return cosine_similarities
    except Exception as e:
        st.error(f"Error in ranking resumes: {str(e)}")
        return []

# Streamlit UI
st.title("📄 AI Resume Screening & Ranking System")
st.sidebar.header("⚡ Features")

# About Section
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    "**Developed by Mohammed Faiz**\n\n"
    "This application helps recruiters efficiently screen resumes based on job descriptions using AI. "
    "By utilizing Natural Language Processing (NLP) techniques, the system ranks resumes based on their relevance to the given job description. "
    "It also provides keyword analysis using word clouds to highlight essential skills and terms.\n\n"
    "👨‍💻 **About the Developer:**\n"
    "Mohammed Faiz is an aspiring software developer with expertise in Python, Java, C, HTML, CSS, JavaScript, and SQL. "
    "He has experience in data analysis, web development, and AI-driven applications. "
    "Connect with Faiz on [GitHub](https://github.com/Faizme) or [LinkedIn](https://www.linkedin.com/in/mohammed-faiz-me)."
)

# Job description input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload resumes (PDF only)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")
    progress = st.progress(0)

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text.startswith("Error extracting text") or text == "No readable text found.":
            st.error(f"❌ Failed to extract text from {file.name}")
        else:
            resumes.append(text)

    if resumes:
        progress.progress(50)
        scores = rank_resumes(job_description, resumes)
        progress.progress(100)

        if scores is not None and scores.size > 0:
            results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
            results = results.sort_values(by="Score", ascending=False)

            # Add a ranking column starting from 1
            results["Rank"] = range(1, len(results) + 1)

            # Reorder columns to show Rank first
            results = results[["Rank", "Resume", "Score"]]

            # Display the results
            st.dataframe(results)

            # Highlight the top candidate
            if not results.empty:
                top_candidate = results.iloc[0]
                st.success(f"🏆 Top Candidate: {top_candidate['Resume']} with Score: {top_candidate['Score']:.2f}")

            # Download results as CSV
            csv = results.to_csv(index=False)
            st.download_button("📥 Download Results as CSV", csv, "ranking_results.csv", "text/csv")
            # Word Cloud Visualization
            st.header("📢 Resume Keyword Analysis")
            all_text = " ".join(preprocess_text(resume) for resume in resumes)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)