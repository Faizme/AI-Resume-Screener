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

# Set up nltk_data path
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure required resources are available
nltk_resources = ["punkt", "stopwords"]
for resource in nltk_resources:
    try:
        if resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

stop_words = set(stopwords.words("english"))

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Screening", layout="wide")

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in pdf.pages])  # Handle None cases
        return text.strip() if text.strip() else None
    except Exception:
        return None

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
        st.error(f"⚠️ Error in ranking resumes: {str(e)}")
        return []

# Streamlit UI
st.title("📄 AI Resume Screening & Ranking System")
st.sidebar.header("⚡ Features")

# About Section
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    "**Developed by Mohammed Faiz**\n\n"
    "This application helps recruiters efficiently screen resumes based on job descriptions using AI. "
    "It ranks resumes using NLP techniques and provides keyword analysis with word clouds.\n\n"
    "👨‍💻 **Connect with Faiz:**\n"
    "[GitHub](https://github.com/Faizme) | [LinkedIn](https://www.linkedin.com/in/mohammed-faiz-me)"
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
    progress = st.progress(10)

    resumes = []
    valid_files = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text:
            resumes.append(text)
            valid_files.append(file.name)
        else:
            st.warning(f"⚠️ Skipped {file.name} (No readable text found)")

    progress.progress(50)

    if resumes:
        scores = rank_resumes(job_description, resumes)
        progress.progress(90)

        if scores is not None and len(scores) > 0:
            results = pd.DataFrame({"Resume": valid_files, "Score": scores})
            results = results.sort_values(by="Score", ascending=False)
            st.dataframe(results)

            if not results.empty:
                top_candidate = results.iloc[0]
                st.success(f"🏆 Top Candidate: {top_candidate['Resume']} with Score: {top_candidate['Score']:.2f}")

            # Download results
            csv = results.to_csv(index=False)
            st.download_button("📥 Download Results as CSV", csv, "ranking_results.csv", "text/csv")

            # Word Cloud Visualization
            if any(resumes):
                st.header("📢 Resume Keyword Analysis")
                all_text = " ".join(preprocess_text(resume) for resume in resumes)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

        progress.progress(100)
    else:
        st.error("❌ No valid resumes found. Please upload readable PDF resumes.")
