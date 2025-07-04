import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from datetime import datetime
import os
import io

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["resumeScreening"]
collection = db["resumes"]

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Helper functions
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_description_vector], resume_vectors).flatten()

# 🌐 Sidebar navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Upload & Rank Resumes", "📄 View Resumes", "🗑️ Delete One", "🧹 Delete All"])

# ------------------ Upload & Rank Section ------------------
if page == "🏠 Upload & Rank Resumes":
    st.title("🤖 AI Resume Screening & Ranking System")

    st.subheader("📋 Enter Job Description")
    job_description = st.text_area("Type the job description here")

    st.subheader("📁 Upload Resume PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        resumes_text, file_names, binary_files = [], [], []

        for file in uploaded_files:
            file_bytes = file.getvalue()
            text = extract_text_from_pdf(io.BytesIO(file_bytes))
            resumes_text.append(text)
            file_names.append(file.name)
            binary_files.append(file_bytes)

        scores = rank_resumes(job_description, resumes_text)

        results = pd.DataFrame({"Resume": file_names, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.success("✅ Resumes Ranked Successfully")
        st.dataframe(results)

        st.markdown("---")
        st.subheader("💾 Saving to Database...")

        for i in range(len(file_names)):
            existing = collection.find_one({"name": file_names[i]})
            if existing:
                st.warning(f"⚠️ `{file_names[i]}` already exists.")
                st.info(f"📝 Existing Score: {round(existing['score'], 4)}")
            else:
                record = {
                    "name": file_names[i],
                    "score": float(scores[i]),
                    "job_description": job_description,
                    "text": resumes_text[i],
                    "resume": binary_files[i],
                    "uploaded_at": datetime.now()
                }
                collection.insert_one(record)
                st.success(f"✅ Inserted: `{file_names[i]}`")

# ------------------ View Resumes ------------------
elif page == "📄 View Resumes":
    st.title("📄 Stored Resumes in MongoDB")
    if st.button("🔄 Load Resumes"):
        resumes = list(collection.find().sort("score", -1))
        if resumes:
            df = pd.DataFrame([{
                "Resume Name": doc["name"],
                "Score": round(doc["score"], 4),
                "Job Description": doc["job_description"][:50] + "...",
                "Uploaded At": doc["uploaded_at"].strftime("%Y-%m-%d %H:%M")
            } for doc in resumes])
            st.success(f"✅ {len(resumes)} resumes loaded.")
            st.dataframe(df)
        else:
            st.info("No resumes found in database.")

# ------------------ Delete One Resume ------------------
elif page == "🗑️ Delete One":
    st.title("🗑️ Delete a Specific Resume")
    names = [doc["name"] for doc in collection.find({}, {"name": 1})]

    if names:
        selected = st.selectbox("Select Resume to Delete", names)
        if st.button("❌ Delete Selected Resume"):
            result = collection.delete_one({"name": selected})
            if result.deleted_count > 0:
                st.success(f"✅ `{selected}` deleted.")
            else:
                st.error("❌ Something went wrong.")
    else:
        st.info("No resumes found in database.")

# ------------------ Delete All Resumes ------------------
elif page == "🧹 Delete All":
    st.title("🧹 Delete All Resumes")
    st.warning("⚠️ This will permanently remove all resumes from the database.")

    confirm = st.checkbox("I understand and want to delete all resumes")

    if confirm and st.button("❌ Confirm Delete All"):
        deleted = collection.delete_many({})
        st.success(f"🗑️ Deleted {deleted.deleted_count} resumes.")
