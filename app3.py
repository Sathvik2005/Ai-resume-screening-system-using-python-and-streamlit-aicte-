import joblib
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load trained model and vectorizer
clf = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectors = vectorizer.transform(documents)
    
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity([job_vector], resume_vectors).flatten()
    
    return similarities

# Input job description
job_description = input("Enter the job description: ")

# Upload and extract resumes
resume_files = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]  # Replace with actual file paths
resumes = [extract_text_from_pdf(file) for file in resume_files]

# Rank resumes
scores = rank_resumes(job_description, resumes)

# Display ranked results
results = pd.DataFrame({"Resume": resume_files, "Score": scores})
results = results.sort_values(by="Score", ascending=False)

print("\nRanked Resumes:")
print(results)