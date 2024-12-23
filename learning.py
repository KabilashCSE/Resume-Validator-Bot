import streamlit as st
import subprocess
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
from PyPDF2 import PdfReader
import json
import re

def preprocess_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s.,;:-]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def query_ollama(prompt, model="qwen:1.8b"):
    """Query the local Ollama model using the `ollama run` command."""
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'  # Explicitly set UTF-8 encoding
        )
        output, error = process.communicate(input=prompt)

        if process.returncode != 0:
            st.error(f"Error querying Ollama: {error}")
            return None
        
        return output.strip()
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return None

def parse_model_output(output):
    """Parse and validate model output"""
    if not output:
        return {
            "Skills": [],
            "Experience": "",
            "Education": ""
        }
    
    try:
        # Try parsing as JSON
        parsed = json.loads(output)
        # Ensure all required fields exist
        required_fields = ["Skills", "Experience", "Education"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = [] if field == "Skills" else ""
        
        # Ensure Skills is always a list
        if isinstance(parsed["Skills"], str):
            parsed["Skills"] = [skill.strip() for skill in parsed["Skills"].split(",")]
        
        return parsed
    except json.JSONDecodeError:
        # Fallback to regex parsing
        fields = {
            "Skills": [],
            "Experience": "",
            "Education": ""
        }
        
        patterns = {
            "Skills": r"Skills:(.+?)(?=Experience:|Education:|$)",
            "Experience": r"Experience:(.+?)(?=Skills:|Education:|$)",
            "Education": r"Education:(.+?)(?=Skills:|Experience:|$)"
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if field == "Skills":
                    fields[field] = [s.strip() for s in content.split(",")]
                else:
                    fields[field] = content
                    
        return fields

def extract_fields_from_jd(jd_text):
    """Extract key fields from a job description"""
    cleaned_text = preprocess_text(jd_text)
    prompt = """Analyze this job description and extract:
    
    1. Skills: List all technical and soft skills required
    2. Experience: Years of experience and type of experience needed
    3. Education: Required education level and qualifications
    
    Job Description:
    {text}
    
    Respond in this JSON format:
    {{
        "Skills": ["skill1", "skill2", ...],
        "Experience": "detailed experience requirements",
        "Education": "education requirements"
    }}"""
    
    result = query_ollama(prompt.format(text=cleaned_text))
    return parse_model_output(result)

def extract_fields_from_resume(resume_text):
    """Extract key fields from a resume"""
    cleaned_text = preprocess_text(resume_text)
    prompt = """Analyze this resume and extract:
    
    1. Skills: List all technical and soft skills mentioned
    2. Experience: Total years and relevant work experience
    3. Education: All education qualifications
    
    Resume Text:
    {text}
    
    Respond in this JSON format:
    {{
        "Skills": ["skill1", "skill2", ...],
        "Experience": "detailed experience",
        "Education": "education details"
    }}"""
    
    result = query_ollama(prompt.format(text=cleaned_text))
    return parse_model_output(result)

def compute_match_score(jd_fields, resume_fields):
    """Compute matching score between JD and Resume fields"""
    def jaccard_similarity(text1, text2):
        if not text1 or not text2:
            return 0.0
        # Create vectors using CountVectorizer
        vec = CountVectorizer(lowercase=True, token_pattern=r'[a-zA-Z0-9]+')
        try:
            vectors = vec.fit_transform([text1, text2]).toarray()
            return jaccard_score(vectors[0], vectors[1], average='binary')
        except:
            return 0.0

    scores = {}
    for field in jd_fields:
        if field in resume_fields:
            scores[field] = jaccard_similarity(
                str(jd_fields[field]), 
                str(resume_fields[field])
            )

    overall_score = sum(scores.values()) / len(scores) if scores else 0
    return {"field_scores": scores, "overall_score": overall_score}

def main():
    st.set_page_config(page_title="Resume and JD Matcher", layout="wide")
    st.title("Resume and JD Matcher")

    st.sidebar.header("Instructions")
    st.sidebar.write("""
        - Upload a resume file (PDF or plain text)
        - Enter the job description
        - Click **Match** to see the analysis
    """)

    uploaded_resume = st.file_uploader("Upload Resume (PDF/Text)", type=["pdf", "txt"])
    jd_text = st.text_area("Enter Job Description", height=200)

    if st.button("Match"):
        if not uploaded_resume or not jd_text.strip():
            st.error("Please upload a resume and enter the job description.")
            return

        try:
            # Read the Resume
            if uploaded_resume.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_resume)
                resume_text = " ".join(
                    page.extract_text() for page in pdf_reader.pages
                )
            else:
                resume_text = uploaded_resume.getvalue().decode('utf-8', errors='ignore')

            # Extract and Match Fields
            with st.spinner("Processing..."):
                jd_fields = extract_fields_from_jd(jd_text)
                resume_fields = extract_fields_from_resume(resume_text)
                match_result = compute_match_score(jd_fields, resume_fields)

            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.write("**JD Fields**")
                st.json(jd_fields)
            with col2:
                st.write("**Resume Fields**")
                st.json(resume_fields)

            st.subheader("Match Analysis")
            df_scores = pd.DataFrame(
                match_result["field_scores"].items(), 
                columns=["Field", "Score"]
            )
            st.dataframe(df_scores)
            st.metric(
                "Overall Match Score", 
                f"{match_result['overall_score']*100:.1f}%"
            )

        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()