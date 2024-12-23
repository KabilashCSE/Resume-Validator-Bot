import streamlit as st
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
import requests
import ssl
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from zipfile import ZipFile
import io

# Add MeaningCloud API key constant
MEANINGCLOUD_API_KEY = "525dd7fb60fca5ac54a49640708396eb"  # Replace with your actual API key

# Custom CSS to enhance UI
def set_custom_css():
    st.markdown("""
    <style>
        .stProgress .st-bo {
            background-color: #f0f2f6;
        }
        .stProgress .st-bp {
            background: linear-gradient(to right, #4CAF50, #8BC34A);
        }
        .skill-tag {
            display: inline-block;
            padding: 5px 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to download NLTK resources
def download_nltk_resources():
    required_resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download(resource)
    return True

# Extract skills from resume
def extract_skills(text):
    skills = []
    predefined_skills = ['python', 'java', 'c++', 'javascript', 'sql', 'aws', 'docker', 'kubernetes']
    for skill in predefined_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            skills.append(skill)
    return skills

# Extract certifications
def extract_certifications(text):
    certifications = []
    predefined_certifications = ['aws certified', 'pmp', 'cfa', 'cpa', 'scrum master']
    for cert in predefined_certifications:
        if re.search(r'\b' + re.escape(cert) + r'\b', text.lower()):
            certifications.append(cert)
    return certifications

# Extract soft skills
def extract_soft_skills(text):
    soft_skills = []
    predefined_soft_skills = ['communication', 'teamwork', 'leadership', 'problem-solving', 'adaptability']
    for skill in predefined_soft_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            soft_skills.append(skill)
    return soft_skills

# Extract previous experience
def extract_experience(text):
    experience = []
    matches = re.findall(r'(\d+)\s+years?\s+of\s+experience', text.lower())
    experience.extend(matches)
    return experience

# Calculate boosted score
def calculate_boosted_score(skills, certifications, soft_skills, experience):
    weights = {
        'skills': 0.5,
        'certifications': 0.15,
        'soft_skills': 0.15,
        'experience': 0.2
    }
    score = (len(skills) * weights['skills'] +
             len(certifications) * weights['certifications'] +
             len(soft_skills) * weights['soft_skills'] +
             len(experience) * weights['experience'])
    return round(score, 2)

def get_meaningcloud_summary(text: str, num_sentences: int = 5) -> Optional[str]:
    url = "https://api.meaningcloud.com/summarization-1.0"
    
    payload = {
        'key': MEANINGCLOUD_API_KEY,
        'txt': text,
        'sentences': str(num_sentences),
        'lang': 'en'
    }

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if response.status_code == 200 and 'summary' in result:
            return result['summary']
        else:
            st.error(f"API Error: {result.get('status', {}).get('msg', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error occurred: {str(e)}")
        return None

def process_resume(file, job_description):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        if not text:
            st.warning(f"No text found in the uploaded resume")
            return None
        
        skills = extract_skills(text)
        certifications = extract_certifications(text)
        soft_skills = extract_soft_skills(text)
        experience = extract_experience(text)
        
        boosted_score = calculate_boosted_score(skills, certifications, soft_skills, experience)
        
        summary = get_meaningcloud_summary(text)
        
        return {
            'name': file.name,
            'score': boosted_score,
            'skills': skills,
            'certifications': certifications,
            'soft_skills': soft_skills,
            'experience': experience,
            'text': summary if summary else text
        }
        
    except Exception as e:
        st.error(f"Error processing the resume: {str(e)}")
        return None

def display_results(scores):
    for score in scores:
        st.write(f"### Resume: {score['name']} - Match: {score['score']}%")
        st.write("### Skills Found:")
        for skill in score['skills']:
            st.markdown(f"- {skill}")
        st.write("### Certifications:")
        for cert in score['certifications']:
            st.markdown(f"- {cert}")
        st.write("### Soft Skills:")
        for skill in score['soft_skills']:
            st.markdown(f"- {skill}")
        st.write("### Experience:")
        for exp in score['experience']:
            st.markdown(f"- {exp} years of experience")
        
        # Display charts
        st.write("### Analysis Breakdown")
        data = {
            'Category': ['Skills', 'Certifications', 'Soft Skills', 'Experience'],
            'Count': [len(score['skills']), len(score['certifications']), len(score['soft_skills']), len(score['experience'])]
        }
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index('Category'))

        if st.button(f"View Detailed Analysis for {score['name']}"):
            st.markdown("### Full Resume Analysis")
            st.text(score['text'])
            st.markdown("### Match Details")
            st.progress(score['score'] / 100)

def main():
    set_custom_css()
    st.title("Enhanced Resume Analyzer")
    
    if not download_nltk_resources():
        st.error("Failed to download required NLTK resources. Please try again.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Upload Resumes (ZIP)")
        uploaded_file = st.file_uploader("Choose a ZIP file containing PDF resumes", type="zip")
    
    with col2:
        st.markdown("### 🎯 Job Requirements")
        job_description = st.text_area("Enter job description:", 
                                     height=150,
                                     placeholder="Paste job description here...")
    
    if st.button("🔍 Analyze Resumes", type="primary"):
        if not uploaded_file:
            st.error("Please upload a ZIP file containing resumes")
            return
        if not job_description:
            st.error("Please enter a job description")
            return
            
        with st.spinner("Processing resumes..."):
            try:
                scores = []
                with ZipFile(uploaded_file) as z:
                    for file_name in z.namelist():
                        with z.open(file_name) as file:
                            score = process_resume(file, job_description)
                            if score:
                                scores.append(score)
                
                if scores:
                    st.success("Analysis complete!")
                    display_results(scores)
                else:
                    st.warning("No valid resumes found to process")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Upload a ZIP file containing multiple resume PDF files.
        2. Paste the job description.
        3. Click 'Analyze Resumes' to start processing.
        4. Results will show matching scores and relevant skills for each resume.
        5. Click 'View Detailed Analysis' for a detailed summary and match details.
        """)

if __name__ == "__main__":
    main()