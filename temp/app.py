import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import pandas as pd
from collections import Counter

# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Downloading language model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Define technical skills categories
TECHNICAL_SKILLS = {
    'programming_languages': set(['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift', 'kotlin', 'r', 'matlab',
                                'typescript', 'scala', 'go', 'rust', 'c#', 'perl', 'html', 'css']),
    'frameworks': set(['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node.js', 'tensorflow',
                      'pytorch', 'keras', 'jquery', 'bootstrap', 'laravel', 'asp.net', 'ruby on rails']),
    'databases': set(['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'elasticsearch', 'cassandra',
                     'dynamodb', 'firebase']),
    'tools': set(['git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 'jira', 'webpack', 'npm', 'yarn',
                  'gradle', 'maven', 'postman', 'gitlab', 'bitbucket']),
    'concepts': set(['api', 'rest', 'graphql', 'microservices', 'ci/cd', 'agile', 'scrum', 'devops', 'tdd', 'oop',
                    'mvc', 'orm', 'soa', 'serverless'])
}

def clean_text(text):
    """Clean and normalize text"""
    # Remove special characters but keep spaces between words
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace and convert to lowercase
    text = ' '.join(text.lower().split())
    return text

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return clean_text(text.strip())
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_technical_skills(text):
    """Extract technical skills from text and categorize them"""
    words = set(text.lower().split())
    skills = {}
    for category, category_skills in TECHNICAL_SKILLS.items():
        found_skills = words.intersection(category_skills)
        if found_skills:
            skills[category] = list(found_skills)
    return skills

def extract_key_phrases(text):
    """Extract key phrases using spaCy"""
    doc = nlp(text)
    phrases = []
    
    # Extract noun phrases and named entities
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2:  # Only phrases with 2 or more words
            phrases.append(chunk.text.lower())
    
    # Count and sort phrases by frequency
    phrase_counts = Counter(phrases)
    return [phrase for phrase, count in phrase_counts.most_common(15)]

def calculate_skill_match(resume_skills, jd_skills):
    """Calculate the percentage match for technical skills"""
    if not jd_skills:
        return 0
    
    all_jd_skills = set()
    all_resume_skills = set()
    
    for skills in jd_skills.values():
        all_jd_skills.update(skills)
    for skills in resume_skills.values():
        all_resume_skills.update(skills)
    
    if not all_jd_skills:
        return 0
        
    matching_skills = all_resume_skills.intersection(all_jd_skills)
    return len(matching_skills) / len(all_jd_skills) * 100

def analyze_documents(resume_text, jd_text):
    """Perform comprehensive analysis of resume and job description"""
    # Extract technical skills
    resume_skills = extract_technical_skills(resume_text)
    jd_skills = extract_technical_skills(jd_text)
    
    # Calculate skill match percentage
    skill_match = calculate_skill_match(resume_skills, jd_skills)
    
    # Extract key phrases
    resume_phrases = extract_key_phrases(resume_text)
    jd_phrases = extract_key_phrases(jd_text)
    
    # Calculate overall similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    
    # Calculate weighted score (60% technical skills, 40% overall content)
    weighted_score = (skill_match * 0.6) + (similarity_score * 0.4)
    
    return {
        'weighted_score': weighted_score,
        'skill_match': skill_match,
        'content_match': similarity_score,
        'resume_skills': resume_skills,
        'jd_skills': jd_skills,
        'resume_phrases': resume_phrases,
        'jd_phrases': jd_phrases
    }

def main():
    st.title("Advanced Resume Analyzer")
    
    # Upload Resume
    st.subheader("Upload Resume (PDF)")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Job Description
    st.subheader("Enter Job Description")
    jd_text = st.text_area("Job Description")
    
    if st.button("Analyze"):
        if pdf_file is not None and jd_text:
            with st.spinner("Performing detailed analysis..."):
                # Extract and process text
                resume_text = extract_text_from_pdf(pdf_file)
                
                if resume_text:
                    # Perform analysis
                    analysis = analyze_documents(resume_text, jd_text)
                    
                    # Display Results
                    st.header("Analysis Results")
                    
                    # Overall Score
                    st.subheader("Match Scores")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Match", f"{analysis['weighted_score']:.1f}%")
                    with col2:
                        st.metric("Technical Skills", f"{analysis['skill_match']:.1f}%")
                    with col3:
                        st.metric("Content Match", f"{analysis['content_match']:.1f}%")
                    
                    # Match Level Assessment
                    if analysis['weighted_score'] >= 75:
                        st.success("Strong Match! Your profile aligns well with the job requirements.")
                    elif analysis['weighted_score'] >= 50:
                        st.warning("Moderate Match. Some improvements recommended.")
                    else:
                        st.error("Low Match. Consider addressing the gaps highlighted below.")
                    
                    # Technical Skills Analysis
                    st.subheader("Technical Skills Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("🎯 Your Technical Skills:")
                        for category, skills in analysis['resume_skills'].items():
                            st.write(f"**{category.title()}:** {', '.join(skills)}")
                    
                    with col2:
                        st.write("📋 Required Technical Skills:")
                        for category, skills in analysis['jd_skills'].items():
                            st.write(f"**{category.title()}:** {', '.join(skills)}")
                    
                    # Key Phrases Comparison
                    st.subheader("Key Phrases Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("📄 Key Phrases in Your Resume:")
                        for phrase in analysis['resume_phrases']:
                            st.write(f"- {phrase}")
                    
                    with col2:
                        st.write("🎯 Key Phrases in Job Description:")
                        for phrase in analysis['jd_phrases']:
                            st.write(f"- {phrase}")
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    missing_skills = set()
                    for category, skills in analysis['jd_skills'].items():
                        if category in analysis['resume_skills']:
                            missing = set(skills) - set(analysis['resume_skills'][category])
                        else:
                            missing = set(skills)
                        missing_skills.update(missing)
                    
                    if missing_skills:
                        st.write("Consider adding these technical skills to your resume:")
                        st.write(", ".join(missing_skills))
                    
                    # Additional Tips
                    if analysis['weighted_score'] < 75:
                        st.write("Additional Recommendations:")
                        st.write("1. Quantify your achievements with metrics and numbers")
                        st.write("2. Use action verbs to describe your experience")
                        st.write("3. Tailor your resume to match the job description keywords")
                        st.write("4. Highlight relevant projects and technologies")
                        
        else:
            st.error("Please upload a resume and enter a job description.")

if __name__ == "__main__":
    main()