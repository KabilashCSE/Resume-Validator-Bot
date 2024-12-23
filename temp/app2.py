import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import pandas as pd
from collections import Counter
from openai import OpenAI
import json

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Downloading language model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_llm_analysis(resume_text, job_description):
    """Use GPT to analyze the resume against the job description"""
    try:
        prompt = f"""
        Analyze the following resume and job description as an expert recruiter. 
        Provide a detailed analysis in JSON format with the following structure:
        {{
            "match_score": (0-100 score indicating overall match),
            "key_matches": [list of strong matching points],
            "gaps": [list of missing or weak areas],
            "skill_analysis": {{
                "technical_skills": {{
                    "present": [skills found in resume],
                    "missing": [required skills not found]
                }},
                "soft_skills": {{
                    "present": [soft skills found],
                    "missing": [required soft skills not found]
                }}
            }},
            "recommendations": [specific suggestions for improvement],
            "overall_assessment": "detailed evaluation of the candidate's fit"
        }}

        Resume:
        {resume_text}

        Job Description:
        {job_description}
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert recruiter and resume analyzer."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error in LLM analysis: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def main():
    st.title("AI-Powered Resume Analyzer")
    
    # Add API Key input
    if 'OPENAI_API_KEY' not in st.secrets:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.secrets["OPENAI_API_KEY"] = api_key
    
    # Upload Resume
    st.subheader("Upload Resume (PDF)")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Job Description
    st.subheader("Enter Job Description")
    jd_text = st.text_area("Job Description")
    
    if st.button("Analyze"):
        if not pdf_file or not jd_text:
            st.error("Please provide both resume and job description.")
            return
            
        if 'OPENAI_API_KEY' not in st.secrets:
            st.error("Please enter your OpenAI API key.")
            return
            
        with st.spinner("Performing AI-powered analysis..."):
            # Extract resume text
            resume_text = extract_text_from_pdf(pdf_file)
            if not resume_text:
                st.error("Could not extract text from the PDF. Please try another file.")
                return
                
            # Get LLM analysis
            analysis = get_llm_analysis(resume_text, jd_text)
            if not analysis:
                st.error("Analysis failed. Please try again.")
                return
            
            # Display Results
            st.header("Analysis Results")
            
            # Overall Match Score
            score = analysis.get('match_score', 0)
            st.subheader("Match Score")
            st.progress(score/100)
            st.metric("Overall Match", f"{score}%")
            
            # Match Level Assessment
            if score >= 75:
                st.success(analysis.get('overall_assessment', "Strong Match!"))
            elif score >= 50:
                st.warning(analysis.get('overall_assessment', "Moderate Match"))
            else:
                st.error(analysis.get('overall_assessment', "Low Match"))
            
            # Key Matches
            st.subheader("💪 Strengths")
            for match in analysis.get('key_matches', []):
                st.write(f"✓ {match}")
            
            # Gaps
            st.subheader("🎯 Areas for Improvement")
            for gap in analysis.get('gaps', []):
                st.write(f"• {gap}")
            
            # Skills Analysis
            st.subheader("🛠️ Skills Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Technical Skills Present:**")
                for skill in analysis.get('skill_analysis', {}).get('technical_skills', {}).get('present', []):
                    st.write(f"✓ {skill}")
                
                st.write("**Soft Skills Present:**")
                for skill in analysis.get('skill_analysis', {}).get('soft_skills', {}).get('present', []):
                    st.write(f"✓ {skill}")
            
            with col2:
                st.write("**Missing Technical Skills:**")
                for skill in analysis.get('skill_analysis', {}).get('technical_skills', {}).get('missing', []):
                    st.write(f"• {skill}")
                
                st.write("**Missing Soft Skills:**")
                for skill in analysis.get('skill_analysis', {}).get('soft_skills', {}).get('missing', []):
                    st.write(f"• {skill}")
            
            # Recommendations
            st.subheader("📋 Recommendations")
            for rec in analysis.get('recommendations', []):
                st.write(f"→ {rec}")
            
            # Export Option
            st.subheader("Export Analysis")
            export_data = {
                "Resume Analysis Report": {
                    "Overall Match": f"{score}%",
                    "Assessment": analysis.get('overall_assessment', ''),
                    "Key Strengths": analysis.get('key_matches', []),
                    "Areas for Improvement": analysis.get('gaps', []),
                    "Skills Analysis": analysis.get('skill_analysis', {}),
                    "Recommendations": analysis.get('recommendations', [])
                }
            }
            st.download_button(
                label="Download Analysis Report",
                data=json.dumps(export_data, indent=2),
                file_name="resume_analysis_report.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()