import streamlit as st
import requests
import json
import time
import subprocess
import re
from typing import Optional, Dict, Any

class DocparserAPI:
    def __init__(self, api_key: str, parser_id: str):
        self.api_key = api_key
        self.parser_id = parser_id
        self.base_url = "https://api.docparser.com/v1"
    
    def extract_fields(self, file) -> Optional[Dict[str, Any]]:
        """Extract fields from a document using Docparser API."""
        url = f"{self.base_url}/document/upload/{self.parser_id}"
        params = {"api_key": self.api_key}
        
        try:
            files = {"file": file}
            upload_response = requests.post(url, params=params, files=files)
            upload_response.raise_for_status()
            
            response_data = upload_response.json()
            
            if isinstance(response_data, list) and len(response_data) > 0:
                doc_id = response_data[0].get("id")
            else:
                doc_id = response_data.get("id")
                
            if not doc_id:
                raise ValueError("No document ID received from upload")
            
            time.sleep(3)
            
            results_url = f"{self.base_url}/results/{self.parser_id}"
            params = {
                "api_key": self.api_key,
                "document_id": doc_id
            }
            
            results_response = requests.get(results_url, params=params)
            results_response.raise_for_status()
            
            return results_response.json()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

def query_ollama(prompt, model="qwen:1.8b"):
    """
    Query the local Ollama model using the `ollama run` command.
    """
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Send the prompt to the model
        output, error = process.communicate(prompt)

        if process.returncode != 0:
            st.error(f"Error querying Ollama: {error}")
            return "Error in response"
        
        return output.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Error in response"

def extract_fields_from_jd(jd_text):
    """
    Extract key fields from a job description using the local model.
    """
    prompt = f"Extract the following fields from this job description: Skills required (programming languages, frameworks), Projects worked on, Experience previous, Certifications if any. Return each field as a list enclosed in brackets. \nJob Description: {jd_text}"
    result = query_ollama(prompt)
    
    # Initialize empty fields
    fields = {
        "skills_required": [],
        "projects_worked_on": [],
        "experience_previous": [],
        "certifications": []
    }
    
    try:
        st.write("Raw response from Ollama:")
        st.write(result)

        # Skills extraction using regex patterns to detect programming languages, frameworks, etc.
        skills_keywords = [
            "JavaScript", "React", "Node.js", "Python", "HTML", "CSS", "Angular", "Java", "SQL", "MongoDB", "AWS", "Kubernetes", "Azure"
        ]
        for skill in skills_keywords:
            if skill.lower() in result.lower():
                fields["skills_required"].append(skill)

        # Projects extraction
        projects_matches = re.findall(r"(Projects to be worked on include.*?)(?:\.|$)", result, re.DOTALL)
        if projects_matches:
            projects = projects_matches[0].strip().strip("[]")
            fields["projects_worked_on"] = [p.strip() for p in projects.split(",")]

        # Experience extraction
        experience_matches = re.findall(r"(\d[\d+]* years of experience)", result)
        if experience_matches:
            fields["experience_previous"] = experience_matches

        # Certifications extraction
        certifications_matches = re.findall(r"(Certified [A-Za-z\s]+|AWS Certified [A-Za-z\s]+|Microsoft Certified: [A-Za-z\s]+)", result)
        if certifications_matches:
            fields["certifications"] = certifications_matches

        # Clean up any extra whitespace in the lists
        fields["skills_required"] = [s.strip() for s in fields["skills_required"]]
        fields["projects_worked_on"] = [p.strip() for p in fields["projects_worked_on"]]
        fields["experience_previous"] = [e.strip() for e in fields["experience_previous"]]
        fields["certifications"] = [c.strip() for c in fields["certifications"]]

        return fields
    
    except Exception as e:
        st.error(f"Failed to parse JD fields: {e}")
        return fields

def display_resume_data(parsed_data):
    """Display the extracted resume data once."""
    if not parsed_data:
        return
    
    # Extract data from first item if list, otherwise use as is
    data = parsed_data[0] if isinstance(parsed_data, list) else parsed_data
    
    skills = data.get('skills', [])
    experiences = data.get('experiences', [])
    projects = data.get('projects', [])
    achievements = data.get('achievements', [])
    
    display_skills(skills)
    display_experience(experiences) 
    display_projects(projects)
    display_achievements(achievements)

def analyze_match(resume_data, jd_fields, weights):
    """Analyze match between resume and JD using Ollama."""
    prompt = f"""
    Compare this resume data and job description requirements:
    
    Resume:
    Skills: {resume_data.get('skills', [])}
    Experience: {resume_data.get('experiences', [])}
    Projects: {resume_data.get('projects', [])}
    Certifications: {resume_data.get('achievements', [])}
    
    Job Requirements:
    Skills: {jd_fields.get('skills_required', [])}
    Experience: {jd_fields.get('experience_previous', [])}
    Projects: {jd_fields.get('projects_worked_on', [])}
    Certifications: {jd_fields.get('certifications', [])}
    
    Weights:
    Skills: {weights['skills']}%
    Experience: {weights['experience']}%
    Projects: {weights['projects']}%
    Certifications: {weights['certs']}%
    
    Provide:
    1. Overall match score (0-100%)
    2. Key matching skills
    3. Experience alignment
    4. Missing requirements
    5. Recommendations
    """
    
    return query_ollama(prompt)

def display_skills(skills):
    st.subheader("🛠️ Skills")
    if not skills:
        return
    cols = st.columns(3)
    for idx, skill in enumerate(skills):
        cols[idx % 3].markdown(f"- {skill}")

def display_experience(experiences):
    st.subheader("💼 Work Experience")
    if not experiences:
        return
    for exp in experiences:
        st.markdown(f"**{exp.get('key_0', '')}** | {exp.get('key_1', '')} | {exp.get('key_2', '')}")
        st.markdown(f"_{exp.get('key_3', '')}_")
        st.markdown("---")

def display_projects(projects):
    st.subheader("🚀 Projects")
    if not projects:
        return
    for project in projects:
        title = project.get('key_0', '')
        tech = project.get('key_1', '')
        description = project.get('key_2', '')
        st.markdown(f"**{title}**")
        if tech:
            st.markdown(f"*Technologies: {tech}*")
        st.markdown(description)
        st.markdown("---")

def display_achievements(achievements):
    st.subheader("🏆 Achievements & Certifications")
    if not achievements:
        return
    for achievement in achievements:
        st.markdown(f"- {achievement.get('key_0', '')}")

def main():
    st.set_page_config(page_title="JD and Resume Matching", layout="wide")
    st.title("Job Description and Resume Matching")

    # Session state for tab control
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'parsed_resume' not in st.session_state:
        st.session_state.parsed_resume = None
    if 'jd_fields' not in st.session_state:
        st.session_state.jd_fields = None

    # Sidebar inputs
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Docparser API Key", type="password")
        parser_id = st.text_input("Parser ID")
        uploaded_resume = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

    tab1, tab2, tab3, tab4 = st.tabs(["Extracted JD Fields", "Extracted Resume Fields", "Analysis", "Match"])
    
    # Tab 1: Job Description
    with tab1:
        jd_text = st.text_area("Enter Job Description", height=200)
        if st.button("Extract JD Fields"):
            if jd_text.strip():
                st.session_state.jd_fields = extract_fields_from_jd(jd_text)
                st.subheader("Extracted JD Fields")
                st.json(st.session_state.jd_fields)
            else:
                st.error("Please provide the Job Description.")
    
    # Tab 2: Resume
    with tab2:
        if uploaded_resume and api_key and parser_id:
            docparser = DocparserAPI(api_key, parser_id)
            st.session_state.parsed_resume = docparser.extract_fields(uploaded_resume)
            display_resume_data(st.session_state.parsed_resume)
    
    # Tab 3: Analysis
    with tab3:
        st.subheader("Analysis Configuration")
        skill_weight = st.slider("Weight for Skills", 0, 100, 30)
        experience_weight = st.slider("Weight for Experience", 0, 100, 40)
        projects_weight = st.slider("Weight for Projects", 0, 100, 20)
        certifications_weight = st.slider("Weight for Certifications", 0, 100, 10)
        
        weights = {
            'skills': skill_weight,
            'experience': experience_weight,
            'projects': projects_weight,
            'certs': certifications_weight
        }
        
        if st.button("Start Analysis"):
            if st.session_state.parsed_resume and st.session_state.jd_fields:
                st.session_state.analysis_complete = True
                st.session_state.current_tab = 3  # Switch to match tab
                st.experimental_rerun()
            else:
                st.error("Please complete Resume and JD extraction first")
    
    # Tab 4: Match
    with tab4:
        if st.session_state.analysis_complete:
            st.subheader("Match Analysis Results")
            
            with st.spinner("Analyzing match..."):
                match_analysis = analyze_match(
                    st.session_state.parsed_resume[0] if isinstance(st.session_state.parsed_resume, list) else st.session_state.parsed_resume,
                    st.session_state.jd_fields,
                    weights
                )
            
            st.markdown("### AI Analysis")
            st.write(match_analysis)
            
            # Reset for new analysis
            if st.button("Start New Analysis"):
                st.session_state.analysis_complete = False
                st.session_state.current_tab = 0
                st.experimental_rerun()
        else:
            st.info("Please complete the analysis in the Analysis tab first")

if __name__ == "__main__":
    main()
