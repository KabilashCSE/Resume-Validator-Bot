import streamlit as st
import requests
import json
import time
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

def display_skills(skills):
    st.subheader("🛠️ Skills")
    if not skills:
        return
    
    # Create a clean list of skills
    skill_list = [skill.get("key_0") for skill in skills if skill.get("key_0")]
    
    # Display skills in columns
    cols = st.columns(3)
    for idx, skill in enumerate(skill_list):
        cols[idx % 3].markdown(f"- {skill}")

def display_experience(experiences):
    st.subheader("💼 Work Experience")
    if not experiences:
        return
    
    for exp in experiences:
        with st.container():
            st.markdown(f"**{exp.get('key_0', '')}** | {exp.get('key_1', '')} | {exp.get('key_2', '')}")
            st.markdown(f"_{exp.get('key_3', '')}_")
            st.markdown("---")

def display_projects(projects):
    st.subheader("🚀 Projects")
    if not projects:
        return
    
    for project in projects:
        with st.container():
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

def display_results(parsed_data):
    """Display only the key resume sections in a clean format."""
    if not parsed_data:
        return
    
    # Handle both list and single document responses
    if isinstance(parsed_data, list):
        data = parsed_data[0] if parsed_data else {}
    else:
        data = parsed_data
    
    # Display name as header
    if name := data.get('name'):
        st.title(name)
    if email := data.get('email'):
        st.markdown(f"📧 {email}")
    
    st.markdown("---")
    
    # Display key sections
    display_skills(data.get('skills', []))
    st.markdown("---")
    display_experience(data.get('experiences', []))
    st.markdown("---")
    display_projects(data.get('projects', []))
    st.markdown("---")
    display_achievements(data.get('achievements', []))

def main():
    st.set_page_config(
        page_title="Resume Parser",
        page_icon="📄",
        layout="wide"
    )
    
    # Move configuration to sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Docparser API Key", type="password")
        parser_id = st.text_input("Parser ID")
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"],
        help="Supported formats: PDF, DOCX"
    )
    
    if st.button("Extract Resume", disabled=not (uploaded_file and api_key and parser_id)):
        if not all([uploaded_file, api_key, parser_id]):
            st.warning("Please provide all required inputs")
            return
        
        with st.spinner("Processing resume..."):
            docparser = DocparserAPI(api_key, parser_id)
            parsed_data = docparser.extract_fields(uploaded_file)
            
            if parsed_data:
                display_results(parsed_data)
            else:
                st.error("Failed to extract information from the resume.")

if __name__ == "__main__":
    main()