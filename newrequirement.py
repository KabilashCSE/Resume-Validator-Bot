import streamlit as st
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import requests
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import openai  # Import OpenAI

# Initialize NLTK resources
def download_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'wordnet': 'corpora/wordnet',
        'stopwords': 'corpora/stopwords'
    }
    for package, resource in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package)

download_nltk_resources()

# Ensure spaCy model is downloaded
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Database setup
Base = declarative_base()

class ResumeScore(Base):
    __tablename__ = 'resume_scores'
    id = Column(Integer, primary_key=True)
    resume_name = Column(String)
    score = Column(Float)
    skills = Column(String)
    certifications = Column(String)
    experience_years = Column(Float)
    education_level = Column(String)
    summary = Column(String)
    job_description = Column(String)
    resume_extracted = Column(String)
    jd_extracted = Column(String)
    matching_details = Column(String)

# Create engine and session
engine = create_engine('sqlite:///resumes.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

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

def get_docparser_data(file, api_key, parser_id) -> Optional[dict]:
    try:
        # First, upload the document
        upload_url = f"https://api.docparser.com/v1/document/upload/{parser_id}"
        
        # Create proper headers with base64 encoded API key
        import base64
        auth_string = base64.b64encode(f"{api_key}:".encode()).decode()
        headers = {
            'Authorization': f'Basic {auth_string}'
        }
        
        # Prepare the file for upload
        files = {
            'file': (file.name, file, 'application/pdf')
        }
        
        # Upload document
        upload_response = requests.post(
            upload_url,
            headers=headers,
            files=files
        )
        upload_response.raise_for_status()
        
        # Get document ID from upload response
        upload_data = upload_response.json()
        
        # Extract document ID from the correct response format
        document_id = upload_data.get('id')
        if not document_id:
            st.error("Failed to get document ID from upload response")
            return None

        # Wait a moment for processing
        import time
        time.sleep(3)  # Increased wait time to ensure document is processed

        # Get parsed results
        results_url = f"https://api.docparser.com/v1/results/{parser_id}/{document_id}"
        results_response = requests.get(
            results_url,
            headers=headers
        )
        results_response.raise_for_status()
        
        # Handle results
        results_data = results_response.json()
        
        if isinstance(results_data, list) and len(results_data) > 0:
            # Map the fields according to your Docparser parser configuration
            result = results_data[0]  # Get the first result
            parsed_data = {
                'name': result.get('name', result.get('full_name', 'Unknown')),
                'email': result.get('email', 'Unknown'),
                'phone': result.get('phone', result.get('phone_number', 'Unknown')),
                'skills': result.get('skills', []),
                'certifications': result.get('certifications', []),
                'experience_years': float(result.get('experience_years', 0)),
                'degree': result.get('degree', result.get('education_degree', 'Not specified')),
                'institution': result.get('institution', result.get('university', 'Not specified')),
                'year': result.get('year', result.get('graduation_year', 'Not specified')),
                'summary': result.get('summary', result.get('profile_summary', 'No summary available')),
                'projects': result.get('projects', [])
            }
            
            # Convert skills from string to list if needed
            if isinstance(parsed_data['skills'], str):
                parsed_data['skills'] = [skill.strip() for skill in parsed_data['skills'].split(',')]
            
            # Convert certifications from string to list if needed
            if isinstance(parsed_data['certifications'], str):
                parsed_data['certifications'] = [cert.strip() for cert in parsed_data['certifications'].split(',')]
            
            return parsed_data
        else:
            st.error(f"No parsed data received from Docparser: {results_data}")
            return None

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        if hasattr(http_err, 'response') and http_err.response is not None:
            st.error(f"Response content: {http_err.response.content}")
    except json.JSONDecodeError as json_err:
        st.error(f"JSON decode error: {json_err}")
        st.error("Raw response content: " + str(upload_response.content if 'upload_response' in locals() else 'No response'))
    except Exception as e:
        st.error(f"Error fetching data from Docparser: {e}")
        st.error(f"Upload data: {upload_data if 'upload_data' in locals() else 'No upload data'}")
        st.error(f"Results data: {results_data if 'results_data' in locals() else 'No results data'}")
    return None

def get_openai_data(file, openai_key: str) -> Optional[dict]:
    openai.api_key = openai_key
    try:
        file_content = file.read()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Extract and analyze the resume content: {file_content}",
            max_tokens=1500
        )
        return response.choices[0].text
    except Exception as e:
        st.error(f"Error fetching data from OpenAI: {e}")
        return None

def get_ollama_data(file_content: str, is_resume: bool = True) -> Optional[dict]:
    try:
        import subprocess
        import json

        # Prepare prompt based on whether it's resume or JD
        if is_resume:
            prompt = f"""Extract the following information from this resume:
            - Personal details (name, email, phone)
            - Skills (as a list)
            - Experience (in years and details)
            - Education (degree, institution)
            - Certifications (as a list)
            - Projects (as a list)
            
            Resume content: {file_content}
            
            Return the information in valid JSON format with these exact keys:
            {{
                "personal_details": {{"name": "", "email": "", "phone": ""}},
                "skills": [],
                "experience_years": 0,
                "education": {{"degree": "", "institution": ""}},
                "certifications": [],
                "projects": [],
                "summary": ""
            }}"""
        else:
            prompt = f"""Extract the following information from this job description:
            - Required skills (as a list)
            - Required experience (in years)
            - Education requirements
            - Other requirements
            
            Job Description: {file_content}
            
            Return the information in valid JSON format with these exact keys:
            {{
                "required_skills": [],
                "required_experience": "",
                "education_requirements": "",
                "other_requirements": []
            }}"""

        # Run Ollama command
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True
        )

        # Extract JSON from response
        response_text = result.stdout
        # Find JSON content between curly braces
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_content = response_text[json_start:json_end]
            return json.loads(json_content)
        else:
            st.error("No valid JSON found in Ollama response")
            return None

    except Exception as e:
        st.error(f"Error processing with Ollama: {e}")
        return None

def calculate_weighted_score(skills, certifications, experience_years, education_level, projects, skill_weight, certification_weight, experience_weight, education_weight, project_weight):
    skill_score = min(len(skills) * 15, 100)
    certification_score = min(len(certifications) * 20, 100)
    experience_score = min(experience_years * 15, 100)
    education_score = 100 if education_level else 0
    project_score = min(len(projects) * 10, 100)  # Assuming each project contributes 10 points

    total_score = (
        skill_score * skill_weight +
        certification_score * certification_weight +
        experience_score * experience_weight +
        education_score * education_weight +
        project_score * project_weight
    )

    return round(min(total_score, 100), 2)

def init_session_state():
    if 'latest_analysis' not in st.session_state:
        st.session_state.latest_analysis = None
    if 'analysis_timestamp' not in st.session_state:
        st.session_state.analysis_timestamp = None

def process_resume(file, job_description, filename, parser_choice, openai_key=None, api_key=None, parser_id=None, matching_weights=None):
    try:
        if parser_choice == "Docparser":
            data = get_docparser_data(file, api_key, parser_id)
        elif parser_choice == "OpenAI":
            data = get_openai_data(file, openai_key)
        elif parser_choice == "Ollama":
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
            
            # Get resume data from Ollama
            data = get_ollama_data(text_content, is_resume=True)
            
            # Get JD data from Ollama
            jd_data = get_ollama_data(job_description, is_resume=False)
        else:
            st.error("Invalid parser choice")
            return None

        if not data:
            st.warning(f"Failed to extract data from the resume {filename}")
            return None

        # Ensure skills is a list of strings
        if isinstance(data.get('skills'), str):
            skills = [skill.strip() for skill in data['skills'].split(',')]
        elif isinstance(data.get('skills'), list):
            skills = [str(skill) for skill in data['skills']]
        else:
            skills = []

        # Extract resume data
        resume_data = {
            'personal_details': {
                'name': data.get('name', 'Unknown'),
                'email': data.get('email', 'Unknown'),
                'phone': data.get('phone', 'Unknown')
            },
            'education': {
                'degree': data.get('degree', 'Not specified'),
                'institution': data.get('institution', 'Not specified'),
                'year': data.get('year', 'Not specified')
            },
            'experience_years': float(data.get('experience_years', 0)),
            'certifications': data.get('certifications', []),
            'skills': skills,
            'projects': data.get('projects', []),
            'summary': data.get('summary', 'No summary available')
        }

        # Extract JD data
        jd_data = extract_jd_details(job_description)

        # Calculate matching score using weights
        matching_details = calculate_matching_score(resume_data, jd_data, matching_weights)

        # Calculate overall score
        overall_score = matching_details['overall_score']

        # Save to database
        resume_score = ResumeScore(
            resume_name=filename,
            score=overall_score,
            skills=json.dumps(skills),  # Store skills as JSON string
            certifications=json.dumps(resume_data['certifications']),
            experience_years=resume_data['experience_years'],
            education_level=resume_data['education']['degree'],
            summary=resume_data['summary'],
            job_description=job_description,
            resume_extracted=json.dumps(resume_data),
            jd_extracted=json.dumps(jd_data),
            matching_details=json.dumps(matching_details)
        )
        session.add(resume_score)
        session.commit()

        result = {
            **resume_data,
            'score': overall_score,
            'jd_extracted': jd_data,
            'matching_details': matching_details
        }

        # Store in session state
        st.session_state.latest_analysis = result
        st.session_state.analysis_timestamp = pd.Timestamp.now()

        return result

    except Exception as e:
        st.error(f"Error processing the resume {filename}: {e}")
        session.rollback()
        return None

def process_resumes(files, job_description, parser_choice, openai_key=None, api_key=None, parser_id=None, matching_weights=None):
    scores = []
    processed_count = 0

    try:
        if not files:
            st.warning("No PDF files uploaded")
            return []

        total_files = len(files)
        progress_bar = st.progress(0)

        for index, file in enumerate(files):
            result = process_resume(
                file, 
                job_description, 
                file.name, 
                parser_choice, 
                openai_key, 
                api_key, 
                parser_id, 
                matching_weights
            )
            if result:
                scores.append(result)
                processed_count += 1

            progress = (index + 1) / total_files
            progress_bar.progress(progress)

        st.success(f"Successfully processed {processed_count} resumes")

        if scores:
            # Store latest batch analysis
            st.session_state.latest_analysis = scores
            st.session_state.analysis_timestamp = pd.Timestamp.now()

        return scores

    except Exception as e:
        st.error(f"Error processing resumes: {e}")
        session.rollback()
        return []

def extract_jd_details(job_description):
    try:
        # Simple field extraction from JD using regex and basic NLP
        jd_data = {
            'required_skills': [],
            'required_experience': '',
            'education_requirements': '',
            'other_requirements': []
        }
        
        # Extract skills - improved pattern matching
        skill_patterns = [
            r'proficient in\s+([\w\s,]+)(?:\.|;|\n)',
            r'experience with\s+([\w\s,]+)(?:\.|;|\n)',
            r'skills:\s*([\w\s,]+)(?:\.|;|\n)',
            r'technologies:\s*([\w\s,]+)(?:\.|;|\n)',
            r'frameworks:\s*([\w\s,]+)(?:\.|;|\n)',
            r'knowledge of\s+([\w\s,]+)(?:\.|;|\n)'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, job_description, re.IGNORECASE)
            for match in matches:
                skills = [s.strip() for s in match.group(1).split(',')]
                jd_data['required_skills'].extend(skills)
        
        # Remove duplicates and empty strings
        jd_data['required_skills'] = list(set(filter(None, jd_data['required_skills'])))
        
        # Extract experience requirements
        exp_pattern = r'(\d+)[\+]?\s*(?:to\s*\d+)?\s*years?.*?experience'
        exp_match = re.search(exp_pattern, job_description, re.IGNORECASE)
        if exp_match:
            jd_data['required_experience'] = exp_match.group(0)
        
        # Extract education requirements - improved pattern
        edu_patterns = [
            r"bachelor'?s?\s*degree",
            r"master'?s?\s*degree",
            r"phd",
            r"b\.?\s*tech",
            r"m\.?\s*tech",
            r"b\.?\s*e",
            r"m\.?\s*e"
        ]
        
        for pattern in edu_patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                jd_data['education_requirements'] = match.group(0)
                break
        
        return jd_data
    except Exception as e:
        st.error(f"Error extracting JD details: {e}")
        return {
            'required_skills': [],
            'required_experience': '',
            'education_requirements': '',
            'other_requirements': []
        }

def calculate_matching_score(resume_data, jd_data, weights):
    try:
        matching_details = {
            'skills_match': {
                'matched': [],
                'missing': [],
                'score': 0
            },
            'experience_match': {
                'details': '',
                'score': 0
            },
            'education_match': {
                'details': '',
                'score': 0
            },
            'overall_score': 0
        }
        
        # Skills matching - improved handling
        resume_skills = []
        if isinstance(resume_data.get('skills'), list):
            resume_skills = [str(skill).lower() for skill in resume_data.get('skills', [])]
        elif isinstance(resume_data.get('skills'), str):
            resume_skills = [skill.lower().strip() for skill in resume_data.get('skills', '').split(',')]
        
        jd_skills = [str(skill).lower() for skill in jd_data.get('required_skills', [])]
        
        matched_skills = []
        missing_skills = []
        
        for jd_skill in jd_skills:
            if any(jd_skill in res_skill or res_skill in jd_skill for res_skill in resume_skills):
                matched_skills.append(jd_skill)
            else:
                missing_skills.append(jd_skill)
        
        matching_details['skills_match']['matched'] = matched_skills
        matching_details['skills_match']['missing'] = missing_skills
        
        skills_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 100
        matching_details['skills_match']['score'] = round(skills_score, 2)
        
        # Experience matching - improved handling
        resume_exp = float(resume_data.get('experience_years', 0))
        required_exp = 0
        if jd_data.get('required_experience'):
            exp_match = re.search(r'(\d+)', jd_data['required_experience'])
            if exp_match:
                required_exp = float(exp_match.group(1))
        
        exp_score = min((resume_exp / required_exp) * 100, 100) if required_exp > 0 else 100
        matching_details['experience_match']['score'] = round(exp_score, 2)
        matching_details['experience_match']['details'] = f"Has {resume_exp} years, Required {required_exp} years"
        
        # Education matching - improved handling
        resume_edu = str(resume_data.get('education', {}).get('degree', '')).lower()
        required_edu = str(jd_data.get('education_requirements', '')).lower()
        
        edu_score = 100 if (resume_edu and (resume_edu in required_edu or required_edu in resume_edu)) else 0
        matching_details['education_match']['score'] = edu_score
        matching_details['education_match']['details'] = (
            f"Required: {jd_data.get('education_requirements', 'Not specified')}\n"
            f"Found: {resume_data.get('education', {}).get('degree', 'Not specified')}"
        )
        
        # Calculate overall score using weights
        overall_score = (
            skills_score * weights['skill_weight'] +
            exp_score * weights['experience_weight'] +
            edu_score * weights['education_weight']
        )
        
        matching_details['overall_score'] = round(overall_score, 2)
        
        return matching_details
    except Exception as e:
        st.error(f"Error calculating matching score: {e}")
        return {
            'skills_match': {'matched': [], 'missing': [], 'score': 0},
            'experience_match': {'details': '', 'score': 0},
            'education_match': {'details': '', 'score': 0},
            'overall_score': 0
        }

def display_analysis_tabs(result):
    tab1, tab2, tab3 = st.tabs(["📄 Resume Extracted", "📋 JD Extracted", "🎯 Matching Analysis"])
    
    with tab1:
        st.header("Resume Information")
        st.subheader("Personal Details")
        personal_details = result.get('personal_details', {})
        st.write(f"Name: {personal_details.get('name', 'Not specified')}")
        st.write(f"Email: {personal_details.get('email', 'Not specified')}")
        st.write(f"Phone: {personal_details.get('phone', 'Not specified')}")
        
        st.subheader("Skills")
        skills = result.get('skills', [])
        if skills:
            for skill in skills:
                st.markdown(f"- {skill}")
        else:
            st.write("No skills found")
        
        st.subheader("Experience")
        experience_years = result.get('experience_years', 0)
        st.write(f"Total Years: {experience_years}")
        
        st.subheader("Education")
        education = result.get('education', {})
        st.write(f"Degree: {education.get('degree', 'Not specified')}")
        st.write(f"Institution: {education.get('institution', 'Not specified')}")
        
    with tab2:
        st.header("Job Description Analysis")
        jd_data = result.get('jd_extracted', {})
        if isinstance(jd_data, str):
            try:
                jd_data = json.loads(jd_data)
            except json.JSONDecodeError:
                jd_data = {}
        
        st.subheader("Required Skills")
        required_skills = jd_data.get('required_skills', [])
        if required_skills:
            for skill in required_skills:
                st.markdown(f"- {skill}")
        else:
            st.write("No required skills specified")
            
        st.subheader("Experience Requirements")
        st.write(jd_data.get('required_experience', 'Not specified'))
        
        st.subheader("Education Requirements")
        st.write(jd_data.get('education_requirements', 'Not specified'))
        
    with tab3:
        st.header("Matching Analysis")
        matching_data = result.get('matching_details', {})
        if isinstance(matching_data, str):
            try:
                matching_data = json.loads(matching_data)
            except json.JSONDecodeError:
                matching_data = {}
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Match", f"{result.get('score', 0)}%")
        
        st.subheader("Skills Match")
        skills_match = matching_data.get('skills_match', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("✅ Matched Skills")
            matched_skills = skills_match.get('matched', [])
            if matched_skills:
                for skill in matched_skills:
                    st.markdown(f"- {skill}")
            else:
                st.write("No matched skills")
                
        with col2:
            st.write("❌ Missing Skills")
            missing_skills = skills_match.get('missing', [])
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f"- {skill}")
            else:
                st.write("No missing skills")
        
        st.subheader("Experience Match")
        exp_match = matching_data.get('experience_match', {})
        st.write(exp_match.get('details', 'Not available'))
        
        st.subheader("Education Match")
        edu_match = matching_data.get('education_match', {})
        st.write(edu_match.get('details', 'Not available'))

def view_scores():
    st.header("Resume Analysis Results")
    
    try:
        # Get all resumes first
        all_resumes = session.query(ResumeScore).all()
        
        if not all_resumes:
            st.info("No resumes have been analyzed yet. Please process some resumes first.")
            return

        # Add filters
        st.sidebar.header("Filters")
        min_score = st.sidebar.slider("Minimum Score", 0, 100, 0)
        
        # Get all unique skills from the database
        all_skills = set()
        for resume in all_resumes:
            try:
                if resume.skills:
                    skills = json.loads(resume.skills)
                    if isinstance(skills, list):
                        all_skills.update([str(skill) for skill in skills])
            except json.JSONDecodeError:
                continue
        
        # Convert to sorted list and filter out empty strings
        all_skills = sorted([skill for skill in all_skills if skill])
        
        if all_skills:
            skill_filter = st.sidebar.multiselect("Filter by Skills", all_skills)
        else:
            st.sidebar.info("No skills found in the database")
            skill_filter = []

        # Query resumes with filters
        query = session.query(ResumeScore)
        
        if min_score > 0:
            query = query.filter(ResumeScore.score >= min_score)
            
        if skill_filter:
            skill_conditions = []
            for skill in skill_filter:
                skill_conditions.append(ResumeScore.skills.like(f'%{skill}%'))
            from sqlalchemy import or_
            query = query.filter(or_(*skill_conditions))
        
        resumes = query.order_by(ResumeScore.score.desc()).all()
        
        if resumes:
            st.success(f"Found {len(resumes)} matching resumes")
            for resume in resumes:
                with st.expander(f"📄 {resume.resume_name} - Match: {resume.score}%"):
                    tabs = st.tabs(["Resume Details", "Job Description", "Matching Analysis"])
                    
                    with tabs[0]:
                        try:
                            resume_data = json.loads(resume.resume_extracted) if resume.resume_extracted else {}
                            
                            st.subheader("Personal Details")
                            personal_details = resume_data.get('personal_details', {})
                            st.write(f"**Name:** {personal_details.get('name', 'Not specified')}")
                            st.write(f"**Email:** {personal_details.get('email', 'Not specified')}")
                            st.write(f"**Phone:** {personal_details.get('phone', 'Not specified')}")
                            
                            st.subheader("Skills")
                            skills = resume_data.get('skills', [])
                            if skills:
                                for skill in skills:
                                    if isinstance(skill, dict):
                                        st.markdown(f"- {skill.get('key_0', '')}")
                                    else:
                                        st.markdown(f"- {skill}")
                            
                            st.subheader("Experience")
                            st.write(f"Total Years: {resume_data.get('experience_years', 0)}")
                            
                            st.subheader("Education")
                            education = resume_data.get('education', {})
                            st.write(f"Degree: {education.get('degree', 'Not specified')}")
                            st.write(f"Institution: {education.get('institution', 'Not specified')}")
                            
                        except json.JSONDecodeError:
                            st.error("Error parsing resume data")
                    
                    with tabs[1]:
                        try:
                            jd_data = json.loads(resume.jd_extracted) if resume.jd_extracted else {}
                            
                            st.subheader("Required Skills")
                            required_skills = jd_data.get('required_skills', [])
                            if required_skills:
                                for skill in required_skills:
                                    st.markdown(f"- {skill}")
                            
                            st.subheader("Experience Requirements")
                            st.write(jd_data.get('required_experience', 'Not specified'))
                            
                            st.subheader("Education Requirements")
                            st.write(jd_data.get('education_requirements', 'Not specified'))
                            
                        except json.JSONDecodeError:
                            st.error("Error parsing job description data")
                    
                    with tabs[2]:
                        try:
                            matching_data = json.loads(resume.matching_details) if resume.matching_details else {}
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Overall Match", f"{resume.score}%")
                            
                            st.subheader("Skills Match")
                            skills_match = matching_data.get('skills_match', {})
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("✅ Matched Skills")
                                matched_skills = skills_match.get('matched', [])
                                if matched_skills:
                                    for skill in matched_skills:
                                        st.markdown(f"- {skill}")
                                else:
                                    st.write("No matched skills")
                            
                            with col2:
                                st.write("❌ Missing Skills")
                                missing_skills = skills_match.get('missing', [])
                                if missing_skills:
                                    for skill in missing_skills:
                                        st.markdown(f"- {skill}")
                                else:
                                    st.write("No missing skills")
                            
                            st.subheader("Experience Match")
                            exp_match = matching_data.get('experience_match', {})
                            st.write(exp_match.get('details', 'Not available'))
                            
                            st.subheader("Education Match")
                            edu_match = matching_data.get('education_match', {})
                            st.write(edu_match.get('details', 'Not available'))
                            
                        except json.JSONDecodeError:
                            st.error("Error parsing matching data")
                    
                    if st.button("Delete", key=f"delete_{resume.id}"):
                        session.delete(resume)
                        session.commit()
                        st.success("Resume deleted successfully!")
                        st.experimental_rerun()
        else:
            st.warning("No resumes found matching the selected criteria. Try adjusting the filters.")
            
    except Exception as e:
        st.error(f"Error viewing scores: {e}")
        session.rollback()

def main():
    init_session_state()  # Initialize session state
    st.title("Advanced Resume Analyzer")
    set_custom_css()
    
    menu = ["Home", "View Scores"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Show latest analysis if available and recent
        if st.session_state.latest_analysis is not None:
            current_time = pd.Timestamp.now()
            if st.session_state.analysis_timestamp is not None:
                time_diff = (current_time - st.session_state.analysis_timestamp).total_seconds()
                if time_diff < 3600:  # Show results if less than 1 hour old
                    st.info("📊 Latest Analysis Results")
                    if isinstance(st.session_state.latest_analysis, list):
                        for result in st.session_state.latest_analysis:
                            display_analysis_tabs(result)
                    else:
                        display_analysis_tabs(st.session_state.latest_analysis)
                    
                    if st.button("Clear Latest Results"):
                        st.session_state.latest_analysis = None
                        st.session_state.analysis_timestamp = None
                        st.rerun()

        # Rest of the home page code...
        st.sidebar.header("Analysis Configuration")
        analysis_type = st.sidebar.radio(
            "Select Analysis Type:",
            ["Single Resume", "Folder Upload"],
            key="analysis_type"
        )
        
        method_choice = st.sidebar.radio(
            "Select Method:",
            ["Use OpenAI", "Use Docparser", "Use Ollama"],
            key="method_choice"
        )

        # Matching Configuration
        st.sidebar.header("Matching Configuration")
        matching_weights = {
            'skill_weight': st.sidebar.slider("Skills Importance", 0.0, 1.0, 0.4),
            'experience_weight': st.sidebar.slider("Experience Importance", 0.0, 1.0, 0.3),
            'education_weight': st.sidebar.slider("Education Importance", 0.0, 1.0, 0.2),
            'certification_weight': st.sidebar.slider("Certifications Importance", 0.0, 1.0, 0.1)
        }

        # Main content area
        st.header("Resume Analysis")
        
        if method_choice == "Use OpenAI":
            openai_key = st.text_input("Enter OpenAI API Key:", type="password")
            parser_choice = "OpenAI"
        elif method_choice == "Use Docparser":
            parser_choice = "Docparser"
            api_key = st.text_input("Enter Docparser API Key:", type="password")
            parser_id = st.text_input("Enter Docparser Parser ID:")
        else:  # Ollama
            parser_choice = "Ollama"
            st.info("Using local Ollama LLM for extraction")
            api_key = None
            parser_id = None
            openai_key = None

        job_description = st.text_area(
            "Enter job description:", 
            height=150, 
            placeholder="Paste job description here...", 
            key="job_desc"
        )

        if analysis_type == "Single Resume":
            uploaded_file = st.file_uploader(
                "Upload a resume PDF file", 
                type="pdf",
                help="Upload a single PDF resume file"
            )

            if st.button("Analyze Resume", type="primary"):
                if not uploaded_file:
                    st.error("Please upload a resume PDF file")
                    return
                if not job_description:
                    st.error("Please enter a job description")
                    return
                if method_choice == "Use LLM" and not openai_key:
                    st.error("Please enter the OpenAI API key")
                    return
                if method_choice == "Use Field Extraction" and (not api_key or not parser_id):
                    st.error("Please enter the Docparser API key and Parser ID")
                    return
                
                with st.spinner("Processing resume..."):
                    result = process_resume(
                        uploaded_file,
                        job_description,
                        uploaded_file.name,
                        parser_choice,
                        openai_key,
                        api_key,
                        parser_id,
                        matching_weights
                    )
                    if result:
                        st.success("Analysis complete!")
                        display_analysis_tabs(result)
                    else:
                        st.warning("Failed to process the resume.")
                        st.session_state.latest_analysis = None
                        st.session_state.analysis_timestamp = None

        elif analysis_type == "Folder Upload":
            uploaded_files = st.file_uploader(
                "Upload multiple resume PDF files", 
                type="pdf", 
                accept_multiple_files=True,
                help="Upload multiple PDF resume files"
            )

            if st.button("Analyze Resumes", type="primary"):
                if not uploaded_files:
                    st.error("Please upload resume PDF files")
                    return
                if not job_description:
                    st.error("Please enter a job description")
                    return
                if method_choice == "Use LLM" and not openai_key:
                    st.error("Please enter the OpenAI API key")
                    return
                if method_choice == "Use Field Extraction" and (not api_key or not parser_id):
                    st.error("Please enter the Docparser API key and Parser ID")
                    return
                
                with st.spinner("Processing resumes..."):
                    scores = process_resumes(
                        uploaded_files,
                        job_description,
                        parser_choice,
                        openai_key,
                        api_key,
                        parser_id,
                        matching_weights
                    )
                    if scores:
                        st.success("Analysis complete!")
                        for result in scores:
                            display_analysis_tabs(result)
                    else:
                        st.warning("No valid resumes found to process")
                        st.session_state.latest_analysis = None
                        st.session_state.analysis_timestamp = None

        with st.sidebar.expander("ℹ️ How to use"):
            st.markdown("""
            1. Select the analysis type from the sidebar
            2. Choose the parsing method
            3. Configure matching weights
            4. Enter API credentials
            5. Upload resume(s)
            6. Paste job description
            7. Click Analyze to process
            8. View results in the tabs
            """)

    elif choice == "View Scores":
        view_scores()
        
        # Add button to return to latest analysis
        if st.session_state.latest_analysis is not None:
            if st.sidebar.button("Return to Latest Analysis"):
                st.session_state.page = "Home"
                st.rerun()

# Add a function to check if analysis is recent
def is_analysis_recent():
    if st.session_state.analysis_timestamp is None:
        return False
    current_time = pd.Timestamp.now()
    time_diff = (current_time - st.session_state.analysis_timestamp).total_seconds()
    return time_diff < 3600  # Returns True if analysis is less than 1 hour old

if __name__ == "__main__":
    main()