# skills_matcher.py

from typing import List, Dict, Tuple
import re
from collections import Counter

def clean_text(text: str) -> str:
    """Clean and normalize text for comparison"""
    return re.sub(r'[^\w\s]', '', text.lower())

def extract_skills(text: str, skills_keywords: List[str]) -> List[str]:
    """Extract skills from text based on keywords"""
    cleaned_text = clean_text(text)
    found_skills = []
    
    for skill in skills_keywords:
        if clean_text(skill) in cleaned_text:
            found_skills.append(skill)
    
    return list(set(found_skills))

def get_skills_match(
    resume_text: str, 
    jd_text: str, 
    skills_keywords: List[str]
) -> Dict:
    """
    Compare skills between resume and job description
    Returns dict with matching and missing skills
    """
    # Extract skills from both texts
    resume_skills = extract_skills(resume_text, skills_keywords)
    jd_skills = extract_skills(jd_text, skills_keywords)
    
    # Find matching and missing skills
    matching_skills = list(set(resume_skills) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(resume_skills))
    
    # Calculate match percentage
    if len(jd_skills) > 0:
        match_percentage = (len(matching_skills) / len(jd_skills)) * 100
    else:
        match_percentage = 0
        
    return {
        'matching_skills': matching_skills,
        'missing_skills': missing_skills,
        'match_percentage': match_percentage,
        'resume_skills': resume_skills,
        'jd_skills': jd_skills
    }

# Common skills keywords list
TECHNICAL_SKILLS = [
    'python', 'java', 'javascript', 'sql', 'aws', 
    'docker', 'kubernetes', 'react', 'node.js',
    'machine learning', 'data analysis', 'git'
]

SOFT_SKILLS = [
    'communication', 'leadership', 'teamwork',
    'problem solving', 'time management', 'critical thinking',
    'project management', 'collaboration'
]

def get_skills_keywords() -> List[str]:
    """Return combined list of technical and soft skills"""
    return TECHNICAL_SKILLS + SOFT_SKILLS