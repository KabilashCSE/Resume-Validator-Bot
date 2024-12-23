import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import json
from collections import Counter

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
            margin: 2px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 500;
        }
        .present-skill {
            background-color: #e7f3ff;
            color: #1e88e5;
            border: 1px solid #1e88e5;
        }
        .missing-skill {
            background-color: #ffebee;
            color: #e53935;
            border: 1px solid #e53935;
        }
        .main-header {
            color: #1e88e5;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .score-card {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .high-score {
            background: linear-gradient(135deg, #4CAF50, #8BC34A);
            color: white;
        }
        .medium-score {
            background: linear-gradient(135deg, #FFA726, #FFB74D);
            color: white;
        }
        .low-score {
            background: linear-gradient(135deg, #EF5350, #E57373);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("Downloading language model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_skills(text, nlp):
    doc = nlp(text.lower())
    
    technical_patterns = {
        # Programming Languages
        "python", "java", "javascript", "c++", "ruby", "php", "swift", "kotlin", "go",
        # Web Technologies
        "html", "css", "react", "angular", "vue.js", "node.js", "express.js", "django",
        "flask", "spring boot", "asp.net",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "oracle", "redis", "elasticsearch",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "terraform",
        "ansible", "devops", "ci/cd",
        # Data Science & AI
        "machine learning", "deep learning", "artificial intelligence", "data analysis",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "nlp",
        # Other Technical Skills
        "git", "rest api", "graphql", "microservices", "linux", "agile", "scrum"
    }
    
    soft_patterns = {
        # Communication
        "communication", "presentation", "public speaking", "writing", "listening",
        # Leadership
        "leadership", "team management", "mentoring", "coaching", "strategic thinking",
        # Collaboration
        "teamwork", "collaboration", "interpersonal", "relationship building",
        # Problem Solving
        "problem solving", "analytical", "critical thinking", "decision making",
        "troubleshooting",
        # Project Management
        "project management", "time management", "organization", "planning",
        "risk management",
        # Other Soft Skills
        "adaptability", "creativity", "innovation", "attention to detail", "multitasking",
        "negotiation", "conflict resolution", "customer service"
    }
    
    found_technical_skills = set()
    found_soft_skills = set()
    
    text_lower = text.lower()
    for skill in technical_patterns:
        if skill in text_lower:
            found_technical_skills.add(skill)
    
    for skill in soft_patterns:
        if skill in text_lower:
            found_soft_skills.add(skill)
    
    return list(found_technical_skills), list(found_soft_skills)

def boost_score(original_score, boost_factor=1.2):
    """Boost the score while keeping it within reasonable bounds"""
    boosted = original_score * boost_factor
    return min(100, max(boosted, original_score))

def calculate_match_score(resume_skills, jd_skills, weight):
    if not jd_skills:
        return 0.0
    
    matched_skills = set(resume_skills) & set(jd_skills)
    base_score = (len(matched_skills) / len(set(jd_skills))) * 100 * weight
    
    # Apply boosting to the base score
    boosted_score = boost_score(base_score)
    return min(100 * weight, boosted_score)

def analyze_resume(resume_text, job_description, nlp):
    try:
        resume_tech_skills, resume_soft_skills = extract_skills(resume_text, nlp)
        jd_tech_skills, jd_soft_skills = extract_skills(job_description, nlp)
        
        weights = {
            'technical': 0.8,
            'soft': 0.2
        }
        
        tech_score = calculate_match_score(resume_tech_skills, jd_tech_skills, weights['technical'])
        soft_score = calculate_match_score(resume_soft_skills, jd_soft_skills, weights['soft'])
        
        # Apply additional boosting for overall score
        overall_score = min(100, boost_score(tech_score + soft_score, 1.15))
        
        tech_match_percent = boost_score((len(set(resume_tech_skills) & set(jd_tech_skills)) / 
                            max(len(set(jd_tech_skills)), 1)) * 100)
        soft_match_percent = boost_score((len(set(resume_soft_skills) & set(jd_soft_skills)) / 
                            max(len(set(jd_soft_skills)), 1)) * 100)
        
        missing_tech_skills = list(set(jd_tech_skills) - set(resume_tech_skills))
        missing_soft_skills = list(set(jd_soft_skills) - set(resume_soft_skills))
        
        recommendations = []
        if missing_tech_skills:
            recommendations.append(f"Consider acquiring these technical skills: {', '.join(missing_tech_skills)}")
        if missing_soft_skills:
            recommendations.append(f"Demonstrate these soft skills: {', '.join(missing_soft_skills)}")
        if tech_match_percent < 75:
            recommendations.append("Focus on gaining more relevant technical skills for this position")
        if soft_match_percent < 75:
            recommendations.append("Emphasize soft skills more in your resume")
        
        if overall_score >= 80:
            assessment = "Excellent match! Your profile strongly aligns with the job requirements."
        elif overall_score >= 65:
            assessment = "Good match! Your profile aligns well with most job requirements."
        elif overall_score >= 50:
            assessment = "Moderate match. Consider improving in the suggested areas."
        else:
            assessment = "Additional skill development recommended to better match the job requirements."
        
        return {
            "match_score": round(overall_score),
            "key_matches": [
                f"Technical skills match: {tech_match_percent:.1f}%",
                f"Soft skills match: {soft_match_percent:.1f}%",
                f"Matched technical skills: {', '.join(sorted(resume_tech_skills))}" if resume_tech_skills else "No technical skills found",
                f"Matched soft skills: {', '.join(sorted(resume_soft_skills))}" if resume_soft_skills else "No soft skills found"
            ],
            "gaps": [
                f"Missing technical skills: {', '.join(sorted(missing_tech_skills))}" if missing_tech_skills else "No major technical skill gaps",
                f"Missing soft skills: {', '.join(sorted(missing_soft_skills))}" if missing_soft_skills else "No major soft skill gaps"
            ],
            "skill_analysis": {
                "technical_skills": {
                    "present": sorted(resume_tech_skills),
                    "missing": sorted(missing_tech_skills)
                },
                "soft_skills": {
                    "present": sorted(resume_soft_skills),
                    "missing": sorted(missing_soft_skills)
                }
            },
            "recommendations": recommendations if recommendations else ["Your profile shows strong alignment with the job requirements"],
            "overall_assessment": f"{assessment} Overall match: {round(overall_score)}%, "
                                f"with technical skills at {tech_match_percent:.1f}% "
                                f"and soft skills at {soft_match_percent:.1f}%"
        }
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

def display_skill_tags(skills, style_class):
    """Display skills as colored tags"""
    if not skills:
        st.write("None")
        return
    
    tags_html = ""
    for skill in skills:
        tags_html += f'<span class="skill-tag {style_class}">{skill}</span>'
    st.markdown(tags_html, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
    set_custom_css()
    
    st.markdown('<h1 class="main-header">🚀 AI-Powered Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Get instant feedback on how well your resume matches the job requirements!")
    
    nlp = load_spacy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📎 Upload Resume")
        pdf_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
        
    with col2:
        st.markdown("### 💼 Job Description")
        jd_text = st.text_area("Paste the job description here")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Resume", use_container_width=True)
    
    if analyze_button:
        if not pdf_file or not jd_text:
            st.error("⚠️ Please provide both resume and job description.")
            return
            
        with st.spinner("🔄 Analyzing your resume..."):
            resume_text = extract_text_from_pdf(pdf_file)
            if not resume_text:
                st.error("📄 Could not extract text from the PDF. Please try another file.")
                return
                
            analysis = analyze_resume(resume_text, jd_text, nlp)
            if not analysis:
                st.error("❌ Analysis failed. Please try again.")
                return
            
            score = analysis.get('match_score', 0)
            
            # Score display with gradient background
            score_class = "high-score" if score >= 80 else "medium-score" if score >= 65 else "low-score"
            st.markdown(f"""
                <div class="score-card {score_class}">
                    <h2>Overall Match Score</h2>
                    <h1>{score}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Create tabs with enhanced styling
            tab1, tab2, tab3 = st.tabs(["💪 Skills Match", "🎯 Areas to Improve", "📋 Recommendations"])
            
            with tab1:
                st.markdown("### 🌟 Present Skills")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Technical Skills")
                    display_skill_tags(analysis['skill_analysis']['technical_skills']['present'], "present-skill")
                with col2:
                    st.markdown("#### Soft Skills")
                    display_skill_tags(analysis['skill_analysis']['soft_skills']['present'], "present-skill")
            
            with tab2:
                st.markdown("### 🎯 Skills to Acquire")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Technical Skills")
                    display_skill_tags(analysis['skill_analysis']['technical_skills']['missing'], "missing-skill")
                with col2:
                    st.markdown("#### Soft Skills")
                    display_skill_tags(analysis['skill_analysis']['soft_skills']['missing'], "missing-skill")
            
            with tab3:
                st.markdown("### 📝 Personalized Recommendations")
                for rec in analysis['recommendations']:
                    st.info(rec)
            
            # Export Option
            st.markdown("---")
            st.markdown("### 📥 Export Your Analysis")
            export_data = {
                "Resume Analysis Report": {
                    "Overall Match": f"{score}%",
                    "Assessment": analysis['overall_assessment'],
                    "Key Strengths": analysis['key_matches'],
                    "Areas for Improvement": analysis['gaps'],
                    "Skills Analysis": analysis['skill_analysis'],
                    "Recommendations": analysis['recommendations']
                }
            }
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=json.dumps(export_data, indent=2),
                    file_name="resume_analysis_report.json",
                    mime="application/json",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()