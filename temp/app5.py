import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re
from nltk.corpus import wordnet
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

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

def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
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
    
    # Define technical and soft skill patterns
    technical_patterns = {
        # Programming Languages
        "python", "java", "javascript", "c++", "ruby", "php", "swift", "kotlin", "go",
        # Web Technologies
        "html", "css", "react", "angular", "vue.js", "node.js", "express.js", "django",
        "flask", "spring boot", "asp.net","ReactJS","React.js","NodeJS","Node.js"
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
    
    # Extract technical skills
    for skill in technical_patterns:
        if skill in text_lower:
            found_technical_skills.add(skill)
    
    # Extract soft skills
    for skill in soft_patterns:
        if skill in text_lower:
            found_soft_skills.add(skill)

    return list(found_technical_skills), list(found_soft_skills)
def calculate_proj_exp_score(text):
    # Keywords for projects and experience sections (you can customize these)
    proj_keywords = ["project", "collaboration", "team", "leadership", "development", "design", "architecture"]
    exp_keywords = ["experience", "role", "responsible", "worked", "achieved", "managed", "participated"]

    proj_score = 0
    exp_score = 0

    text_lower = text.lower()

    # Calculate project score based on proj_keywords
    for keyword in proj_keywords:
        if keyword in text_lower:
            proj_score += 1  # Increment score for each found keyword

    # Calculate experience score based on exp_keywords
    for keyword in exp_keywords:
        if keyword in text_lower:
            exp_score += 1  # Increment score for each found keyword

    return proj_score, exp_score



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
        # Extract skills and experience from both resume and job description
        resume_tech_skills, resume_soft_skills = extract_skills(resume_text, nlp)
        jd_tech_skills, jd_soft_skills = extract_skills(job_description, nlp)
        
        # Extract projects and experience (assuming these functions exist or need to be created)
        resume_projects, resume_experience = calculate_proj_exp_score(resume_text)
        jd_projects, jd_experience = calculate_proj_exp_score(job_description)
        
        # Define weight distribution for different components
        weights = {
            'technical': 0.4,
            'soft': 0.2,
            'proj': 0.2,
            'exp': 0.2
        }
        
        # Calculate scores for each component
        tech_score = calculate_match_score(resume_tech_skills, jd_tech_skills, weights['technical'])
        soft_score = calculate_match_score(resume_soft_skills, jd_soft_skills, weights['soft'])
        proj_score = calculate_match_score(resume_projects, jd_projects, weights['proj'])
        exp_score = calculate_match_score(resume_experience, jd_experience, weights['exp'])
        
        # Apply additional boosting for overall score
        overall_score = min(100, boost_score(tech_score + soft_score + proj_score + exp_score, 1.15))
        
        # Calculate match percentages for technical and soft skills
        tech_match_percent = boost_score((len(set(resume_tech_skills) & set(jd_tech_skills)) / 
                                          max(len(set(jd_tech_skills)), 1)) * 100)
        soft_match_percent = boost_score((len(set(resume_soft_skills) & set(jd_soft_skills)) / 
                                          max(len(set(jd_soft_skills)), 1)) * 100)
        
        # Calculate match percentages for projects and experience
        proj_match_percent = boost_score((len(set(resume_projects) & set(jd_projects)) / 
                                          max(len(set(jd_projects)), 1)) * 100)
        exp_match_percent = boost_score((len(set(resume_experience) & set(jd_experience)) / 
                                         max(len(set(jd_experience)), 1)) * 100)
        
        # Identify missing skills, projects, and experience
        missing_tech_skills = list(set(jd_tech_skills) - set(resume_tech_skills))
        missing_soft_skills = list(set(jd_soft_skills) - set(resume_soft_skills))
        missing_projects = list(set(jd_projects) - set(resume_projects))
        missing_experience = list(set(jd_experience) - set(resume_experience))
        
        # Prepare recommendations based on the gaps
        recommendations = []
        if missing_tech_skills:
            recommendations.append(f"Consider acquiring these technical skills: {', '.join(missing_tech_skills)}")
        if missing_soft_skills:
            recommendations.append(f"Demonstrate these soft skills: {', '.join(missing_soft_skills)}")
        if missing_projects:
            recommendations.append(f"Gain experience with these types of projects: {', '.join(missing_projects)}")
        if missing_experience:
            recommendations.append(f"Add more relevant experience in these areas: {', '.join(missing_experience)}")
        if tech_match_percent < 75:
            recommendations.append("Focus on gaining more relevant technical skills for this position")
        if soft_match_percent < 75:
            recommendations.append("Emphasize soft skills more in your resume")
        if proj_match_percent < 75:
            recommendations.append("Include more relevant projects to strengthen your profile")
        if exp_match_percent < 75:
            recommendations.append("Highlight more relevant experience in your resume")
        
        # Determine overall assessment based on the score
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
                f"Projects match: {proj_match_percent:.1f}%",
                f"Experience match: {exp_match_percent:.1f}%",
                f"Matched technical skills: {', '.join(sorted(resume_tech_skills))}" if resume_tech_skills else "No technical skills found",
                f"Matched soft skills: {', '.join(sorted(resume_soft_skills))}" if resume_soft_skills else "No soft skills found",
                f"Matched projects: {', '.join(sorted(resume_projects))}" if resume_projects else "No projects found",
                f"Matched experience: {', '.join(sorted(resume_experience))}" if resume_experience else "No experience found"
            ],
            "gaps": [
                f"Missing technical skills: {', '.join(sorted(missing_tech_skills))}" if missing_tech_skills else "No major technical skill gaps",
                f"Missing soft skills: {', '.join(sorted(missing_soft_skills))}" if missing_soft_skills else "No major soft skill gaps",
                f"Missing projects: {', '.join(sorted(missing_projects))}" if missing_projects else "No major project gaps",
                f"Missing experience: {', '.join(sorted(missing_experience))}" if missing_experience else "No major experience gaps"
            ],
            "skill_analysis": {
                "technical_skills": {
                    "present": sorted(resume_tech_skills),
                    "missing": sorted(missing_tech_skills)
                },
                "soft_skills": {
                    "present": sorted(resume_soft_skills),
                    "missing": sorted(missing_soft_skills)
                },
                "projects": {
                    "present": sorted(resume_projects),
                    "missing": sorted(missing_projects)
                },
                "experience": {
                    "present": sorted(resume_experience),
                    "missing": sorted(missing_experience)
                }
            },
            "recommendations": recommendations if recommendations else ["Your profile shows strong alignment with the job requirements"],
            "overall_assessment": f"{assessment} Overall match: {round(overall_score)}%, "
                                f"with technical skills at {tech_match_percent:.1f}%, "
                                f"soft skills at {soft_match_percent:.1f}%, "
                                f"projects at {proj_match_percent:.1f}%, "
                                f"and experience at {exp_match_percent:.1f}%"
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
def calculate_semantic_similarity(text1, text2, nlp):
    """Calculate semantic similarity between two texts using spaCy"""
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def analyze_resume_with_dual_jd(resume_text, original_jd, abstracted_jd, nlp):
    """Analyze resume against both original and abstracted JDs with differentiated scoring"""
    try:
        # Get basic skill analysis for both JDs
        original_analysis = analyze_resume(resume_text, original_jd, nlp)
        abstracted_analysis = analyze_resume(resume_text, abstracted_jd, nlp)
        
        # Validate analysis results
        if not original_analysis or not isinstance(original_analysis, dict):
            original_analysis = create_empty_analysis()
        if not abstracted_analysis or not isinstance(abstracted_analysis, dict):
            abstracted_analysis = create_empty_analysis()
        
        # Calculate semantic similarities
        resume_original_similarity = calculate_semantic_similarity(resume_text, original_jd, nlp)
        resume_abstracted_similarity = calculate_semantic_similarity(resume_text, abstracted_jd, nlp)
        jd_similarity = calculate_semantic_similarity(original_jd, abstracted_jd, nlp)
        
        return {
            "original_jd": {
                "analysis": original_analysis,
                "semantic_match": round(resume_original_similarity * 100, 2),
            },
            "abstracted_jd": {
                "analysis": abstracted_analysis,
                "semantic_match": round(resume_abstracted_similarity * 100, 2),
            },
            "jd_comparison": {
                "similarity": round(jd_similarity * 100, 2),
                "recommendations": generate_jd_comparison_recommendations(
                    original_analysis,
                    abstracted_analysis,
                    resume_original_similarity,
                    resume_abstracted_similarity,
                    jd_similarity   
                )
            }
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return create_empty_dual_analysis()

def create_empty_analysis():
    """Create an empty analysis structure with default values"""
    return {
        "match_score": 0,
        "skill_analysis": {
            "technical_skills": {"present": [], "missing": []},
            "soft_skills": {"present": [], "missing": []}
        },
        "recommendations": ["Unable to generate recommendations"],
        "overall_assessment": "Analysis failed"
    }

def create_empty_dual_analysis():
    """Create an empty dual analysis structure"""
    empty = create_empty_analysis()
    return {
        "original_jd": {
            "analysis": empty,
            "semantic_match": 0,
        },
        "abstracted_jd": {
            "analysis": empty,
            "semantic_match": 0,
        },
        "jd_comparison": {
            "similarity": 0,
            "recommendations": ["Analysis failed"]
        }
    }

def display_analysis_results(analysis, semantic_score, jd_type):
    """Helper function to display analysis results for each JD type"""
    try:
        if not isinstance(analysis, dict):
            st.error(f"Invalid analysis format for {jd_type}")
            analysis = create_empty_analysis()
        
        score = analysis.get('match_score', 0)
        score_class = "high-score" if score >= 80 else "medium-score" if score >= 65 else "low-score"
        
        st.markdown(f"""
            <div class="score-card {score_class}">
                <h2>{jd_type} Match Score</h2>
                <h1>{score}%</h1>
                <h3>Semantic Match: {semantic_score}%</h3>
            </div>
        """, unsafe_allow_html=True)
        
        subtab1, subtab2, subtab3 = st.tabs(["💪 Skills Match", "🎯 Areas to Improve", "📋 Recommendations"])
        
        with subtab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Technical Skills")
                display_skill_tags(analysis.get('skill_analysis', {}).get('technical_skills', {}).get('present', []), "present-skill")
            with col2:
                st.markdown("#### Soft Skills")
                display_skill_tags(analysis.get('skill_analysis', {}).get('soft_skills', {}).get('present', []), "present-skill")
        
        with subtab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Missing Technical Skills")
                display_skill_tags(analysis.get('skill_analysis', {}).get('technical_skills', {}).get('missing', []), "missing-skill")
            with col2:
                st.markdown("#### Missing Soft Skills")
                display_skill_tags(analysis.get('skill_analysis', {}).get('soft_skills', {}).get('missing', []), "missing-skill")
        
        with subtab3:
            for rec in analysis.get('recommendations', ["No recommendations available"]):
                st.info(rec)
                
    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")

def update_assessment(score):
    """Generate an updated assessment message based on the new scoring system"""
    if score >= 85:
        return f"Excellent match! Your profile strongly aligns with the job requirements. Overall match: {round(score)}%"
    elif score >= 70:
        return f"Good match! Your profile aligns well with most job requirements. Overall match: {round(score)}%"
    elif score >= 55:
        return f"Moderate match. Consider improving in the suggested areas. Overall match: {round(score)}%"
    else:
        return f"Additional skill development recommended to better match the job requirements. Overall match: {round(score)}%"


def generate_jd_comparison_recommendations(orig_analysis, abs_analysis, orig_sim, abs_sim, jd_sim):
    """Generate enhanced recommendations based on comparing both JD analyses"""
    recommendations = []
    
    if not orig_analysis or not abs_analysis:
        return ["Unable to generate recommendations due to incomplete analysis."]
    
    # Compare skill matches with more nuanced thresholds
    orig_score = orig_analysis.get('match_score', 0)
    abs_score = abs_analysis.get('match_score', 0)
    
    score_diff = abs(orig_score - abs_score)
    
    if score_diff <= 5:
        recommendations.append("✨ Your resume shows consistent matching across both specific and general job requirements.")
    elif orig_score > abs_score + 15:
        recommendations.append("✅ Strong alignment with specific role requirements - this is excellent for specialized positions.")
        if orig_score - abs_score > 25:
            recommendations.append("💡 Your expertise is highly specialized for this role.")
    elif abs_score > orig_score + 15:
        recommendations.append("⚠️ Your resume matches general industry requirements but may need more specific role-related keywords.")
        recommendations.append("💡 Consider tailoring your resume to include more role-specific terminology and experiences.")
    
    # Analyze semantic similarities with enhanced insights
    sem_diff = abs(orig_sim - abs_sim)
    if sem_diff > 0.2:
        if orig_sim > abs_sim:
            recommendations.append("💫 Your resume's language strongly matches the specific job description - great for direct applications.")
        else:
            recommendations.append("📝 Consider revising your resume to better match the specific terminology used in the job description.")
    
    # JD similarity analysis with actionable insights
    if jd_sim < 0.5:
        recommendations.append("⚠️ The job description contains unique requirements that may need special attention in your application.")
        if orig_score < 70:
            recommendations.append("💡 Focus on addressing the specific technical requirements mentioned in the original job description.")
    
    return recommendations
def download_nltk_resources():
    """Download required NLTK resources with proper error handling"""
    required_resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]
    
    for resource in required_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            st.error(f"Error downloading NLTK resource {resource}: {str(e)}")
            return False
    return True

def get_wordnet_synonyms(word):
    """Get broader/more general synonyms for a word"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        # Get hypernyms (more general terms)
        for hypernym in syn.hypernyms():
            synonyms.update([lemma.name() for lemma in hypernym.lemmas()])
        # Get similar general terms
        for similar in syn.similar_tos():
            synonyms.update([lemma.name() for lemma in similar.lemmas()])
    return list(synonyms)

def abstract_technical_terms(text, nlp):
    """Replace specific technical terms with more general categories"""
    tech_categories = {
        'programming_languages': [
            'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift', 'kotlin',
            'golang', 'scala', 'rust', 'typescript'
        ],
        'frameworks': [
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel',
            'express', 'node.js', 'nextjs', 'nuxt'
        ],
        'databases': [
            'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'redis',
            'cassandra', 'elasticsearch', 'dynamodb'
        ],
        'cloud_services': [
            'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
            'kubernetes', 'docker', 'terraform'
        ]
    }
    
    abstracted_text = text.lower()
    for category, terms in tech_categories.items():
        for term in terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            abstracted_text = re.sub(pattern, category, abstracted_text)
    
    return abstracted_text

def abstract_job_description(original_jd, nlp):
    """Generate an abstracted version of the job description"""
    try:
        # Ensure NLTK resources are downloaded
        if not download_nltk_resources():
            st.error("Failed to download required NLTK resources. Please try again.")
            return original_jd
        
        # First pass: Abstract technical terms
        abstracted_jd = abstract_technical_terms(original_jd, nlp)
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(abstracted_jd)
        except LookupError:
            # Fallback to simple sentence splitting if NLTK fails
            sentences = [s.strip() for s in abstracted_jd.split('.') if s.strip()]
        
        abstracted_sentences = []
        
        for sentence in sentences:
            try:
                # Tokenize and tag parts of speech
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                
                # Replace specific terms with more general ones
                abstracted_tokens = []
                for word, tag in tagged:
                    if tag.startswith('NN'):  # Nouns
                        try:
                            synonyms = get_wordnet_synonyms(word)
                            if synonyms:
                                # Use the first, most common synonym
                                abstracted_tokens.append(synonyms[0])
                            else:
                                abstracted_tokens.append(word)
                        except Exception:
                            abstracted_tokens.append(word)
                    else:
                        abstracted_tokens.append(word)
                
                abstracted_sentences.append(' '.join(abstracted_tokens))
            except Exception as e:
                # If processing fails for a sentence, keep it unchanged
                abstracted_sentences.append(sentence)
        
        # Join sentences and clean up the text
        abstracted_jd = ' '.join(abstracted_sentences)
        
        # Clean up formatting
        abstracted_jd = re.sub(r'\s+', ' ', abstracted_jd)  # Remove extra whitespace
        abstracted_jd = re.sub(r'_', ' ', abstracted_jd)    # Replace underscores with spaces
        abstracted_jd = abstracted_jd.capitalize()          # Capitalize first letter
        
        # Add a summary section
        try:
            key_requirements = extract_key_requirements(original_jd, nlp)
            abstracted_jd += "\n\nKey Requirements:\n" + "\n".join([f"- {req}" for req in key_requirements])
        except Exception as e:
            st.warning("Could not extract key requirements. Continuing with basic abstraction.")
        
        return abstracted_jd
        
    except Exception as e:
        st.error(f"Error in JD abstraction: {str(e)}")
        return original_jd 

def extract_key_requirements(text, nlp):
    """Extract key requirements from the job description"""
    doc = nlp(text)
    
    # Define weights for different categories
    weights = {
        'skills': 0.3,
        'qualifications': 0.2,
        'certifications': 0.25,
        'experience': 0.25
    }
    
    # Extract phrases that often indicate requirements
    requirement_patterns = [
        r'required|requirements?|must have|essential|necessary',
        r'qualifications?|skills?|experiences?|expertise',
        r'proficiency|proficient|knowledge of|familiarity with',
        r'ability to|capable of|responsible for',
        r'certifications?|certified|accredited|licensed',
        r'years? of experience|\d+ years?|\d+\+ years?|work experience'
    ]
    
    requirements = {
        'skills': [],
        'qualifications': [],
        'certifications': [],
        'experience': []
    }
    
    def calculate_score(matched_requirements):
        scores = {}
        for category, reqs in matched_requirements.items():
            score = len(reqs) * weights[category]
            scores[category] = score
        
        overall_score = sum(scores.values())
        return scores, overall_score

    # Generate PDF report
    def generate_pdf_report(requirements, scores, overall_score):
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Job Requirements Analysis Report', 0, 1, 'C')
        
        pdf.set_font('Arial', '', 12)
        for category, reqs in requirements.items():
            pdf.cell(0, 10, f'\n{category.title()} (Score: {scores[category]:.2f})', 0, 1)
            for req in reqs:
                pdf.cell(0, 10, f'- {req}', 0, 1)
        
        pdf.cell(0, 10, f'\nOverall Score: {overall_score:.2f}', 0, 1)
        pdf.output('requirements_analysis.pdf')

    try:
        # Extract requirements logic here
        # ... (keep existing matching logic)
        
        scores, overall_score = calculate_score(requirements)
        generate_pdf_report(requirements, scores, overall_score)
        
        return {
            'requirements': requirements,
            'scores': scores,
            'overall_score': overall_score,
            'report_path': 'requirements_analysis.pdf'
        }
    
    except Exception as e:
        st.error(f"Error in requirements extraction: {str(e)}")
        return None

def main():
    st.title("📊 Resume Analysis System")
    nlp = load_spacy()
    
    # Initialize session state
    if 'jd_generated' not in st.session_state:
        st.session_state.jd_generated = False
        st.session_state.jd_text = ""

    # Step 1: Job Description Section
    st.markdown("### 💼 Job Description Generator")
    job_title = st.text_input("Enter Job Title")
    job_level = st.selectbox("Select Job Level", 
                            ["Entry Level", "Mid Level", "Senior Level"])
    
    if st.button("Generate JD"):
        # Generate JD based on title and level
        st.session_state.jd_text = generate_job_description(job_title, job_level)
        st.session_state.jd_generated = True
        st.text_area("Generated Job Description", st.session_state.jd_text, height=200)

    # Step 2: Resume Upload (only shown after JD is generated)
    if st.session_state.jd_generated:
        st.markdown("### 📎 Resume Analysis")
        pdf_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
        
        if pdf_file and st.button("Analyze Resume"):
            resume_text = extract_text_from_pdf(pdf_file)
            
            # Extract all components
            tech_skills = extract_technical_skills(resume_text)
            soft_skills = extract_soft_skills(resume_text)
            certifications = extract_certifications(resume_text)
            experience_years = extract_experience(resume_text)
            
            # Calculate scores
            scores = {
                'Technical Skills': calculate_skill_match(tech_skills, st.session_state.jd_text) * 0.2,
                'Soft Skills': calculate_skill_match(soft_skills, st.session_state.jd_text) * 0.2,
                'Experience': min(experience_years/10, 1) * 0.2,
                'Certifications': min(len(certifications)/5, 1) * 0.2,
                'Education': calculate_education_score(resume_text) * 0.2
            }
            
            total_score = sum(scores.values())
            
            # Display Results
            st.markdown("### 📊 Analysis Results")
            st.progress(total_score)
            st.write(f"Overall Match Score: {total_score:.2%}")
            
            # Display detailed scores
            for category, score in scores.items():
                st.write(f"{category}: {score:.2%}")
            
            # Export data preparation
            export_data = {
                "scores": scores,
                "total_score": total_score,
                "details": {
                    "technical_skills": list(tech_skills),
                    "soft_skills": list(soft_skills),
                    "certifications": certifications,
                    "experience_years": experience_years
                }
            }
            
            # Download button
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

def display_comparison_tab(dual_analysis):
    """Helper function to safely display comparison tab content"""
    try:
        st.markdown("### 📊 JD Comparison Analysis")
        
        similarity_score = dual_analysis.get('jd_comparison', {}).get('similarity', 0)
        st.metric("JD Similarity Score", f"{similarity_score}%")
        
        st.markdown("### 🔍 Comparative Insights")
        recommendations = dual_analysis.get('jd_comparison', {}).get('recommendations', [])
        
        if isinstance(recommendations, list) and recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.warning("No comparative insights available")
            
        # Export functionality
        if isinstance(dual_analysis, dict):
            export_data = {
                "Resume Analysis Report": {
                    "Original JD Analysis": dual_analysis.get('original_jd', {}),
                    "Abstracted JD Analysis": dual_analysis.get('abstracted_jd', {}),
                    "Comparison Results": dual_analysis.get('jd_comparison', {})
                }
            }
            st.download_button(
                label="������ Download Analysis Report",
                data=json.dumps(export_data, indent=2),
                file_name="resume_analysis_report.json",
                mime="application/json"
            )
    except Exception as e:
        st.error(f"Error displaying comparison: {str(e)}")