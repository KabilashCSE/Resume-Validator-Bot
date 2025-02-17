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
import openai


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
def extract_certifications(text, nlp):
    """Extract certifications from text"""
    doc = nlp(text)
    certifications = []
    cert_keywords = ["certified", "certification", "certificate", "AWS", "PMP", "CISSP", "CISA"]
    
    for sent in doc.sents:
        if any(keyword.lower() in sent.text.lower() for keyword in cert_keywords):
            certifications.append(sent.text.strip())
    return list(set(certifications))

def extract_experience(text, nlp):
    """Extract work experience details"""
    doc = nlp(text)
    experience = []
    exp_keywords = ["experience", "worked", "work", "years", "position", "role"]
    
    for sent in doc.sents:
        if any(keyword.lower() in sent.text.lower() for keyword in exp_keywords):
            experience.append(sent.text.strip())
    return list(set(experience))
def analyze_resume(resume_text, job_description, nlp):
    try:
        # Extract skills, certifications and experience
        resume_tech_skills, resume_soft_skills = extract_skills(resume_text, nlp)
        jd_tech_skills, jd_soft_skills = extract_skills(job_description, nlp)
        
        certifications = extract_certifications(resume_text, nlp)
        experience = extract_experience(resume_text, nlp)
        
        # Updated weights
        weights = {
            'technical': 0.4,
            'soft': 0.2,
            'certifications': 0.2,
            'experience': 0.2
        }
        
        # Calculate scores
        technical_score = len(set(resume_tech_skills) & set(jd_tech_skills)) / max(len(jd_tech_skills), 1) * 100
        soft_score = len(set(resume_soft_skills) & set(jd_soft_skills)) / max(len(jd_soft_skills), 1) * 100
        cert_score = min(len(certifications) * 25, 100)  # 25 points per certification, max 100
        exp_score = min(len(experience) * 20, 100)  # 20 points per experience entry, max 100
        
        # Calculate weighted score
        total_score = (
            technical_score * weights['technical'] +
            soft_score * weights['soft'] +
            cert_score * weights['certifications'] +
            exp_score * weights['experience']
        )
        
        return {
            'total_score': total_score,
            'technical_score': technical_score,
            'soft_score': soft_score,
            'certification_score': cert_score,
            'experience_score': exp_score,
            'technical_skills': {
                'matched': list(set(resume_tech_skills) & set(jd_tech_skills)),
                'missing': list(set(jd_tech_skills) - set(resume_tech_skills))
            },
            'soft_skills': {
                'matched': list(set(resume_soft_skills) & set(jd_soft_skills)),
                'missing': list(set(jd_soft_skills) - set(resume_soft_skills))
            },
            'certifications': certifications,
            'experience': experience
        }
        
    except Exception as e:
        print(f"Error in analyze_resume: {str(e)}")
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
    # Get basic skill analysis for both JDs
    original_analysis = analyze_resume(resume_text, original_jd, nlp)
    abstracted_analysis = analyze_resume(resume_text, abstracted_jd, nlp)
    
    # Calculate semantic similarities with higher weight for original JD
    resume_original_similarity = calculate_semantic_similarity(resume_text, original_jd, nlp)
    resume_abstracted_similarity = calculate_semantic_similarity(resume_text, abstracted_jd, nlp)
    jd_similarity = calculate_semantic_similarity(original_jd, abstracted_jd, nlp)
    
    # Adjust scores based on semantic similarity and specificity
    if original_analysis and abstracted_analysis:
        # Original JD score gets a boost for specific matches
        original_score = original_analysis['match_score'] * (1 + resume_original_similarity * 0.2)
        original_score = min(100, original_score)  # Cap at 100
        original_analysis['match_score'] = round(original_score)
        
        # Abstracted JD score is weighted differently to reflect broader matching
        abstracted_score = abstracted_analysis['match_score'] * (1 + resume_abstracted_similarity * 0.1)
        abstracted_score = min(100, abstracted_score)  # Cap at 100
        abstracted_analysis['match_score'] = round(abstracted_score)
        
        # Update the overall assessments with the new scores
        original_analysis['overall_assessment'] = update_assessment(original_score)
        abstracted_analysis['overall_assessment'] = update_assessment(abstracted_score)
    
    # Create enhanced analysis results
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
    
    # Extract phrases that often indicate requirements
    requirement_patterns = [
        r'required|requirements?|must have|essential|necessary',
        r'qualifications?|skills?|experiences?|expertise',
        r'proficiency|proficient|knowledge of|familiarity with',
        r'ability to|capable of|responsible for'
    ]
    
    requirements = []
    for sentence in doc.sents:
        for pattern in requirement_patterns:
            if re.search(pattern, sentence.text.lower()):
                # Clean and normalize the requirement
                req = re.sub(r'\s+', ' ', sentence.text.strip())
                requirements.append(req)
                break
    
    # Deduplicate and limit to most important requirements
    return list(set(requirements))[:5]  
def initialize_openai():
    """Initialize OpenAI client with hardcoded API key"""
    try:
        # Hardcoded OpenAI API key - Replace with your actual key
        api_key = "#"
        openai.api_key = api_key  # Set the API key for the OpenAI module
        return openai  # Return the openai module itself
    except Exception as e:
        st.error(f"Error initializing OpenAI: {str(e)}")
        return None

def generate_abstracted_jd_with_gpt(original_jd):
    """Generate an abstracted job description using OpenAI's GPT model"""
    client = initialize_openai()
    if not client:
        return None

    try:
        prompt = f"""
        Convert this job description into a generalized version that maintains the core requirements 
        while expressing them in broader, more abstract terms. Create a flowing paragraph that 
        captures the essence of the role without specific technical details.

        Original JD: {original_jd}

        Create a professional but less rigid description that:
        1. Generalizes technical skills into broader categories
        2. Maintains core responsibilities and experience level
        3. Expresses requirements in more universal terms
        4. Keeps the essence while being less specific
        5. Makes it more accessible to a wider audience
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional job description writer who specializes in creating clear, abstracted job descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Format the response
        abstracted_jd = response.choices[0].message.content.strip()
        return abstracted_jd
    except Exception as e:
        st.error(f"Error generating abstracted JD: {str(e)}")
        return None 

def main():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
    set_custom_css()
    
    st.markdown('<h1 class="main-header">🚀 AI-Powered Resume Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize NLTK resources at startup
    with st.spinner("Initializing NLTK resources..."):
        if not download_nltk_resources():
            st.error("Failed to initialize required resources. Some features may be limited.")
    
    nlp = load_spacy()
    
    # Initialize session state
    if 'jd_generated' not in st.session_state:
        st.session_state.jd_generated = False
    if 'abstracted_jd' not in st.session_state:
        st.session_state.abstracted_jd = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📎 Upload Resume")
        pdf_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
        
    with col2:
        st.markdown("### 🔒 Original Job Description")
        original_jd = st.text_area("Paste the original job description", key="original_jd")
        
        # Create Public JD button
        if original_jd and not st.session_state.jd_generated:
            if st.button("🔄 Create Public JD", use_container_width=True):
                with st.spinner("Generating abstracted JD..."):
                    abstracted_jd = generate_abstracted_jd_with_gpt(original_jd)
                    if abstracted_jd:
                        st.session_state.abstracted_jd = abstracted_jd
                        st.session_state.jd_generated = True
                        st.experimental_rerun()
        
        # Show abstracted JD if generated
        if st.session_state.jd_generated and st.session_state.abstracted_jd:
            st.markdown("### 📝 Public Job Description")
            st.write(st.session_state.abstracted_jd)
            st.download_button(
                label="📥 Download Public JD",
                data=st.session_state.abstracted_jd,
                file_name="public_jd.txt",
                mime="text/plain"
            )
    
    # Only show Analyze Resume button after JD is generated
    if st.session_state.jd_generated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Resume", use_container_width=True)
            
        if analyze_button and original_jd:
            if not pdf_file:
                st.error("⚠️ Please upload your resume to continue.")
                return
                
            with st.spinner("🔄 Analyzing your resume..."):
                resume_text = extract_text_from_pdf(pdf_file)
                if not resume_text:
                    st.error("📄 Could not extract text from the PDF. Please try again.")
                    return
                
                dual_analysis = analyze_resume_with_dual_jd(
                    resume_text, 
                    original_jd, 
                    st.session_state.abstracted_jd, 
                    nlp
                )
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📊 Original JD Analysis", "🎯 Public JD Analysis", "🔄 Comparison"])
                
                with tab1:
                    display_analysis_results(
                        dual_analysis["original_jd"]["analysis"],
                        dual_analysis["original_jd"]["semantic_match"],
                        "Original JD"
                    )
                
                with tab2:
                    display_analysis_results(
                        dual_analysis["abstracted_jd"]["analysis"],
                        dual_analysis["abstracted_jd"]["semantic_match"],
                        "Public JD"
                    )
                    
                with tab3:
                    display_comparison_results(dual_analysis)


def display_comparison_results(dual_analysis):
    """Helper function to display comparison results"""
    st.markdown("### 📊 JD Comparison Analysis")
    st.metric("JD Similarity Score", f"{dual_analysis['jd_comparison']['similarity']}%")
    
    st.markdown("### 🔍 Comparative Insights")
    for rec in dual_analysis["jd_comparison"]["recommendations"]:
        st.info(rec)
    
    # Export functionality
    export_data = {
        "Resume Analysis Report": {
            "Original JD Analysis": dual_analysis["original_jd"],
            "Public JD Analysis": dual_analysis["abstracted_jd"],
            "Comparison Analysis": dual_analysis["jd_comparison"]
        }
    }
    
    st.download_button(
        label="📥 Download Complete Analysis Report",
        data=json.dumps(export_data, indent=2),
        file_name="complete_resume_analysis_report.json",
        mime="application/json",
        use_container_width=True
    )
def display_analysis_results(analysis, semantic_score, jd_type):
    """Helper function to display analysis results for each JD type"""
    score = analysis.get('match_score', 0)
    score_class = "high-score" if score >= 80 else "medium-score" if score >= 65 else "low-score"
    
    st.markdown(f"""
        <div class="score-card {score_class}">
            <h2>{jd_type} Match Score</h2>
            <h1>{score}%</h1>
            <h3>Semantic Match: {semantic_score}%</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Create subtabs for detailed analysis
    subtab1, subtab2, subtab3 = st.tabs(["💪 Skills Match", "🎯 Areas to Improve", "📋 Recommendations"])
    
    with subtab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Technical Skills")
            display_skill_tags(analysis['skill_analysis']['technical_skills']['present'], "present-skill")
        with col2:
            st.markdown("#### Soft Skills")
            display_skill_tags(analysis['skill_analysis']['soft_skills']['present'], "present-skill")
    
    with subtab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Missing Technical Skills")
            display_skill_tags(analysis['skill_analysis']['technical_skills']['missing'], "missing-skill")
        with col2:
            st.markdown("#### Missing Soft Skills")
            display_skill_tags(analysis['skill_analysis']['soft_skills']['missing'], "missing-skill")
    
    with subtab3:
        for rec in analysis['recommendations']:
            st.info(rec)

if __name__ == "__main__":
    main()