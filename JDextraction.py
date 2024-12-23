import streamlit as st
import subprocess
import re

# Function to query Ollama model using subprocess
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

# Function to extract fields from JD
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
        # Display raw result for debugging
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

        # Return the cleaned up extracted fields
        return fields
    
    except Exception as e:
        st.error(f"Failed to parse JD fields: {e}")
        return fields

# Streamlit UI
def main():
    st.set_page_config(page_title="JD Extractor", layout="wide")
    st.title("Job Description Field Extractor")

    # JD Input
    jd_text = st.text_area("Enter Job Description (as a paragraph)", height=200)

    # Extract Fields Button
    if st.button("Extract Fields"):
        if not jd_text.strip():
            st.error("Please enter the job description.")
        else:
            try:
                # Extract fields from JD
                with st.spinner("Extracting fields..."):
                    fields = extract_fields_from_jd(jd_text)

                # Display extracted fields in JSON format
                if fields:
                    st.subheader("Extracted Job Description Fields (in JSON format)")

                    # Convert fields to JSON and display
                    st.json(fields)

                else:
                    st.write("No fields were extracted.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
