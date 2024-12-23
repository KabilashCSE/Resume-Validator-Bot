import streamlit as st
import requests
from typing import Optional

# MeaningCloud API configuration
API_KEY = "525dd7fb60fca5ac54a49640708396eb"
API_URL = "https://api.meaningcloud.com/summarization-1.0"

def get_summary(text: str, num_sentences: int = 3) -> Optional[str]:
    try:
        payload = {
            'key': API_KEY,
            'txt': text,
            'sentences': str(num_sentences)
        }
        
        response = requests.post(API_URL, data=payload)
        response.raise_for_status()
        
        result = response.json()
        if response.status_code == 200 and 'summary' in result:
            return result['summary']
        return None
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None

# Streamlit Interface
st.title("Text Summarizer")

input_text = st.text_area("Enter your text here:", height=200)
num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)

if st.button("Generate Summary"):
    if input_text:
        with st.spinner("Processing..."):
            summary = get_summary(input_text, num_sentences)
            if summary:
                st.subheader("Generated Summary:")
                st.write(summary)
    else:
        st.warning("Please enter some text to summarize")