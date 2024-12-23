import streamlit as st
from gradio_client import Client
import time

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the Gradio client with the full Space URL
try:
    client = Client(
        "https://kabilash10-huggingfacetb-smollm2-1-7b-instruct.hf.space",
        hf_token="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"
    )
except Exception as e:
    st.error(f"Failed to initialize client: {str(e)}")
    st.stop()

# Streamlit app
st.title("Chat Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input text box
user_input = st.chat_input("You:")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    try:
        # Show spinner while processing
        with st.spinner("Thinking..."):
            result = client.predict(
                message=user_input,
                api_name="/chat"
            )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(result)
                
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")