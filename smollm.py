import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
from typing import Tuple, Optional
import gc  # For manual garbage collection
import psutil  # For system information

# Page configuration
st.set_page_config(
    page_title="SmolLM Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize debug container in sidebar
with st.sidebar:
    st.title("Debug Info")
    debug_container = st.empty()

def update_debug(message: str) -> None:
    """Update debug message in sidebar."""
    with debug_container:
        st.write(f"🔍 {message}")

@st.cache_resource
def load_model() -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Load the model and tokenizer with optimized settings for CPU.
    """
    try:
        update_debug("Loading model and tokenizer...")
        checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        
        # Load tokenizer with minimal settings
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            model_max_length=512,  # Limit context size
            padding_side='left'
        )
        tokenizer.pad_token = tokenizer.eos_token
        update_debug("Tokenizer loaded successfully")
        
        # Load model with optimized settings for CPU
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        model.eval()  # Set to evaluation mode
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        update_debug("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        update_debug(f"❌ Error loading model: {str(e)}")
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def format_chat_prompt(messages: list) -> str:
    """Format the chat messages into a single prompt."""
    formatted_messages = []
    for message in messages[-3:]:  # Only use last 3 messages for context
        role = message["role"]
        content = message["content"]
        if role == "user":
            formatted_messages.append(f"Human: {content}")
        elif role == "assistant":
            formatted_messages.append(f"Assistant: {content}")
    formatted_messages.append("Assistant:")
    return "\n".join(formatted_messages)

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list,
    max_new_tokens: int = 50  # Reduced for faster responses
) -> str:
    """Generate response with optimized settings."""
    try:
        # Format prompt
        prompt = format_chat_prompt(messages)
        
        # Tokenize with efficient settings
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,  # Reduced context size
            return_attention_mask=True
        )
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Lower temperature for faster, more focused responses
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                num_beams=3,  # No beam search for faster generation
                early_stopping=True
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clear memory
        del outputs
        gc.collect()
        
        return response.strip()
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface
st.title("SmolLM Chat Interface")

# Load model and tokenizer
model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.error("Failed to initialize model and tokenizer. Please check debug info.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What would you like to know?"):
        try:
            update_debug("Processing new user input...")
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(model, tokenizer, st.session_state.messages)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    update_debug("Response generated successfully")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
            update_debug(f"❌ {error_msg}")
            st.error(error_msg)

    # Sidebar controls
    with st.sidebar:
        st.divider()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()
        
        # Memory management button
        if st.button("Clear Memory"):
            gc.collect()
            torch.cuda.empty_cache()
            update_debug("Memory cleared")
        
        # System info
        st.divider()
        st.write("System Information")
        memory_usage = psutil.Process().memory_info().rss / 1024 ** 2
        st.write(f"Memory Usage: {memory_usage:.2f} MB")
