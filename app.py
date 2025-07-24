import os
import streamlit as st
import uuid
import tempfile
from dotenv import load_dotenv
from agents.coordinator import CoordinatorAgent

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()

# Page title
st.title("Agentic RAG Chatbot for Multi-Format Document QA")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, PPTX, CSV, DOCX, TXT, MD)", 
        accept_multiple_files=True,
        type=["pdf", "pptx", "csv", "docx", "txt", "md"]
    )
    
    # Process uploaded files
    if uploaded_files and len(uploaded_files) > 0:
        new_files = [f for f in uploaded_files if f.name not in [existing.name for existing in st.session_state.uploaded_files]]
        
        if new_files:
            with st.spinner("Processing documents..."):
                for file in new_files:
                    # Save the file temporarily
                    temp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    # Process the file with the coordinator agent
                    st.session_state.coordinator.process_document(temp_path)
                    
                    # Add to session state
                    st.session_state.uploaded_files.append(file)
                
                st.success(f"Processed {len(new_files)} new documents")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.write("Uploaded Documents:")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file.name}")

# Main chat interface
st.header("Chat with your documents")

# Display conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.uploaded_files:
        st.error("Please upload at least one document before asking questions.")
    else:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Get response from coordinator agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.coordinator.process_query(prompt)
                
                # Display response
                st.write(response["content"])
                
                # Display sources if available
                if "sources" in response and response["sources"]:
                    with st.expander("Sources"):
                        for i, source in enumerate(response["sources"]):
                            st.markdown(f"**Source {i+1}:** {source}")
                
                # Add assistant message to conversation
                st.session_state.conversation.append({
                    "role": "assistant", 
                    "content": response["content"],
                    "sources": response.get("sources", [])
                })

# Footer
st.markdown("---")
st.markdown(
    """<div style='text-align: center'>
    <p>Built with Langgraph, Langchain, and Model Context Protocol (MCP)</p>
    </div>""", 
    unsafe_allow_html=True
)