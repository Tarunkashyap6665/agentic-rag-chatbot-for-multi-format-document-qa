import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

# Load environment variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_embeddings_model():
    """Get the embeddings model.
    
    Returns:
        Embeddings model instance
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(GEMINI_API_KEY)
    )