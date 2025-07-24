import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

# Get environment variables
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

# Ensure the persist directory exists
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Global vector store instance
_vector_store = None

def get_vector_store(embeddings: Embeddings):
    """Get the vector store instance.
    
    Args:
        embeddings: Embeddings model to use
        
    Returns:
        Vector store instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = Chroma(
            # persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    
    return _vector_store