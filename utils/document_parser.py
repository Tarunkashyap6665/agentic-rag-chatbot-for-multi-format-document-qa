import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

def parse_document(file_path: str) -> str:
    """Parse a document based on its file extension using LangChain document loaders.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Extracted text content from the document
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Load documents
        documents = loader.load()
        
        # Combine document contents
        text = ""
        for i, doc in enumerate(documents):
            # Add page/section separator if multiple documents
            if i > 0:
                text += "\n\n" + "-" * 40 + "\n\n"
            
            # Add metadata header if available
            if doc.metadata:
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page", None)
                if page is not None:
                    text += f"\n--- Page {page + 1} ---\n"
                elif "slide" in doc.metadata:
                    text += f"\n--- Slide {doc.metadata['slide']} ---\n"
            
            # Add document content
            text += doc.page_content
        
        return text
    
    except Exception as e:
        raise ValueError(f"Error parsing document: {str(e)}")
