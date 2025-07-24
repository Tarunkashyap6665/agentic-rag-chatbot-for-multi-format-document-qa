from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class MCPPayload(BaseModel):
    """Base class for MCP message payloads."""
    pass

class DocumentIngestionPayload(MCPPayload):
    """Payload for document ingestion requests."""
    document_path: str = Field(..., description="Path to the document to ingest")

class IngestionResultPayload(MCPPayload):
    """Payload for document ingestion results."""
    status: str = Field(..., description="Status of the ingestion process")
    document_path: str = Field(..., description="Path to the ingested document")
    num_chunks: int = Field(..., description="Number of chunks created from the document")

class RetrievalRequestPayload(MCPPayload):
    """Payload for retrieval requests."""
    query: str = Field(..., description="Query to retrieve relevant documents")

class RetrievalResultPayload(MCPPayload):
    """Payload for retrieval results."""
    retrieved_context: list[str] = Field(..., description="List of retrieved document chunks")
    sources: list[str] = Field(default_factory=list, description="List of sources for the retrieved chunks")
    query: str = Field(..., description="Original query")

class ResponseRequestPayload(MCPPayload):
    """Payload for response requests."""
    query: str = Field(..., description="User query")
    retrieved_context: list[str] = Field(..., description="List of retrieved document chunks")

class ResponseResultPayload(MCPPayload):
    """Payload for response results."""
    answer: str = Field(..., description="Generated answer")
    sources: list[str] = Field(default_factory=list, description="List of sources for the answer")

class ErrorPayload(MCPPayload):
    """Payload for error messages."""
    error: str = Field(..., description="Error message")
    query: Optional[str] = Field(None, description="Original query if available")
    document_path: Optional[str] = Field(None, description="Document path if available")