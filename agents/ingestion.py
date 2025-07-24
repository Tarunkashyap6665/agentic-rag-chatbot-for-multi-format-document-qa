import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from utils.document_parser import parse_document
from utils.vector_store import get_vector_store
from utils.embeddings import get_embeddings_model
from mcp.protocol import MCPMessage

# Load environment variables
load_dotenv()

# Get environment variables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

class IngestionAgent:
    """Agent responsible for document ingestion and preprocessing."""
    
    def __init__(self):
        """Initialize the ingestion agent."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        self.embeddings = get_embeddings_model()
        self.vector_store = get_vector_store(self.embeddings)
    
    def process_message(self, message: MCPMessage) -> MCPMessage:
        """Process an incoming MCP message."""
        if message.type == "DOCUMENT_INGESTION":
            return self._handle_document_ingestion(message)
        else:
            raise ValueError(f"Unsupported message type: {message.type}")
    
    def _handle_document_ingestion(self, message: MCPMessage) -> MCPMessage:
        """Handle document ingestion request."""
        document_path = message.payload.get("document_path")
        if not document_path:
            return MCPMessage(
                sender="IngestionAgent",
                receiver=message.sender,
                type="ERROR",
                trace_id=message.trace_id,
                payload={"error": "No document path provided"}
            )
        
        try:
            # Parse the document
            document_content = parse_document(document_path)
            
            # Split the document into chunks
            chunks = self.text_splitter.split_text(document_content)
            
            # Create metadata for each chunk
            metadatas = [{
                "source": os.path.basename(document_path),
                "chunk_id": str(uuid.uuid4()),
                "document_path": document_path
            } for _ in chunks]
            
            # Add chunks to vector store
            self.vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
            
            # Return success message
            return MCPMessage(
                sender="IngestionAgent",
                receiver=message.sender,
                type="INGESTION_RESULT",
                trace_id=message.trace_id,
                payload={
                    "status": "success",
                    "document_path": document_path,
                    "num_chunks": len(chunks)
                }
            )
        
        except Exception as e:
            # Return error message
            return MCPMessage(
                sender="IngestionAgent",
                receiver=message.sender,
                type="ERROR",
                trace_id=message.trace_id,
                payload={
                    "error": f"Error processing document: {str(e)}",
                    "document_path": document_path
                }
            )