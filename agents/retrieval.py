import os
from dotenv import load_dotenv

from utils.vector_store import get_vector_store
from utils.embeddings import get_embeddings_model
from mcp.protocol import MCPMessage
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import tool
from pydantic import SecretStr,BaseModel,Field
from mcp.protocol import MCPMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



# Load environment variables
load_dotenv()

# Configure Gemini API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="Gemma2-9b-It",api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY else None)

class RewriteQuery(BaseModel):
    """Model for rewriting queries."""
    query: str = Field(..., description="The rewritten query")

# Define a tool for rewriting queries
@tool
def rewrite_query_tool(query: str,chat_history:str):
    """
    Rewrite the user query if it fails to return relevant documents."""

    # Use this system prompt
    system_message = (
        "You are a question re-writer that converts a vague or context-dependent input "
        "question into a clearer and more specific version optimized for document or web search. "
        "Use the chat history to understand the question's intent and make it self-contained. "
        "Do not include explanations—only output the rewritten question."
    )

    # Human message template with chat history and current query
    human_message = (
        "Conversation History:\n{chat_history}\n\n"
        "Original Query:\n{question}\n\n"
        "Rewritten Query (only output the improved question):"
    )

    # Combine into a ChatPromptTemplate
    rewrite_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


    llm_with_structured_output = rewrite_prompt_template | llm.with_structured_output(schema=RewriteQuery)
    
    return llm_with_structured_output.invoke({"question": query, "chat_history": chat_history})


class RetrievalAgent:
    """Agent responsible for retrieving relevant document chunks."""
    
    def __init__(self, top_k: int = 5):
        """Initialize the retrieval agent.
        
        Args:
            top_k: Number of top results to retrieve
        """
        self.top_k = top_k
        self.embeddings = get_embeddings_model()
        self.vector_store = get_vector_store(self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self.retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            name="document_retriever",
            description="Useful for answering questions about the uploaded document. Ask specific questions about the content."
        )

    
    
    def process_message(self, message: MCPMessage,chat_history:str) -> MCPMessage:
        """Process an incoming MCP message."""
        if message.type == "RETRIEVAL_REQUEST":
            return self._handle_retrieval_request(message,chat_history)
        else:
            raise ValueError(f"Unsupported message type: {message.type}")
    
    def _handle_retrieval_request(self, message: MCPMessage,chat_history:str) -> MCPMessage:
        """Handle retrieval request."""
        query = message.payload.get("query")
        if not query:
            return MCPMessage(
                sender="RetrievalAgent",
                receiver=message.sender,
                type="ERROR",
                trace_id=message.trace_id,
                payload={"error": "No query provided"}
            )
        
        try:
            # Perform similarity search
            results = self.retriever.invoke(query)
            results = [doc for doc in results if self._is_valid_content(doc.page_content)]
            
            if not results:
                rewritten_query = rewrite_query_tool.invoke({"query": query, "chat_history": chat_history})
                query = rewritten_query.query if rewritten_query.query else query
                # Retry retrieval
                results = self.retriever.invoke(query)
                results = [doc for doc in results if self._is_valid_content(doc.page_content)]
            
            # Extract document chunks and their sources
            retrieved_context = []
            sources = []
            
            for doc in results:
                # Add document content to retrieved context
                retrieved_context.append(doc.page_content)
                
                # Add source information
                source = f"{doc.metadata.get('source', 'Unknown')}"
                if source not in sources:
                    sources.append(source)
            
            # Return retrieval results
            return MCPMessage(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",  # Forward to LLM response agent
                type="RETRIEVAL_RESULT",
                trace_id=message.trace_id,
                payload={
                    "retrieved_context": retrieved_context,
                    "sources": sources,
                    "query": query
                }
            )
        
        except Exception as e:
            # Return error message
            return MCPMessage(
                sender="RetrievalAgent",
                receiver=message.sender,
                type="ERROR",
                trace_id=message.trace_id,
                payload={
                    "error": f"Error retrieving documents: {str(e)}",
                    "query": query
                }
            )
        
    def _is_valid_content(self, content: str) -> bool:
        """Check if content is meaningful (not just dashes or empty)."""
        stripped = content.strip().replace("-", "")
        return bool(stripped)

