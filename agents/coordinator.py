import uuid
from typing import Dict, Any
from typing import TypedDict
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, END
from agents.ingestion import IngestionAgent
from agents.retrieval import RetrievalAgent
from agents.llm_response import LLMResponseAgent
from mcp.protocol import MCPMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from pydantic import SecretStr, BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

# Configure Gemini API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="Gemma2-9b-It",api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY else None)

class RewriteQuery(BaseModel):
    """Model for rewriting queries."""
    query: str = Field(..., description="The rewritten query")

# Define a tool for rewriting queries
@tool
def rewrite_query_tool(query: str, chat_history: str):
    """
    Rewrite the user query by combining it with relevant context from the chat history to form a complete, self-contained question.
    """

    # System message: guide the assistant to create an improved, standalone question
    system_message = (
        "You are a query reformulation assistant. Your job is to take a user query and the preceding chat history, "
        "and rewrite the query as a complete, clear, and self-contained question. \n\n"
        "If the user’s current query is a follow-up or refers to something mentioned earlier, use the relevant context from the chat history "
        "to make the new query meaningful on its own. \n\n"
        "Your output must be a rewritten query that stands alone without requiring the previous conversation. "
        "Do not provide any explanations—just return the rewritten query."
    )

    # Human message with formatted chat history and current query
    human_message = (
        "Conversation History:\n{chat_history}\n\n"
        "User's Latest Query:\n{question}\n\n"
        "Rewritten Self-Contained Query:"
    )

    # Create the prompt template
    rewrite_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])

    # Apply LLM with structured output
    llm_with_structured_output = rewrite_prompt_template | llm.with_structured_output(schema=RewriteQuery)

    return llm_with_structured_output.invoke({
        "question": query,
        "chat_history": chat_history
    })



memory= []

# Define the state schema as a TypedDict
class WorkflowState(TypedDict):
    """State schema for the RAG workflow."""
    document_path: NotRequired[str]
    query: NotRequired[str]
    trace_id: str
    ingestion_result: NotRequired[Dict[str, Any]]
    retrieval_result: NotRequired[Dict[str, Any]]
    final_response: NotRequired[Dict[str, Any]]


class CoordinatorAgent:
    """Coordinator agent that orchestrates the workflow between other agents."""
    
    def __init__(self):
        """Initialize the coordinator agent with its component agents."""
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.llm_response_agent = LLMResponseAgent()
        self.graph = self._build_graph()
        

    
    def _build_graph(self):
        """Build the workflow graph using LangGraph."""
        # Define the graph with the state schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("ingestion", self._run_ingestion)
        workflow.add_node("retrieval", self._run_retrieval)
        workflow.add_node("llm_response", self._run_llm_response)
        
        # Define the edges
        # Only go from ingestion to retrieval if a query is present in the state
        workflow.add_edge("__start__", "ingestion")
        workflow.add_conditional_edges(
            "ingestion",
            self._router,
            {"retrieval": "retrieval", END: END}
        )
        workflow.add_edge("retrieval", "llm_response")
        workflow.add_edge("llm_response", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _router(self, state: WorkflowState) -> str:
        """Route the workflow based on the state."""
        if "query" in state:
            return "retrieval"
        else:
            return END
    
    def _run_ingestion(self, state: WorkflowState) -> WorkflowState:
        """Run the ingestion agent."""
        document_path = state.get("document_path")
        trace_id = state.get("trace_id", str(uuid.uuid4()))
        
        # Create MCP message for ingestion agent
        message = MCPMessage(
            sender="CoordinatorAgent",
            receiver="IngestionAgent",
            type="DOCUMENT_INGESTION",
            trace_id=trace_id,
            payload={"document_path": document_path}
        )
        
        # Process the message with ingestion agent
        response = self.ingestion_agent.process_message(message)
        
        # Update state with response
        return {**state, "ingestion_result": response.payload}
    
    def _run_retrieval(self, state: WorkflowState) -> WorkflowState:
        """Run the retrieval agent."""
        query = state.get("query")
        trace_id = state.get("trace_id", str(uuid.uuid4()))
        
        # Create MCP message for retrieval agent
        message = MCPMessage(
            sender="CoordinatorAgent",
            receiver="RetrievalAgent",
            type="RETRIEVAL_REQUEST",
            trace_id=trace_id,
            payload={"query": query}
        )
        
        # Process the message with retrieval agent
        response = self.retrieval_agent.process_message(message,self.get_chat_history())
        
        # Update state with response
        return {**state, "retrieval_result": response.payload}
    
    def _run_llm_response(self, state: WorkflowState) -> WorkflowState:
        """Run the LLM response agent."""
        query = state.get("query")
        retrieval_result = state.get("retrieval_result", {})
        trace_id = state.get("trace_id", str(uuid.uuid4()))
        
        if query is None:
            raise ValueError("Missing 'query' in workflow state for LLM response.")
        
        # Create MCP message for LLM response agent
        message = MCPMessage(
            sender="CoordinatorAgent",
            receiver="LLMResponseAgent",
            type="RESPONSE_REQUEST",
            trace_id=trace_id,
            payload={
                "query": query,
                "retrieved_context": retrieval_result.get("retrieved_context", []),
                "sources": retrieval_result.get("sources", [])
            }
        )
        
        # Process the message with LLM response agent
        response = self.llm_response_agent.process_message(message,chat_history=self.get_chat_history())
        
        # Update state with response
        return {**state, "final_response": response.payload}
    
    def process_document(self, document_path: str) -> None:
        """Process a document through the ingestion pipeline."""
        trace_id = str(uuid.uuid4())
        initial_state: WorkflowState = {"document_path": document_path, "trace_id": trace_id}
        self.graph.invoke(initial_state)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the retrieval and response pipeline."""
        trace_id = str(uuid.uuid4())
        initial_state: WorkflowState = {"query": query, "trace_id": trace_id}
        result = self.graph.invoke(initial_state)
        memory.append(result)
        
        # Extract the final response
        final_response = result.get("final_response", {})
        if final_response.get("answer") =="false":
            rewritten_query = rewrite_query_tool.invoke({"query": query, "chat_history": self.get_chat_history()})
            query = rewritten_query.query if rewritten_query.query else query
            initial_state: WorkflowState = {"query": query, "trace_id": trace_id}
            result = self.graph.invoke(initial_state)
            memory.append(result)
            final_response = result.get("final_response", {})

            if final_response.get("answer") =="false":
                return {
                    "content": "I couldn't find an answer to your question.",
                    "sources": []
                }
     
        # Format the response for the UI
        return {
            "content": final_response.get("answer", "I couldn't find an answer to your question."),
            "sources": final_response.get("sources", [])
        }

    def plot_graph(self):
        """Plot the workflow graph."""
        print("Drawing the workflow graph...")
        return self.graph.get_graph().draw_mermaid_png()

    def get_chat_history(self)->str:
        """Get the chat history from the memory."""
        if not memory:
            return ""
        text=""
        for item in memory:
            text+=f"Question: {item.get('query', '')}\n"
            text+=f"Answer: {item.get('final_response', {}).get('answer', 'No answer found')}\n\n"
        
        return text
        
       
