import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from mcp.protocol import MCPMessage

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class LLMResponseAgent:
    """Agent responsible for generating responses using an LLM."""
    
    def __init__(self):
        """Initialize the LLM response agent."""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=SecretStr(GEMINI_API_KEY) if GEMINI_API_KEY else None)
        self.prompt_template = ChatPromptTemplate.from_template(
            """
                You are a helpful assistant designed to answer questions based strictly on provided context or prior chat history. Do not use external knowledge or assumptions. If the answer is not found in the context or chat history, respond with "I don't know based on the provided information."

                Chat History:
                {chat_history}

                Context Information (from uploaded document):
                {context}

                User Question:
                {question}

                Instructions:
                - Only answer if the answer can be found in either the chat history or the context information.
                - If the answer is not explicitly available, return: "false"
                - Do not make up information.
                - Be concise and accurate.
            """
        )
    
    def process_message(self, message: MCPMessage,chat_history:str) -> MCPMessage:
        """Process an incoming MCP message."""
        if message.type == "RESPONSE_REQUEST" or message.type == "RETRIEVAL_RESULT":
            return self._handle_response_request(message,chat_history)
        else:
            raise ValueError(f"Unsupported message type: {message.type}")
    
    def _handle_response_request(self, message: MCPMessage,chat_history:str) -> MCPMessage:
        """Handle response request."""
        query = message.payload.get("query")
        retrieved_context = message.payload.get("retrieved_context", [])
        sources = message.payload.get("sources", [])
        
        if not query:
            return MCPMessage(
                sender="LLMResponseAgent",
                receiver=message.sender,
                type="ERROR",
                trace_id=message.trace_id,
                payload={"error": "No query provided"}
            )
            
        try:
            # Format the context
            formatted_context = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_context)])
            
            # Generate response using the LLM
            chain = self.prompt_template | self.llm
            response = chain.invoke({"context": formatted_context, "question": query, "chat_history":chat_history})
            answer = response.content

            
            # Return response result
            return MCPMessage(
                sender="LLMResponseAgent",
                receiver=message.sender,
                type="RESPONSE_RESULT",
                trace_id=message.trace_id,
                payload={
                    "answer": answer,
                    "sources": sources
                }
            )
        
        except Exception as e:
            # Return error message
            return MCPMessage(
                sender="LLMResponseAgent",
                receiver=message.sender,
                type="ERROR",
                trace_id=message.trace_id,
                payload={
                    "error": f"Error generating response: {str(e)}",
                    "query": query
                }
            )