# Agentic RAG Chatbot for Multi-Format Document QA using MCP

A chatbot that answers questions based on uploaded documents using an agent-based architecture and Model Context Protocol (MCP).

## Overview

This project implements an agent-based Retrieval-Augmented Generation (RAG) chatbot that can answer user questions using uploaded documents of various formats. The architecture follows an agentic structure and incorporates Model Context Protocol (MCP) as the mechanism for communication between agents.

![Architecture Diagram](/docs/architecture.png)

## Features

- **Multi-format Document Support**:

  - PDF
  - PPTX
  - CSV
  - DOCX
  - TXT/Markdown

- **Agentic Architecture**:

  - **Coordinator Agent**: Orchestrates workflow between agents
  - **Ingestion Agent**: Parses & preprocesses documents
  - **Retrieval Agent**: Handles embedding + semantic retrieval
  - **LLM Response Agent**: Forms final LLM query and generates answer

- **MCP Integration**:

  - Structured message passing between agents
  - JSON-based protocol with sender, receiver, type, trace_id, and payload

- **Vector Store + Embeddings**:

  - Chroma for vector database
  - Gemini Embeddings for document embedding

- **Streamlit UI**:
  - Document upload interface
  - Chat interface with question and answer
  - Source context display

## System Flow

![System Flow](/docs/flow-diagram.png)

### Document Upload Flow

1. User uploads document through Streamlit UI
2. Coordinator sends document to Ingestion Agent
3. Ingestion Agent parses document, splits into chunks
4. Chunks are embedded and stored in Chroma vector store

### Query Flow

1. User asks question through Streamlit UI
2. Coordinator sends query to Ingestion Agent
3. Ingestion Agent check document is available in vector store or not
4. If available, Ingestion Agent sends query to Retrieval Agent
5. Retrieval Agent retrieves relevant chunks from vector store
6. Chunks and query sent to LLM Response Agent
7. LLM generates answer based on context
8. Answer and sources displayed to user

## Tech Stack

- **Frontend**: Streamlit
- **Backend**:
  - LangChain & LangGraph for agent orchestration
  - Chroma for vector database
  - Gemini for embeddings and LLM
- **Document Processing**:
  Document parsing is handled by LangChain community loaders -
  - PyPDFLoader for PDF
  - UnstructuredPowerPointLoader for PPTX
  - CSVLoader for CSV
  - UnstructuredWordDocumentLoader for DOCX
  - TextLoader for TXT
  - UnstructuredMarkdownLoader for MD

## Setup Instructions

### Prerequisites

- Python 3.11+
- API keys for Gemini and Groq

### Installation

1. Clone the repository

```bash
git clone https://github.com/Tarunkashyap6665/agentic-rag-chatbot-for-multi-format-document-qa.git
cd agentic-rag-chatbot-for-multi-format-document-qa
```

2. Create a virtual environment and activate it

```bash
conda create -n qna_rag python=3.11
conda activate qna_rag
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

1. Upload documents using the sidebar
2. Wait for the documents to be processed
3. Ask questions about the documents in the chat interface
4. View the answers and sources

## Project Structure

```
├── agents/                  # Agent implementations
│   ├── coordinator.py       # Coordinator agent
│   ├── ingestion.py         # Ingestion agent
│   ├── llm_response.py      # LLM response agent
│   └── retrieval.py         # Retrieval agent
├── docs/                    # Documentation
│   ├── architecture.png     # Architecture diagram
│   ├── flow-diagram.png     # Flow diagram
│   └── presentation.pptx    # Presentation slides
├── mcp/                     # Model Context Protocol implementation
│   └── protocol.py          # MCP protocol implementation
├── utils/                   # Utility functions
│   ├── document_parser.py   # Document parsing utilities
│   ├── embeddings.py        # Embeddings model utilities
│   └── vector_store.py      # Vector store utilities
├── .env.example             # Environment variables
├── app.py                   # Main application
└── requirements.txt         # Dependencies
```

## Challenges Faced

- **Document Parsing**: Handling various formats with different structures
- **Agent Communication**: Implementing MCP for structured message passing
- **Vector Search Optimization**: Tuning parameters for relevant retrieval
- **UI/UX Design**: Creating an intuitive interface for document upload and Q&A

## Future Scope

- **Enhanced Document Processing**: Support for more formats (HTML, images with OCR)
- **Advanced Retrieval**: Hybrid search combining semantic and keyword search
- **Multi-Agent Reasoning**: More complex agent interactions for better answers
