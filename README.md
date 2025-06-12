# RAG Chatbot with Role-Based Access Control (RBAC)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to provide information to company employees while strictly enforcing data access permissions based on their job roles. It allows users to ask questions in natural language and receive answers derived solely from authorized company documents.

The primary function is to serve as an internal knowledge base assistant, ensuring data security and compliance by restricting access to sensitive information according to predefined roles.

## Features

*   **Role-Based Access Control (RBAC):** Restricts document access based on user roles.
*   **Retrieval-Augmented Generation (RAG) Pipeline:** Uses relevant document snippets to ground LLM responses, minimizing hallucination and providing context-aware answers.
*   **FastAPI Backend:** A robust and efficient asynchronous API to serve the RAG pipeline.
*   **Streamlit Frontend:** An interactive web interface for users to select roles and chat with the system.
*   **Source Citation:** Provides references to the source documents used to generate an answer, allowing users to verify information.

## Architecture Overview

The system is composed of several key components that work together to deliver role-aware, document-grounded answers:

1.  **Data (`resources/data/`):** Contains raw documents (Markdown, CSV) categorized by department (e.g., engineering, finance, HR).
2.  **RBAC Configuration (`src/rbac_config.py`):** Defines user roles and maps them to specific document paths they are permitted to access.
3.  **Ingestion (`src/ingest_data.py`):**
    *   Reads documents from `resources/data/`.
    *   Processes and splits document content.
    *   Generates embeddings using a sentence-transformer model (`sentence-transformers/all-MiniLM-L6-v2`).
    *   Stores these embeddings in a FAISS vector store located at `vector_store/faiss_index`.
    *   *Note: If actual files are missing, this script creates dummy files for demonstration.*
4.  **RAG Pipeline (`src/rag_pipeline.py`):**
    *   Utilizes LangChain for its core logic.
    *   When a query is received with a user role:
        *   Retrieves the list of allowed documents for the given role from `rbac_config.py`.
        *   Performs a semantic search over the FAISS vector store for documents relevant to the query.
        *   Filters these retrieved documents against the user's allowed documents (RBAC check).
        *   Constructs a prompt by augmenting the user's query with the content of the filtered, relevant documents.
        *   Sends the augmented prompt to a Large Language Model (LLM, e.g., GPT-3.5-turbo via OpenAI API).
        *   Formats the LLM's response and extracts source document paths.
5.  **Backend API (`src/api/main.py`):**
    *   A FastAPI application.
    *   Exposes a `/chat/query` POST endpoint that accepts a user query and role.
    *   Invokes the RAG Pipeline to get an answer and sources.
    *   Returns the response to the frontend.
6.  **Frontend (`app.py`):**
    *   A Streamlit web application.
    *   Allows users to select their role.
    *   Provides a chat interface to send queries to the backend API.
    *   Displays the chatbot's responses and cited sources.

**Simplified Flow:**

User (Streamlit UI) ➔ Selects Role ➔ Enters Query ➔ FastAPI Backend (`/chat/query`) ➔ RAG Pipeline (RBAC Filter applied to Retriever ➔ LLM with filtered context) ➔ Response with Sources (Streamlit UI)

## User Roles & Data Permissions

The following roles are defined, with access restricted to specific document categories:

*   **Employee Level:** Access to general company documents (e.g., `resources/data/general/employee_handbook.md`).
*   **Finance Team:** Access to finance documents (e.g., `resources/data/finance/*`), plus specific marketing reports for expense analysis and general documents for policies.
*   **Marketing Team:** Access to marketing documents (e.g., `resources/data/marketing/*`).
*   **HR Team:** Access to HR documents (e.g., `resources/data/hr/*`) and general company documents.
*   **Engineering Department:** Access to engineering documents (e.g., `resources/data/engineering/*`).
*   **C-Level Executives:** Access to all documents across all departments.

## Technology Stack

*   **Core Language**: Python 3.10+
*   **Backend**: FastAPI
*   **Frontend**: Streamlit
*   **AI/NLP**:
    *   LangChain (for RAG pipeline orchestration)
    *   OpenAI compatible LLM (e.g., GPT-3.5-turbo, configurable via environment variables)
    *   SentenceTransformers (`all-MiniLM-L6-v2` for embeddings)
    *   FAISS (for efficient similarity search in the vector store)
*   **Testing**: Python `unittest` module

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.9+
    *   Git

2.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

3.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    # venv\Scripts\activate
    ```

4.  **Install dependencies**:
    It is recommended to create a `requirements.txt` file for production. For now, key dependencies can be installed manually or via a preliminary list:
    ```bash
    pip install fastapi uvicorn[standard] streamlit langchain langchain-core langchain-community langchain-openai faiss-cpu sentence-transformers torch python-dotenv requests
    ```
    *(Note: `faiss-cpu` is for CPU-only. If you have a compatible GPU, you might consider `faiss-gpu`.)*

5.  **Environment Variables**:
    Create a `.env` file in the project root directory with the following content:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # Optional: If using a proxy or a non-OpenAI compatible endpoint for your LLM
    # OPENAI_API_BASE="your_llm_api_base_url_here"

    # Optional: If your Streamlit app needs to connect to a backend API not on localhost:8000
    # API_URL="http://your_backend_host:port/chat/query"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

6.  **Data**:
    *   Place your company documents in the `resources/data/` directory, following the departmental subfolder structure (e.g., `resources/data/finance/report.md`).
    *   If no actual data is provided, the ingestion script (`src/ingest_data.py`) will create dummy files for demonstration purposes. For meaningful results, use actual project-specific documents.

## Running the Application

1.  **Prepare Data & Vector Store**:
    Run the data ingestion script from the project root directory. This script processes documents in `resources/data/`, generates embeddings, and creates/updates the FAISS vector store in `vector_store/faiss_index`.
    ```bash
    python -m src.ingest_data
    ```

2.  **Start the FastAPI Backend**:
    From the project root directory, run:
    ```bash
    uvicorn src.api.main:app --reload --port 8000
    ```
    The API server will start, typically available at `http://localhost:8000`. You can access its interactive documentation at `http://localhost:8000/docs`.

3.  **Start the Streamlit Frontend**:
    In a new terminal, from the project root directory, run:
    ```bash
    streamlit run app.py
    ```
    The Streamlit application will automatically open in your web browser, usually at `http://localhost:8501`.

## Running Tests

To run the unit and integration tests, execute the following command from the project root directory:
```bash
python -m unittest discover src/tests
```

## Code Structure

```
.
├── app.py                  # Streamlit frontend application
├── resources/
│   └── data/               # Raw data files (categorized by department)
│       ├── engineering/
│       ├── finance/
│       ├── general/
│       ├── hr/
│       └── marketing/
├── src/
│   ├── __init__.py
│   ├── api/                # FastAPI backend application
│   │   ├── __init__.py
│   │   └── main.py
│   ├── tests/              # Unit and integration tests
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_rbac_config.py
│   ├── ingest_data.py      # Script for data processing and vector store creation
│   ├── rag_pipeline.py     # Core RAG logic with RBAC filtering
│   └── rbac_config.py      # RBAC role and permission definitions
├── vector_store/           # Stores the FAISS index (created by ingest_data.py)
│   └── faiss_index/
├── .env                    # User-created file for environment variables (ignored by git)
└── README.md               # This file
```

## Future Enhancements / To-Do

*   **Refined HR Data Handling:** Implement more sophisticated parsing and representation for CSV data from HR to improve retrieval quality.
*   **User Authentication:** Replace the current role selection dropdown with a proper user authentication system (e.g., OAuth2).
*   **Expanded Data Source Support:** Add loaders for other document types (e.g., DOCX, PDF with better parsing).
*   **Comprehensive Logging & Monitoring:** Integrate more detailed logging across all services and set up monitoring dashboards.
*   **Containerization:** Dockerize the backend and frontend applications for easier deployment and scalability.
*   **CI/CD Pipeline:** Implement a CI/CD pipeline for automated testing and deployment.
*   **Granular RBAC:** Explore document section-level or content-aware RBAC for even finer-grained access control.
*   **Chat History:** Implement persistent chat history per user/session.
*   **Requirements File:** Generate and maintain a `requirements.txt` file.
```
