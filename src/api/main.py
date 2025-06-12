import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables at the very beginning
from dotenv import load_dotenv
load_dotenv()

# Attempt to import query_rag from src.rag_pipeline
# This assumes the application is run from the project root (e.g., /app)
# and PYTHONPATH includes /app.
try:
    from src.rag_pipeline import query_rag, VECTOR_STORE_PATH
except ModuleNotFoundError:
    # This fallback is for scenarios where 'src' might not be directly in PYTHONPATH
    # e.g. if 'api' is the current working directory. More robust solutions involve setting PYTHONPATH.
    logging.warning("Could not import from src.rag_pipeline, attempting relative import for ..rag_pipeline")
    from ..rag_pipeline import query_rag, VECTOR_STORE_PATH
except ImportError as e:
    logging.error(f"Failed to import query_rag: {e}. Ensure rag_pipeline.py is accessible.")
    # Define a dummy query_rag if import fails, so FastAPI can still start and report error via endpoint
    def query_rag(user_query: str, user_role: str):
        return {"answer": "Error: RAG pipeline not loaded.", "sources": [], "error": "RAG pipeline import failed."}
    VECTOR_STORE_PATH = "vector_store/faiss_index" # Dummy path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class ChatQueryRequest(BaseModel):
    query: str
    role: str
    session_id: Optional[str] = None

class ChatQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    error: Optional[str] = None

# FastAPI App Initialization
app = FastAPI(
    title="FinSolve RAG Chatbot API",
    description="API for querying the FinSolve RAG chatbot.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Startup Checks
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set. RAG queries requiring LLM will fail.")
    else:
        logger.info("OPENAI_API_KEY is set.")

    # OPENAI_API_BASE is optional for OpenAI, but good to check if using other providers
    if not os.getenv("OPENAI_API_BASE"):
        logger.info("OPENAI_API_BASE environment variable not set. Using default OpenAI base URL if applicable.")
    else:
        logger.info(f"OPENAI_API_BASE is set to: {os.getenv('OPENAI_API_BASE')}")

    if not os.path.exists(VECTOR_STORE_PATH):
        logger.warning(f"Vector store not found at {VECTOR_STORE_PATH}. Ingestion may be required.")
    else:
        logger.info(f"Vector store found at {VECTOR_STORE_PATH}.")


# API Endpoints
@app.get("/")
async def root():
    """A simple health check endpoint."""
    return {"message": "FinSolve RAG Chatbot API is running!"}

@app.post("/chat/query", response_model=ChatQueryResponse)
async def handle_chat_query(request: ChatQueryRequest):
    """
    Handles chat queries by invoking the RAG pipeline.
    Requires `query` and `role` in the request body.
    """
    logger.info(f"Received query: '{request.query}' for role: '{request.role}'")

    # Check for essential environment variables again, in case they changed post-startup
    # query_rag itself also checks for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set at query time.")
        # Using HTTPException to return a proper HTTP status code
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: OPENAI_API_KEY not set."
        )
        # Alternatively, return ChatQueryResponse as per original spec:
        # return ChatQueryResponse(answer="", sources=[], error="OPENAI_API_KEY not set.")

    if not os.path.exists(VECTOR_STORE_PATH):
        logger.error(f"Vector store not found at {VECTOR_STORE_PATH} at query time.")
        raise HTTPException(
            status_code=500,
            detail=f"Server error: Vector store not found at {VECTOR_STORE_PATH}. Please run ingestion."
        )
        # return ChatQueryResponse(answer="", sources=[], error=f"Server error: Vector store not found at {VECTOR_STORE_PATH}")


    try:
        # Call the RAG pipeline function
        result = query_rag(user_query=request.query, user_role=request.role)

        if result.get("error"):
            logger.error(f"Error from RAG pipeline: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))

        answer = result.get("answer", "No answer found.")
        sources = result.get("sources", [])

        logger.info(f"Sending response. Answer: {answer[:50]}... Sources: {len(sources)}")
        return ChatQueryResponse(answer=answer, sources=sources)

    except HTTPException:  # Re-raise HTTPException to let FastAPI handle it
        raise
    except FileNotFoundError as e:
        logger.exception("FileNotFoundError during RAG query.")
        raise HTTPException(status_code=500, detail=f"Server error: A required file was not found: {str(e)}")
    except Exception as e:
        logger.exception("An unexpected error occurred during RAG query.")
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred. Please check server logs.")

# To run this application (from the project root /app):
# uvicorn src.api.main:app --reload --port 8000
# Ensure PYTHONPATH includes the project root if not already the case.
# export PYTHONPATH=$PYTHONPATH:$(pwd) (from /app directory) might be needed in some environments.
