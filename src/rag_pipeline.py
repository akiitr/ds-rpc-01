import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
# Attempting alternative import for StringOutputParser -> StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.rbac_config import get_allowed_documents

# Configuration Constants
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-3.5-turbo" # or another model like gpt-4

def format_docs_for_context(docs: list[Document]) -> str:
    """
    Formats a list of LangChain Document objects into a single string for context.
    Each document's content is prepended with its source metadata.
    """
    if not docs:
        return "No documents found or provided for context."

    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Source')
        # For CSVs, include row number if available
        if 'row' in doc.metadata:
            source += f" (Row: {doc.metadata['row']})"
        content = doc.page_content
        formatted_docs.append(f"Source: {source}\n{content}")
    return "\n\n".join(formatted_docs)

def query_rag(user_query: str, user_role: str) -> dict:
    """
    Queries the RAG pipeline with RBAC filtering.

    Args:
        user_query: The question asked by the user.
        user_role: The role of the user, used for RBAC filtering.

    Returns:
        A dictionary containing the 'answer' and 'sources'.
        Example: {"answer": "The leave policy is...", "sources": ["path/to/doc.md"]}
    """
    # Load Embeddings and Vector Store
    # Add a check for OPENAI_API_KEY and OPENAI_API_BASE early on
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        return {
            "answer": "Error: OPENAI_API_KEY environment variable not set. Cannot query the LLM.",
            "sources": []
        }
    # openai_api_base might be optional for some setups, but good practice to check
    if not openai_api_base:
        print("Warning: OPENAI_API_BASE environment variable not set. Using default OpenAI base URL.")


    print(f"Loading vector store from: {VECTOR_STORE_PATH}")
    if not os.path.exists(VECTOR_STORE_PATH):
        return {
            "answer": f"Error: Vector store not found at {VECTOR_STORE_PATH}. Please run the ingestion script first.",
            "sources": []
        }

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    except Exception as e:
        return {
            "answer": f"Error loading vector store or embeddings: {str(e)}",
            "sources": []
        }

    # Create Retriever
    # k can be adjusted. More documents might give more context but increase prompt size.
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Define Prompt Template
    template = """
You are an AI assistant for FinSolve Technologies. Your role is to provide information based ONLY on the provided company documents.
Answer the user's question based strictly on the context given.
If the information is not found in the provided context, you MUST state that the information is not available in the documents you have access to.
Do not make up information or use external knowledge.
After your answer, list all unique source document(s) you used to answer the question. Each source should be on a new line and start with "Source: ".

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base, # Optional: defaults to OpenAI if not set or None
        temperature=0.3 # Lower temperature for more factual, less creative answers
    )

    # Helper function for filtering documents based on RBAC
    # This function is defined inside query_rag to easily capture 'retriever'
    def _get_filtered_docs(input_dict: dict) -> list[Document]:
        allowed_sources = get_allowed_documents(input_dict['role'])
        if not allowed_sources:
            print(f"No documents allowed for role: {input_dict['role']}")
            return [] # No documents allowed for this role

        print(f"Role '{input_dict['role']}' allowed sources: {allowed_sources}")

        # Retrieve documents first
        retrieved_docs = retriever.get_relevant_documents(input_dict['question'])
        print(f"Retrieved {len(retrieved_docs)} documents initially for query: \"{input_dict['question']}\"")

        # Filter based on RBAC
        filtered_docs = [
            doc for doc in retrieved_docs if doc.metadata.get('source') in allowed_sources
        ]
        print(f"Filtered down to {len(filtered_docs)} documents based on RBAC for role '{input_dict['role']}'.")
        return filtered_docs

    # LCEL Chain Construction
    lcel_chain = (
        # Input to the chain is a dict: {"question": user_query, "role": user_role}
        # This first step retrieves and filters documents, then passes them along with original input
        RunnableLambda(lambda input_dict: {
            "filtered_docs": _get_filtered_docs(input_dict),
            "original_input": input_dict
        })
        # This step formats the context and prepares inputs for the prompt
        | RunnableLambda(lambda x: {
            "context": format_docs_for_context(x['filtered_docs']),
            "question": x['original_input']['question'],
            # Extract unique sources from the filtered documents for final output
            "sources": sorted(list(set(doc.metadata['source'] for doc in x['filtered_docs'] if 'source' in doc.metadata)))
        })
        # This step runs the prompt formatting, LLM, and output parsing in parallel with passing sources
        | RunnableParallel(
            answer=(
                # Create the prompt string
                RunnableLambda(lambda x: prompt.format(context=x['context'], question=x['question']))
                | llm
                | StringOutputParser()
            ),
            sources=RunnableLambda(lambda x: x['sources']) # Pass the list of sources through
        )
    )

    try:
        print("Invoking RAG chain...")
        response = lcel_chain.invoke({"question": user_query, "role": user_role})
        print("RAG chain invocation complete.")
        return response # Expected: {"answer": "...", "sources": ["..."]}
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return {
            "answer": f"An error occurred while processing your request: {str(e)}",
            "sources": []
        }

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file

    print("Starting RAG pipeline example execution...")

    # Check for necessary environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE") # Optional, but good to check

    if not api_key:
        print("\n" + "="*50)
        print("ERROR: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your environment or in a .env file to run the RAG query.")
        print("Example: export OPENAI_API_KEY='your_api_key_here'")
        print("If using a local LLM compatible with OpenAI's API, also set OPENAI_API_BASE.")
        print("Example: export OPENAI_API_BASE='http://localhost:8080/v1'")
        print("="*50 + "\n")
    else:
        if not os.path.exists(VECTOR_STORE_PATH):
            print("\n" + "="*50)
            print(f"ERROR: Vector store not found at {VECTOR_STORE_PATH}.")
            print("Please ensure you have run the `src/ingest_data.py` script successfully before this script.")
            print("="*50 + "\n")
        else:
            print("\nAttempting a sample query...")
            # Sample query
            sample_query = "What is the company's policy on remote work?"
            sample_role = "employee_level" # This role has access to general_files, like employee_handbook.md

            print(f"Query: \"{sample_query}\" for role: \"{sample_role}\"")

            response = query_rag(sample_query, sample_role)

            print("\n--- RAG Response ---")
            print(f"Answer: {response.get('answer')}")
            if response.get('sources'):
                print("\nSources:")
                for source in response.get('sources', []):
                    print(f"- {source}")
            else:
                print("No sources provided or found for this answer.")
            print("--- End of RAG Response ---\n")

    print("RAG pipeline example execution finished.")
