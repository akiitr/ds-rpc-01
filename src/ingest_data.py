import os
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document # Added for explicit Document creation if needed

# Attempt to import ALL_FILES from rbac_config. If it's in a parent directory, this might need adjustment
# For now, assuming it's directly importable or rbac_config.py will be in PYTHONPATH
from src.rbac_config import ALL_FILES

# Define constants
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_data():
    """
    Loads documents from specified file paths, splits them into chunks,
    creates embeddings, and stores them in a FAISS vector store.
    """
    all_docs_content = []

    print(f"Starting data ingestion. Processing {len(ALL_FILES)} files.")

    for file_path in ALL_FILES:
        try:
            print(f"Processing file: {file_path}")
            if not os.path.exists(file_path):
                print(f"Warning: File not found at {file_path}. Skipping.")
                continue

            if file_path.endswith(".csv"):
                # For CSVLoader, by default, it creates one document per CSV file.
                # To have one document per row, we can use csv_args to specify the field names
                # and then process each row.
                # However, a simpler approach for now is to let CSVLoader load the whole file
                # or specify a source_column if one clearly represents the document.
                # For this task, let's assume each row should be a document.
                # CSVLoader by default will load the whole file as one document if source_column is not set.
                # To make each row a document, we can iterate through loader.load() if it yields dicts per row
                # or use specific parameters.
                # A common pattern for CSVLoader is to use the `source_column` to identify the main content,
                # or to load it and then transform.
                # Let's assume CSVLoader with no special args makes one doc per file.
                # For hr_data.csv, if we want each row as a doc:
                if "hr_data.csv" in file_path:
                    loader = CSVLoader(file_path=file_path, encoding="utf-8") # Autodetect delimiter
                    # This loads the CSV. Depending on CSVLoader's version/implementation,
                    # it might load each row as a dict or the whole file.
                    # If it loads rows as dicts in `doc.page_content`:
                    raw_documents = loader.load() # loader.load() usually returns a list of Document objects

                    processed_csv_docs = []
                    for i, doc in enumerate(raw_documents):
                        # If page_content is a dict (common if no source_column and it's a structured CSV)
                        # We need to convert dict to string.
                        # If page_content is already a string, this part might not be needed.
                        # Let's assume page_content might be a dict for CSVs.
                        # Based on langchain CSVLoader, page_content is a string from one column (if source_column set)
                        # or a string dump of the row.
                        # Let's ensure it's a string:
                        current_page_content = ""
                        if isinstance(doc.page_content, dict): # Should not happen with default CSVLoader
                             current_page_content = ", ".join([f"{k}: {v}" for k, v in doc.page_content.items()])
                        elif isinstance(doc.page_content, str):
                             current_page_content = doc.page_content # It's already a string
                        else:
                             current_page_content = str(doc.page_content)


                        # Create a new Document to ensure metadata is clean and content is string
                        # For CSVs, it's common for CSVLoader to create one Document per row if no source_column is specified,
                        # and page_content is a string representation of that row.
                        # Let's assume this default behavior.
                        metadata = {'source': file_path, 'row': i + 1}
                        processed_csv_docs.append(Document(page_content=current_page_content, metadata=metadata))
                    print(f"Loaded {len(processed_csv_docs)} rows from {file_path}")
                    all_docs_content.extend(processed_csv_docs)
                else: # For other CSVs, load as single doc for now or assume similar row-based structure
                    loader = CSVLoader(file_path=file_path, encoding="utf-8")
                    documents = loader.load()
                    for doc in documents:
                        doc.metadata['source'] = file_path
                    all_docs_content.extend(documents)

            elif file_path.endswith(".md"):
                loader = TextLoader(file_path=file_path, encoding="utf-8")
                documents = loader.load()
                for doc in documents: # TextLoader loads one doc per file
                    doc.metadata['source'] = file_path
                all_docs_content.extend(documents)
            else:
                print(f"Warning: Unsupported file type for {file_path}. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading file {file_path}: {e}. Skipping.")
            continue

    if not all_docs_content:
        print("No documents were loaded. Exiting.")
        return

    print(f"Loaded {len(all_docs_content)} documents in total.")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs_content)
    print(f"Split {len(all_docs_content)} documents into {len(chunks)} chunks.")

    # Initialize embeddings
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create FAISS vector store
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Ensure the vector_store directory exists
    store_dir = os.path.dirname(VECTOR_STORE_PATH)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=True)
        print(f"Created directory: {store_dir}")

    # Save FAISS index
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store created and saved at {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    # Check if resource files exist before running ingestion
    # This is a simplified check; a more robust check would verify each file in ALL_FILES
    base_resource_path = "resources/data"
    if not os.path.exists(base_resource_path):
        print(f"Error: The directory '{base_resource_path}' does not exist.")
        print("Please ensure the data files are correctly placed or run the script to create them if applicable.")
        # Example: Create dummy files if they don't exist for demonstration
        print("Attempting to create dummy files for demonstration...")

        # Create dummy files based on ALL_FILES from rbac_config
        # This part is for ensuring the script can run in a test environment.
        # In a real scenario, these files should already exist.
        for file_path in ALL_FILES:
            dir_name = os.path.dirname(file_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    if file_path.endswith(".csv"):
                        # Create a dummy CSV with a header and one data row
                        f.write("Header1,Header2\nValue1,Value2\n")
                    elif file_path.endswith(".md"):
                        f.write(f"# Dummy content for {os.path.basename(file_path)}\n")
                    else:
                        f.write(f"Dummy content for {os.path.basename(file_path)}\n")
                print(f"Created dummy file: {file_path}")
        print("Dummy files created. Proceeding with ingestion...")

    ingest_data()
