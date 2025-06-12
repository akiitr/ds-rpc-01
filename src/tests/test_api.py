import unittest
from unittest.mock import patch, MagicMock
import os
from fastapi.testclient import TestClient

# Attempt to import the app instance from src.api.main
# This path needs to be correct relative to the project root when running tests
# e.g. python -m unittest discover src/tests
try:
    from src.api.main import app
except ModuleNotFoundError as e:
    # Handle cases where PYTHONPATH might not be set up as expected during testing
    # This is a common issue with test discovery and relative imports.
    # A more robust solution is ensuring PYTHONPATH includes the project root.
    print(f"Initial import failed: {e}. Ensure PYTHONPATH includes project root or tests are run as module.")
    # Fallback for some environments, though not ideal:
    if os.getcwd().endswith("/src/tests"): # If CWD is tests directory
        # This is highly dependent on test runner's CWD.
        # A better approach is to ensure test runner sets PYTHONPATH correctly.
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..'))) # Add project root
        from src.api.main import app
    else: # If running from project root, this should generally not be an issue.
        raise


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Ensure a default OPENAI_API_KEY is set for most tests, can be overridden per test
        self.original_openai_key = os.environ.get("OPENAI_API_KEY")
        self.original_vector_store_path = os.environ.get("VECTOR_STORE_PATH") # If used by API directly
        os.environ["OPENAI_API_KEY"] = "test_api_key" # Mock API key
        # Mock vector store path for tests if API checks it directly
        # For these tests, we mock query_rag, so vector store existence is less critical
        # unless the API endpoint itself checks for the path before calling query_rag.
        # Based on src.api.main, it does check VECTOR_STORE_PATH.
        # Let's assume a dummy one exists or mock that check if needed.
        # For now, we'll assume the check in query_rag is what matters or we mock it.
        # The provided api/main.py checks os.path.exists(VECTOR_STORE_PATH) in /chat/query
        # So, we need to ensure this check passes or mock it.
        # Easiest to mock os.path.exists for this.

    def tearDown(self):
        # Restore original environment variables
        if self.original_openai_key is None:
            del os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = self.original_openai_key

        if self.original_vector_store_path is None and "VECTOR_STORE_PATH" in os.environ:
             del os.environ["VECTOR_STORE_PATH"] # if it was set by test
        elif self.original_vector_store_path:
             os.environ["VECTOR_STORE_PATH"] = self.original_vector_store_path


    @patch('src.api.main.os.path.exists') # Mock os.path.exists used by the API endpoint
    @patch('src.api.main.query_rag')    # Patch where query_rag is LOOKED UP by main.py
    def test_chat_query_employee_allowed(self, mock_query_rag, mock_os_path_exists):
        mock_os_path_exists.return_value = True # Simulate vector store path exists
        mock_query_rag.return_value = {"answer": "This is the employee handbook content.", "sources": ["resources/data/general/employee_handbook.md"]}

        response = self.client.post("/chat/query", json={"query": "What is the leave policy?", "role": "employee_level"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "This is the employee handbook content.")
        self.assertIn("resources/data/general/employee_handbook.md", data["sources"])
        self.assertIsNone(data.get("error"))
        mock_query_rag.assert_called_once_with(user_query="What is the leave policy?", user_role="employee_level")

    @patch('src.api.main.os.path.exists')
    @patch('src.api.main.query_rag')
    def test_chat_query_employee_denied_for_finance_data(self, mock_query_rag, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        # Simulate query_rag finding no relevant (allowed) documents for the role
        mock_query_rag.return_value = {"answer": "Information not available in your accessible documents.", "sources": []}

        response = self.client.post("/chat/query", json={"query": "What are the company profits?", "role": "employee_level"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Information not available in your accessible documents.")
        self.assertEqual(data["sources"], [])
        self.assertIsNone(data.get("error"))
        mock_query_rag.assert_called_once_with(user_query="What are the company profits?", user_role="employee_level")

    @patch('src.api.main.os.path.exists')
    @patch('src.api.main.query_rag')
    def test_chat_query_finance_allowed_for_finance_data(self, mock_query_rag, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        mock_query_rag.return_value = {"answer": "The company profits are $X million.", "sources": ["resources/data/finance/financial_summary.md"]}

        response = self.client.post("/chat/query", json={"query": "What are the company profits?", "role": "finance_team"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "The company profits are $X million.")
        self.assertIn("resources/data/finance/financial_summary.md", data["sources"])
        self.assertIsNone(data.get("error"))
        mock_query_rag.assert_called_once_with(user_query="What are the company profits?", user_role="finance_team")

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True) # Temporarily unset API key
    @patch('src.api.main.os.path.exists')
    @patch('src.api.main.query_rag')
    def test_chat_query_no_openai_key(self, mock_query_rag, mock_os_path_exists):
        mock_os_path_exists.return_value = True # Vector store exists

        # This test assumes your /chat/query endpoint checks for OPENAI_API_KEY
        # and returns an HTTPException if it's missing. TestClient will handle this.
        response = self.client.post("/chat/query", json={"query": "Any query", "role": "employee_level"})

        # The API raises HTTPException(status_code=500, detail="Server configuration error: OPENAI_API_KEY not set.")
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("OPENAI_API_KEY not set", data.get("detail", ""))
        mock_query_rag.assert_not_called()

    @patch('src.api.main.os.path.exists') # Mock os.path.exists
    @patch('src.api.main.query_rag')    # Patch query_rag
    def test_chat_query_vector_store_not_found(self, mock_query_rag, mock_os_path_exists):
        mock_os_path_exists.return_value = False # Simulate vector store path does NOT exist

        response = self.client.post("/chat/query", json={"query": "Any query", "role": "employee_level"})

        # The API raises HTTPException(status_code=500, detail="Server error: Vector store not found...")
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Vector store not found", data.get("detail", ""))
        mock_query_rag.assert_not_called() # query_rag should not be called

    @patch('src.api.main.os.path.exists')
    @patch('src.api.main.query_rag')
    def test_chat_query_rag_pipeline_internal_error(self, mock_query_rag, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        # Simulate query_rag itself returning an error structure
        mock_query_rag.return_value = {"answer": "", "sources": [], "error": "Internal RAG error"}

        response = self.client.post("/chat/query", json={"query": "Test query", "role": "employee_level"})

        # The API raises HTTPException(status_code=500, detail="Internal RAG error")
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data.get("detail"), "Internal RAG error")
        mock_query_rag.assert_called_once_with(user_query="Test query", user_role="employee_level")

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "FinSolve RAG Chatbot API is running!"})


if __name__ == '__main__':
    unittest.main()
