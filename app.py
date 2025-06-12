import streamlit as st
import requests
import json
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/chat/query")
USER_ROLES = [
    "employee_level",
    "finance_team",
    "marketing_team",
    "hr_team",
    "engineering_department",
    "c_level_executives"
]

st.set_page_config(page_title="FinSolve RAG Chatbot", layout="wide")

# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_role" not in st.session_state:
    st.session_state.selected_role = USER_ROLES[0] # Default to the first role

# --- UI Layout ---
st.title("FinSolve RAG Chatbot 🏦")

# Sidebar for Role Selection
with st.sidebar:
    st.header("User Settings")
    # Use the current session state value for selected_role to find the index,
    # or default to the first role if it's somehow not in USER_ROLES.
    try:
        current_role_index = USER_ROLES.index(st.session_state.selected_role)
    except ValueError:
        current_role_index = 0 # Default to first role if current is invalid
        st.session_state.selected_role = USER_ROLES[0]

    st.session_state.selected_role = st.selectbox(
        "Select your role:",
        USER_ROLES,
        index=current_role_index
    )
    st.info(f"Current Role: **{st.session_state.selected_role}**")
    st.markdown("---")
    st.markdown(
        "**Note:** The chatbot will only answer based on documents "
        "accessible to your selected role."
    )

# Chat History Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown("\n*Sources:*")
            for source in message["sources"]:
                # Displaying as code block for clear differentiation
                st.markdown(f"- `{source}`")

# Chat Input and Processing
if prompt := st.chat_input("Ask a question about FinSolve documents..."):
    # Add user's query to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant's thinking placeholder and then the actual response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...") # Initial placeholder content

        full_response_content = ""
        sources_to_display = []

        try:
            payload = {
                "query": prompt,
                "role": st.session_state.selected_role,
                "session_id": "streamlit_session" # Example session_id
            }
            response = requests.post(API_URL, json=payload, timeout=90) # Increased timeout

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer")
                sources = data.get("sources", [])
                error_message_from_api = data.get("error")

                if error_message_from_api:
                    full_response_content = f"Error from API: {error_message_from_api}"
                elif answer:
                    full_response_content = answer
                    if sources: # Only assign sources if answer is valid and sources exist
                        sources_to_display = sources
                else:
                    full_response_content = "Sorry, I received an empty or unexpected response from the server."
            else:
                # Attempt to parse error from response body if available
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", response.text)
                except json.JSONDecodeError:
                    detail = response.text
                full_response_content = f"Error: Received status code {response.status_code}. Detail: {detail}"

        except requests.exceptions.Timeout:
            full_response_content = f"Error: The request to the API timed out after {90} seconds."
        except requests.exceptions.ConnectionError:
            full_response_content = f"Error: Could not connect to the API at {API_URL}. Please ensure the backend server is running."
        except requests.exceptions.RequestException as e:
            full_response_content = f"Error connecting to API: {e}"
        except json.JSONDecodeError: # Should be rare if status_code based error handling is robust
            full_response_content = "Error: Received invalid JSON response from the server."

        message_placeholder.markdown(full_response_content)
        if sources_to_display:
            st.markdown("\n*Sources:*")
            for source in sources_to_display:
                st.markdown(f"- `{source}`")

        # Append the final assistant message (with content and sources if any) to session state
        assistant_final_message = {"role": "assistant", "content": full_response_content}
        if sources_to_display:
            assistant_final_message["sources"] = sources_to_display
        st.session_state.messages.append(assistant_final_message)

# Add a footer or some information
st.markdown("---")
st.markdown("Built with Streamlit and FastAPI. Ensure the backend API is running.")
st.markdown(f"API Endpoint: `{API_URL}`")
