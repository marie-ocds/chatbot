import streamlit as st
from dotenv import load_dotenv, find_dotenv
from src.indexing import build_all_indices
from src.retrieval import answer_query, display_sources

# Load environment variables
load_dotenv(find_dotenv())


@st.cache_resource
def load_indices():
    """
    Load or build all indices. Uses @st.cache_resource to persist
    indices across reruns without reloading.
    """
    book_index, chapter_index, scenes_index = build_all_indices()
    return book_index, chapter_index, scenes_index


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load indices (cached)
book_index, chapter_index, scenes_index = load_indices()

# Display title
st.title("Doctor Dolittle Chatbot")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_query = st.chat_input("Ask any question")
if user_query:
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Get answer
    with st.spinner("Thinking..."):
        answer = answer_query(user_query, book_index, chapter_index, scenes_index)

    # Add assistant message to history and display
    st.session_state.messages.append({"role": "assistant", "content": answer.response})
    with st.chat_message("assistant"):
        st.write(answer.response)
        sources =display_sources(answer)
        st.markdown(sources)
