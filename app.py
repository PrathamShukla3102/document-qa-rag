"""
app.py
Streamlit front-end for Document Q&A Assistant (RAG)
"""

import streamlit as st
from rag_utils import (
    build_index_from_files,
    answer_question_rag,
)

# Page config
st.set_page_config(
    page_title="Document Q&A Assistant (RAG)",
    layout="wide"
)

st.title("Document Q&A Assistant (RAG)")
st.caption(
    "Upload documents and ask questions. "
    "Answers are generated strictly from uploaded content with citations."
)

# Session state
if "documents" not in st.session_state:
    st.session_state.documents = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "index_built" not in st.session_state:
    st.session_state.index_built = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("1️⃣ Upload & Index Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT / MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.sidebar.button("Build / Rebuild Index"):
    if not uploaded_files:
        st.sidebar.error("Please upload at least one document.")
    else:
        with st.spinner("Indexing documents..."):
            docs, index = build_index_from_files(uploaded_files)
            st.session_state.documents = docs
            st.session_state.faiss_index = index
            st.session_state.index_built = True
            st.sidebar.success("Index built successfully.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

if st.sidebar.button("Reset Knowledge Base"):
    st.session_state.documents = []
    st.session_state.faiss_index = None
    st.session_state.index_built = False
    st.session_state.messages = []
    st.sidebar.success("Knowledge base reset.")

# Chat section
st.header("2️⃣ Ask Questions")

if not st.session_state.index_built:
    st.warning("Upload documents and build the index to start asking questions.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about the uploaded documents...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer = answer_question_rag(
                    question=user_query,
                    index=st.session_state.faiss_index,
                    documents=st.session_state.documents,
                )
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
