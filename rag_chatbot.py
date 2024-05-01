import streamlit as st
import os
from pages.backend import rag_functions


st.title("RAG Chatbot")

# Setting the LLM
with st.expander("Setting the LLM"):
    st.markdown("This page is used to have a chat with the uploaded documents")
    with st.form("setting"):
        row_1 = st.columns(3)
        with row_1[0]:
            token = st.text_input("Hugging Face Token", type="password")

        with row_1[1]:
            llm_model = st.text_input("LLM model", value="tiiuae/falcon-7b-instruct")

        with row_1[2]:
            instruct_embeddings = st.text_input("Instruct Embeddings", value="hkunlp/instructor-xl")

        row_2 = st.columns(3)
        with row_2[0]:
            vector_store_list = os.listdir("vector store/")
            default_choice = (
                vector_store_list.index('naruto_snake')
                if 'naruto_snake' in vector_store_list
                else 0
            )
            existing_vector_store = st.selectbox("Vector Store", vector_store_list, default_choice)
        
        with row_2[1]:
            temperature = st.number_input("Temperature", value=1.0, step=0.1)

        with row_2[2]:
            max_length = st.number_input("Maximum character length", value=300, step=1)

        create_chatbot = st.form_submit_button("Create chatbot")


# Prepare the LLM model
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if token:
    st.session_state.conversation = rag_functions.prepare_rag_llm(
        token, llm_model, instruct_embeddings, existing_vector_store, temperature, max_length
    )

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Source documents
if "source" not in st.session_state:
    st.session_state.source = []

# Display chats
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ask a question
if question := st.chat_input("Ask a question"):
    # Append user question to history
    st.session_state.history.append({"role": "user", "content": question})
    # Add user question
    with st.chat_message("user"):
        st.markdown(question)

    # Answer the question
    answer, doc_source = rag_functions.generate_answer(question, token)
    with st.chat_message("assistant"):
        st.write(answer)
    # Append assistant answer to history
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Append the document sources
    st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})


# Source documents
with st.expander("Source documents"):
    st.write(st.session_state.source)
