import streamlit as st
import os
from pages.backend import rag_functions


st.title("Document Embedding")
st.markdown("This page is used to upload the documents as the custom knowledge for the chatbot.")


with st.form("document_input"):
    
    document = st.file_uploader("Knowledge Documents", type=['pdf', 'txt'], help=".pdf or .txt file")

    row_1 = st.columns([2, 1, 1])
    with row_1[0]:
        instruct_embeddings = st.text_input(
            "Model Name of the Instruct Embeddings", value="hkunlp/instructor-xl"
        )
    
    with row_1[1]:
        chunk_size = st.number_input(
            "Chunk Size", value=200, min_value=0, step=1,
        )
    
    with row_1[2]:
        chunk_overlap = st.number_input(
            "Chunk Overlap", value=10, min_value=0, step=1, help="higher that chunk size"
        )
    
    row_2 = st.columns(2)
    with row_2[0]:
        # List the existing vector stores
        vector_store_list = os.listdir("vector store/")
        vector_store_list = ["<New>"] + vector_store_list
        
        existing_vector_store = st.selectbox(
            "Vector Store to Merge the Knowledge", vector_store_list,
            help="Which vector store to add the new documents. Choose <New> to create a new vector store."
        )

    with row_2[1]:
        # List the existing vector stores     
        new_vs_name = st.text_input(
            "New Vector Store Name", value="new_vector_store_name",
            help="If choose <New> in the dropdown / multiselect box, name the new vector store. Otherwise, fill in the existing vector store to merge."
        )

    save_button = st.form_submit_button("Save vector store")

if save_button:
    # Read the uploaded file
    if document.name[-4:] == ".pdf":
        document = rag_functions.read_pdf(document)
    elif document.name[-4:] == ".txt":
        document = rag_functions.read_txt(document)
    else:
        st.error("Check if the uploaded file is .pdf or .txt")

    # Split document
    split = rag_functions.split_doc(document, chunk_size, chunk_overlap)

    # Check whether to create new vector store
    create_new_vs = None
    if existing_vector_store == "<New>" and new_vs_name != "":
        create_new_vs = True
    elif existing_vector_store != "<New>" and new_vs_name != "":
        create_new_vs = False
    else:
        st.error("Check the 'Vector Store to Merge the Knowledge' and 'New Vector Store Name'")
    
    # Embeddings and storing
    rag_functions.embedding_storing(
        instruct_embeddings, split, create_new_vs, existing_vector_store, new_vs_name
    )
