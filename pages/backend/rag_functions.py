import streamlit as st
from langchain.document_loaders import TextLoader
from pypdf import PdfReader
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory


def read_pdf(file):
    document = ""

    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()

    return document


def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")

    return document


def split_doc(document, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)

    return split


def embedding_storing(model_name, split, create_new_vs, existing_vector_store, new_vs_name):
    if create_new_vs is not None:
        # Load embeddings instructor
        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name, model_kwargs={"device":"cuda"}
        )

        # Implement embeddings
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs == True:
            # Save db
            db.save_local("vector store/" + new_vs_name)
        else:
            # Load existing db
            load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            # Merge two DBs and save
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)

        st.success("The document has been saved.")


def prepare_rag_llm(
    token, llm_model, instruct_embeddings, vector_store_list, temperature, max_length
):
    # Load embeddings instructor
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=instruct_embeddings, model_kwargs={"device":"cuda"}
    )

    # Load db
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Load LLM
    llm = HuggingFaceHub(
        repo_id=llm_model,
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=token
    )

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Create the chatbot
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation


def generate_answer(question, token):
    answer = "An error has occured"

    if token == "":
        answer = "Insert the Hugging Face token"
        doc_source = ["no source"]
    else:
        response = st.session_state.conversation({"question": question})
        answer = response.get("answer").split("Helpful Answer:")[-1].strip()
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]

    return answer, doc_source
    