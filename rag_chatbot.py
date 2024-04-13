import streamlit as st


st.title("RAG assistant")
st.markdown("This page is used to have a chat with the uploaded documents")

row_1 = st.columns(2)
with row_1[0]:
    token = st.text_input("Hugging Face Token", type="password")

with row_1[1]:
    llm_model = st.text_input("LLM model", value="tiiuae/falcon-7b-instruct")

row_2 = st.columns(2)
with row_2[0]:
    temp = st.number_input("Temperature", value=1.0, step=0.1)

with row_2[1]:
    max_length = st.number_input("Maximum character length", value=300, step=1)


def generate_answer():
    answer = "I'm under development."
    return answer 


# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chats
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question box
if question := st.chat_input("Ask a question"):
    # Append user question to history
    st.session_state.history.append({"role": "user", "content": question})
    # Add user question
    with st.chat_message("user"):
        st.markdown(question)

    # Answer the question
    with st.chat_message("assistant"):
        answer = st.write(generate_answer())
    # Add assistant answer to history
    st.session_state.history.append({"role": "assistant", "content": answer})
    