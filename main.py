# import important files
import streamlit as st
import PyPDF2

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# getting text from pdf document
def get_text(pdf_file):
    text=""
    for pdf in pdf_file:
        pdf_reader=PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# divide raw data into chunks
def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore=FAISS.from_documents(text_chunks,embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llms=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = conversational_retrieval.from_llm(
        llm=llms,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# handling query as user-input
def handle_userinput(query):
    response = st.session_state.conversation({'question': query})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(message)


def main():
    # loading environment file
    load_dotenv()
    # setup page config
    st.set_page_config(page_title="CHATBOT",
                       page_icon=":fire:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # set heading
    st.header("Chat based on your file :fire:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # sidebar 
    with st.sidebar:
        st.subheader("Your documents")
        # upload your pdf file
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

# run main file
if __name__ == '__main__':
    main()

