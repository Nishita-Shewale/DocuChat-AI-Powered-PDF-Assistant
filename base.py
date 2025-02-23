import streamlit as st
import pdfplumber
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from streamlit_chat import message
import openai
from dotenv import load_dotenv, find_dotenv
import pinecone

# Load environment variables
load_dotenv(find_dotenv())

# Initialize LangChain LLM
llm = ChatOpenAI(temperature=0.0)

#############################################
# Main Function
#############################################
def has_been_processed(file_name):
    """Check if the PDF has already been processed"""
    processed_files = set()
    if os.path.exists("processed_files.txt"):
        with open("processed_files.txt", "r") as file:
            processed_files = set(file.read().splitlines())
    return file_name in processed_files

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return pages

def embed_and_store(pages, embeddings_model):
    # Embedding the documents and storing them in Pinecone
    docsearch = Pinecone.from_texts(pages, embeddings_model, index_name="pdf562")
    return docsearch

def mark_as_processed(file_name):
    """Mark the PDF as processed."""
    with open("processed_files.txt", "a") as file:
        file.write(file_name + "\n")

def handle_enter():
    if 'retriever' in st.session_state:
        user_input = st.session_state.user_input
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Please wait..."):
                try:
                    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.retriever)
                    bot_response = qa.run(user_input)
                    st.session_state.chat_history.append(("Bot", bot_response))
                except Exception as e:
                    st.session_state.chat_history.append(("Bot", f"Error = {e}"))
            st.session_state.user_input = ""  # Clear input box

def main():
    st.title("Ask a PDF Question")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

    if uploaded_file:
        file_name = uploaded_file.name
        if not has_been_processed(file_name):
            with st.spinner("Processing PDF..."):
                pages = extract_text_from_pdf(uploaded_file)
                embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
                vectordb = embed_and_store(pages, embeddings_model)
                st.session_state.retriever = vectordb.as_retriever()
                mark_as_processed(file_name)
                st.success("PDF Processed and Stored!")
                st.session_state.pdf_processed = True
        else:
            if 'retriever' not in st.session_state:
                with st.spinner("Loading existing data..."):
                    index_name = "pdf562"
                    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                    docsearch = Pinecone.from_existing_index(index_name, embeddings)
                    st.session_state.retriever = docsearch.as_retriever()
                st.info("PDF already processed. Using existing data.")
                st.session_state.pdf_processed = True

    if st.session_state.pdf_processed:
        for idx, (speaker, text) in enumerate(st.session_state.chat_history):
            if speaker == "Bot":
                message(text, key=f"msg-{idx}")
            else:
                message(text, is_user=True, key=f"msg-{idx}")

        st.text_input("Enter your question here:", key="user_input", on_change=handle_enter)

if __name__ == "__main__":
    main()
