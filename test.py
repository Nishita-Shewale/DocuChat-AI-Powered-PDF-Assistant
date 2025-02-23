import streamlit as st
import pdfplumber
import os
import random
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_= load_dotenv(find_dotenv())  #read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize LangChain LLM and Memory
llm_model = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(temperature=0.0)

# Main Function
def embed_and_store(pages, embeddings_model):
    # Using LangChain Pinecone to embed and store documents
    docsearch = Pinecone.from_texts(pages, embeddings_model, index_name="pdf562")
    return docsearch

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
        # Extract text and embed it using OpenAI embeddings
        with st.spinner("Processing PDF..."):
            pages = extract_text_from_pdf(uploaded_file)
            embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
            vectordb = embed_and_store(pages, embeddings_model)
            st.session_state.retriever = vectordb.as_retriever()
            st.success("PDF Processed and Stored!")
            st.session_state.pdf_processed = True

    if st.session_state.pdf_processed:
        for idx, (speaker, text) in enumerate(st.session_state.chat_history):
            if speaker == "Bot":
                message(text, key=f"msg-{idx}")
            else:
                message(text, is_user=True, key=f"msg-{idx}")

        st.text_input("Enter your question here:", key="user_input", on_change=handle_enter)

        if st.session_state.user_input:
            handle_enter()

if __name__ == "__main__":
    main()
