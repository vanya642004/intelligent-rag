import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os
# Streamlit page config
st.set_page_config(page_title="Free RAG PDF QA", page_icon="📄", layout="wide")
st.title("📄 Ask Questions to Your PDF Document")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split the PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Local model using Hugging Face (Flan-T5)
    st.info("⏳ Loading local FLAN-T5 model... (first run may take time)")
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    # RAG (Retrieval-Augmented Generation) chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User query input
    query = st.text_input("❓ Ask something about the document:")

    if query:
        with st.spinner("Thinking... 🤔"):
            answer = qa_chain.run(query)
        st.success(answer)

    # Cleanup temp file
    os.remove(tmp_path)

