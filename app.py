import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile
import os

st.set_page_config(page_title="RAG PDF QA")
st.title("📚 Ask Your PDF ")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and chunk PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Embeddings from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Free HF model for Q&A
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.2, "max_length": 256}
    )

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("❓ Ask a question based on the PDF:")
    if query:
        answer = qa_chain.run(query)
        st.success(answer)

    os.remove(tmp_path)
