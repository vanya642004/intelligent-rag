import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from chromadb.config import Settings

# Hugging Face Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.set_page_config(page_title="📖 Intelligent Academic Search with RAG")
st.title("📖 Intelligent Academic Search with RAG")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.info("📡 Creating vector database...")
    all_texts = []

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)
        all_texts.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_settings = Settings(anonymized_telemetry=False, persist_directory="./chroma_db")
    vector_db = Chroma.from_documents(all_texts, embedding=embeddings, client_settings=chroma_settings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = st.text_input("🔍 Ask your academic query")
    if query:
        st.success("📖 Answer:")
        response = qa.run(query)
        st.write(response)
